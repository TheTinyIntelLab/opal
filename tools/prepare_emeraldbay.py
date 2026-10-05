"""Download a pinned bounded release and prepare observed-single chemical transfer."""
import argparse
import ast
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, write_json, sha256
from opal.contextual.contracts import ReferenceBank, SCHEMA, prepare_dataset
from tools.download import download


def condition(value):
    rows = ast.literal_eval(value)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Invalid treatment identity')
    result = []
    for row in rows:
        if len(row) != 3 or not isinstance(row[0], str) or row[2] != 'uM' or not np.isfinite(float(row[1])) or float(row[1]) < 0:
            raise ValueError('Invalid compound/dose/unit')
        result.append((row[0], float(row[1]), row[2]))
    if len(set(r[0] for r in result)) != len(result):
        raise ValueError('Repeated compound')
    return tuple(sorted(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/datasets/emeraldbay-pilot-v1.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = read_json(args.config)
    raw = ROOT / 'data/raw/emeraldbay'
    base = f"https://huggingface.co/datasets/{config['repository']}/resolve/{config['revision']}/"
    import urllib.request, json
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{config['repository']}/tree/{config['revision']}?recursive=true&limit=1000"))
    entries = {r['path']: r for r in tree if r['type'] == 'file'}
    names = ['README.md', 'LICENSE.md', 'metadata/gene_metadata.parquet', 'metadata/cell_line_metadata.parquet']
    names += [f'expression_data/train-{i:05d}-of-00116.parquet' for i in config['shards']]
    hashes = {}
    for name in names:
        path = raw / name
        if not path.exists():
            result = download(base + name, path)
            if result['status'] != 'downloaded_not_accepted':
                raise RuntimeError(f'Download failed: {name}: {result.get("error")}')
        digest = sha256(path)
        expected = entries[name].get('lfs', {}).get('oid')
        if path.stat().st_size != entries[name]['size'] or expected and digest != expected:
            raise ValueError(f'Publisher size/hash mismatch: {name}')
        hashes[name] = digest
    paths = [raw / n for n in names if n.startswith('expression_data/')]
    counts, identities = Counter(), set()
    for path in paths:
        metadata = pq.read_table(path, columns=['drugname_drugconc', 'cell_line', 'sample', 'BARCODE_SUB_LIB_ID']).to_pandas()
        for row in metadata.itertuples(index=False, name=None):
            treatment, context, sample, barcode = row
            key = (sample, barcode)
            if key in identities:
                raise ValueError('Duplicate sample/cell barcode across pilot shards')
            identities.add(key)
            counts[context, condition(treatment)] += 1
    minimum = config['minimum_cells']
    controls = (('DMSO_TF', 0., 'uM'),)
    singles = sorted({c[0] for (_, c), n in counts.items() if len(c) == 1 and not c[0][0].startswith('DMSO')})
    single_set = set(singles)
    pairs = sorted({c for (_, c), n in counts.items() if len(c) == 2 and all(s in single_set for s in c)})
    coverage = {}
    for context in sorted({k[0] for k in counts}):
        eligible = [pair for pair in pairs if counts[context, pair] >= minimum and all(counts[context, (s,)] >= minimum for s in pair)]
        if counts[context, controls] >= minimum and len(eligible) >= 3:
            coverage[context] = eligible
    contexts = sorted(sorted(coverage, key=lambda c: (-len(coverage[c]), c))[:config['maximum_contexts']])
    if len(contexts) < 3:
        write_json(raw / 'pilot_feasibility.json', {'qualified': False, 'coverage': {c: len(v) for c, v in coverage.items()}, 'hashes': hashes})
        raise ValueError('Pilot needs three contexts with controls, exact-dose singles and at least three pairs')
    selected = sorted({s for c in contexts for pair in coverage[c] for s in pair})
    groups = [(c, controls) for c in contexts] + [(c, (s,)) for c in contexts for s in selected if counts[c, (s,)] >= minimum]
    groups += [(c, pair) for c in contexts for pair in coverage[c]]
    group_map = {k: i for i, k in enumerate(groups)}
    vocabulary = pd.read_parquet(raw / 'metadata/gene_metadata.parquet')
    token_count = int(vocabulary.token_id.max()) + 1
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This pilot explicitly requires the authorized CUDA environment')
    print('CUDA device:', torch.cuda.get_device_name(), 'contexts:', contexts, 'eligible rows:', len(groups), flush=True)
    # First pass sees vehicle-control expression only. The second pass uses the
    # frozen feature set; fitting/calibration/evaluation are isolated afterward.
    control_score = np.zeros(token_count, dtype=np.float64)
    selected_tokens = None
    means = None
    used_counts = np.zeros(len(groups), dtype=np.int64)
    for phase in ['control_features', 'profiles']:
        accumulated = torch.zeros((len(groups), token_count if selected_tokens is None else len(selected_tokens)), device='cuda', dtype=torch.float64) if phase == 'profiles' else None
        lookup = np.full(token_count, -1, dtype=np.int64)
        if selected_tokens is not None:
            lookup[selected_tokens] = np.arange(len(selected_tokens))
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                rows = batch.to_pylist()
                active = [(row, group_map.get((row['cell_line'], condition(row['drugname_drugconc'])))) for row in rows]
                active = [(r, g) for r, g in active if g is not None and (phase == 'profiles' or groups[g][1] == controls)]
                if not active:
                    continue
                width = token_count if selected_tokens is None else len(selected_tokens)
                dense = torch.zeros((len(active), width), device='cuda', dtype=torch.float64)
                for i, (row, group) in enumerate(active):
                    tokens = np.asarray(row['genes'], dtype=np.int64)
                    values = np.asarray(row['expressions'], dtype=np.float64)
                    if len(tokens) != len(values) or len(np.unique(tokens)) != len(tokens) or np.any(tokens < 0) or np.any(tokens >= token_count) or not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
                        raise ValueError('Invalid sparse raw counts')
                    normalized = values * (10000. / values.sum())
                    mapped = tokens if selected_tokens is None else lookup[tokens]
                    keep = mapped >= 0
                    dense[i, torch.as_tensor(mapped[keep], device='cuda')] = torch.as_tensor(normalized[keep], device='cuda')
                dense = torch.log2(1 + dense)
                if phase == 'control_features':
                    control_score += dense.sum(0).cpu().numpy()
                else:
                    indices = np.asarray([g for _, g in active], dtype=np.int64)
                    accumulated.index_add_(0, torch.as_tensor(indices, device='cuda'), dense)
                    np.add.at(used_counts, indices, 1)
        if phase == 'control_features':
            eligible_tokens = vocabulary.loc[vocabulary.ensembl_id.notna(), 'token_id'].to_numpy(dtype=int)
            selected_tokens = np.sort(eligible_tokens[np.argsort(-control_score[eligible_tokens], kind='stable')[:config['genes']]])
        else:
            means = accumulated.cpu().numpy() / used_counts[:, None]
        print('Completed', phase, flush=True)
    gene_map = vocabulary.set_index('token_id').ensembl_id
    genes = np.asarray([str(gene_map.loc[i]) for i in selected_tokens])
    if len(set(genes)) != len(genes):
        raise ValueError('Selected Ensembl identifiers are not unique')
    control_profiles = np.array([means[group_map[c, controls]] for c in contexts])
    effects = np.zeros((len(contexts), len(selected), len(genes)))
    available = np.zeros(effects.shape[:2], dtype=bool)
    for ci, c in enumerate(contexts):
        for ti, target in enumerate(selected):
            if (c, (target,)) in group_map:
                effects[ci, ti] = means[group_map[c, (target,)]] - control_profiles[ci]
                available[ci, ti] = True
    labels = np.asarray([f'{s[0]}@{s[1]:g}{s[2]}:5d' for s in selected])
    metadata = {'schema': SCHEMA, 'scale': 'log2', 'synthetic': False,
                'reference_access': 'controls_and_measured_singles_all_declared_contexts',
                'context_features_source': 'controls_and_metadata', 'feature_names': genes.tolist(),
                'context_records': {c: {'biological_state': c, 'modality': 'chemical', 'assay': 'MOSAIC_scRNAseq', 'study': 'EmeraldBay', 'source': base} for c in contexts},
                'intervention_records': {label: {'canonical_target': s[0], 'reagent': s[0], 'modality': 'chemical', 'dose': f'{s[1]} {s[2]}', 'time': '5 days'} for label, s in zip(labels, selected)},
                'normalization': 'Per-cell library size 10000, log2(1+x), condition cell mean minus matched vehicle mean',
                'replication': 'Cells pooled within context/condition across bounded shards; no independent biological replicate claim',
                'source_revision': config['revision'], 'source_hashes': hashes, 'config_sha256': sha256(args.config),
                'license': 'CC-BY-4.0; attribute Tahoe Bio EmeraldBay', 'scope': config['scope']}
    bank = ReferenceBank({'singles': effects, 'available': available, 'controls': control_profiles, 'features': control_profiles,
                          'targets': labels, 'genes': genes, 'contexts': np.asarray(contexts)}, metadata)
    pair_rows, context_rows, responses = [], [], []
    for ci, c in enumerate(contexts):
        for pair in coverage[c]:
            pair_rows.append([selected.index(s) for s in pair]); context_rows.append(ci)
            responses.append(means[group_map[c, pair]] - control_profiles[ci])
    held = contexts[-2:] if len(contexts) >= 5 else contexts[-1:]
    prepare_dataset(args.output, bank, np.asarray(pair_rows), np.asarray(context_rows), np.asarray(responses), held_contexts=held,
                    calibration_fraction=config['calibration_fraction'], seed=config['seed'])
    write_json(args.output / 'source_audit.json', {'contexts': contexts, 'held_contexts': held, 'matched_pair_conditions': len({tuple(p) for p in pair_rows}),
               'combination_context_rows': len(responses), 'source_cells': len(identities), 'condition_cell_counts': [{'context': c, 'condition': list(condition), 'cells': int(used_counts[i])} for i, (c, condition) in enumerate(groups)],
               'cuda_device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
               'hashes': hashes, 'biological_confirmation': False})
    print('Prepared', args.output, flush=True)


if __name__ == '__main__':
    main()
