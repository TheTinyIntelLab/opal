"""Prepare matched-dose independent-study evidence with lineage-wide isolation."""
import argparse
import re
from pathlib import Path
import sys
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, write_json, sha256, load_npz
from opal.contextual.contracts import ReferenceBank, SCHEMA, prepare_dataset


def treatment(name):
    if re.fullmatch(r'CONTROL_CD3_[123]', name):
        return ()
    match = re.fullmatch(r'([A-Za-z]+)_([0-9]+(?:p[0-9]+)?)_([A-Za-z]+)_([0-9]+(?:p[0-9]+)?)_CD3(?:_[12])?', name)
    if not match:
        raise ValueError(f'Unmapped stimulated treatment: {name}')
    a, da, b, db = match.groups()
    return tuple(sorted((drug, float(dose.replace('p', '.'))) for drug, dose in [(a, da), (b, db)] if float(dose.replace('p', '.')) > 0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/datasets/dspin-confirmation-v1.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = read_json(args.config)
    raw = Path(config['raw_path'])
    if sha256(raw) != config['sha256']:
        raise ValueError('Independent source checksum mismatch')
    import anndata as ad
    data = ad.read_h5ad(raw, backed='r')
    obs = data.obs
    condition = {str(n): treatment(str(n)) for n in obs.loc[obs.CD3 == 1, 'sample_id'].unique()}
    row_keys = [(str(r.cell_type_coarse) + '@' + str(r.batch), condition[str(r.sample_id)]) if int(r.CD3) == 1 else None
                for r in obs.itertuples()]
    counts = Counter(k for k in row_keys if k is not None)
    minimum = config['minimum_cells']
    singles = sorted({key[1][0] for key, count in counts.items() if len(key[1]) == 1})
    pairs = sorted({key[1] for key, count in counts.items() if len(key[1]) == 2})
    coverage = {}
    for context in sorted({k[0] for k in counts}):
        eligible = [pair for pair in pairs if counts[context, pair] >= minimum and all(counts[context, (s,)] >= minimum for s in pair)]
        if counts[context, ()] >= minimum and len(eligible) >= 3:
            coverage[context] = eligible
    contexts = sorted(coverage)
    held = [c for c in contexts if c.split('@')[0] in config['held_lineages']]
    if not held or len({c.split('@')[0] for c in contexts if c not in held}) < 2:
        raise ValueError('Lineage-wide evaluation lacks source or query coverage')
    selected = sorted({s for c in contexts for pair in coverage[c] for s in pair})
    groups = [(c, ()) for c in contexts] + [(c, (s,)) for c in contexts for s in selected if counts[c, (s,)] >= minimum]
    groups += [(c, pair) for c in contexts for pair in coverage[c]]
    mapping = {k: i for i, k in enumerate(groups)}
    ids = data.var.gene_ENSG.astype(str).to_numpy()
    reference = load_npz(ROOT / config['gene_reference'])['genes']
    idmap = {g: i for i, g in enumerate(ids)}
    genes = np.asarray([g for g in reference if g in idmap])
    columns = np.asarray([idmap[g] for g in genes])
    if len(genes) < 500:
        raise ValueError('Insufficient overlap with frozen gene vocabulary')
    # Freeze identity decisions and gene order before inspecting combinations.
    audit = {'config': config, 'config_sha256': sha256(args.config), 'contexts': contexts, 'held_contexts': held,
             'coverage': {c: len(v) for c, v in coverage.items()}, 'genes': len(genes), 'source_cells': data.n_obs,
             'matched_control': 'Same CD3 status, lineage and batch', 'biological_replication_claim': False,
             'post_treatment_lineage_annotations': True, 'source_sha256': config['sha256']}
    write_json(ROOT / 'results/next/dspin-preparation-v1.json', audit)
    sums = np.zeros((len(groups), len(genes)))
    used = np.zeros(len(groups), dtype=np.int64)
    for start in range(0, data.n_obs, 2048):
        end = min(start + 2048, data.n_obs)
        chosen = [(i, mapping[k]) for i, k in enumerate(row_keys[start:end]) if k in mapping]
        if not chosen:
            continue
        block = data.X[start:end].tocsr()
        row_indices, group_indices = np.asarray(chosen).T
        total = np.asarray(block.sum(1)).ravel()[row_indices]
        if np.any(total <= 0) or not np.isfinite(block.data).all() or np.any(block.data < 0):
            raise ValueError('Invalid raw counts')
        values = block[row_indices][:, columns].toarray().astype(np.float64)
        values = np.log2(1 + values * (10000. / total[:, None]))
        np.add.at(sums, group_indices, values)
        np.add.at(used, group_indices, 1)
    data.file.close()
    if np.any(used < minimum):
        raise ValueError('Expression and metadata cell counts disagree')
    means = sums / used[:, None]
    controls = np.asarray([means[mapping[c, ()]] for c in contexts])
    effects = np.zeros((len(contexts), len(selected), len(genes)))
    available = np.zeros(effects.shape[:2], dtype=bool)
    for ci, c in enumerate(contexts):
        for ti, intervention in enumerate(selected):
            if (c, (intervention,)) in mapping:
                effects[ci, ti] = means[mapping[c, (intervention,)]] - controls[ci]
                available[ci, ti] = True
    labels = np.asarray([f'{drug}@{dose:g}nM:source_endpoint' for drug, dose in selected])
    metadata = {'schema': SCHEMA, 'scale': 'log2', 'synthetic': False,
                'reference_access': 'controls_and_measured_singles_all_declared_contexts', 'context_features_source': 'controls_and_metadata',
                'feature_names': genes.tolist(),
                'context_records': {c: {'biological_state': c.split('@')[0], 'selection_group': c.split('@')[0],
                    'batch': c.split('@')[1], 'stimulation': 'CD3', 'modality': 'chemical', 'assay': 'D_SPIN_scRNAseq',
                    'study': 'D_SPIN_dosage_combinations', 'source': config['source']} for c in contexts},
                'intervention_records': {label: {'canonical_target': drug, 'reagent': drug, 'dose': f'{dose:g} nM',
                    'modality': 'chemical', 'time': 'Shared source dosage-combination endpoint; not used for time transfer'} for label, (drug, dose) in zip(labels, selected)},
                'normalization': 'Per-cell library 10000, log2(1+x), matched batch/CD3/lineage vehicle mean subtraction',
                'source_sha256': config['sha256'], 'gene_policy': 'Intersection with frozen EmeraldBay control-selected gene vocabulary',
                'limitations': ['Published coarse lineage labels are post-treatment', 'Batch is not a biological replicate',
                               'Dosage pairs share compound identities', 'No cross-assay fitted-weight pooling']}
    bank = ReferenceBank({'singles': effects, 'available': available, 'controls': controls, 'features': controls,
                          'contexts': np.asarray(contexts), 'targets': labels, 'genes': genes}, metadata)
    pr, cr, responses = [], [], []
    for ci, c in enumerate(contexts):
        for pair in coverage[c]:
            pr.append([selected.index(s) for s in pair]); cr.append(ci)
            responses.append(means[mapping[c, pair]] - controls[ci])
    prepare_dataset(args.output, bank, np.asarray(pr), np.asarray(cr), np.asarray(responses), held_contexts=held,
                    calibration_fraction=config['calibration_fraction'], seed=config['seed'])
    write_json(args.output / 'source_audit.json', {**audit, 'pair_context_observations': len(pr),
               'matched_pair_conditions': len(set(map(tuple, pr))), 'used_cell_counts': used.tolist()})
    print('Prepared', args.output, 'contexts', len(contexts), 'genes', len(genes), 'pair/context rows', len(pr), flush=True)


if __name__ == '__main__':
    main()
