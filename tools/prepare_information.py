"""Extract only permitted control/single summaries from the cached CSC source."""
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, sha256, write_json
from opal.information import reference_projection, projected_variance, stratified_shuffle
from prepare_pertspectra_series import column, label, csc_columns
from measure_information_process import retained

OUT = ROOT / 'results/opal_information_v1'
LOCAL = Path('C:/Users/USER/Desktop/opal-information-20261011')


def resource_check(projected=0):
    import shutil
    old = read_json(ROOT / 'results/opal_model_series_v2/verification/final_resources.json')
    sizes = retained(list(old['physical_roots']) + [OUT, LOCAL])
    if sum(sizes.values()) + projected > 3000000000:
        raise RuntimeError('Inherited artifact cap would overflow')
    if shutil.disk_usage('C:/').free - projected < 1073741824:
        raise RuntimeError('C reserve would overflow')
    return sizes


def run():
    resource_check(120000000)
    LOCAL.mkdir(exist_ok=True)
    if list(LOCAL.iterdir()):
        raise FileExistsError('Preparation output already exists; preserve before retry')
    cfg = read_json(ROOT / 'configs/experiments/opal_information_v1.json')
    prepared = read_json(ROOT / 'results/opal_model_series_v1/pertspectra/access-v4/prepared.json')
    raw = ROOT / 'data/raw/pertspectra-model-series/inhouse.h5ad'
    if sha256(raw) != prepared['full_file_sha256']:
        raise ValueError('Raw source changed')
    records = []
    with h5py.File(raw, 'r') as f, threadpool_limits(limits=1):
        ids, contexts, targets, guides = [column(f['obs'][k]) for k in ['SampleIndex', 'Treatment', 'target_gene_name', 'feature_call']]
        assigned = [label(t)[0] if len(str(t).split('|')) == len(str(g).split('|')) else None for t, g in zip(targets, guides)]
        # Read backed sparse blocks, but discard every pair row before any statistics.
        refrows = np.asarray([i for i, p in enumerate(assigned) if p is not None and len(p) <= 1])
        names = column(f['var'][f['var'].attrs['_index']])
        matrix = f['layers/counts']
        library = np.zeros(len(refrows))
        for start in range(0, len(names), 128):
            block = csc_columns(matrix, start, min(start+128, len(names)), len(ids))[refrows]
            if not np.isfinite(block.data).all() or np.any(block.data < 0) or not np.allclose(block.data, np.rint(block.data), atol=1e-6):
                raise ValueError('Reference counts invalid')
            library += np.asarray(block.sum(1)).ravel()
        for context, info in prepared['contexts'].items():
            source = 'TNFA-positive' if context == 'TNFA+' else 'TNFA-negative'
            bankpath = Path(info['local']) / 'canonical_bank.npz'
            if sha256(bankpath) != info['bank_sha256']:
                raise ValueError('Canonical bank changed')
            with np.load(bankpath, allow_pickle=False) as z:
                bank = {k: z[k] for k in ('genes', 'singles', 'pair_ids', 'culture_ids')}
            panel = [int(np.flatnonzero(names == g)[0]) for g in bank['genes']]
            normalized = np.column_stack([csc_columns(matrix, g, g+1, len(ids))[refrows].toarray().ravel() for g in panel]).astype(np.float64)
            normalized *= (10000 / np.maximum(library, 1))[:, None]
            group = {}
            for j, i in enumerate(refrows):
                if contexts[i] == context and library[j] > 0:
                    group.setdefault((str(ids[i]), assigned[i]), []).append(j)
            shape = (len(bank['pair_ids']), 2, 8)
            arrays = {k: np.empty(shape) for k in ('means_projected', 'variance', 'shuffle', 'precision_projected', 'precision_variance', 'precision_shuffle')}
            arrays['precision_singles'] = np.empty_like(bank['singles'])
            arrays['reference_counts'] = np.empty((len(bank['pair_ids']), 2), dtype=int)
            projections = []
            source_records = []
            for stratum in sorted(set(bank['culture_ids'])):
                rows = np.flatnonzero(bank['culture_ids'] == stratum)
                controls = normalized[group[stratum, ()]]
                _, basis = reference_projection(np.log2(1+controls), 8)
                projections.append(basis)
                rng = np.random.default_rng(cfg['precision_seed'])
                cdraw = rng.choice(len(controls), cfg['precision_cells'], replace=False)
                cmean = np.log2(1+controls[cdraw].mean(0))
                names_single = sorted({t for i in rows for t in json.loads(bank['pair_ids'][i])})
                variances, precision_variances, precision_means, counts = [], [], [], []
                for t in names_single:
                    cells = normalized[group[stratum, (t,)]]
                    if len(cells) < 10:
                        raise ValueError('Original eligible single support lost')
                    draw = rng.choice(len(cells), cfg['precision_cells'], replace=False)
                    variances.append(projected_variance(np.log2(1+cells), basis))
                    precision_variances.append(projected_variance(np.log2(1+cells[draw]), basis))
                    precision_means.append(np.log2(1+cells[draw].mean(0))-cmean)
                    counts.append(len(cells))
                shuffled, bins = stratified_shuffle(np.asarray(variances), counts, cfg['shuffle_seed'])
                # All precision references have ten cells, hence one compatible bin.
                pshuffled, pbins = stratified_shuffle(np.asarray(precision_variances), np.repeat(10, len(counts)), cfg['shuffle_seed'])
                lookup = {t: i for i, t in enumerate(names_single)}
                for i in rows:
                    ix = [lookup[t] for t in json.loads(bank['pair_ids'][i])]
                    arrays['means_projected'][i] = bank['singles'][i] @ basis
                    arrays['variance'][i] = np.asarray(variances)[ix]
                    arrays['shuffle'][i] = shuffled[ix]
                    arrays['precision_singles'][i] = np.asarray(precision_means)[ix]
                    arrays['precision_projected'][i] = arrays['precision_singles'][i] @ basis
                    arrays['precision_variance'][i] = np.asarray(precision_variances)[ix]
                    arrays['precision_shuffle'][i] = pshuffled[ix]
                    arrays['reference_counts'][i] = np.asarray(counts)[ix]
                    # Verify the unchanged original aggregation on precisely these references.
                    original = np.asarray([np.log2(1+normalized[group[stratum, (t,)]].mean(0))-np.log2(1+controls.mean(0)) for t in json.loads(bank['pair_ids'][i])])
                    if not np.allclose(original, bank['singles'][i], atol=1e-10, rtol=0):
                        raise ValueError('Original mean transform parity failed')
                source_records.append({'stratum': str(stratum), 'single_targets': names_single,
                    'single_cells': list(map(int, counts)), 'control_cells': len(controls), 'shuffle_bins': bins,
                    'precision_bins': pbins, 'shared_draws': True, 'independent_cultures': None})
            arrays['projections'] = np.asarray(projections)
            arrays['pair_ids'] = bank['pair_ids']
            arrays['culture_ids'] = bank['culture_ids']
            resource_check(10000000)
            destination = LOCAL / (source + '-summaries.npz')
            np.savez_compressed(destination, **arrays)
            record = {'source': source, 'bank': str(bankpath), 'bank_sha256': sha256(bankpath),
                'summary': str(destination), 'summary_sha256': sha256(destination), 'summary_bytes': destination.stat().st_size,
                'mean_parity_atol': 1e-10, 'pair_cells_summarized': 0, 'references': source_records,
                'genes': bank['genes'].tolist(), 'units': info['units'], 'raw_sha256': prepared['full_file_sha256']}
            records.append(record)
            write_json(OUT / 'measurement' / (source + '.json'), record)
            print(source, destination.stat().st_size, flush=True)
    write_json(OUT / 'measurement/summaries.json', records)


if __name__ == '__main__':
    run()
