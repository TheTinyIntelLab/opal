"""Replay safe native-adaptation states without retraining or fitting transforms."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.artifacts import read_json, sha256, write_json
from run_native_series_aggregate import gears_fit, cpa_fit
from run_native_pertspectra_adapter import SPECTRA, column, label, csc


def run(args):
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    prepared = read_json(ROOT / 'results/opal_model_series_v1/pertspectra/access-v4/prepared.json')
    with gzip.open(args.artifacts / 'native-assets/gene2go.json.gz', 'rt') as handle:
        go = json.load(handle)
    records = []
    raw = h5py.File(ROOT / 'data/raw/pertspectra-model-series/inhouse.h5ad', 'r')
    try:
        ids, treatment, targets_raw, guides = [column(raw['obs'][k]) for k in
                                              ['SampleIndex', 'Treatment', 'target_gene_name', 'feature_call']]
        labels = np.asarray([label(t) if len(str(t).split('|')) == len(str(g).split('|')) else None
                             for t, g in zip(targets_raw, guides)], dtype=object)
        names = column(raw['var'][raw['var'].attrs.get('_index', '_index')])
        libraries = np.zeros(len(ids))
        matrix = raw['layers/counts']
        for start in range(0, len(names), 128):
            libraries += np.asarray(csc(matrix, start, min(start + 128, len(names)), len(ids)).sum(1)).ravel()
        for context, info in prepared['contexts'].items():
            source = 'TNFA-positive' if context == 'TNFA+' else 'TNFA-negative'
            with np.load(Path(info['local']) / 'canonical_bank.npz', allow_pickle=False) as archive:
                bank = {k: archive[k] for k in ['singles', 'responses', 'pair_ids', 'culture_ids', 'genes']}
            primary = read_json(ROOT / 'results/opal_model_series_v2/primary' / source / 'conditions.json')
            folds = read_json(ROOT / 'results/opal_model_series_v1/pertspectra' / source / 'scope_before_fit.json')['global_fold_assignment']
            for row in [r for r in primary if r['budget'] == 1 and r['order'] == 'uniform'
                        and r['acquisition_seed'] == 11 and r['training_seed'] == 20261017]:
                identity = f'{source}-{row["stratum"]}-{row["outer"]}'
                contract = read_json(Path(row['checkpoints'][0]['path']).parent.parent / 'contract.json')
                dev = (bank['culture_ids'] == row['stratum']) & np.isin(bank['pair_ids'], contract['acquired_pair_ids'])
                test = (bank['culture_ids'] == row['stratum']) & np.isin(bank['pair_ids'], contract['held_pair_ids'])
                targets = sorted({t for p in bank['pair_ids'][dev | test] for t in json.loads(p)})
                index = {t: i for i, t in enumerate(targets)}
                profiles = np.empty((len(targets), len(bank['genes'])))
                for pair, singles in zip(bank['pair_ids'][bank['culture_ids'] == row['stratum']],
                                         bank['singles'][bank['culture_ids'] == row['stratum']]):
                    for target, vector in zip(json.loads(pair), singles):
                        profiles[index[target]] = vector
                pairs = np.asarray([[index[t] for t in json.loads(p)] for p in bank['pair_ids'][dev]])
                query = np.asarray([[index[t] for t in json.loads(p)] for p in bank['pair_ids'][test]])
                validation = np.asarray([folds[p] == (row['outer'] + 1) % 3 for p in bank['pair_ids'][dev]])
                for method in ['GEARS', 'CPA', 'PertSpectra']:
                    directory = 'native-results' if method == 'PertSpectra' else method + '-aggregate-amended-results'
                    record = read_json(args.artifacts / directory / (identity + '.json'))
                    checkpoint = Path(record['checkpoint'])
                    filename = 'native_parameters.npz' if method == 'PertSpectra' else 'parameters.npz'
                    started = time.monotonic()
                    with np.load(checkpoint / filename, allow_pickle=False) as saved:
                        state = {k: saved[k] for k in saved.files}
                    if method == 'GEARS':
                        prediction, _, _, _ = gears_fit(pairs, bank['responses'][dev], query, profiles,
                            {'hidden_size': record['selected_width']}, 20261017, go, targets, restored_state=state)
                    elif method == 'CPA':
                        prediction, _, _, _ = cpa_fit(pairs, bank['responses'][dev], query, profiles,
                            {'width': record['selected_width'], 'splits': np.where(validation, 'valid', 'train')},
                            20261017, restored_state=state)
                    else:
                        ordered = ['__control__'] + targets
                        model = SPECTRA(X=np.zeros((1, len(bank['genes'])), np.float32), labels=None,
                            adj_matrix=np.array([]), L=record['selected_factors'], pert_idx=np.repeat(-1, len(ordered)),
                            pert_labels=ordered, use_cell_types=False, device=torch.device('cpu'))
                        model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
                        panel = np.asarray([np.flatnonzero(names == gene)[0] for gene in bank['genes']])
                        mask = (ids == row['stratum']) & (treatment == context) & (labels == '[]') & (libraries > 0)
                        control = np.asarray([np.asarray(csc(matrix, int(i), int(i) + 1, len(ids))[mask]
                            .multiply(10000 / libraries[mask, None]).mean(0)).ravel()[0] for i in panel])
                        control = np.log2(1 + control)
                        design = np.zeros((test.sum(), len(ordered)), np.float32)
                        lookup = {t: i for i, t in enumerate(ordered)}
                        for i, pair in enumerate(bank['pair_ids'][test]):
                            for target in json.loads(pair):
                                design[i, lookup[target]] = 1
                        with torch.no_grad():
                            prediction = model.loss_no_cell_types(torch.zeros((test.sum(), len(panel))),
                                torch.ones(test.sum()), torch.from_numpy(design), forward=True)[1].numpy() - control
                    with np.load(checkpoint / 'evaluator_arrays.npz', allow_pickle=False) as saved:
                        difference = float(np.max(np.abs(prediction - saved['prediction'])))
                        truth_difference = float(np.max(np.abs(bank['responses'][test] - saved['truth'])))
                    if difference > 1e-6 or truth_difference > 1e-10:
                        raise ValueError(f'{method} {identity} replay failed: {difference}, {truth_difference}')
                    records.append({'method': method, 'identity': identity, 'max_abs': difference,
                        'truth_max_abs': truth_difference, 'parameters_sha256': sha256(checkpoint / filename),
                        'checkpoint_bytes': sum(p.stat().st_size for p in checkpoint.iterdir() if p.is_file()),
                        'reconstruction_and_inference_seconds': time.monotonic() - started,
                        'replay_peak_process_rss_bytes': psutil.Process().memory_info().peak_wset,
                        'training_ram': 'Not measured in original native training; replay RSS is not training RAM'})
                    print(method, identity, difference, flush=True)
    finally:
        raw.close()
    write_json(args.output, {'records': records, 'count': len(records), 'retrained': False,
        'tolerance': 1e-6, 'reason': 'Float32 upstream GPU/CPU inference; Opal float64 replay uses 1e-10',
        'predictor_outer_outcomes': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('D:/opal-model-series-20261010-v2-continuation'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
