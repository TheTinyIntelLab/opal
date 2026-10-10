"""Source-qualified shared-contract fits on existing genetic development banks."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import fingerprint, read_json, sha256, write_json
from opal.series import OpalGuard, OpalBase, Unavailable, subset, freeze_predictions
from opal.replicated.retrospective import pair_folds
from run_model_series_bilinear import variants, metrics, resource_check


def run(local, output):
    output.mkdir(parents=True, exist_ok=True)
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    menu = read_json(ROOT / 'configs/experiments/replicated_fixed_state_v1.json')['menu']
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for name in ['norman-log2', 'wessels-log2', 'norman-count', 'wessels-count']:
            source = ROOT / 'data/prepared' / name
            directory = output / name
            directory.mkdir(exist_ok=True)
            preprocessing = read_json(source / 'preprocessing.json')
            with np.load(source / 'profiles.npz', allow_pickle=False) as f:
                profiles, pairs, responses, targets, genes = [f[k] for k in ('singles', 'pairs', 'doubles', 'targets', 'genes')]
            singles = profiles[pairs].copy()
            for i, pair in enumerate(pairs):
                singles[i] = singles[i, np.argsort([str(targets[t]) for t in pair])]
            bank = {'singles': singles, 'responses': responses.astype(float),
                    'pair_ids': np.asarray([json.dumps(sorted([str(targets[a]), str(targets[b])]), separators=(',', ':')) for a, b in pairs]),
                    'culture_ids': np.repeat(name, len(pairs))}
            # Only the coherent historical DEVELOPMENT bank is adapted; the
            # protected original evaluator bank is not opened or relabeled.
            assignment = pair_folds(bank['pair_ids'], settings['outer_seed'])
            freeze = {'source': name, 'profiles_sha256': sha256(source / 'profiles.npz'),
                      'preprocessing_sha256': sha256(source / 'preprocessing.json'), 'gene_order_sha256': fingerprint(genes.tolist()),
                      'source_genes': len(genes), 'pairs': len(pairs), 'scale': preprocessing['scale'],
                      'normalizer': preprocessing['normalizer'], 'panel': 'unchanged coherent development-bank genes',
                      'exposure': 'Previously scored retrospective development; original protected evaluator bank untouched',
                      'fold_assignment': {p: int(f) for p, f in zip(bank['pair_ids'], assignment)},
                      'configuration_sha256': sha256(ROOT / 'configs/experiments/opal_model_series_v2.json'),
                      'information_tier': 'historical equal-stratum aggregate observed singles',
                      'historical_query_labels_accessed': 0, 'runner_sha256': sha256(Path(__file__)),
                      'single_order': 'Each single vector aligned with its sorted canonical target identity'}
            scope_path = directory / 'scope_before_fit.json'
            if scope_path.exists() and read_json(scope_path) != freeze:
                raise ValueError('Refuse changed retrospective scope; use a new output directory')
            write_json(scope_path, freeze)
            records = []
            for outer in range(3):
                if (directory / f'fold-{outer}.json').exists():
                    records.append(read_json(directory / f'fold-{outer}.json'))
                    continue
                work = local / name / str(outer)
                resource_check(local, output, settings)
                work.mkdir(parents=True, exist_ok=False)
                train, test = assignment != outer, assignment == outer
                development = subset(bank, train)
                development['partitions'] = np.where(assignment[train] == (outer + 1) % 3, 'validation', 'train')
                query = {k: v[test] for k, v in bank.items() if k != 'responses'}
                contract = {'genes': genes.tolist(), 'units': 'normalized_count_change' if preprocessing['scale'] == 'count' else 'deposited_log_change',
                            'source': name, 'assay': 'historical library-normalized genetic aggregate',
                            'reference_provenance': freeze['profiles_sha256'], 'split_lineage': fingerprint(freeze),
                            'held_pair_ids': bank['pair_ids'][test].tolist(), 'acquired_pair_ids': bank['pair_ids'][train].tolist(),
                            'checkpoint_role': 'frozen_retrospective_evaluation', 'information_tier': freeze['information_tier'],
                            'context_contract': name, 'preprocessing': {k: v for k, v in preprocessing.items() if k != 'gene_mask'}}
                write_json(work / 'contract.json', contract)
                started, cpu = time.monotonic(), time.process_time()
                record = {'outer': outer, 'source': name, 'unique_labels': int(train.sum()), 'pair_stratum_labels': int(train.sum()),
                          'held_pairs': int(test.sum()), 'metrics': {}, 'unavailable': {}, 'checkpoints': []}
                predictions = {'no-change': np.zeros_like(bank['responses'][test]), 'additivity': query['singles'].sum(1)}
                try:
                    guard = OpalGuard(settings, menu).fit(development, contract)
                    for label, model in variants(guard).items():
                        checkpoint = model.save(work / label / 'checkpoint')
                        result = freeze_predictions(work / label / 'freeze', model, query, contract)
                        restored = OpalBase.load(checkpoint)
                        difference = float(np.max(np.abs(result['predictions'] - restored.predict(query, contract)['predictions'])))
                        if difference > settings['replay_atol']:
                            raise ValueError('Retrospective checkpoint replay failed')
                        predictions[label] = result['predictions']
                        record['checkpoints'].append({'model': label, 'path': str(checkpoint), 'replay_max_abs': difference,
                            'model_sha256': sha256(checkpoint / 'model.json'), 'parameters_sha256': sha256(checkpoint / 'parameters.npz')})
                except Unavailable as error:
                    record['unavailable'] = {label: str(error) for label in ['Opal-Base', 'Opal-Interaction', 'Opal-Guard']}
                base = predictions.get('Opal-Base', predictions['no-change'])
                record['metrics'] = {label: metrics(values, bank['responses'][test], base) for label, values in predictions.items()}
                record.update(wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu,
                              peak_rss_bytes=psutil.Process().memory_info().peak_wset)
                write_json(directory / f'fold-{outer}.json', record)
                records.append(record)
                print(name, outer, {k: round(v['mse'], 6) for k, v in record['metrics'].items()}, flush=True)
            write_json(directory / 'results.json', {'records': records, 'role': freeze['exposure']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', type=Path, default=Path('C:/Users/USER/Desktop/opal-model-series-20261010-v2/retrospective'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/opal_model_series_v2/retrospective')
    args = parser.parse_args()
    run(args.local, args.output)
