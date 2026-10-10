"""Globally held-pair transport across source SampleIndex labels, never cultures."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import fingerprint, read_json, sha256, write_json
from opal.replicated.retrospective import pair_folds
from opal.series import OpalBase, OpalGuard, evaluate_predictions, freeze_predictions, subset
from run_model_series import extract_experts


def run(args):
    prepared = read_json(args.preparation / 'prepared.json')
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v1.json')
    menu = read_json(ROOT / 'configs/experiments/replicated_fixed_state_v1.json')['menu']
    banks = {}
    for context, info in prepared['contexts'].items():
        with np.load(Path(info['local']) / 'canonical_bank.npz', allow_pickle=False) as f:
            banks[context] = {k: f[k] for k in ('singles', 'responses', 'pair_ids', 'culture_ids')}
    universe = sorted(set(p for bank in banks.values() for p in bank['pair_ids']))
    assignment = dict(zip(universe, pair_folds(universe, settings['outer_seed'])))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'scope_before_fit.json', {'utc': datetime.now(timezone.utc).isoformat(),
               'comparison_budget': '24 complete fits: two separate TNF contexts, four held sample labels and three global outer folds; unchanged finite model menu',
               'code_sha256': sha256(Path(__file__)), 'model_code_sha256': sha256(ROOT / 'src/series.py'),
               'configuration': settings, 'source_full_file_sha256': prepared['full_file_sha256'],
               'task': 'Transport across source sample labels within TNF context. Exclude query pair from every training sample, use query sample controls/singles. Equal training-stratum weights.',
               'guard_policy': 'Unknown sample context forces declared Base fallback; not calibrated uncertainty',
               'independent_culture_claim': False, 'TNF_state_transfer': False})
    torch.set_num_threads(1)
    started, records = time.monotonic(), []
    with threadpool_limits(limits=1):
        for context, bank in banks.items():
            info = prepared['contexts'][context]
            folds = np.asarray([assignment[p] for p in bank['pair_ids']])
            for sample in info['strata']:
                for outer in range(3):
                    test = (bank['culture_ids'] == sample) & (folds == outer)
                    mask = (bank['culture_ids'] != sample) & (folds != outer)
                    held = sorted(set(bank['pair_ids'][test]))
                    if set(bank['pair_ids'][mask]) & set(held):
                        raise ValueError('Transport global pair leakage')
                    development = subset(bank, mask)
                    development['partitions'] = np.where(folds[mask] == (outer + 1) % 3, 'validation', 'train')
                    query = {k: bank[k][test] for k in ('singles', 'pair_ids', 'culture_ids')}
                    identity = fingerprint({'context': context, 'sample': sample, 'fold': outer, 'held': held})
                    contract = {'source': info['source'], 'assay': 'PertSpectra deposited gene-label-conditional aggregate',
                                'units': info['units'], 'genes': info['genes'], 'reference_provenance': prepared['full_file_sha256'],
                                'split_lineage': identity, 'held_pair_ids': held, 'acquired_pair_ids': sorted(set(development['pair_ids'])),
                                'checkpoint_role': 'sample_label_transport_evaluation'}
                    guard = OpalGuard(settings, menu).fit(development, contract)
                    models = extract_experts(guard)
                    work = Path(info['local']) / 'transport' / identity
                    work.mkdir(parents=True, exist_ok=False)
                    write_json(work / 'contract.json', contract)
                    row = {'context': context, 'held_sample': sample, 'outer': outer,
                           'training_strata': sorted(set(development['culture_ids'])), 'training_pair_ids': contract['acquired_pair_ids'],
                           'held_pair_ids': held, 'test_pairs': int(test.sum()), 'metrics': {}, 'fallback_counts': {}, 'checkpoints': []}
                    for name, model in models.items():
                        result = freeze_predictions(work / name / 'freeze', model, query, contract)
                        score = evaluate_predictions(work / name / 'freeze', query, contract, bank['responses'][test])
                        checkpoint = model.save(work / name / 'checkpoint')
                        difference = float(np.max(np.abs(OpalBase.load(checkpoint).predict(query, contract)['predictions'] - result['predictions'])))
                        if difference > settings['replay_atol']:
                            raise ValueError('Transport replay mismatch')
                        row['metrics'][name] = score['mse']
                        row['fallback_counts'][name] = int((~result['support']).sum())
                        row['checkpoints'].append({'model': name, 'path': str(checkpoint), 'model_sha256': sha256(checkpoint / 'model.json'),
                                                   'parameters_sha256': sha256(checkpoint / 'parameters.npz'), 'replay_max_abs': difference})
                    row['metrics']['additivity'] = float(np.mean((query['singles'].sum(1) - bank['responses'][test]) ** 2))
                    row['metrics']['no-change'] = float(np.mean(bank['responses'][test] ** 2))
                    records.append(row)
                    write_json(args.output / 'progress.json', {'completed': len(records), 'elapsed_seconds': time.monotonic() - started})
                    print(f'Transport {context}/{sample}/{outer}', flush=True)
    summary = {}
    for context in banks:
        per_sample = []
        for sample in prepared['contexts'][context]['strata']:
            rows = [r for r in records if r['context'] == context and r['held_sample'] == sample]
            per_sample.append({'sample': sample, 'metrics': {n: float(np.average([r['metrics'][n] for r in rows],
                              weights=[r['test_pairs'] for r in rows])) for n in rows[0]['metrics']}})
        summary[context] = {'per_sample': per_sample, 'equal_sample_mse': {n: float(np.mean([r['metrics'][n] for r in per_sample])) for n in per_sample[0]['metrics']}}
    write_json(args.output / 'report.json', {'records': records, 'contexts': summary,
               'scope': 'source sample-label transport only, not independent cultures or TNF-state transfer',
               'nonlinear_variants': 'Both use documented Base fallback in a query SampleIndex absent from training',
               'wall_seconds': time.monotonic() - started, 'replay_max_abs': 0.})
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, default=ROOT / 'results/opal_model_series_v1/pertspectra/access-v4')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/opal_model_series_v1/pertspectra/sample-label-transport')
    run(parser.parse_args())
