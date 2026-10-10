"""Independent saved-bank replay and access audit of every external checkpoint."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import fingerprint, read_json, sha256, write_json
from opal.replicated.models import shuffled_profiles
from opal.series import OpalBase, acquisition_order, evaluate_predictions


def verify(args):
    started = time.monotonic()
    prepared = read_json(args.preparation / 'prepared.json')
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v1.json')
    torch.set_num_threads(1)
    results = {}
    with threadpool_limits(limits=1):
        for context, info in prepared['contexts'].items():
            directory = args.output / ('TNFA-positive' if context == 'TNFA+' else 'TNFA-negative')
            scope = read_json(directory / 'scope_before_fit.json')
            bank_path = Path(info['local']) / 'canonical_bank.npz'
            if sha256(bank_path) != scope['input_bank_sha256']:
                raise ValueError('External bank hash changed')
            with np.load(bank_path, allow_pickle=False) as f:
                bank = {k: f[k] for k in ('singles', 'responses', 'pair_ids', 'culture_ids')}
                genes = f['genes'].tolist()
            if genes != scope['endpoint']:
                raise ValueError('External gene endpoint changed')
            assignments = scope['global_fold_assignment']
            folds = np.asarray([assignments[p] for p in bank['pair_ids']])
            inputs = {'aligned': bank, **{f'shuffle-{s}': shuffled_profiles(bank, s) for s in settings['sampling_seeds']}}
            conditions = read_json(directory / 'conditions.json')
            if len(conditions) != 4 * 3 * 2 * 3 * 4 + 3 * 4 * 3:
                raise ValueError('Missing external budget/seed/order/stratum conditions')
            manifests = {r['path']: r for r in read_json(directory / 'checkpoint_manifest.json')}
            checked, models, max_abs, unavailable = set(), 0, 0., 0
            for row in conditions:
                test = (bank['culture_ids'] == row['run']) & (folds == row['outer'])
                dev = (bank['culture_ids'] == row['run']) & (folds != row['outer'])
                order = acquisition_order(bank['pair_ids'][dev], row['order'], row['sampling_seed'])
                acquired = order[:int(row['budget'] * len(order))]
                held = sorted(set(bank['pair_ids'][test]))
                if (row['labels_acquired'] != len(acquired) or row['acquired_pair_hash'] != fingerprint(sorted(acquired))
                        or row['test_pair_hash'] != fingerprint(held) or set(acquired) & set(held)):
                    raise ValueError('External nested budget/global holdout changed')
                if row['fit_id'] in checked:
                    continue
                checked.add(row['fit_id'])
                work = Path(info['local']) / 'evaluation' / row['fit_id']
                receipt = read_json(work / 'receipt.json')
                if row['metrics'] != receipt['metrics'] or row['unavailable'] != receipt['unavailable']:
                    raise ValueError('External compact receipt mismatch')
                unavailable += len(row['unavailable'])
                if row['budget'] == 0 and set(row['metrics']) != {'no-change', 'additivity'}:
                    raise ValueError('Invalid external zero-budget result')
                contract = read_json(work / 'contract.json')
                if set(contract['acquired_pair_ids']) != set(acquired) or set(contract['held_pair_ids']) != set(held):
                    raise ValueError('External access contract mismatch')
                query = {k: inputs[row['variant']][k][test] for k in ('singles', 'pair_ids', 'culture_ids')}
                for name in ('Opal-Base', 'Opal-Interaction', 'Opal-Guard'):
                    if name not in receipt['metrics']:
                        continue
                    checkpoint = work / name / 'checkpoint'
                    manifest = manifests[str(checkpoint)]
                    if sha256(checkpoint / 'model.json') != manifest['model_sha256'] or sha256(checkpoint / 'parameters.npz') != manifest['parameters_sha256']:
                        raise ValueError('Checkpoint differs from external manifest')
                    model = OpalBase.load(checkpoint)
                    if set(model.access['fitting_and_selection_pair_ids']) != set(acquired):
                        raise ValueError('Unacquired external labels entered fitting')
                    for record in getattr(model, 'crossfit_records', []):
                        if set(record['validation_pair_ids']) & set(record['train_pair_ids']):
                            raise ValueError('External Guard cross-fit leakage')
                        if set(record['expert_access']['fitting_and_selection_pair_ids']) != set(record['train_pair_ids']):
                            raise ValueError('Guard cross-fit selection access mismatch')
                    prediction = model.predict(query, contract)['predictions']
                    with np.load(work / name / 'freeze/predictions.npz', allow_pickle=False) as f:
                        difference = float(np.max(np.abs(prediction - f['predictions'])))
                    if difference > settings['replay_atol']:
                        raise ValueError('External prediction replay mismatch')
                    score = evaluate_predictions(work / name / 'freeze', query, contract, bank['responses'][test])
                    if abs(score['mse'] - receipt['metrics'][name]) > 1e-12:
                        raise ValueError('External score replay mismatch')
                    max_abs = max(max_abs, difference)
                    models += 1
            result = {'status': 'verified', 'conditions': len(conditions), 'unique_fits': len(checked),
                      'checkpoints_replayed': models, 'prediction_max_abs': max_abs,
                      'unavailable_unique_fit_model_records': unavailable,
                      'all_canonical_holdouts_and_budget_prefixes_checked': True,
                      'all_guard_crossfit_access_checked': True, 'full_file_sha256': prepared['full_file_sha256'],
                      'gene_order_and_units_hash_bound': True, 'source': info['source'],
                      'verification_code_sha256': sha256(Path(__file__)), 'wall_seconds_cumulative': time.monotonic() - started,
                      'independent_cultures_claimed': False}
            write_json(directory / 'independent_replay_verification.json', result)
            results[context] = result
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, default=ROOT / 'results/opal_model_series_v1/pertspectra/access-v4')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/opal_model_series_v1/pertspectra')
    verify(parser.parse_args())
