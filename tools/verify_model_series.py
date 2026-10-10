"""Replay every budget fit and verify source, split, selection and score records."""
import argparse
import copy
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
from opal.replicated.retrospective import pair_folds
from opal.series import (OpalBase, OpalGuard, acquisition_order, evaluate_predictions, subset)
from run_model_series import extract_experts


def verify(args):
    started = time.monotonic()
    compact, local = args.output, args.local
    conditions = read_json(compact / 'conditions.json')
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v1.json')
    menu = read_json(ROOT / 'configs/experiments/replicated_fixed_state_v1.json')['menu']
    manifest = read_json(compact / 'input_manifest.json')
    if sha256(local / 'canonical_bank.npz') != manifest['bank_sha256']:
        raise ValueError('Input bank changed')
    with np.load(local / 'canonical_bank.npz', allow_pickle=False) as archive:
        bank = {k: archive[k] for k in ('singles', 'responses', 'pair_ids', 'culture_ids')}
        genes = archive['genes'].tolist()
    if genes != manifest['gene_order']:
        raise ValueError('Gene panel changed')
    folds = pair_folds(bank['pair_ids'], settings['outer_seed'])
    inputs = {'aligned': bank, **{f'shuffle-{s}': shuffled_profiles(bank, s) for s in settings['sampling_seeds']}}
    checked, checked_models, max_abs, unavailable, checkpoint_rows = set(), 0, 0., 0, []
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for condition in conditions:
            run, outer, variant = condition['run'], condition['outer'], condition['variant']
            test = (bank['culture_ids'] == run) & (folds == outer)
            development = (bank['culture_ids'] == run) & (folds != outer)
            held = sorted(set(bank['pair_ids'][test]))
            order = acquisition_order(bank['pair_ids'][development], condition['order'], condition['sampling_seed'])
            acquired = order[:int(condition['budget'] * len(order))]
            if (condition['labels_acquired'] != len(acquired) or condition['full_development_pairs'] != len(order)
                    or condition['test_pair_hash'] != fingerprint(held)
                    or condition['acquired_pair_hash'] != fingerprint(sorted(acquired)) or set(acquired) & set(held)):
                raise ValueError('Acquisition/population isolation changed')
            if condition['fit_id'] in checked:
                continue
            checked.add(condition['fit_id'])
            work = local / 'fits' / condition['fit_id']
            receipt = read_json(work / 'receipt.json')
            if receipt['metrics'] != condition['metrics'] or receipt['test_pairs'] != bank['pair_ids'][test].tolist():
                raise ValueError('Receipt or query coverage changed')
            if not acquired:
                if set(receipt['metrics']) != {'no-change', 'additivity'} or len(receipt['unavailable']) != 6:
                    raise ValueError('Invalid zero-budget result')
                unavailable += len(receipt['unavailable'])
                continue
            mask = development & np.isin(bank['pair_ids'], acquired)
            training = subset(inputs[variant], mask)
            training['partitions'] = np.where(folds[mask] == (outer + 1) % 3, 'validation', 'train')
            query = {k: inputs[variant][k][test] for k in ('singles', 'pair_ids', 'culture_ids')}
            contract = read_json(work / 'contract.json')
            if set(contract['acquired_pair_ids']) != set(acquired) or contract['held_pair_ids'] != held:
                raise ValueError('Training contract lineage changed')
            if args.refit:
                guard = OpalGuard(settings, menu).fit(training, contract)
                models = extract_experts(guard)
            else:
                models = {}
            for name in ('Opal-Base', 'Opal-Interaction', 'Opal-Guard'):
                if name in receipt['unavailable']:
                    unavailable += 1
                    continue
                checkpoint = work / name / 'checkpoint'
                if args.refit:
                    prediction = models[name].predict(query, contract)['predictions']
                    if receipt['selection_records'][name] != {k: getattr(models[name], k) for k in
                         ('selection', 'branch_selection', 'gate_selection', 'crossfit_records') if hasattr(models[name], k)}:
                        raise ValueError('Selection or epoch history replay mismatch')
                    if not checkpoint.exists():
                        models[name].save(checkpoint)
                else:
                    prediction = OpalBase.load(checkpoint).predict(query, contract)['predictions']
                with np.load(work / name / 'freeze/predictions.npz', allow_pickle=False) as archive:
                    difference = float(np.max(np.abs(prediction - archive['predictions'])))
                if difference > settings['replay_atol']:
                    raise ValueError('Prediction replay mismatch')
                max_abs = max(max_abs, difference)
                restored = OpalBase.load(checkpoint)
                difference = float(np.max(np.abs(restored.predict(query, contract)['predictions'] - prediction)))
                if difference > settings['replay_atol']:
                    raise ValueError('Restored replay mismatch')
                score = evaluate_predictions(work / name / 'freeze', query, contract, bank['responses'][test])
                if abs(score['mse'] - receipt['metrics'][name]) > 1e-12 or not np.allclose(score['pair_mse'], receipt['pair_metrics'][name], atol=1e-12, rtol=0):
                    raise ValueError('Evaluator score replay mismatch')
                for record in getattr(restored, 'crossfit_records', []):
                    if set(record['validation_pair_ids']) & set(record['train_pair_ids']):
                        raise ValueError('Guard OOF leakage')
                    if set(record['expert_access']['fitting_and_selection_pair_ids']) != set(record['train_pair_ids']):
                        raise ValueError('Guard expert access mismatch')
                checkpoint_rows.append({'model': name, 'path': str(checkpoint), 'run': run, 'outer': outer,
                       'variant': variant, 'budget': condition['budget'], 'labels': len(acquired),
                       'model_sha256': sha256(checkpoint / 'model.json'), 'parameters_sha256': sha256(checkpoint / 'parameters.npz'),
                       'fit_hash': restored.fit_hash, 'selected_reference': restored.config,
                       'role': 'frozen_outer_evaluation', 'replay_max_abs': difference})
                checked_models += 1
            print(f"Verified {len(checked)} unique fits; {checked_models} models", flush=True)
    expected = 3 * 3 * 2 * 3 * 4 + 3 * 3 * 3
    if len(conditions) != expected:
        raise ValueError('Missing declared budget/order/seed/shuffle conditions')
    write_json(compact / 'all_budget_checkpoint_manifest.json', checkpoint_rows)
    result = {'status': 'verified', 'conditions': len(conditions), 'unique_fit_receipts': len(checked),
              'model_checkpoints_replayed': checked_models, 'prediction_max_abs': max_abs,
              'unavailable_model_records': unavailable, 'exact_selection_history_replayed': args.refit,
              'held_population_unchanged': True, 'zero_budget_checked': True,
              'guard_oof_pair_isolation_checked': True, 'gene_and_input_hashes_checked': True,
              'configuration_sha256': sha256(ROOT / 'configs/experiments/opal_model_series_v1.json'),
              'verification_code_sha256': sha256(Path(__file__)), 'current_model_code_sha256': sha256(ROOT / 'src/series.py'),
              'post_fit_integrity_amendment': 'Canonical access declarations now refused if reversed/aliased. No fitting arithmetic changed; all actual refits replay frozen predictions/selection histories.',
              'wall_seconds': time.monotonic() - started, 'biological_gates_passed': False}
    write_json(compact / 'verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/opal_model_series_v1/combi-seq')
    parser.add_argument('--local', type=Path, default=Path('E:/opal-model-series-20261010'))
    parser.add_argument('--refit', action='store_true')
    print(verify(parser.parse_args()))
