"""Source-specific PertSpectra within-stratum budgets and sample-label transport."""
import argparse
from datetime import datetime, timezone
import gzip
import csv
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
from opal.replicated.models import fit_candidate, predict_candidate, select, shuffled_profiles
from opal.replicated.retrospective import pair_folds
from opal.series import (OpalBase, OpalInteraction, OpalGuard, Unavailable, acquisition_order,
                         check_partition, evaluate_predictions, freeze_predictions, subset)
from run_model_series import extract_experts
from report_model_series import summarize


def execute(args):
    preparation = read_json(args.preparation / 'prepared.json')
    settings_path = ROOT / 'configs/experiments/opal_model_series_v1.json'
    settings = read_json(settings_path)
    menu_path = ROOT / 'configs/experiments/replicated_fixed_state_v1.json'
    menu = read_json(menu_path)['menu']
    started = time.monotonic()
    torch.set_num_threads(1)
    source_banks = {}
    for context, info in preparation['contexts'].items():
        path = Path(info['local']) / 'canonical_bank.npz'
        if sha256(path) != info['bank_sha256']:
            raise ValueError('Prepared source hash mismatch')
        with np.load(path, allow_pickle=False) as archive:
            source_banks[context] = {k: archive[k] for k in ('singles', 'responses', 'pair_ids', 'culture_ids')}
    universe = sorted(set(p for b in source_banks.values() for p in b['pair_ids']))
    assignments = dict(zip(universe, pair_folds(universe, settings['outer_seed'])))
    with threadpool_limits(limits=1):
        for context, bank in source_banks.items():
            info = preparation['contexts'][context]
            directory = args.output / ('TNFA-positive' if context == 'TNFA+' else 'TNFA-negative')
            directory.mkdir(parents=True, exist_ok=True)
            local = Path(info['local']) / 'evaluation'
            local.mkdir(exist_ok=True)
            freeze = {'utc': datetime.now(timezone.utc).isoformat(), 'source': info['source'],
                      'configuration_sha256': sha256(settings_path), 'configuration': settings,
                      'menu_sha256': sha256(menu_path), 'code_sha256': sha256(Path(__file__)),
                      'model_code_sha256': sha256(ROOT / 'src/series.py'),
                      'preparation_sha256': sha256(args.preparation / 'prepared.json'),
                      'input_bank_sha256': info['bank_sha256'], 'full_file_sha256': preparation['full_file_sha256'],
                      'expression_exposure': 'Preparation completed; no prior Opal fitting on this source',
                      'scale': info['units'], 'endpoint': info['genes'],
                      'global_fold_assignment': {p: int(f) for p, f in assignments.items()},
                      'strata': info['strata'], 'independent_cultures_claimed': False,
                      'transport': 'Leave SampleIndex out within the same TNF context; all occurrences of each query pair globally excluded. Equal training-stratum weights. Unknown sample context triggers Base fallback for nonlinear variants.'}
            scope = directory / 'scope_before_fit.json'
            if scope.exists():
                previous = read_json(scope)
                for key in ('configuration_sha256', 'code_sha256', 'model_code_sha256', 'input_bank_sha256'):
                    if previous[key] != freeze[key]:
                        raise ValueError('External resume frozen hash mismatch')
            else:
                write_json(scope, freeze)
            folds = np.asarray([assignments[p] for p in bank['pair_ids']])
            variants = [('aligned', bank)] + [(f'shuffle-{s}', shuffled_profiles(bank, s)) for s in settings['sampling_seeds']]
            conditions, checkpoints, pair_rows, transport = [], [], [], []
            peak = psutil.Process().memory_info().rss
            for variant, inputs in variants:
                for stratum in info['strata']:
                    for outer in range(3):
                        test = (bank['culture_ids'] == stratum) & (folds == outer)
                        dev = (bank['culture_ids'] == stratum) & (folds != outer)
                        if not test.any():
                            raise ValueError('Empty required outer population')
                        held = sorted(set(bank['pair_ids'][test]))
                        population = sorted(set(bank['pair_ids'][dev]))
                        query = {k: inputs[k][test] for k in ('singles', 'pair_ids', 'culture_ids')}
                        for method in settings['orders'] if variant == 'aligned' else ['uniform']:
                            for seed in settings['sampling_seeds'] if variant == 'aligned' else [11]:
                                order = acquisition_order(population, method, seed)
                                for budget in settings['budgets'] if variant == 'aligned' else [1.]:
                                    acquired = order[:int(budget * len(order))]
                                    identity = fingerprint({'stratum': stratum, 'outer': outer, 'variant': variant, 'acquired': sorted(acquired)})
                                    work = local / identity
                                    condition = {'variant': variant, 'run': stratum, 'outer': outer, 'order': method,
                                                 'sampling_seed': seed, 'budget': budget, 'labels_acquired': len(acquired),
                                                 'full_development_pairs': len(population), 'test_pair_count': int(test.sum()),
                                                 'test_pair_hash': fingerprint(held), 'acquired_pair_hash': fingerprint(sorted(acquired)), 'fit_id': identity}
                                    if (work / 'receipt.json').exists():
                                        receipt = read_json(work / 'receipt.json')
                                    else:
                                        work.mkdir(exist_ok=False)
                                        write_json(work / 'acquisition.json', {'order': order, 'acquired': acquired, 'held': held,
                                                    'method': method, 'seed': seed, 'validation_labels_counted': True})
                                        mask = dev & np.isin(bank['pair_ids'], acquired)
                                        training = subset(inputs, mask)
                                        training['partitions'] = np.where(folds[mask] == (outer + 1) % 3, 'validation', 'train')
                                        contract = {'source': info['source'], 'assay': 'PertSpectra deposited gene-label-conditional aggregate',
                                                    'units': info['units'], 'genes': info['genes'], 'reference_provenance': preparation['full_file_sha256'],
                                                    'split_lineage': identity, 'held_pair_ids': held, 'acquired_pair_ids': sorted(acquired),
                                                    'input_bank_sha256': info['bank_sha256'], 'code_sha256': freeze['model_code_sha256'],
                                                    'checkpoint_role': 'frozen_outer_evaluation'}
                                        write_json(work / 'contract.json', contract)
                                        y = bank['responses'][test]
                                        estimates = {'no-change': np.zeros_like(y), 'additivity': query['singles'].sum(1)}
                                        receipt = {'metrics': {}, 'pair_metrics': {}, 'unavailable': {}, 'selection_records': {},
                                                   'checkpoint_manifests': [], 'test_pairs': query['pair_ids'].tolist(), 'gate_summary': None}
                                        if acquired:
                                            try:
                                                check_partition(training, settings['minimum_partition_pairs'])
                                                selection = select(training, menu)
                                                receipt['selection_records']['historical-menu'] = selection
                                                for role in ['reference', 'candidate']:
                                                    expert = fit_candidate(selection[role], training, np.ones(len(training['pair_ids']), bool))
                                                    estimates['historical-' + role] = predict_candidate(selection[role], expert, query)
                                                try:
                                                    guard = OpalGuard(settings, menu).fit(training, contract)
                                                    models = extract_experts(guard)
                                                except Unavailable as error:
                                                    models = {}
                                                    receipt['unavailable']['Opal-Guard'] = str(error)
                                                    for name, cls in [('Opal-Base', OpalBase), ('Opal-Interaction', OpalInteraction)]:
                                                        try:
                                                            models[name] = cls(settings, menu).fit(training, contract)
                                                        except Unavailable as exc:
                                                            receipt['unavailable'][name] = str(exc)
                                                for name, model in models.items():
                                                    prediction = freeze_predictions(work / name / 'freeze', model, query, contract)
                                                    score = evaluate_predictions(work / name / 'freeze', query, contract, y)
                                                    estimates[name] = prediction['predictions']
                                                    checkpoint = model.save(work / name / 'checkpoint')
                                                    restored = OpalBase.load(checkpoint).predict(query, contract)['predictions']
                                                    difference = float(np.max(np.abs(restored - estimates[name])))
                                                    if difference > settings['replay_atol'] or abs(score['mse'] - np.mean((estimates[name] - y) ** 2)) > 1e-12:
                                                        raise ValueError('External checkpoint/evaluator replay failed')
                                                    receipt['selection_records'][name] = {k: getattr(model, k) for k in
                                                        ('selection', 'branch_selection', 'gate_selection', 'crossfit_records') if hasattr(model, k)}
                                                    receipt['checkpoint_manifests'].append({'model': name, 'path': str(checkpoint), 'source': info['source'],
                                                        'stratum': stratum, 'outer': outer, 'variant': variant, 'budget': budget, 'labels': len(acquired),
                                                        'model_sha256': sha256(checkpoint / 'model.json'), 'parameters_sha256': sha256(checkpoint / 'parameters.npz'),
                                                        'fit_hash': model.fit_hash, 'selected_reference': model.config, 'replay_max_abs': difference,
                                                        'role': 'frozen_outer_evaluation'})
                                                    if name == 'Opal-Guard':
                                                        _, support = model.support(query)
                                                        estimates['fixed-half-gate'] = model._base(query) + .5 * support[:, None] * model._interaction(query)
                                                        receipt['gate_summary'] = {'mean': float(prediction['gate'].mean()), 'fallback_count': int((~support).sum())}
                                            except Unavailable as error:
                                                receipt['unavailable'].update({n: str(error) for n in ['historical-reference', 'historical-candidate', 'Opal-Base', 'Opal-Interaction', 'Opal-Guard', 'fixed-half-gate']})
                                        else:
                                            receipt['unavailable'] = {n: 'Zero pair labels' for n in ['historical-reference', 'historical-candidate', 'Opal-Base', 'Opal-Interaction', 'Opal-Guard', 'fixed-half-gate']}
                                        for name, estimate in estimates.items():
                                            error = np.mean((estimate - y) ** 2, axis=1)
                                            receipt['metrics'][name] = float(error.mean())
                                            receipt['pair_metrics'][name] = error.tolist()
                                        write_json(work / 'receipt.json', receipt)
                                    condition.update(metrics=receipt['metrics'], unavailable=receipt['unavailable'], gate_summary=receipt['gate_summary'])
                                    conditions.append(condition)
                                    checkpoints.extend(receipt['checkpoint_manifests'])
                                    for name, errors in receipt['pair_metrics'].items():
                                        for pair, error in zip(receipt['test_pairs'], errors):
                                            pair_rows.append([variant, stratum, outer, method, seed, budget, len(acquired), name, pair, error])
                                    peak = max(peak, psutil.Process().memory_info().rss)
                                    print(f'{context} {len(conditions)} {stratum}/{outer} {method}/{seed} {budget}', flush=True)
                                    if time.monotonic() - started > settings['wall_seconds_cap'] or peak > settings['rss_bytes_cap']:
                                        raise RuntimeError('External finite training resource cap')
            write_json(directory / 'conditions.json', conditions)
            write_json(directory / 'checkpoint_manifest.json', list({r['path']: r for r in checkpoints}.values()))
            with gzip.open(directory / 'pair_metrics.csv.gz', 'wt', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['variant', 'stratum', 'outer', 'order', 'seed', 'budget', 'labels', 'model', 'pair', 'mse'])
                writer.writerows(pair_rows)
            summarize(directory)
            write_json(directory / 'runtime.json', {'wall_seconds_cumulative': time.monotonic() - started, 'peak_sampled_rss_bytes': peak,
                       'conditions': len(conditions), 'unique_fit_receipts': len(list(local.glob('*/receipt.json'))),
                       'local_storage_bytes': sum(p.stat().st_size for p in local.rglob('*') if p.is_file())})
            write_json(directory / 'verification.json', {'status': 'verified during execution',
                       'checkpoint_replay_max_abs': max((r['replay_max_abs'] for r in checkpoints), default=0.),
                       'model_checkpoints': len({r['path'] for r in checkpoints}), 'conditions': len(conditions),
                       'globally_shared_fold_assignment': True, 'complete_stratum_predictions': True,
                       'original_biological_gates_passed': False, 'independent_cultures_claimed': False})
    print('External model execution complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, default=ROOT / 'results/opal_model_series_v1/pertspectra/access-v4')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/opal_model_series_v1/pertspectra')
    execute(parser.parse_args())
