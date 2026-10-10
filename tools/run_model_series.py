"""Execute the frozen Combi-seq model series and nested label-budget study."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import array_hash, environment, fingerprint, read_json, sha256, write_json
from opal.replicated.models import fit_candidate, predict_candidate, select, shuffled_profiles
from opal.replicated.public_diagnostic import nominal_panel
from opal.replicated.retrospective import nested_pair_benchmark, pair_folds, reference_gene_panel
from opal.series import (OpalBase, OpalInteraction, OpalGuard, Unavailable, acquisition_order,
                         evaluate_predictions, freeze_predictions, subset)
from diagnose_public_combi import load_matrix


def extract_experts(guard):
    """The three variants share the same fitted final experts, avoiding extra fits."""
    base = OpalBase(guard.settings, guard.menu)
    interaction = OpalInteraction(guard.settings, guard.menu)
    common = ('settings', 'menu', 'contract', 'fit_hash', 'access', 'selection', 'config',
              'contexts', 'degrees', 'expert', 'train_profiles', 'input_mean', 'input_scale')
    for key in common:
        setattr(base, key, copy.deepcopy(getattr(guard, key)))
        setattr(interaction, key, copy.deepcopy(getattr(guard, key)))
    interaction.branch = copy.deepcopy(guard.branch)
    interaction.branch_selection = copy.deepcopy(guard.branch_selection)
    return {'Opal-Base': base, 'Opal-Interaction': interaction, 'Opal-Guard': guard}


def run(args):
    compact, local = args.output, args.local
    compact.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / 'configs/experiments/opal_model_series_v1.json'
    menu_path = ROOT / 'configs/experiments/replicated_fixed_state_v1.json'
    settings, menu = read_json(config_path), read_json(menu_path)['menu']
    source = ROOT / 'data/source_audits/replicated_fixed_state_v1/source_snapshots'
    historical = ROOT / 'results/replicated_fixed_state_v1/retrospective-combi-20261009-v1'
    code = [ROOT / 'src/series.py', Path(__file__), ROOT / 'src/replicated/models.py',
            ROOT / 'src/replicated/retrospective.py', ROOT / 'src/replicated/public_diagnostic.py',
            ROOT / 'tools/diagnose_public_combi.py']
    fixed = {'utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha256(config_path),
             'menu_sha256': sha256(menu_path), 'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in code},
             'metadata_sha256': sha256(source / 'combi_treatments.csv.gz'),
             'matrix_sha256': sha256(source / 'combi_large_scale_matrix.csv.gz'),
             'historical_report_sha256': sha256(historical / 'report.json'),
             'exposure': 'Combi outcomes already exposed; this is a bounded new model-development cycle',
             'configuration': settings, 'storage_free_bytes': shutil.disk_usage(local).free,
             'memory_available_bytes': psutil.virtual_memory().available,
             'expert_reuse': 'One final Guard fit produces identical fitted Base and Interaction experts for all three variants; each variant has its own inference and checkpoint. All cross-fit and selection attempts preserved.',
             'role': 'frozen evaluation checkpoints, not all-development deployment refits'}
    scope_path = compact / 'execution_freeze.json'
    if scope_path.exists():
        old = read_json(scope_path)
        for key in ('config_sha256', 'menu_sha256', 'code_sha256', 'metadata_sha256', 'matrix_sha256'):
            if old[key] != fixed[key]:
                raise ValueError('Resume refused: changed frozen ' + key)
    else:
        write_json(scope_path, fixed)
        for p in code:
            destination = local / 'code_snapshot' / p.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, destination)
    started = time.monotonic()
    torch.set_num_threads(settings['threads'])
    bank_path = local / 'canonical_bank.npz'
    if bank_path.exists():
        with np.load(bank_path, allow_pickle=False) as archive:
            bank = {k: archive[k] for k in ('singles', 'responses', 'pair_ids', 'culture_ids')}
            genes = archive['genes'].tolist()
    else:
        with gzip.open(source / 'combi_treatments.csv.gz', 'rt', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        counts, genes = load_matrix(source / 'combi_large_scale_matrix.csv.gz', [r[''] for r in rows])
        panel = nominal_panel(rows, counts, genes)
        del counts
        indices = reference_gene_panel(panel['singles'], genes)
        genes = [genes[i] for i in indices]
        n = len(panel['runs']) * len(panel['pairs'])
        bank = {'singles': panel['singles'][:, :, :, indices].reshape(n, 2, len(genes)),
                'responses': panel['responses'][:, :, indices].reshape(n, len(genes)),
                'pair_ids': np.tile([json.dumps(p, separators=(',', ':')) for p in panel['pairs']], len(panel['runs'])),
                'culture_ids': np.repeat(panel['runs'], len(panel['pairs']))}
        np.savez_compressed(bank_path, **bank, genes=np.asarray(genes))
        write_json(compact / 'input_manifest.json', {'bank_sha256': sha256(bank_path),
                    'gene_order': genes, 'gene_sha256': fingerprint(genes),
                    'source_genes': 25912, 'canonical_pairs': len(panel['pairs']), 'strata': panel['runs'],
                    'reference_lineage': panel['lineage'], 'exclusions': panel['exclusions'],
                    'excluded_pairs': sorted(panel['all_pairs'] - set(panel['pairs'])),
                    'zero_library_source_ids': panel['zero_library_ids'], 'historical_panel_reused': True})
        del panel
    original = read_json(historical / 'report.json')
    if genes != original['endpoint_genes']:
        raise ValueError('Historical endpoint mismatch')
    folds = pair_folds(bank['pair_ids'], settings['outer_seed'])
    if {p: int(f) for p, f in zip(bank['pair_ids'], folds)} != original['fold_assignment']:
        raise ValueError('Historical outer split mismatch')
    # Reproduce the full historical budget before any new model fitting.
    replay_path = compact / 'historical_reproduction.json'
    with threadpool_limits(limits=1):
        if not replay_path.exists():
            report, predictions = nested_pair_benchmark(bank, menu)
            diffs = {}
            with np.load(historical / 'predictions.npz', allow_pickle=False) as archive:
                for label, values in predictions.items():
                    for role, value in values.items():
                        diff = float(np.max(np.abs(value - archive[f'{label}_{role}'])))
                        if diff > settings['replay_atol']:
                            raise ValueError('Historical prediction replay failed')
                        diffs[f'{label}/{role}'] = diff
            if report['fold_records'] != original['fold_records']:
                raise ValueError('Historical selections/records differ')
            write_json(replay_path, {'equal_run_mean_mse': report['equal_run_mean_mse'],
                       'prediction_max_abs_differences': diffs, 'fold_records_exact': True,
                       'historical_report_sha256': sha256(historical / 'report.json')})
            del predictions
        conditions, checkpoints, pair_rows = [], [], []
        peak = psutil.Process().memory_info().rss
        variants = [('aligned', bank)] + [(f'shuffle-{s}', shuffled_profiles(bank, s)) for s in settings['sampling_seeds']]
        for variant, inputs in variants:
            for run_id in sorted(set(bank['culture_ids'])):
                for outer in range(3):
                    test = (bank['culture_ids'] == run_id) & (folds == outer)
                    dev = (bank['culture_ids'] == run_id) & (folds != outer)
                    query = {k: inputs[k][test] for k in ('singles', 'pair_ids', 'culture_ids')}
                    held = sorted(set(bank['pair_ids'][test]))
                    population = sorted(set(bank['pair_ids'][dev]))
                    base_contract = {'genes': genes, 'units': 'log2_library_10000_change',
                                     'source': 'GSE174695', 'assay': 'Combi-seq nominal aggregate reads',
                                     'reference_provenance': 'source-hashed nominal controls/singles; see input_manifest.json',
                                     'split_lineage': fingerprint({'outer': outer, 'held': held, 'run': str(run_id), 'variant': variant}),
                                     'held_pair_ids': held, 'checkpoint_role': 'frozen_outer_evaluation',
                                     'code_sha256': fixed['code_sha256'], 'input_bank_sha256': sha256(bank_path)}
                    for method in settings['orders'] if variant == 'aligned' else ['uniform']:
                        for seed in settings['sampling_seeds'] if variant == 'aligned' else [11]:
                            order = acquisition_order(population, method, seed)
                            for budget in settings['budgets'] if variant == 'aligned' else [1.0]:
                                acquired = order[:int(budget * len(order))]
                                identity = fingerprint({'run': str(run_id), 'outer': outer, 'variant': variant,
                                                        'acquired': sorted(acquired)})
                                work = local / 'fits' / identity
                                receipt_path = work / 'receipt.json'
                                condition = {'variant': variant, 'run': str(run_id), 'outer': outer,
                                             'order': method, 'sampling_seed': seed, 'budget': budget,
                                             'labels_acquired': len(acquired), 'full_development_pairs': len(population),
                                             'test_pair_count': int(test.sum()), 'test_pair_hash': fingerprint(held),
                                             'acquired_pair_hash': fingerprint(sorted(acquired)),
                                             'fit_id': identity, 'metrics': {}, 'unavailable': {}}
                                if receipt_path.exists():
                                    receipt = read_json(receipt_path)
                                else:
                                    work.mkdir(parents=True, exist_ok=False)
                                    acquisition = {'order': order, 'prefix': acquired, 'method': method, 'seed': seed,
                                                   'counts_include_validation': True, 'outer_population': held}
                                    write_json(work / 'acquisition.json', acquisition)
                                    y = bank['responses'][test]  # evaluator only; never handed to fitting.
                                    estimates = {'no-change': np.zeros_like(y), 'additivity': query['singles'].sum(1)}
                                    receipt = {'metrics': {}, 'unavailable': {}, 'checkpoint_manifests': [],
                                               'pair_metrics': {}, 'selection_records': {}, 'replay': {}}
                                    if not acquired:
                                        receipt['unavailable'] = {name: 'Zero pair labels' for name in
                                                                 ['historical-reference', 'historical-candidate', 'Opal-Base', 'Opal-Interaction', 'Opal-Guard', 'fixed-half-gate']}
                                    else:
                                        mask = dev & np.isin(bank['pair_ids'], acquired)
                                        development = subset(inputs, mask)
                                        development['partitions'] = np.where(folds[mask] == (outer + 1) % 3, 'validation', 'train')
                                        contract = dict(base_contract, acquired_pair_ids=sorted(acquired))
                                        write_json(work / 'contract.json', contract)
                                        selection = select(development, menu)
                                        receipt['selection_records']['historical'] = selection
                                        for role in ('reference', 'candidate'):
                                            expert = fit_candidate(selection[role], development, np.ones(len(acquired), bool))
                                            estimates['historical-' + role] = predict_candidate(selection[role], expert, query)
                                        try:
                                            guard = OpalGuard(settings, menu).fit(development, contract)
                                            models = extract_experts(guard)
                                        except Unavailable as exc:
                                            models = {}
                                            for name, cls in [('Opal-Base', OpalBase), ('Opal-Interaction', OpalInteraction)]:
                                                try:
                                                    models[name] = cls(settings, menu).fit(development, contract)
                                                except Unavailable as error:
                                                    receipt['unavailable'][name] = str(error)
                                            receipt['unavailable']['Opal-Guard'] = str(exc)
                                        for name, model in models.items():
                                            prediction = freeze_predictions(work / name / 'freeze', model, query, contract)
                                            score = evaluate_predictions(work / name / 'freeze', query, contract, y)
                                            estimates[name] = prediction['predictions']
                                            receipt['metrics'][name] = score['mse']
                                            receipt['selection_records'][name] = {k: getattr(model, k) for k in
                                                  ('selection', 'branch_selection', 'gate_selection', 'crossfit_records') if hasattr(model, k)}
                                            if budget == 1.0:
                                                checkpoint = work / name / 'checkpoint'
                                                model.save(checkpoint)
                                                restored = OpalBase.load(checkpoint).predict(query, contract)
                                                difference = float(np.max(np.abs(restored['predictions'] - estimates[name])))
                                                if difference > settings['replay_atol']:
                                                    raise ValueError('Checkpoint replay mismatch')
                                                receipt['replay'][name] = difference
                                                receipt['checkpoint_manifests'].append({'model': name, 'source': 'GSE174695', 'run': str(run_id),
                                                  'outer': outer, 'variant': variant, 'path': str(checkpoint),
                                                  'model_sha256': sha256(checkpoint / 'model.json'),
                                                  'parameters_sha256': sha256(checkpoint / 'parameters.npz'),
                                                  'fit_hash': model.fit_hash, 'selected_reference': model.config,
                                                  'role': 'frozen_outer_evaluation', 'labels_acquired': len(acquired), 'replay_max_abs': difference})
                                            if name == 'Opal-Guard':
                                                features, support = model.support(query)
                                                estimates['fixed-half-gate'] = model._base(query) + .5 * support[:, None] * model._interaction(query)
                                                receipt['gate_summary'] = {'mean': float(prediction['gate'].mean()),
                                                    'min': float(prediction['gate'].min()), 'max': float(prediction['gate'].max()),
                                                    'base_fallback_count': int((~prediction['support']).sum())}
                                        # Example bundles use one frozen evaluation checkpoint population, not a deployment refit.
                                        if variant == 'aligned' and outer == 0 and run_id == 'rep1' and budget == 1:
                                            example = local / 'example'
                                            example.mkdir(exist_ok=True)
                                            np.savez_compressed(example / 'query.npz', **query)
                                            np.savez_compressed(example / 'development.npz', **development)
                                            np.savez_compressed(example / 'outcomes.npz', responses=y)
                                            write_json(example / 'contract.json', contract)
                                            write_json(example / 'checkpoints.json', receipt['checkpoint_manifests'])
                                    for name, estimate in estimates.items():
                                        errors = np.mean((estimate - y) ** 2, axis=1)
                                        receipt['metrics'][name] = float(errors.mean())
                                        receipt['pair_metrics'][name] = errors.tolist()
                                    receipt['test_pairs'] = query['pair_ids'].tolist()
                                    receipt['elapsed_seconds'] = time.monotonic() - started
                                    write_json(receipt_path, receipt)
                                condition['metrics'], condition['unavailable'] = receipt['metrics'], receipt['unavailable']
                                condition['gate_summary'] = receipt.get('gate_summary')
                                conditions.append(condition)
                                for name, errors in receipt['pair_metrics'].items():
                                    for pair, error in zip(receipt['test_pairs'], errors):
                                        pair_rows.append([variant, str(run_id), outer, method, seed, budget, len(acquired), name, pair, error])
                                if budget == 1:
                                    checkpoints.extend(receipt['checkpoint_manifests'])
                                peak = max(peak, psutil.Process().memory_info().rss)
                                write_json(local / 'progress.json', {'conditions_complete': len(conditions), 'current': condition,
                                                                  'elapsed_seconds': time.monotonic() - started, 'rss_bytes': peak})
                                print(f"{len(conditions)} {variant} {run_id} outer={outer} {method}/{seed} budget={budget}: {condition['metrics']}", flush=True)
                                if peak > settings['rss_bytes_cap'] or time.monotonic() - started > settings['wall_seconds_cap']:
                                    raise RuntimeError('Frozen resource cap reached; retained attempts, no expansion')
        write_json(compact / 'conditions.json', conditions)
        unique_checkpoints = {c['path']: c for c in checkpoints}
        write_json(compact / 'checkpoint_manifest.json', list(unique_checkpoints.values()))
        # Compact pair errors, never prediction/expression arrays.
        with gzip.open(compact / 'pair_metrics.csv.gz', 'wt', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['variant', 'run', 'outer', 'order', 'sampling_seed', 'budget', 'labels', 'model', 'pair_id', 'mse'])
            writer.writerows(pair_rows)
        write_json(compact / 'runtime.json', {'wall_seconds': time.monotonic() - started, 'peak_sampled_rss_bytes': peak,
                   'rss_scope': 'sampled after each fit, not OS continuous peak', 'environment': environment(),
                   'storage': str(local), 'storage_bytes': sum(p.stat().st_size for p in local.rglob('*') if p.is_file()),
                   'conditions': len(conditions), 'unique_fit_receipts': len(list((local / 'fits').glob('*/receipt.json')))})
    return len(conditions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/opal_model_series_v1/combi-seq')
    parser.add_argument('--local', type=Path, default=Path('E:/opal-model-series-20261010'))
    args = parser.parse_args()
    print(json.dumps({'conditions': run(args)}))
