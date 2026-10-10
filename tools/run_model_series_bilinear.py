"""Execute frozen acquired-label fits and retain every score/refusal locally."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import array_hash, fingerprint, read_json, sha256, write_json
from opal.series import MODEL_REGISTRY, OpalBase, OpalGuard, Unavailable, acquisition_order, evaluate_predictions, freeze_predictions, subset
from opal.series_bilinear import (FIT_CACHE, base_oof, expert_oof, expert_predict, fit_branch, fit_memory,
                                fit_reference, predict_branch, predict_reference, retrieve, weights)
from opal.replicated.models import select, fit_candidate, predict_candidate
from opal.replicated.retrospective import pair_folds


def source_banks():
    prepared = read_json(ROOT / 'results/opal_model_series_v1/pertspectra/access-v4/prepared.json')
    paths = {'combi-seq': (Path('E:/opal-model-series-20261010/canonical_bank.npz'), ROOT / 'results/opal_model_series_v1/combi-seq')}
    for context, info in prepared['contexts'].items():
        name = 'TNFA-positive' if context == 'TNFA+' else 'TNFA-negative'
        paths[name] = (Path(info['local']) / 'canonical_bank.npz', ROOT / 'results/opal_model_series_v1/pertspectra' / name)
    return paths


def metrics(prediction, truth, base):
    errors = np.mean((prediction - truth) ** 2, axis=1)
    p, y = prediction - prediction.mean(1, keepdims=True), truth - truth.mean(1, keepdims=True)
    norm = np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1)
    correlations = [float(p[i] @ y[i] / norm[i]) if norm[i] > 1e-12 else None for i in range(len(y))]
    distances = np.mean((p[:, None] - y[None, :]) ** 2, axis=2)
    ranks = [float(np.sum(distances[i] < distances[i, i] - 1e-12) + (np.sum(np.abs(distances[i] - distances[i, i]) <= 1e-12) + 1) / 2) for i in range(len(y))]
    null = float(np.mean(truth ** 2))
    reference = float(np.mean((base - truth) ** 2))
    return {'mse': float(errors.mean()), 'mae': float(np.mean(np.abs(prediction - truth))),
            'skill_no_change': 1 - float(errors.mean()) / null if null > 0 else None,
            'skill_reference': 1 - float(errors.mean()) / reference if reference > 0 else None,
            'signed_correction_mean': float(np.mean(prediction - base)),
            'correction_rms': float(np.sqrt(np.mean((prediction - base) ** 2))),
            'centered_pearson_mean': float(np.mean([c for c in correlations if c is not None])) if any(c is not None for c in correlations) else None,
            'undefined_pearson_rows': sum(c is None for c in correlations),
            'undefined_reason': 'Zero centered profile norm' if any(c is None for c in correlations) else None,
            'retrieval_mean_rank': float(np.mean(ranks)), 'retrieval_pool': len(y),
            'retrieval_rule': 'Gene-centered squared Euclidean distance, average rank for ties, same-stratum held population'}


def variants(guard):
    result = {}
    for name, cls in MODEL_REGISTRY.items():
        model = cls(guard.settings, guard.menu)
        vars(model).update(copy.deepcopy(vars(guard)))
        if name == 'Opal-Base':
            model.fitted.update(branch=None, memory=None, router=None, expert_mask=np.asarray([True, False, False]))
            model.fitted.pop('branch_selection', None)
            model.fitted.pop('gate_selection', None)
        elif name == 'Opal-Interaction':
            model.fitted.update(memory=None, router=None, expert_mask=np.asarray([True, model.fitted['branch'] is not None, False]))
            model.fitted.pop('gate_selection', None)
        result[name] = model
    return result


def ablations(model, development, query):
    fitted, settings, menu = model.fitted, model.settings, model.menu
    base = model._base(query)
    predictions, refusals = {'zero-branch': base}, {}
    for route in settings['residual_routes']:
        for diagonal in [False, True]:
            label = ('diagonal' if diagonal else 'cross-factor') + '-' + route
            try:
                branch = fit_branch(development, {'input_rank': 4, 'output_rank': 4, 'alpha': 1, 'route': route, 'diagonal': diagonal},
                                    model.config, fitted['reference'], settings, menu)
                predictions[label] = base + predict_branch(branch, model.config, fitted['reference'], query)
            except Unavailable as error:
                refusals[label] = str(error)
    try:
        config = {'kind': 'sum_pca', 'rank': 4, 'alpha': 1}
        predictions['matched-main-effect'] = predict_reference(config, fit_reference(config, development), query)
    except Unavailable as error:
        refusals['matched-main-effect'] = str(error)
    for k in settings['retrieval_k']:
        try:
            memory = fit_memory(development, settings, menu, k)
            predictions[f'retrieval-{k}'] = base + retrieve(memory, query)[0]
        except Unavailable as error:
            refusals[f'retrieval-{k}'] = str(error)
    if model.fitted['router'] is not None:
        experts, _, _, _ = expert_predict(fitted, query)
        available = fitted['expert_mask']
        predictions['equal-mixture'] = experts[:, available].mean(1)
        for index, label in enumerate(['base-alone', 'interaction-alone', 'memory-alone']):
            if available[index]:
                predictions[label] = experts[:, index]
            else:
                refusals[label] = 'Expert unavailable in required nested router folds'
        try:
            oof, _, availability, _ = expert_oof(development, settings, menu, fitted['memory']['k'] if fitted['memory'] else 4)
            availability &= available
            e = oof[:, availability]
            w = weights(development)
            def objective(mixture):
                pred = (e * mixture[None, :, None]).sum(1)
                return float(w @ np.mean((pred - development['responses']) ** 2, axis=1))
            constant = minimize(objective, np.ones(e.shape[1]) / e.shape[1], method='SLSQP',
                                bounds=[(0, 1)] * e.shape[1], constraints={'type': 'eq', 'fun': lambda v: v.sum() - 1},
                                options={'ftol': 1e-12, 'maxiter': 100})
            if not constant.success:
                raise Unavailable('Constant convex fit failed: ' + constant.message)
            predictions['constant-mixture'] = (experts[:, availability] * constant.x[None, :, None]).sum(1)
        except Unavailable as error:
            refusals['constant-mixture'] = str(error)
    else:
        for name in ['equal-mixture', 'constant-mixture', 'interaction-alone', 'memory-alone']:
            refusals[name] = 'Router expert set unavailable; Base fallback retained'
    return predictions, refusals


def resource_check(local, output, settings):
    for volume, reserve in settings['reserve_bytes'].items():
        if shutil.disk_usage(volume + ':/').free < reserve:
            raise RuntimeError('Disk reserve reached on ' + volume)
    # Include every compact series record, not only this runner's output.
    # Overlapping roots describe the same physical files and count once.
    candidates = {Path(p).resolve() for p in settings.get('artifact_roots', [])}
    candidates.update([Path(local).resolve(), Path(output).resolve(),
                       ROOT / 'results/opal_model_series_v2'])
    roots = [root for root in candidates
             if not any(root != parent and root.is_relative_to(parent) for parent in candidates)]
    retained = sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())
    if retained > settings['new_artifact_bytes_cap']:
        raise RuntimeError('New artifact cap reached')
    return retained


def run(args):
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    menu = read_json(ROOT / 'configs/experiments/replicated_fixed_state_v1.json')['menu']
    local, output = args.local, args.output
    local.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_banks()
    code = {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / 'src/series.py', ROOT / 'src/series_bilinear.py']}
    freeze = {'settings': settings, 'configuration_sha256': sha256(ROOT / 'configs/experiments/opal_model_series_v2.json'),
              'code_sha256': code, 'source_banks': {name: {'path': str(p), 'sha256': sha256(p)} for name, (p, _) in sources.items()},
              'role': 'Retrospective evaluation; deployment refits produced separately after scoring',
              'information_tier': 'strict observed singles and acquired pair aggregates',
              'primary_metric': settings['primary_metric'], 'native_scores': None}
    if (output / 'execution_freeze.json').exists():
        if read_json(output / 'execution_freeze.json') != freeze:
            old = read_json(output / 'execution_freeze.json')
            old_code, current_code = old.pop('code_sha256'), dict(freeze['code_sha256'])
            runner_key = str(Path(__file__).relative_to(ROOT))
            old_runner, current_runner = old_code.pop(runner_key), current_code.pop(runner_key)
            comparable = dict(freeze)
            comparable.pop('code_sha256')
            if args.amend_resources:
                resource_keys = {'new_artifact_bytes_cap', 'reserve_bytes', 'readonly_volumes', 'resource_amendment', 'artifact_roots'}
                old_settings = {k: v for k, v in old['settings'].items() if k not in resource_keys}
                new_settings = {k: v for k, v in comparable['settings'].items() if k not in resource_keys}
                old.pop('settings')
                comparable.pop('settings')
                old.pop('configuration_sha256')
                comparable.pop('configuration_sha256')
                if old_settings != new_settings:
                    raise ValueError('Resource amendment cannot change scientific configuration')
            if old != comparable or (old_code != current_code and not args.amend_storage):
                raise ValueError('Resume refused after source/model/config change; preserve attempt and amend explicitly')
            write_json(output / 'runner_manifest_amendment.json', {'original_runner_sha256': old_runner,
                        'amended_runner_sha256': current_runner, 'original_model_code': old_code, 'current_model_code': current_code,
                        'reason': 'Verified metadata adapter defect: PertSpectra rows use stratum. Storage-only amendment: future arrays deduplicated by content; completed fits retained and replayed, partial directories preserved. No fitting arithmetic, populations, selection or endpoints changed.',
                        'original_freeze_sha256': sha256(output / 'execution_freeze.json'), 'amended_freeze': freeze})
            if args.amend_resources:
                write_json(output / 'resource_amendment.json', {'authorization': 'Owner explicitly replied: Allow 2 GB, spill to C:',
                           'original_freeze_sha256': sha256(output / 'execution_freeze.json'), 'amended_configuration': settings,
                           'scientific_configuration_unchanged': True, 'old_artifacts_readonly': True})
    else:
        write_json(output / 'execution_freeze.json', freeze)
    started, cpu_start = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    all_conditions = []
    with threadpool_limits(limits=1):
        for source_name, (bank_path, historical) in sources.items():
            if args.source and source_name not in args.source:
                continue
            directory = output / source_name
            directory.mkdir(exist_ok=True)
            with np.load(bank_path, allow_pickle=False) as f:
                bank = {k: f[k] for k in ('singles', 'responses', 'pair_ids', 'culture_ids')}
                genes = f['genes'].tolist()
            old_rows = read_json(historical / 'conditions.json')
            if source_name == 'combi-seq':
                assignment = pair_folds(bank['pair_ids'], settings['outer_seed'])
            else:
                fold_map = read_json(historical / 'scope_before_fit.json')['global_fold_assignment']
                assignment = np.asarray([fold_map[p] for p in bank['pair_ids']])
            source_rows = []
            for stratum in sorted(set(bank['culture_ids'])):
                for outer in range(3):
                    test = (bank['culture_ids'] == stratum) & (assignment == outer)
                    dev = (bank['culture_ids'] == stratum) & (assignment != outer)
                    q = {k: bank[k][test] for k in ('singles', 'pair_ids', 'culture_ids')}
                    held, population = sorted(set(bank['pair_ids'][test])), sorted(set(bank['pair_ids'][dev]))
                    historical_contract = next((read_json(Path(r['path']).parent.parent / 'contract.json') for r in read_json(historical / ('all_budget_checkpoint_manifest.json' if source_name == 'combi-seq' else 'checkpoint_manifest.json'))
                                                if r.get('run', r.get('stratum')) == stratum and r['outer'] == outer), None)
                    if historical_contract is None:
                        raise ValueError('Missing historical source contract')
                    for order_name in settings['orders']:
                        for acquisition_seed in settings['sampling_seeds']:
                            order = acquisition_order(population, order_name, acquisition_seed)
                            for budget in settings['budgets']:
                                acquired = sorted(order[:int(budget * len(order))])
                                seeds = [settings['training_seed']]
                                if budget == 1 and order_name == 'uniform' and acquisition_seed == 11:
                                    seeds += settings['stability_seeds']
                                for training_seed in seeds:
                                    config = dict(settings, training_seed=training_seed)
                                    contract = dict(historical_contract, acquired_pair_ids=acquired, held_pair_ids=held,
                                                    checkpoint_role='frozen_outer_evaluation', context_contract=source_name,
                                                    information_tier='strict', code_sha256=code)
                                    mask = dev & np.isin(bank['pair_ids'], acquired)
                                    development = subset(bank, mask)
                                    development['partitions'] = np.where(assignment[mask] == (outer + 1) % 3, 'validation', 'train')
                                    identity = fingerprint({'source': source_name, 'stratum': str(stratum), 'outer': outer, 'acquired': acquired, 'training_seed': training_seed})
                                    work = local / source_name / identity
                                    preserved = args.preserved_local / source_name / identity if args.preserved_local else None
                                    if preserved and (preserved / 'receipt.json').exists():
                                        work = preserved
                                    if work.exists() and not (work / 'receipt.json').exists():
                                        # Preserve interrupted output; never overwrite partial weights.
                                        attempt = 1
                                        while (local / source_name / (identity + f'-retry-{attempt}')).exists():
                                            attempt += 1
                                        work = local / source_name / (identity + f'-retry-{attempt}')
                                    condition = {'source': source_name, 'stratum': str(stratum), 'outer': outer, 'order': order_name,
                                                 'acquisition_seed': acquisition_seed, 'training_seed': training_seed, 'budget': budget,
                                                 'unique_labels': len(acquired), 'pair_stratum_labels': int(mask.sum()), 'held_population_sha256': fingerprint(held),
                                                 'fit_id': identity, 'local_fit_directory': str(work), 'role': 'frozen_outer_evaluation'}
                                    if (work / 'receipt.json').exists():
                                        receipt = read_json(work / 'receipt.json')
                                    else:
                                        resource_check(local, output, settings)
                                        work.mkdir(parents=True, exist_ok=False)
                                        write_json(work / 'contract.json', contract)
                                        write_json(work / 'acquisition.json', {'population': population, 'acquired': acquired, 'held': held,
                                                   'every_label_access_stage_counted': ['fit', 'representation', 'stopping', 'cross-fitting', 'router-validation']})
                                        fit_start, fit_cpu = time.monotonic(), time.process_time()
                                        predictions = {'no-change': np.zeros_like(bank['responses'][test]), 'additivity': q['singles'].sum(1)}
                                        receipt = {'unavailable': {}, 'checkpoints': [], 'metrics': {}, 'fallbacks': {}, 'selections': {}}
                                        models = {}
                                        try:
                                            guard = OpalGuard(config, menu).fit(development, contract)
                                            models = variants(guard)
                                            predictions['training-mean'] = np.broadcast_to(weights(development) @ development['responses'], q['singles'][:, 0].shape)
                                            selected = select(development, menu)
                                            for role in ('reference', 'candidate'):
                                                expert = fit_candidate(selected[role], development, np.ones(len(development['pair_ids']), bool))
                                                predictions['historical-' + role] = predict_candidate(selected[role], expert, q)
                                        except Unavailable as error:
                                            receipt['unavailable'].update({name: str(error) for name in MODEL_REGISTRY})
                                        for name, model in models.items():
                                            checkpoint = model.save(work / name / 'checkpoint')
                                            prediction = freeze_predictions(work / name / 'freeze', model, q, contract)
                                            restored = OpalBase.load(checkpoint)
                                            replay = restored.predict(q, contract)
                                            difference = float(np.max(np.abs(replay['predictions'] - prediction['predictions'])))
                                            if difference > settings['replay_atol']:
                                                raise ValueError('New checkpoint replay failed')
                                            predictions[name] = prediction['predictions']
                                            receipt['fallbacks'][name] = {'unsupported_fraction': float(np.mean(~prediction['support'])),
                                                                         'expert_availability': prediction['expert_availability']}
                                            receipt['checkpoints'].append({'model': name, 'path': str(checkpoint), 'model_sha256': sha256(checkpoint / 'model.json'),
                                                                          'parameters_sha256': sha256(checkpoint / 'parameters.npz'), 'replay_max_abs': difference,
                                                                          'bytes': sum(p.stat().st_size for p in checkpoint.iterdir())})
                                            receipt['selections'][name] = {k: getattr(model, k) for k in ('selection', 'branch_selection', 'gate_selection') if hasattr(model, k)}
                                        if models and budget == 1:
                                            additional, refusals = ablations(guard, development, q)
                                            predictions.update(additional)
                                            receipt['unavailable'].update(refusals)
                                        base = predictions.get('Opal-Base', predictions['no-change'])
                                        # All outcomes below belong to the sealed evaluator path.
                                        for name, prediction in predictions.items():
                                            receipt['metrics'][name] = metrics(prediction, bank['responses'][test], base)
                                        old = next((r for r in old_rows if r['variant'] == 'aligned' and r['run'] == stratum and r['outer'] == outer and
                                                    r['order'] == order_name and r['sampling_seed'] == acquisition_seed and r['budget'] == budget), None)
                                        if old:
                                            receipt['preserved_v1_metrics'] = old['metrics']
                                        array_pool = local / 'array_pool'
                                        array_pool.mkdir(exist_ok=True)
                                        receipt['evaluator_array_hashes'] = {}
                                        for name, prediction in predictions.items():
                                            content_hash = array_hash(prediction)
                                            destination = array_pool / (content_hash + '.npz')
                                            if not destination.exists():
                                                np.savez_compressed(destination, prediction=prediction)
                                            receipt['evaluator_array_hashes'][name] = content_hash
                                        receipt.update(wall_seconds=time.monotonic() - fit_start, cpu_seconds=time.process_time() - fit_cpu,
                                                       process_peak_rss_bytes=psutil.Process().memory_info().peak_wset,
                                                       representation_labels=len(acquired), training_and_validation_labels=len(acquired))
                                        # Complete histories remain in a safe compressed JSON file.
                                        history_path = work / 'selection_histories.json.gz'
                                        with gzip.open(history_path, 'wt', encoding='utf-8') as stream:
                                            json.dump(receipt.pop('selections'), stream, sort_keys=True)
                                        receipt['selection_history_sha256'] = sha256(history_path)
                                        write_json(work / 'receipt.json', receipt)
                                    condition.update(metrics=receipt['metrics'], unavailable=receipt['unavailable'], checkpoints=receipt['checkpoints'],
                                                     wall_seconds=receipt['wall_seconds'], cpu_seconds=receipt['cpu_seconds'])
                                    source_rows.append(condition)
                                    all_conditions.append(condition)
                                    write_json(directory / 'conditions.json', source_rows)
                                    print(source_name, stratum, outer, order_name, acquisition_seed, budget, training_seed,
                                          {k: round(v['mse'], 6) for k, v in receipt['metrics'].items() if k.startswith('Opal')}, flush=True)
                                    if (time.process_time() - cpu_start) / 3600 > settings['allocation_core_hours']['models_ablations']:
                                        raise RuntimeError('Model allocation exhausted; preserve completed and missing conditions')
            write_json(directory / 'completion.json', {'conditions': len(source_rows), 'status': 'declared primary budgets and full-budget stability seeds executed'})
    write_json(output / 'runtime.json', {'wall_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu_start,
               'peak_rss_bytes': psutil.Process().memory_info().peak_wset, 'new_artifact_bytes': resource_check(local, output, settings), 'conditions': len(all_conditions)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', type=Path, default=Path('D:/opal-model-series-20261010-v2/fits'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/opal_model_series_v2/primary')
    parser.add_argument('--source', nargs='+')
    parser.add_argument('--amend-storage', action='store_true', help='Record reviewed serialization-only amendment; preserve every existing fit')
    parser.add_argument('--amend-resources', action='store_true', help='Record explicit owner cap/spill authorization without changing scientific settings')
    parser.add_argument('--preserved-local', type=Path)
    run(parser.parse_args())
