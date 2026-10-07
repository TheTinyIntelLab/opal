"""Registered, bounded exposed-development studies; never opens evaluator outcomes."""
import argparse
import csv
import itertools
import json
import platform
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from opal.artifacts import load_npz, read_json, sha256, write_json, environment
from opal.models.kernel import smoother, prediction_error
from opal.contextual.contracts import ReferenceBank
from opal.state.benchmark import groups_for, descriptors, query_rows
from opal.state.models import MainReference
from opal.state.contracts import registered_split, validate_isolation, error_decomposition
from opal.training import fit, predict
from tools.improve_residual_model import fold_sets

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '84de7e36eae20531e815be9de5a4cb2fa9b94ce9'


def rbf(x, bandwidth):
    gram = x @ x.T / max(x.shape[1], 1)
    distance = np.maximum(np.diag(gram)[:, None] + np.diag(gram)[None] - 2 * gram, 0)
    return np.exp(-distance / bandwidth)


def scale_features(x, train):
    center = x[train].mean(0)
    scale = np.maximum(x[train].std(0), 1e-5)
    return (x - center) / scale


def feature_candidates(x, train, config, standardized=False):
    x = x if standardized else scale_features(x, train)
    kernels = [('ridge', 1., x @ x.T / x.shape[1])]
    kernels += [('rbf', b, rbf(x, b)) for b in config['bandwidths']]
    return [(dict(kernel=k, bandwidth=b, alpha=a), g) for k, b, g in kernels for a in config['alphas']]


def context_pc(controls, rows, train, rank=3):
    source = np.unique(rows[train])
    center = controls[source].mean(0)
    _, _, vt = np.linalg.svd(controls[source] - center, full_matrices=False)
    return (controls[rows] - center) @ vt[:min(rank, len(source) - 1)].T


def input_features(data, train, label, config):
    # Only declared measured references and pretreatment controls enter features.
    base = data['a'] + data['b']
    if label in {'measured_effect_only', 'no_context', 'global_correction'}:
        return base
    extra = []
    if label in {'context', 'context_identity', 'shuffled_context'}:
        rows = data['contexts'].copy()
        if label == 'shuffled_context':
            order = np.random.default_rng(config['seed']).permutation(len(data['controls']))
            rows = order[rows]
        extra.append(context_pc(data['controls'], rows, train))
    if label in {'identity', 'context_identity', 'shuffled_identity'}:
        p = data['pairs']
        identities = np.eye(len(data['targets']))
        if label == 'shuffled_identity':
            # One-hot renaming is an exact negative control, not semantic destruction.
            identities = identities[np.random.default_rng(config['seed']).permutation(len(identities))]
        extra.append(identities[p[:, 0]] + identities[p[:, 1]])
    if label in {'descriptor', 'shuffled_descriptor'}:
        values = data['descriptors'].copy()
        if label == 'shuffled_descriptor':
            values = values[np.random.default_rng(config['seed']).permutation(len(values))]
        p = data['pairs']; av, bv = values[p[:, 0]], values[p[:, 1]]
        extra.append(np.concatenate([av + bv, np.abs(av - bv)], 1))
    if not extra:
        raise ValueError('Unknown feature route')
    # Equal weight per information block, avoiding dilution by gene dimension.
    blocks = [scale_features(base, train) / np.sqrt(base.shape[1])]
    blocks += [scale_features(v, train) / np.sqrt(v.shape[1]) for v in extra]
    return np.concatenate(blocks, 1) * np.sqrt(sum(v.shape[1] for v in [base] + extra))


def candidates(data, train, family, config):
    if family == 'mean_residual':
        return [(dict(alpha=1.), np.zeros((len(data['pairs']),) * 2))]
    if family not in {'single_sum', 'pair_product', 'mixed'}:
        return feature_candidates(input_features(data, train, family, config), train, config,
                                  standardized=family not in {'measured_effect_only', 'no_context', 'global_correction'})
    # Per-fold scales use measured-single features, with no response preprocessing.
    both = np.concatenate([data['a'][train], data['b'][train]])
    center, scale = both.mean(0), np.maximum(both.std(0), 1e-5)
    a, b = [(data[k] - center) / scale for k in ['a', 'b']]
    summed = feature_candidates(data['a'] + data['b'], train, config)
    out = []
    for bw in config['bandwidths']:
        z = np.concatenate([a, b]); sk = rbf(z, bw); n = len(a)
        ac, ad, bc, bd = sk[:n, :n], sk[:n, n:], sk[n:, :n], sk[n:, n:]
        kernel = (ac + ad + bc + bd) / 4 if family == 'single_sum' else (ac * bd + ad * bc) / 2
        if family == 'mixed':
            # Nonlinear shared distortion plus symmetric pair product.
            kernel = kernel + next(g for c, g in summed if c['kernel'] == 'rbf' and c['bandwidth'] == bw)
        out += [(dict(bandwidth=bw, alpha=alpha), kernel) for alpha in config['alphas']]
    return out


def kernel_prediction(data, family, config, train, query, chosen):
    menu = candidates(data, train, family, config)
    kernel = next(g for c, g in menu if c == chosen)
    scale = max(float(np.diag(kernel)[train].mean()), 1e-12)
    weights = smoother(kernel / scale, train, query, chosen['alpha'])
    return weights @ data['residual'][train]


def select_kernel(data, train, inner, family, config):
    menu = candidates(data, train, family, config)
    scores = [[] for _ in menu]
    for ti, vi in inner:
        t, v = train[ti], train[vi]
        inner_menu = candidates(data, t, family, config)
        for j, (candidate, kernel) in enumerate(inner_menu):
            scale = max(float(np.diag(kernel)[t].mean()), 1e-12)
            weights = smoother(kernel / scale, t, v, candidate['alpha'])
            scores[j].append((prediction_error(weights, t, v, data['residual_gram']), len(v)))
    records = [dict(config=c, mse=float(np.average([e for e, _ in es], weights=[n for _, n in es])))
               for (c, _), es in zip(menu, scores)]
    return min(records, key=lambda r: r['mse']), records


def splits(data, kind, config):
    if kind == 'context':
        groups = data['groups']
        return [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in np.unique(groups)]
    return fold_sets(data['pairs'], kind, config['seed'], config['outer_folds'])


def inner_splits(data, train, kind, config, f):
    if kind == 'context':
        groups = data['groups'][train]
        return [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in np.unique(groups)]
    return fold_sets(data['pairs'][train], kind, config['seed'] + f + 1, config['inner_folds'])


def interval(error, reference, groups, seed=923, draws=2000):
    units = np.unique(groups)
    rng = np.random.default_rng(seed)
    # Keep all conditions sharing a state together. Genetic targets use a crossed
    # pigeonhole bootstrap below, preserving shared-single dependence diagnostically.
    if np.asarray(groups).ndim == 2:
        units = np.unique(groups)
        samples = []
        for _ in range(draws):
            count = np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units))
            positions = np.searchsorted(units, groups)
            w = count[positions[:, 0]] * count[positions[:, 1]]
            if w.sum():
                samples.append(float(np.average(reference - error, weights=w)))
        method = 'crossed target bootstrap; fixed profiles and common control; diagnostic only'
    else:
        samples = []
        for _ in range(draws):
            chosen = rng.choice(units, len(units), replace=True)
            w = sum((groups == g).astype(float) for g in chosen)
            samples.append(float(np.average(reference - error, weights=w)))
        method = 'whole-state bootstrap; fixed shared measured references; diagnostic only'
    return {'paired_mse_reduction': float(np.mean(reference - error)),
            'diagnostic_95_interval': np.quantile(samples, [.025, .975]).tolist(),
            'method': method, 'independent_biological_interval': None, 'draws': draws}


def summarize(data, predictions, rows, config):
    y = data['y'][rows]
    errors = {k: np.mean((p - y) ** 2, 1) for k, p in predictions.items()}
    units = data['groups'][rows] if data['contextual'] else data['pairs'][rows]
    results = {}
    for label, err in errors.items():
        mse = float(err.mean()); gain = 1 - mse / float(errors['selected_simple'].mean())
        uncertainty = interval(err, errors['selected_simple'], units)
        lower = uncertainty['diagnostic_95_interval'][0]
        results[label] = {'mse': mse, 'relative_gain': gain, **uncertainty,
                          'materiality_screen': gain >= config['minimum_gain'],
                          'development_gate': gain >= config['minimum_gain'] and lower > 0,
                          'by_state': {str(g): float(err[data['groups'][rows] == g].mean()) for g in np.unique(data['groups'][rows])},
                          'by_intervention': {str(t): float(err[np.any(data['pairs'][rows] == i, axis=1)].mean())
                              for i, t in enumerate(data['targets']) if np.any(data['pairs'][rows] == i)}}
    return results


def neural_menu(config):
    out = []
    for family in ['shared', 'pair']:
        for width, rank, lr, decay in itertools.product(config['neural']['widths'], config['neural']['ranks'],
                                                       config['neural']['learning_rates'], config['neural']['weight_decays']):
            out.append(dict(family='sum_network' if family == 'shared' else 'network',
                shared_only=family == 'shared', audit_family=family, embedding=16, features=16, depth=1,
                width=width, rank=rank, lr=lr, weight_decay=decay, epochs=config['neural']['updates'],
                offset=True, maximum_fit_seconds=config['neural']['per_fit_seconds']))
    # Original schedule is kept intact; original sum head also includes |a-b|.
    for family in ['sum_network', 'network']:
        out.append(dict(family=family, audit_family='original_' + family, embedding=16, features=16,
                        width=64, rank=4, depth=1, lr=.01, weight_decay=.01, epochs=80,
                        offset=True, maximum_fit_seconds=config['neural']['per_fit_seconds']))
    return out


def neural_audit(data, train, query, inner, config, out, fold, deadline):
    search = []
    for j, candidate in enumerate(neural_menu(config)):
        errors, sizes, stopping, trajectories = [], [], [], []
        for k, (ti, vi) in enumerate(inner):
            t, v = train[ti], train[vi]
            members = []
            for seed in config['neural']['seeds']:
                if time.perf_counter() > deadline:
                    raise TimeoutError('Frozen neural audit wall budget exhausted')
                model = fit(candidate, data['singles'], data['pairs'][t], data['y'][t], seed,
                            (data['pairs'][v], data['y'][v]))
                members.append(predict(model, data['pairs'][v])); stopping.append(model['epochs'])
                trajectories.append({'inner_fold': k, 'seed': seed, 'curve': model['curve'],
                    'training_mse': float(np.mean((predict(model, data['pairs'][t]) - data['y'][t]) ** 2)),
                    'stopped_epoch': model['epochs'], 'parameters': model['parameters']})
            # The ensemble is the selection unit at both inner and outer levels.
            errors.append(float(np.mean((np.mean(members, 0) - data['y'][v]) ** 2))); sizes.append(len(v))
        record = dict(config=candidate, mse=float(np.average(errors, weights=sizes)),
                      refit_epochs=max(1, int(np.median(stopping))), trajectories=trajectories)
        search.append(record)
        write_json(out / f'neural-fold-{fold}-search.json', search)
        print(data['name'], 'neural', fold, j + 1, '/', len(neural_menu(config)), flush=True)
    selected = {family: min([r for r in search if r['config']['audit_family'] == family], key=lambda r: r['mse'])
                for family in sorted({r['config']['audit_family'] for r in search})}
    predictions = {}
    for family, record in selected.items():
        members = []
        for seed in config['neural']['seeds']:
            if time.perf_counter() > deadline:
                raise TimeoutError('Frozen neural refit wall budget exhausted')
            candidate = {**record['config'], 'epochs': record['refit_epochs']}
            model = fit(candidate, data['singles'], data['pairs'][train], data['y'][train], seed)
            members.append(predict(model, data['pairs'][query]))
        predictions['neural_' + family] = np.mean(members, 0)
    return predictions, selected


def technical(data, train, query, chosen, config):
    halves = data.get('halves')
    if halves is None:
        return None
    records = []
    for h in [0, 1]:
        s, y = halves[f'singles_{h}'], halves[f'doubles_{h}']
        p = data['pairs']; main = s[p].sum(1)
        other_main = halves[f'singles_{1-h}'][p].sum(1)
        other_y = halves[f'doubles_{1-h}']
        temp = {**data, 'a': s[p[:, 0]], 'b': s[p[:, 1]], 'residual': y - main}
        residual = kernel_prediction(temp, 'global_correction', config, train, query, chosen)
        # Entire half reference matrices remain shared across every pair.
        record = error_decomposition(main[query] + residual, main[query], other_main[query], other_y[query])
        left, right = (y - main)[query], (other_y - other_main)[query]
        cosine = np.sum(left * right, 1) / np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12)
        records.append({'fit_half': h, **record, 'residual_half_cosine': float(cosine.mean()),
                        'status': 'disjoint cell halves; frozen QC/normalizer/gene panel; not independent cultures',
                        'selection': 'full-reference inner-selected global configuration; fixed-selection sensitivity'})
    return records


def study(data, output, config, kind, neural=False):
    output.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    deadline = started + config['neural']['maximum_seconds_per_dataset']
    families = ['mean_residual', 'global_correction', 'single_sum', 'pair_product', 'mixed',
                'measured_effect_only', 'no_context', 'identity', 'shuffled_identity']
    if data['contextual']:
        families += ['context', 'context_identity', 'shuffled_context', 'descriptor', 'shuffled_descriptor']
    all_preds, all_rows, folds = {}, [], []
    for f, (train, query) in enumerate(splits(data, kind, config)):
        inner = inner_splits(data, train, kind, config, f)
        if len(inner) < 2 or any(not len(t) or not len(v) for t, v in inner):
            raise ValueError('Registered split infeasible; no redraw')
        if kind == 'context':
            validate_isolation(train, query, data['groups'])
        selections, searches, predictions = {}, {}, {'additive': (data['a'] + data['b'])[query]}
        for family in families:
            chosen, search = select_kernel(data, train, inner, family, config)
            selections[family], searches[family] = chosen, search
            predictions[family] = predictions['additive'] + kernel_prediction(data, family, config, train, query, chosen['config'])
        simple = min(['mean_residual', 'global_correction', 'single_sum'], key=lambda k: selections[k]['mse'])
        predictions['selected_simple'] = predictions[simple]
        tech = technical(data, train, query, selections['global_correction']['config'], config)
        if neural:
            npreds, nselect = neural_audit(data, train, query, inner, config, output, f, deadline)
            predictions.update(npreds); selections.update(nselect)
        decomposition = None
        if data['contextual']:
            # Separate baseline-only route: source exact-intervention means,
            # shared pair offset. No query singles enter fitting or preprocessing.
            source_contexts = np.unique(data['contexts'][train]); query_contexts = data['contexts'][query]
            bank = data['bank']; a = bank.arrays
            supported = a['available'][source_contexts].any(0)
            if not supported[data['pairs'][query]].all():
                raise ValueError('Unregistered baseline-only unsupported intervention exclusion')
            main_model = MainReference().fit(a['controls'][source_contexts], a['singles'][source_contexts], a['available'][source_contexts])
            main = sum(main_model.predict(a['controls'][query_contexts], data['pairs'][query, i]) for i in [0, 1])
            estimated = main + data['residual'][train].mean(0)
            predictions['baseline_only_mean_plus_offset'] = estimated
            decomposition = error_decomposition(estimated, main, predictions['additive'], data['y'][query])
        # Persist predictions and selection before the outer metric summaries.
        np.savez_compressed(output / f'fold-{f}-predictions.npz', rows=query, **predictions)
        folds.append({'fold': f, 'train': train.tolist(), 'query': query.tolist(),
                      'inner': [{'train': train[t].tolist(), 'query': train[v].tolist()} for t, v in inner],
                      'selection': selections, 'search': searches, 'selected_simple_family': simple,
                      'technical_cross_reference': tech, 'estimated_single_decomposition': decomposition})
        write_json(output / 'folds.json', folds)
        for label, pred in predictions.items():
            all_preds.setdefault(label, []).append(pred)
        all_rows.extend(query.tolist())
        if psutil.Process().memory_info().rss > config['maximum_rss_bytes']:
            raise MemoryError('Registered process memory cap exceeded')
        print(data['name'], kind, 'fold', f, 'saved', flush=True)
    rows = np.asarray(all_rows); prediction = {k: np.concatenate(v) for k, v in all_preds.items()}
    np.savez_compressed(output / 'predictions.npz', rows=rows, pairs=data['pairs'][rows], contexts=data['contexts'][rows], **prediction)
    result = {'status': 'complete_exposed_development', 'name': data['name'], 'split': kind,
              'conditions_per_occurrence': len(rows), 'unique_conditions': len(np.unique(rows)),
              'models': summarize(data, prediction, rows, config), 'folds': folds,
              'seconds': time.perf_counter() - started, 'process_peak_working_set_bytes': psutil.Process().memory_info().peak_wset,
              'prediction_sha256': sha256(output / 'predictions.npz'), 'evaluator_access': False,
              'independent_culture_diagnostics': 'unavailable; require independently initiated cultures with matched controls and singles',
              'identity_shuffle_scope': 'One-hot bijective renaming is invariant for ridge/RBF; not a test of external semantic identity',
              'descriptor_scope': 'Exact dose only; no chemical identity or unseen-intervention claim',
              'technical_diagnostics': 'shared full-half matrices preserve common single/control dependence; fixed-selection sensitivity'}
    write_json(output / 'results.json', result)
    return result


def qualify(data_root, output):
    """Inspect provenance/metadata and hash files without candidate score access."""
    output.mkdir(parents=True, exist_ok=False)
    registry = read_json(data_root / 'state_source_registry.json')
    genetic = read_json(data_root / 'source_registry.json')
    sources = []
    for name in ['norman-log2', 'wessels-log2', 'emeraldbay-pilot-v1', 'sciplex-state-v1']:
        directory = data_root / 'prepared' / name
        files = ['profiles.npz', 'profile_uncertainty.npz', 'source_manifest.json', 'preprocessing.json',
                 'protocol.json', 'references.json', 'references.npz', 'source_audit.json']
        hashes = {f: sha256(directory / f) for f in files if (directory / f).exists()}
        sources.append({'id': name, 'available': bool(hashes), 'files': hashes,
                        'exposure': 'all previous outcomes exposed; development only',
                        'independent_cultures_verified': False, 'sealed_confirmation': False,
                        'license': 'CC-BY-4.0' if name != 'wessels-log2' else 'GEO public scientific reuse; raw redistribution unspecified',
                        'accepted': ['development observed-single pairs'] if name != 'sciplex-state-v1' else ['development baseline-only singles'],
                        'missing': ['independent culture mapping', 'sealed joint pair/context outcomes', 'qualified chemical identity descriptors']})
    import torch
    record = {'baseline': BASELINE, 'sources': sources, 'state_registry': registry,
              'genetic_protocols': genetic['prepared_protocols'],
              'capabilities': {'implemented': ['state models', 'context kernels', 'claim-specific splits', 'safe exports'],
                               'demonstrated': ['historical predictive corrections and exposed development only'],
                               'unestablished': ['independent joint transfer', 'interaction recovery', 'calibrated OOD risk']},
              'hardware': {'platform': platform.platform(), 'cpu': platform.processor(),
                           'logical_cores': psutil.cpu_count(), 'ram_bytes': psutil.virtual_memory().total,
                           'gpu': str(torch.cuda.get_device_properties(0)) if torch.cuda.is_available() else None,
                           'fit_device': 'cpu; CUDA is available but historical float64 implementation is CPU'},
              'dependencies_reused': True, 'environment': environment(),
              'resource_exclusions': 'No unresolved-license external priors or raw datasets redistributed',
              'source_order': 'qualification completed before successor candidate fitting'}
    write_json(output / 'qualification.json', record)
    return record


def load_data(root, name):
    directory = root / 'prepared' / name
    p = load_npz(directory / 'profiles.npz')
    if name == 'emeraldbay-pilot-v1':
        bank = ReferenceBank.load(directory); pairs, ci = bank.validate_rows(p['pairs'], p['contexts'])
        a = bank.arrays; s = a['singles']; y = p['responses']
        desc, mask = descriptors(bank)
        if not mask.all():
            raise ValueError('Dose descriptors not qualified for all interventions')
        data = dict(name=name, pairs=pairs, contexts=ci, a=s[ci, pairs[:, 0]], b=s[ci, pairs[:, 1]],
                    y=y, controls=a['controls'], targets=a['targets'], descriptors=desc,
                    groups=groups_for(bank)[ci], bank=bank, contextual=True)
    else:
        s, pairs = p['singles'].astype(float), p['pairs']; y = p['doubles'].astype(float)
        data = dict(name=name, pairs=pairs, contexts=np.zeros(len(pairs), int), a=s[pairs[:, 0]],
                    b=s[pairs[:, 1]], y=y, controls=p['control'][None], singles=s, targets=p['targets'],
                    groups=np.repeat(name, len(pairs)), contextual=False,
                    halves=load_npz(directory / 'profile_uncertainty.npz'))
    data['residual'] = y - data['a'] - data['b']
    data['residual_gram'] = data['residual'] @ data['residual'].T / y.shape[1]
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/stage1-20261007.json')
    p.add_argument('--qualify-only', action='store_true')
    args = p.parse_args(); config = read_json(args.config)
    if args.qualify_only:
        qualify(args.data_root, args.output); return
    if args.output.exists():
        raise FileExistsError('Preserve every prior run; choose a new output ID')
    qualification = read_json(args.output.parent / 'qualification' / 'qualification.json')
    args.output.mkdir(parents=True)
    write_json(args.output / 'protocol.json', {**config, 'config_sha256': sha256(args.config),
               'qualification_sha256': sha256(args.output.parent / 'qualification/qualification.json'),
               'neural_candidates': neural_menu(config)})
    with zipfile.ZipFile(args.output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for f in list((ROOT / 'src').rglob('*.py')) + [Path(__file__), ROOT / 'tools/improve_residual_model.py']:
            archive.write(f, f.relative_to(ROOT))
    results = {}
    try:
        with threadpool_limits(limits=1):
            for name in ['emeraldbay-pilot-v1', 'norman-log2', 'wessels-log2']:
                # Verify qualified bytes immediately before fitting.
                source = next(s for s in qualification['sources'] if s['id'] == name)
                for file, digest in source['files'].items():
                    if sha256(args.data_root / 'prepared' / name / file) != digest:
                        raise ValueError('Qualified source changed')
                data = load_data(args.data_root, name)
                for kind in ['context'] if data['contextual'] else ['random', 'partner']:
                    result = study(data, args.output / name / kind, config, kind, neural=kind == 'random')
                    results[name + '/' + kind] = result
                    write_json(args.output / 'summary.json', results)
        write_json(args.output / 'status.json', {'status': 'complete_registered_supported_pair_studies',
                   'confirmation': 'unavailable', 'neural_context_audit': 'not registered; existing state prototype not expanded'})
    except BaseException as exc:
        write_json(args.output / 'status.json', {'status': 'interrupted_or_failed', 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
