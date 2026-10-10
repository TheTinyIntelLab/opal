"""Finite float64 model-series fits; historical neural fits live in series.py.

This module receives acquired labels only. Every nested call receives its own
globally restricted bank; no transform is fitted on a validation query.
"""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve
import torch
from torch import nn

from .artifacts import array_hash, read_json, sha256, write_json
from .replicated.models import balanced_weights, fit_candidate, predict_candidate


FIT_CACHE = {}


def cache_fit(function):
    """Reuse identical acquired fits within a top-level call, never across labels."""
    def cached(bank, settings, menu, *args):
        key = (function.__name__, array_hash(*(bank[k] for k in sorted(bank))),
               json.dumps(settings, sort_keys=True), json.dumps(menu, sort_keys=True), args)
        if key not in FIT_CACHE:
            FIT_CACHE[key] = function(bank, settings, menu, *args)
        return FIT_CACHE[key]
    return cached


def weights(bank):
    return balanced_weights(bank['pair_ids'], bank['culture_ids'])


def take(bank, mask):
    return {k: np.asarray(v)[mask].copy() for k, v in bank.items() if k != 'partitions'}


def query(bank):
    return {k: bank[k] for k in ('singles', 'pair_ids', 'culture_ids')}


def partition(bank, settings):
    from .series import inner_partition, check_partition
    if 'partitions' not in bank:
        bank = inner_partition(bank, settings['training_seed'], settings['minimum_partition_pairs'])
    train, val = check_partition(bank, settings['minimum_partition_pairs'])
    return take(bank, train), take(bank, val)


def folds(bank, settings):
    identities = np.random.default_rng(settings['training_seed']).permutation(sorted(set(bank['pair_ids'])))
    assignment = {p: i % settings['crossfit_folds'] for i, p in enumerate(identities)}
    return np.asarray([assignment[p] for p in bank['pair_ids']])


def reference_menu(menu):
    historical = [copy.deepcopy(c) for c in menu if c['role'] == 'reference']
    extra = [{'id': 'null', 'kind': 'no_change', 'role': 'reference'},
             {'id': 'response-mean', 'kind': 'response_mean', 'role': 'reference'}]
    extra += [{'id': f'affine-{a}', 'kind': 'affine', 'alpha': a, 'role': 'reference'} for a in [.1, 1, 10]]
    extra += [{'id': f'sum-pca-{r}-{a}', 'kind': 'sum_pca', 'rank': r, 'alpha': a, 'role': 'reference'}
              for r in [4, 8] for a in [.1, 1, 10]]
    return historical + extra


def pca(x, rank):
    from .series import Unavailable
    key = ('pca', array_hash(x), rank)
    if key in FIT_CACHE:
        return FIT_CACHE[key]
    profiles = np.unique(x.reshape(-1, x.shape[-1]), axis=0)
    mean = profiles.mean(0)
    _, singular, vt = np.linalg.svd(profiles - mean, full_matrices=False)
    support = int(np.sum(singular > max(profiles.shape) * np.finfo(float).eps * max(singular[0], 1)))
    if support < rank:
        raise Unavailable(f'Input PCA rank {rank} requires numerical rank {rank}; available {support}')
    result = {'mean': mean, 'basis': vt[:rank].T}
    FIT_CACHE[key] = result
    return result


def encode(x, encoder):
    return (x - encoder['mean']) @ encoder['basis']


def ridge(x, y, w, alpha):
    xm, ym = w @ x, w @ y
    xc, yc = x - xm, y - ym
    coefficient = solve(xc.T @ (w[:, None] * xc) + alpha * np.eye(x.shape[1]),
                        xc.T @ (w[:, None] * yc), assume_a='pos')
    return {'coefficients': coefficient, 'intercept': ym - xm @ coefficient}


def fit_reference(config, bank):
    kind, x, y, w = config['kind'], bank['singles'], bank['responses'], weights(bank)
    if kind == 'no_change':
        return {}
    if kind == 'response_mean':
        return {'mean': w @ y}
    if kind == 'affine':
        s = x.sum(1)
        sm, ym = w @ s, w @ y
        sc, yc = s - sm, y - ym
        # Gene-averaged normalized squared error plus alpha*(gamma-1)^2.
        variance = float(np.sum(w[:, None] * sc ** 2) / y.shape[1])
        covariance = float(np.sum(w[:, None] * sc * yc) / y.shape[1])
        gamma = float(np.clip((covariance + config['alpha']) / (variance + config['alpha']), 0, 1.5))
        return {'gamma': gamma, 'intercept': ym - gamma * sm}
    if kind == 'sum_pca':
        encoder = pca(x, config['rank'])
        result = ridge(encode(x, encoder).sum(1), y, w, config['alpha'])
        return dict(result, encoder=encoder)
    expert = fit_candidate(config, bank, np.ones(len(x), bool))
    return {'legacy': expert}


def predict_reference(config, fitted, bank):
    kind, x = config['kind'], bank['singles']
    if kind == 'no_change':
        return np.zeros_like(x[:, 0])
    if kind == 'response_mean':
        return np.broadcast_to(fitted['mean'], x[:, 0].shape).copy()
    if kind == 'affine':
        return fitted['gamma'] * x.sum(1) + fitted['intercept']
    if kind == 'sum_pca':
        return encode(x, fitted['encoder']).sum(1) @ fitted['coefficients'] + fitted['intercept']
    return predict_candidate(config, fitted['legacy'], bank)


@cache_fit
def select_reference(bank, settings, menu):
    from .series import Unavailable
    train, validation = partition(bank, settings)
    records = []
    for config in reference_menu(menu):
        try:
            fitted = fit_reference(config, train)
            prediction = predict_reference(config, fitted, validation)
            records.append({'config': config, 'validation_mse': float(weights(validation) @ np.mean((prediction - validation['responses']) ** 2, axis=1))})
        except Unavailable as error:
            records.append({'config': config, 'unavailable': str(error)})
    winner = min((r for r in records if 'validation_mse' in r), key=lambda r: (r['validation_mse'], r['config']['id']))
    config = winner['config']
    return config, fit_reference(config, bank), {'reference': config, 'comparisons': records, 'search_count': len(records),
                                               'tie_breaker': 'MSE then lexical configuration ID'}


@cache_fit
def base_oof(bank, settings, menu):
    from .series import Unavailable
    assignment = folds(bank, settings)
    prediction = np.empty_like(bank['responses'])
    records = []
    for fold in range(settings['crossfit_folds']):
        train, validation = take(bank, assignment != fold), take(bank, assignment == fold)
        if len(set(train['pair_ids'])) < settings['minimum_pairs'] or len(set(validation['pair_ids'])) < settings['minimum_partition_pairs']:
            raise Unavailable('Insufficient unique pairs for honest Base residual cross-fitting')
        config, fitted, selection = select_reference(train, settings, menu)
        prediction[assignment == fold] = predict_reference(config, fitted, validation)
        records.append({'fold': fold, 'train_pair_ids': sorted(set(train['pair_ids'])),
                        'validation_pair_ids': sorted(set(validation['pair_ids'])), 'selection': selection})
    return prediction, records


def main_design(config, fitted, bank):
    kind = config['kind']
    if kind == 'component_additive':
        values = fitted['legacy'].design(bank['pair_ids'])
    elif kind == 'sum_pca':
        values = encode(bank['singles'], fitted['encoder']).sum(1)
    elif kind in {'global', 'affine', 'additive'}:
        values = bank['singles'].sum(1)
    elif kind in {'no_change', 'response_mean', 'shared_offset'}:
        values = np.empty((len(bank['pair_ids']), 0))
    else:
        return None
    return np.column_stack((np.ones(len(values)), values))


def cross_features(z, diagonal=False):
    a, b = z[:, 0], z[:, 1]
    if diagonal:
        return a * b
    return np.column_stack([a[:, i] * b[:, i] if i == j else
                            (a[:, i] * b[:, j] + a[:, j] * b[:, i]) / np.sqrt(2)
                            for i in range(a.shape[1]) for j in range(i, a.shape[1])])


def fit_branch(bank, config, reference_config, reference, settings, menu, residual=None):
    from .series import Unavailable
    dimension = config['input_rank'] if config.get('diagonal') else config['input_rank'] * (config['input_rank'] + 1) // 2
    if len(set(bank['pair_ids'])) < settings['minimum_bilinear_pairs_per_feature'] * dimension:
        raise Unavailable(f'Bilinear map requires {2 * dimension} unique training pairs')
    encoder = pca(bank['singles'], config['input_rank'])
    f = cross_features(encode(bank['singles'], encoder), config.get('diagonal', False))
    design = main_design(reference_config, reference, bank)
    w = weights(bank)
    projection = None
    if design is not None:
        projection = np.linalg.lstsq(np.sqrt(w)[:, None] * design, np.sqrt(w)[:, None] * f, rcond=None)[0]
        f = f - design @ projection
    center = w @ f
    scale = np.maximum(np.sqrt(w @ (f - center) ** 2), 1e-6)
    f = (f - center) / scale
    crossfit = []
    if residual is None:
        if config['route'] == 'cross-fitted':
            prediction, crossfit = base_oof(bank, settings, menu)
        else:
            prediction = predict_reference(reference_config, reference, bank)
        residual = bank['responses'] - prediction
    mean = w @ residual
    basis_key = ('residual_svd', array_hash(residual, w))
    if basis_key not in FIT_CACHE:
        _, singular, vt = np.linalg.svd(np.sqrt(w)[:, None] * (residual - mean), full_matrices=False)
        FIT_CACHE[basis_key] = (singular, vt)
    singular, vt = FIT_CACHE[basis_key]
    available = int(np.sum(singular > max(residual.shape) * np.finfo(float).eps * max(singular[0], 1)))
    if available < config['output_rank']:
        raise Unavailable(f'Residual SVD rank {config["output_rank"]} unavailable; numerical rank {available}')
    basis = vt[:config['output_rank']].T
    coefficients = ridge(f, residual @ basis, w, config['alpha'])
    return dict(coefficients, config=copy.deepcopy(config), encoder=encoder, feature_mean=center, feature_scale=scale,
                output_basis=basis, projection=projection, main_orthogonality=design is not None,
                crossfit_records=crossfit, training_residual_rms=float(np.sqrt(w @ np.mean(residual ** 2, axis=1))))


def predict_branch(branch, reference_config, reference, bank):
    if branch is None:
        return np.zeros_like(bank['singles'][:, 0])
    f = cross_features(encode(bank['singles'], branch['encoder']), branch['config'].get('diagonal', False))
    if branch['projection'] is not None:
        f -= main_design(reference_config, reference, bank) @ branch['projection']
    f = (f - branch['feature_mean']) / branch['feature_scale']
    return (f @ branch['coefficients'] + branch['intercept']) @ branch['output_basis'].T


def select_branch(bank, settings, menu, reference_config, reference):
    from .series import Unavailable
    train, validation = partition(bank, settings)
    tc, tr, _ = select_reference(train, settings, menu)
    base = predict_reference(tc, tr, validation)
    records = [{'config': None, 'validation_mse': float(weights(validation) @ np.mean((base - validation['responses']) ** 2, axis=1))}]
    residuals = {}
    for route in settings['residual_routes']:
        try:
            if route == 'cross-fitted':
                prediction, _ = base_oof(train, settings, menu)
            else:
                prediction = predict_reference(tc, tr, train)
            residuals[route] = train['responses'] - prediction
        except Unavailable as error:
            residuals[route] = str(error)
    for ir, outr, alpha, route in itertools.product(settings['input_ranks'], settings['output_ranks'], settings['penalties'], settings['residual_routes']):
        config = {'input_rank': ir, 'output_rank': outr, 'alpha': alpha, 'route': route}
        try:
            if isinstance(residuals[route], str):
                raise Unavailable(residuals[route])
            branch = fit_branch(train, config, tc, tr, settings, menu, residuals[route])
            prediction = base + predict_branch(branch, tc, tr, validation)
            records.append({'config': config, 'validation_mse': float(weights(validation) @ np.mean((prediction - validation['responses']) ** 2, axis=1))})
        except Unavailable as error:
            records.append({'config': config, 'unavailable': str(error)})
    # Favor the zero branch on exact ties; all other ties are lexical.
    winner = min((r for r in records if 'validation_mse' in r), key=lambda r: (r['validation_mse'], json.dumps(r['config'], sort_keys=True)))
    branch = None if winner['config'] is None else fit_branch(bank, winner['config'], reference_config, reference, settings, menu)
    return branch, {'attempts': records, 'selected': winner['config'], 'search_count': len(records),
                    'residual_target_refit': 'Selected route repeated on full acquired set; cross-fit training sets smaller than final Base refit'}


def profile_distance(left, right, mean, scale):
    l, r = (left - mean) / scale, (right - mean) / scale
    direct = np.mean((l[:, None, 0] - r[None, :, 0]) ** 2 + (l[:, None, 1] - r[None, :, 1]) ** 2, axis=2) / 2
    swap = np.mean((l[:, None, 0] - r[None, :, 1]) ** 2 + (l[:, None, 1] - r[None, :, 0]) ** 2, axis=2) / 2
    return np.minimum(direct, swap)


@cache_fit
def fit_memory(bank, settings, menu, k):
    from .series import Unavailable
    if len(set(bank['pair_ids'])) < k + 1:
        raise Unavailable('Insufficient distinct retrieval neighbors')
    prediction, records = base_oof(bank, settings, menu)
    profiles = np.unique(bank['singles'].reshape(-1, bank['singles'].shape[-1]), axis=0)
    mean, scale = profiles.mean(0), np.maximum(profiles.std(0), 1e-6)
    distances = profile_distance(bank['singles'], bank['singles'], mean, scale)
    allowed = bank['pair_ids'][:, None] != bank['pair_ids'][None, :]
    values = distances[allowed & (distances > 0)]
    temperature = float(np.median(values)) if len(values) else 1.
    return {'singles': bank['singles'].copy(), 'pair_ids': bank['pair_ids'].copy(), 'culture_ids': bank['culture_ids'].copy(),
            'residuals': bank['responses'] - prediction, 'mean': mean, 'scale': scale, 'temperature': temperature,
            'k': k, 'crossfit_records': records}


def retrieve(memory, bank):
    from .series import Unavailable
    distances = profile_distance(bank['singles'], memory['singles'], memory['mean'], memory['scale'])
    prediction, summaries, neighbors = [], [], []
    for i, pair in enumerate(bank['pair_ids']):
        eligible = sorted(set(memory['pair_ids']) - {str(pair)})
        candidates = []
        for p in eligible:
            rows = np.flatnonzero(memory['pair_ids'] == p)
            candidates.append((float(np.mean(distances[i, rows])), str(p), rows))
        chosen = sorted(candidates, key=lambda v: (v[0], v[1]))[:memory['k']]
        if len(chosen) < memory['k']:
            raise Unavailable('Query exclusion leaves insufficient distinct retrieval neighbors')
        d = np.asarray([c[0] for c in chosen])
        w = np.exp(-(d - d.min()) / max(memory['temperature'], 1e-12))
        w /= w.sum()
        # Each canonical neighbor receives equal stratum weight before RBF mixing.
        residuals = []
        for _, _, rows in chosen:
            strata = memory['culture_ids'][rows]
            residuals.append(np.mean([memory['residuals'][rows[strata == c]].mean(0) for c in sorted(set(strata))], axis=0))
        residuals = np.asarray(residuals)
        prediction.append(w @ residuals)
        # Outcome-dependent residual dispersion cannot enter the router.
        summaries.append([float(np.sqrt(d.min())), float(np.std(np.sqrt(d)))])
        neighbors.append([{'pair_id': p, 'distance': float(distance), 'weight': float(weight),
                           'strata': sorted(set(memory['culture_ids'][rows]))}
                          for (distance, p, rows), weight in zip(chosen, w)])
    return np.asarray(prediction), np.asarray(summaries), neighbors


def support(fitted, bank, transport=False):
    degrees = np.asarray([[fitted['degrees'].get(c, 0) for c in json.loads(p)] for p in bank['pair_ids']])
    z = (bank['singles'] - fitted['mean']) / fitted['scale']
    refs = (fitted['profiles'] - fitted['mean']) / fitted['scale']
    distance = np.asarray([np.sqrt(np.min(np.mean((s - refs) ** 2, axis=1))) for s in z.reshape(-1, z.shape[-1])]).reshape(-1, 2).max(1)
    context = np.asarray([c in fitted['contexts'] or transport for c in bank['culture_ids']])
    supported = (degrees.min(1) > 0) & context & (distance <= fitted['distance_limit'])
    return np.column_stack((np.log1p(degrees.min(1)), np.log1p(degrees.max(1)), distance, context.astype(float))), supported


def fit_support(bank):
    from collections import Counter
    profiles = np.unique(bank['singles'].reshape(-1, bank['singles'].shape[-1]), axis=0)
    mean, scale = profiles.mean(0), np.maximum(profiles.std(0), 1e-6)
    z = (profiles - mean) / scale
    distances = np.sqrt(np.mean((z[:, None] - z[None, :]) ** 2, axis=2))
    distances[np.eye(len(z), dtype=bool)] = np.inf
    limit = float(np.max(np.min(distances, axis=1))) if len(z) > 1 else 0.
    return {'profiles': profiles, 'mean': mean, 'scale': scale,
            'degrees': dict(Counter(c for p in set(bank['pair_ids']) for c in json.loads(p))),
            'contexts': sorted(set(bank['culture_ids'])), 'distance_limit': max(limit, 1e-10)}


@cache_fit
def expert_fit(bank, settings, menu, k):
    from .series import Unavailable
    config, reference, selection = select_reference(bank, settings, menu)
    branch, branch_selection = select_branch(bank, settings, menu, config, reference)
    memory, memory_reason = None, None
    try:
        memory = fit_memory(bank, settings, menu, k)
    except Unavailable as error:
        memory_reason = str(error)
    return {'config': config, 'reference': reference, 'selection': selection, 'branch': branch,
            'branch_selection': branch_selection, 'memory': memory, 'memory_reason': memory_reason,
            'support': fit_support(bank)}


def expert_predict(fitted, bank, transport=False):
    base = predict_reference(fitted['config'], fitted['reference'], bank)
    f, supported = support(fitted['support'], bank, transport)
    delta = predict_branch(fitted['branch'], fitted['config'], fitted['reference'], bank)
    neighbors = [[] for _ in base]
    residual, sf = np.zeros_like(base), np.zeros((len(base), 2))
    if fitted['memory'] is not None:
        residual, sf, neighbors = retrieve(fitted['memory'], bank)
    experts = np.stack((base, base + supported[:, None] * delta, base + supported[:, None] * residual), axis=1)
    return experts, np.column_stack((f, sf)), supported, neighbors


@cache_fit
def expert_oof(bank, settings, menu, k):
    from .series import Unavailable
    assignment = folds(bank, settings)
    predictions = np.empty((len(bank['responses']), 3, bank['responses'].shape[1]))
    features = np.empty((len(predictions), 6))
    availability = np.ones(3, dtype=bool)
    records = []
    for fold in range(settings['crossfit_folds']):
        train, validation = take(bank, assignment != fold), take(bank, assignment == fold)
        if len(set(train['pair_ids'])) < settings['minimum_pairs'] or len(set(validation['pair_ids'])) < settings['minimum_partition_pairs']:
            raise Unavailable('Insufficient unique pairs in nested expert OOF partition')
        fitted = expert_fit(train, settings, menu, k)
        predictions[assignment == fold], features[assignment == fold], _, _ = expert_predict(fitted, validation)
        availability &= [True, fitted['branch'] is not None, fitted['memory'] is not None]
        records.append({'fold': fold, 'train_pair_ids': sorted(set(train['pair_ids'])),
                        'validation_pair_ids': sorted(set(validation['pair_ids'])),
                        'selection': fitted['selection'], 'branch_selection': fitted['branch_selection'],
                        'available': [True, fitted['branch'] is not None, fitted['memory'] is not None]})
    return predictions, features, availability, records


def router_weights(router, features, available):
    logits = ((features - router['mean']) / router['scale']) @ router['coefficients'].T + router['intercept']
    logits[:, ~available] = -np.inf
    logits -= np.max(logits, axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(1, keepdims=True)


def optimize_router(features, experts, y, w, available, settings, penalty, validation=None, epochs=None):
    mean, scale = features.mean(0), np.maximum(features.std(0), 1e-6)
    x = torch.as_tensor((features - mean) / scale, dtype=torch.float64)
    e, target, wt = [torch.as_tensor(v, dtype=torch.float64) for v in (experts, y, w)]
    torch.manual_seed(settings['training_seed'])
    network = nn.Linear(features.shape[1], 3).double()
    optimizer = torch.optim.Adam(network.parameters(), lr=settings['learning_rate'])
    mask = torch.as_tensor(~available)
    best, state, best_epoch, stale, history = float('inf'), None, 0, 0, []
    def prediction(inputs, expert):
        logits = network(inputs).masked_fill(mask, -torch.inf)
        return (torch.softmax(logits, 1)[:, :, None] * expert).sum(1)
    for epoch in range(1, (epochs or settings['epoch_cap']) + 1):
        optimizer.zero_grad()
        loss = (wt * (prediction(x, e) - target).square().mean(1)).sum()
        objective = loss + penalty * sum(v.square().sum() for v in network.parameters()) / sum(v.numel() for v in network.parameters())
        if not torch.isfinite(objective):
            raise ValueError('Nonfinite router optimization')
        objective.backward()
        optimizer.step()
        score = None
        if validation is not None:
            vf, ve, vy, vw = validation
            with torch.no_grad():
                vp = prediction(torch.as_tensor((vf - mean) / scale, dtype=torch.float64), torch.as_tensor(ve, dtype=torch.float64))
                score = float((torch.as_tensor(vw) * (vp - torch.as_tensor(vy)).square().mean(1)).sum())
            if score < best - settings['minimum_improvement']:
                best, best_epoch, stale = score, epoch, 0
                state = copy.deepcopy(network.state_dict())
            else:
                stale += 1
        history.append({'epoch': epoch, 'train_mse': float(loss.detach()), 'validation_mse': score})
        if validation is not None and stale >= settings['patience']:
            break
    if state is not None:
        network.load_state_dict(state)
    router = {'mean': mean, 'scale': scale, 'coefficients': network.weight.detach().numpy().copy(),
              'intercept': network.bias.detach().numpy().copy()}
    return router, {'penalty': penalty, 'selected_epoch': best_epoch if validation is not None else (epochs or settings['epoch_cap']),
                    'validation_mse': best if validation is not None else None, 'history': history}


def fit_guard(bank, settings, menu, fitted):
    from .series import Unavailable
    if len(set(bank['pair_ids'])) < settings['minimum_router_pairs']:
        return dict(fitted, router=None, expert_mask=np.asarray([True, False, False]),
                    gate_selection={'fallback': 'Fewer than 24 acquired unique pairs'})
    train, validation = partition(bank, settings)
    attempts, objects = [], []
    for k in settings['retrieval_k']:
        try:
            te, tf, available, records = expert_oof(train, settings, menu, k)
            ve_fit = expert_fit(train, settings, menu, k)
            ve, vf, _, _ = expert_predict(ve_fit, validation)
            available &= [True, ve_fit['branch'] is not None, ve_fit['memory'] is not None]
            for penalty in settings['gate_regularization']:
                router, record = optimize_router(tf, te, train['responses'], weights(train), available, settings, penalty,
                                                 (vf, ve, validation['responses'], weights(validation)))
                record.update(k=k, available=available.tolist(), nested_records=records)
                attempts.append(record)
                objects.append((router, available))
        except Unavailable as error:
            attempts.append({'k': k, 'unavailable': str(error)})
            objects.append(None)
    eligible = [i for i, r in enumerate(attempts) if 'validation_mse' in r]
    if not eligible:
        return dict(fitted, router=None, expert_mask=np.asarray([True, False, False]), gate_selection={'attempts': attempts, 'fallback': 'Nested expert support unavailable'})
    chosen = min(eligible, key=lambda i: (attempts[i]['validation_mse'], attempts[i]['k'], attempts[i]['penalty']))
    best = attempts[chosen]
    full_experts, full_features, available, full_records = expert_oof(bank, settings, menu, best['k'])
    # Availability can shrink on refit, but an expert absent during selection cannot appear.
    available &= np.asarray(best['available'])
    final = expert_fit(bank, settings, menu, best['k'])
    available &= [True, final['branch'] is not None, final['memory'] is not None]
    router, refit = optimize_router(full_features, full_experts, bank['responses'], weights(bank), available,
                                    settings, best['penalty'], epochs=best['selected_epoch'])
    final.update(router=router, expert_mask=available, gate_selection={'attempts': attempts, 'selected': {key: best[key] for key in ('k', 'penalty', 'selected_epoch')},
                 'router_train_pair_ids': sorted(set(train['pair_ids'])), 'router_validation_pair_ids': sorted(set(validation['pair_ids'])),
                 'full_oof_records': full_records, 'refit': refit})
    return final


def fit(model, development, contract):
    FIT_CACHE.clear()
    bank = model._prepare(development, contract)
    model.cycle = 2
    config, reference, selection = select_reference(bank, model.settings, model.menu)
    fitted = {'config': config, 'reference': reference, 'selection': selection, 'support': fit_support(bank),
              'branch': None, 'memory': None, 'router': None, 'expert_mask': np.asarray([True, False, False])}
    if model.name != 'Opal-Base':
        fitted['branch'], fitted['branch_selection'] = select_branch(bank, model.settings, model.menu, config, reference)
        fitted['expert_mask'][1] = fitted['branch'] is not None
    if model.name == 'Opal-Guard':
        fitted = fit_guard(bank, model.settings, model.menu, fitted)
    model.fitted = fitted
    model.config, model.selection = fitted['config'], fitted['selection']
    for key in ('branch_selection', 'gate_selection'):
        if key in fitted:
            setattr(model, key, fitted[key])
    model.contexts = fitted['support']['contexts']
    model.degrees = fitted['support']['degrees']
    model.train_profiles = fitted['support']['profiles']
    model.input_mean, model.input_scale = fitted['support']['mean'], fitted['support']['scale']
    return model


def parameter_counts(fitted):
    """Separate fitted response coefficients, representations and local memory."""
    reference = fitted['reference']
    if 'legacy' in reference:
        reference = vars(reference['legacy']) if reference['legacy'] is not None else {}
    def count(state, keys):
        return sum(int(np.asarray(state[k]).size) for k in keys if k in state and state[k] is not None)
    coefficients = count(reference, ('coefficients', 'intercept', 'gamma'))
    if 'intercept' not in reference:
        coefficients += count(reference, ('mean',))
    representations = count(reference.get('encoder', {}), ('basis', 'mean'))
    branch = fitted['branch']
    if branch is not None:
        coefficients += count(branch, ('coefficients', 'intercept'))
        representations += count(branch, ('output_basis', 'projection', 'feature_mean', 'feature_scale'))
        representations += count(branch['encoder'], ('basis', 'mean'))
    if fitted['router'] is not None:
        coefficients += count(fitted['router'], ('coefficients', 'intercept'))
        representations += count(fitted['router'], ('mean', 'scale'))
    memory = fitted['memory']
    return {'fitted_response_coefficient_values': coefficients,
            'fitted_representation_values': representations,
            'acquired_residual_memory_values': 0 if memory is None else int(memory['residuals'].size),
            'support_reference_values': int(fitted['support']['profiles'].size),
            'counting_rule': 'Coefficient/intercept values separate from PCA/SVD/projection/scales and stored acquired references/residuals'}


def predict(model, bank, contract):
    if contract.get('reference_provenance') != model.contract['reference_provenance']:
        raise ValueError('Changed reference provenance')
    x = model._query(bank, contract)
    transport = contract.get('transport_mode') == 'same-context'
    if transport:
        if not model.contract.get('context_contract') or contract.get('context_contract') != model.contract['context_contract']:
            raise ValueError('Same-context transport requires unchanged explicit context contract')
    if contract.get('context_contract') != model.contract.get('context_contract'):
        raise ValueError('Changed source/TNF context contract')
    experts, features, supported, neighbors = expert_predict(model.fitted, x, transport)
    mixture = np.zeros((len(x['pair_ids']), 3))
    mixture[:, 0] = 1
    if model.name == 'Opal-Interaction' and model.fitted['branch'] is not None:
        mixture[supported] = [0, 1, 0]
    if model.name == 'Opal-Guard' and model.fitted['router'] is not None:
        mixture[supported] = router_weights(model.fitted['router'], features[supported], model.fitted['expert_mask'])
    prediction = np.sum(mixture[:, :, None] * experts, axis=1)
    result = model._result(x, prediction, supported, mixture[:, 1])
    result.update(expert_weights=mixture, neighbor_attribution=neighbors,
                  expert_availability=model.fitted['expert_mask'].tolist(),
                  reference_family=model.config['kind'], architecture_id=model.settings['architecture_id'])
    result['provenance']['checkpoint_format'] = 2
    result['provenance']['checkpoint_role'] = model.contract.get('checkpoint_role', 'evaluation')
    result['provenance']['parameter_counts'] = parameter_counts(model.fitted)
    return result


def save(model, directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    arrays = {}
    # Safe content-addressed rows share full-gene reference vectors across
    # checkpoints without changing fitted arithmetic or older checkpoints.
    store = model.settings.get('checkpoint_array_store')
    store = Path(store).resolve() if store else None
    store_hashes = {}
    logical_pooled_values = 0
    if store:
        store.mkdir(parents=True, exist_ok=True)
    def pack(value):
        nonlocal logical_pooled_values
        if isinstance(value, np.ndarray):
            if store and value.ndim == 2 and value.shape[1] >= 1000 and value.dtype.kind in 'fiu':
                rows = []
                for row in value:
                    key = array_hash(row)
                    destination = store / (key + '.npz')
                    if not destination.exists():
                        np.savez_compressed(destination, value=row)
                    else:
                        with np.load(destination, allow_pickle=False) as saved:
                            if array_hash(saved['value']) != key:
                                raise ValueError('Corrupt checkpoint array store')
                    rows.append(key)
                    store_hashes[key] = sha256(destination)
                logical_pooled_values += value.size
                return {'array_rows': rows, 'shape': list(value.shape), 'dtype': value.dtype.str}
            key = f'array_{len(arrays)}'
            arrays[key] = value
            return {'array': key}
        if isinstance(value, dict):
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                # Safe JSON histories compress with the NPZ, avoiding repeated
                # multi-megabyte text in every evaluation checkpoint.
                key = f'array_{len(arrays)}'
                arrays[key] = np.asarray([json.dumps(value, sort_keys=True)])
                return {'json_array': key}
            return [pack(v) for v in value]
        if hasattr(value, '__dict__'):
            return {'legacy_type': type(value).__name__, 'state': pack(vars(value))}
        return value
    state = pack({k: getattr(model, k) for k in ('settings', 'menu', 'contract', 'fit_hash', 'access', 'fitted')})
    np.savez_compressed(path / 'parameters.npz', **arrays)
    parameter_count = sum(v.size for v in arrays.values() if v.dtype.kind in 'fiu')
    metadata = {'model': model.name, 'format_version': 2, 'architecture_id': model.settings['architecture_id'],
                'state': state, 'stored_numeric_values': parameter_count, 'parameter_counts': parameter_counts(model.fitted),
                'parameters_sha256': sha256(path / 'parameters.npz')}
    if store:
        import os
        metadata.update(array_store=os.path.relpath(store, path.resolve()), array_store_hashes=store_hashes,
                        logical_numeric_values=parameter_count + logical_pooled_values)
    write_json(path / 'model.json', metadata)
    write_json(path / 'integrity.json', {'model_sha256': sha256(path / 'model.json'), 'parameters_sha256': metadata['parameters_sha256']})
    return path


def load(path, metadata, array_store=None):
    from .series import MODEL_REGISTRY, validate_contract
    from .replicated.models import ComponentAdditive, ProfileKernel
    if metadata['architecture_id'] != 'symmetric-bilinear-nested-memory':
        raise ValueError('Unknown safe checkpoint architecture')
    with np.load(path / 'parameters.npz', allow_pickle=False) as file:
        arrays = {k: file[k] for k in file.files}
    for value in arrays.values():
        if value.dtype.kind not in 'fiuUb' or (value.dtype.kind in 'fiu' and not np.isfinite(value).all()):
            raise ValueError('Invalid safe checkpoint arrays')
    def unpack(value):
        if isinstance(value, dict) and set(value) == {'array_rows', 'shape', 'dtype'}:
            if 'array_store' not in metadata or len(value['shape']) != 2:
                raise ValueError('Invalid shared-array checkpoint schema')
            directory = Path(array_store).resolve() if array_store else (path / metadata['array_store']).resolve()
            rows = []
            for key in value['array_rows']:
                if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                    raise ValueError('Invalid checkpoint array key')
                source = directory / (key + '.npz')
                if sha256(source) != metadata['array_store_hashes'].get(key):
                    raise ValueError('Shared checkpoint array hash mismatch')
                with np.load(source, allow_pickle=False) as saved:
                    row = saved['value']
                if (row.ndim != 1 or row.dtype.kind not in 'fiu' or not np.isfinite(row).all()
                        or array_hash(row) != key or row.dtype.str != value['dtype']):
                    raise ValueError('Invalid shared checkpoint array')
                rows.append(row)
            result = np.asarray(rows, dtype=value['dtype'])
            if list(result.shape) != value['shape']:
                raise ValueError('Shared checkpoint array shape mismatch')
            return result
        if isinstance(value, dict) and set(value) == {'array'}:
            return arrays[value['array']]
        if isinstance(value, dict) and set(value) == {'json_array'}:
            return json.loads(str(arrays[value['json_array']][0]))
        if isinstance(value, dict) and set(value) == {'legacy_type', 'state'}:
            state = unpack(value['state'])
            if value['legacy_type'] == 'ComponentAdditive':
                expert = ComponentAdditive(state['alpha'])
            elif value['legacy_type'] == 'ProfileKernel':
                expert = ProfileKernel(state['config'])
            else:
                raise ValueError('Unknown legacy expert type')
            vars(expert).update(state)
            return expert
        if isinstance(value, dict):
            return {k: unpack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [unpack(v) for v in value]
        return value
    state = unpack(metadata['state'])
    model = MODEL_REGISTRY[metadata['model']](state['settings'], state['menu'])
    vars(model).update(state)
    validate_contract(model.contract, len(model.contract['genes']))
    model.cycle = 2
    model.config, model.selection = model.fitted['config'], model.fitted['selection']
    for key in ('branch_selection', 'gate_selection'):
        if key in model.fitted:
            setattr(model, key, model.fitted[key])
    model.contexts = model.fitted['support']['contexts']
    model.degrees = model.fitted['support']['degrees']
    model.train_profiles = model.fitted['support']['profiles']
    model.input_mean, model.input_scale = model.fitted['support']['mean'], model.fitted['support']['scale']
    return model
