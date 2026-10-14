"""Bounded checks on retained predictions, without fitting or acceptance creation."""
import numpy as np


def checked_losses(predictions, truth, identities, population, seen):
    ids = np.asarray(identities)
    truth = np.asarray(truth)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(set(ids.tolist())) != len(ids):
        raise ValueError('Unique integer query identities required')
    if np.any(ids < 0) or np.any(ids >= population) or seen[ids].any():
        raise ValueError('Duplicated or out-of-population query identities')
    if truth.ndim != 2 or len(truth) != len(ids) or not np.isfinite(truth).all():
        raise ValueError('Finite complete evaluator required')
    losses = {}
    for name, value in predictions.items():
        value = np.asarray(value)
        if value.shape != truth.shape or not np.isfinite(value).all():
            raise ValueError('Finite complete prediction population required')
        losses[name] = np.mean((value - truth)**2, axis=1)
    seen[ids] = True
    return losses


def conditional_bounds(candidate, reference, draws):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.ndim != 1 or candidate.shape != reference.shape or not np.isfinite([candidate, reference]).all():
        raise ValueError('Finite paired losses required')
    return np.quantile((reference-candidate)[draws].mean(1), [.025, .975])


def measurement_summary(record):
    rows = record['rows']
    if len(rows) != record['declared_population']:
        raise ValueError('Incomplete declared measurement population')
    keys = [(r['context'], r['drug']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicated measurement identity')
    names = ('response_half_mse', 'residual_half_mse', 'zero_response_mse', 'zero_residual_mse')
    contexts = {}
    for context in sorted({r['context'] for r in rows}):
        selected = [r for r in rows if r['context'] == context]
        supported = [r for r in selected if r['status'] == 'technical_support_qualified']
        if any(r['status'] not in ('technical_support_qualified', 'unsupported') for r in selected):
            raise ValueError('Unknown measurement status')
        if any(not r.get('reasons') for r in selected if r['status'] == 'unsupported'):
            raise ValueError('Unsupported measurement lacks reason')
        if any(not np.isfinite(r[n]) or r[n] < 0 for r in supported for n in names):
            raise ValueError('Invalid measurement error')
        contexts[context] = dict(declared=len(selected), supported=len(supported),
            means={n: float(np.mean([r[n] for r in supported])) if supported else None for n in names})
    count = sum(c['supported'] for c in contexts.values())
    if count != record['supported_population']:
        raise ValueError('Supported population mismatch')
    return contexts
