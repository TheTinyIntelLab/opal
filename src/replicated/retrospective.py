"""Exposed-source nested pair prediction; no prospective inference or gates."""
import json

import numpy as np

from .models import fit_candidate, predict_candidate, select, shuffled_profiles


def pair_folds(pair_ids, seed=20261009, folds=3):
    """Assign identities once, so every run and variant uses the same fold."""
    identities = sorted(set(map(str, pair_ids)))
    if folds < 3 or len(identities) < folds * 2:
        raise ValueError('At least three folds with two identities each required')
    order = np.random.default_rng(seed).permutation(len(identities))
    assignment = {identities[int(i)]: j % folds for j, i in enumerate(order)}
    return np.asarray([assignment[str(p)] for p in pair_ids])


def reference_gene_panel(singles, genes, maximum=512):
    """Fixed endpoint from observed monotherapies only, never pair responses.

    Deduplicate component profiles within each run, avoiding graph-degree
    weighting. All query singles are permitted under the observed-single task.
    This is a reference-conditional endpoint, not an all-gene endpoint.
    """
    x = np.asarray(singles, float)
    if x.ndim != 4 or x.shape[2:] != (2, len(genes)) or not np.isfinite(x).all():
        raise ValueError('Expected finite run/pair/two-single/gene references')
    if maximum < 1 or not genes or len(set(genes)) != len(genes):
        raise ValueError('Positive panel size and unique genes required')
    profiles = np.concatenate([np.unique(run.reshape(-1, len(genes)), axis=0) for run in x])
    variance = profiles.var(axis=0)
    order = sorted(range(len(genes)), key=lambda i: (-variance[i], str(genes[i])))
    return np.asarray(order[:min(maximum, len(genes))], dtype=int)


def nested_pair_benchmark(bank, menu, seed=20261009, shuffle_seeds=(11, 29, 47)):
    """Three outer folds, rotating inner validation, separate fits per run.

    Scaling and kernel distances are fitted inside each inner training bank.
    Penalty/family selection uses inner validation only. Outer predictions are
    made after refitting on both remaining folds. No cross-run pair outcomes
    enter any prediction; folds are computational splits, never cultures.
    """
    pair_ids = np.asarray(bank['pair_ids'])
    cultures = np.asarray(bank['culture_ids'])
    y = np.asarray(bank['responses'], float)
    singles = np.asarray(bank['singles'], float)
    if (y.ndim != 2 or singles.shape != (len(y), 2, y.shape[1])
            or len(pair_ids) != len(y) or len(cultures) != len(y)
            or not np.isfinite(y).all() or not np.isfinite(singles).all()):
        raise ValueError('Invalid retrospective bank')
    if len(set(zip(cultures, pair_ids))) != len(y):
        raise ValueError('Collapse technical variants before fitting')
    for pair in pair_ids:
        components = json.loads(str(pair))
        if len(components) != 2 or components != sorted(set(components)):
            raise ValueError('Canonical distinct two-component identities required')
    fold_ids = pair_folds(pair_ids, seed)
    variants = [('aligned', bank)] + [(f'shuffle-{s}', shuffled_profiles(bank, s)) for s in shuffle_seeds]
    records, predictions = [], {}
    for label, inputs in variants:
        predicted = {role: np.full_like(y, np.nan) for role in ('reference', 'candidate')}
        for run in sorted(set(cultures)):
            for outer in range(3):
                test = (cultures == run) & (fold_ids == outer)
                development = (cultures == run) & (fold_ids != outer)
                inner = {k: np.asarray(inputs[k])[development] for k in
                         ('singles', 'responses', 'pair_ids', 'culture_ids')}
                inner['partitions'] = np.where(fold_ids[development] == (outer + 1) % 3,
                                               'validation', 'train')
                selection = select(inner, menu)
                query = {k: np.asarray(inputs[k])[test] for k in ('singles', 'pair_ids', 'culture_ids')}
                record = {'variant': label, 'run': str(run), 'outer_fold': outer,
                          'test_pairs': pair_ids[test].tolist(), 'selection': selection, 'losses': {}}
                for role in ('reference', 'candidate'):
                    config = selection[role]
                    model = fit_candidate(config, inner, np.ones(len(inner['pair_ids']), dtype=bool))
                    estimate = predict_candidate(config, model, query)
                    predicted[role][test] = estimate
                    record['losses'][role] = float(np.mean((estimate - y[test]) ** 2))
                records.append(record)
        if any(not np.isfinite(value).all() for value in predicted.values()):
            raise ValueError('Incomplete outer predictions')
        predictions[label] = predicted
    per_run = []
    for run in sorted(set(cultures)):
        mask = cultures == run
        losses = {'zero': float(np.mean(y[mask] ** 2)),
                  'additive': float(np.mean((singles[mask].sum(1) - y[mask]) ** 2))}
        for label, values in predictions.items():
            for role, estimate in values.items():
                losses[f'{label}/{role}'] = float(np.mean((estimate[mask] - y[mask]) ** 2))
        per_run.append({'run': str(run), 'pair_count': int(mask.sum()), 'mse': losses})
    return {'role': 'exposed_source_retrospective_nominal_pair_prediction',
            'primary_claim_pass': False, 'biological_measurement_gate_pass': False,
            'new_confirmation_performed': False, 'confidence_intervals_computed': False,
            'fold_assignment': {str(p): int(f) for p, f in zip(pair_ids, fold_ids)},
            'per_run': per_run, 'fold_records': records,
            'equal_run_mean_mse': {key: float(np.mean([r['mse'][key] for r in per_run]))
                                   for key in per_run[0]['mse']}}, predictions
