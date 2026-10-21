"""Outcome-blind nested pair-label budgets and common lookup support."""
import hashlib

import numpy as np


BUDGETS = ((5000, 4000, 1000), (20000, 16000, 4000))


def budget_split(screen, original, train_labels, validation_labels):
    """Reserve the lowest full-SHA pair hashes within the original partitions.

    -1 is inaccessible, 0 reserved training, 1 reserved validation, 2 the
    unchanged original test population. No support-driven edge is added.
    """
    pairs, genes = np.asarray(screen['pairs']), np.asarray(screen['genes'])
    original = np.asarray(original)
    if original.shape != (len(pairs),) or not np.isin(original, [0, 1, 2]).all():
        raise ValueError('Exact original train/validation/test membership required')
    if train_labels < 2 or validation_labels < 3:
        raise ValueError('Positive declared label budgets required')
    hashes = [hashlib.sha256(f'opal-budget-v1|{screen["accession"]}|{genes[a]}|{genes[b]}'.encode()).hexdigest()
              for a, b in pairs]
    result = np.full(len(pairs), -1, dtype=np.int8)
    result[original == 2] = 2
    for part, count in ((0, train_labels), (1, validation_labels)):
        rows = np.flatnonzero(original == part)
        if len(rows) < count:
            raise ValueError('Original partition cannot supply the registered budget')
        order = sorted(rows.tolist(), key=lambda row: hashes[row])
        result[order[:count]] = part
    return result


def supported_queries(pairs, split, n_genes):
    pairs, split = np.asarray(pairs), np.asarray(split)
    if split.shape != (len(pairs),) or not np.isin(split, [-1, 0, 1, 2]).all():
        raise ValueError('Aligned budget membership required')
    support = np.bincount(pairs[split == 0].ravel(), minlength=n_genes) > 0
    supported = support[pairs].all(1)
    return dict(support=support, train=np.flatnonzero(split == 0),
                validation=np.flatnonzero((split == 1) & supported),
                test=np.flatnonzero((split == 2) & supported),
                full_test=np.flatnonzero(split == 2),
                reserved_validation=int((split == 1).sum()),
                unused_validation=int(((split == 1) & ~supported).sum()))


def masked_native_inputs(screen, split):
    """Pass only supported rows; unavailable and test labels remain NaN."""
    q = supported_queries(screen['pairs'], split, len(screen['genes']))
    rows = np.r_[q['train'], q['validation'], q['test']]
    membership = np.r_[np.zeros(len(q['train']), dtype=np.int8),
                       np.ones(len(q['validation']), dtype=np.int8),
                       np.full(len(q['test']), 2, dtype=np.int8)]
    if min(len(q[k]) for k in ('train', 'validation', 'test')) < 3:
        raise ValueError('Insufficient common supported population')
    targets = np.full(len(rows), np.nan)
    used = membership != 2
    targets[used] = screen['responses'][rows[used]]
    return rows, targets, membership


def choose_family(targets, predictions):
    families = ('simple', 'rank8', 'native')
    targets = np.asarray(targets)
    if set(predictions) != set(families) or targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all():
        raise ValueError('Finite validation targets and exactly three families required')
    if any(np.asarray(predictions[k]).shape != targets.shape or not np.isfinite(predictions[k]).all() for k in families):
        raise ValueError('Aligned finite predictions required')
    losses = {k: float(np.mean((np.asarray(predictions[k])-targets)**2)) for k in families}
    return min(families, key=losses.__getitem__), losses


def scientific_gate(coverage, assessment):
    """A missing or failed task cannot become a smaller-budget nominee."""
    return bool(coverage >= .90 and assessment.get('passes_gate', False))


def nominee(results, tasks):
    for total, _, _ in BUDGETS:
        r = results.get(str(total), {})
        if set(r) == set(tasks) and all(v.get('passes_budget_gate', False) for v in r.values()):
            return total
    return None
