"""Exact within-source guide/arm singles for already processed gamma.

Transcript and intentional mismatch annotations are explicitly unavailable.
This adapter never interprets gamma as raw counts or uses double targets for singles.
"""
import csv
import gzip
from collections import Counter
import numpy as np

COLUMNS = ((13, 14), (4, 5))
CONTEXTS = ('K562', 'Jurkat')


def library(path):
    guides = {}
    sequences = set()
    with gzip.open(path, 'rt') as stream:
        for row in csv.reader(stream, delimiter='\t'):
            if len(row) != 9 or row[1] != row[2] or row[3] != 'Y':
                raise ValueError('Guide library schema differs')
            name, sequence = row[1], row[5]
            if name in guides or sequence in sequences or len(sequence) != 20 or set(sequence)-set('ACGT'):
                raise ValueError('Unique named 20nt guide sequence required')
            sequences.add(sequence)
            guides[name] = dict(index=len(guides), gene=name.split('_')[0], sequence=sequence,
                control=name.startswith('negative_control-'), transcript=None, mismatch=None,
                barcode_A=row[6], barcode_B=row[7], vector=row[8])
    if not guides or not any(g['control'] for g in guides.values()):
        raise ValueError('Explicit controls required')
    return guides


def rows(path, guides, check=lambda: None):
    """Join literal IDs before converting four explicitly declared outcomes."""
    seen = np.zeros((len(guides), len(guides)), dtype=bool)
    with gzip.open(path, 'rt') as stream:
        reader = csv.reader(stream, delimiter='\t')
        headers = [next(reader) for _ in range(3)]
        for c, columns in enumerate(COLUMNS):
            for r, col in enumerate(columns):
                if [h[col] for h in headers] != [CONTEXTS[c], 'Triple Sequencing', f'Replicate {r+1}']:
                    raise ValueError('Context/mode/infection header mismatch')
        for number, row in enumerate(reader):
            if number % 2048 == 0:
                check()
            if not row or not row[0]:
                continue
            if len(row) <= max(max(c) for c in COLUMNS):
                raise ValueError('Truncated phenotype row')
            names = row[0].split('++')
            if len(names) != 2 or any(name not in guides for name in names):
                raise ValueError('Unmatched exact guide identifier')
            a, b = (guides[name] for name in names)
            key = a['index'], b['index']
            if seen[key]:
                raise ValueError('Duplicate ordered construct')
            seen[key] = True
            values = np.full((2, 2), np.nan)
            for c, columns in enumerate(COLUMNS):
                for r, col in enumerate(columns):
                    try:
                        value = float(row[col])
                    except ValueError:
                        continue
                    if np.isfinite(value):
                        values[c, r] = value
            yield a, b, values


def collect_singles(iterator, n_guides, minimum=2):
    if minimum < 1:
        raise ValueError('Positive control support required')
    sums = np.zeros((2, 2, n_guides, 2))
    counts = np.zeros(sums.shape, dtype=np.int32)
    tally = Counter()
    for a, b, values in iterator:
        tally['source_rows'] += 1
        if a['control'] == b['control']:
            continue
        target, arm = (b, 1) if a['control'] else (a, 0)
        valid = np.isfinite(values)
        sums[:, :, target['index'], arm] += np.where(valid, values, 0)
        counts[:, :, target['index'], arm] += valid
        tally['target_control_rows'] += 1
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts >= minimum)
    return means, counts, dict(tally)


def panel_predictions(a, b):
    """Equal ordered-construct weights; symmetric canonical endpoint columns."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite paired guide singles required')
    return np.array([a.mean(), b.mean(), np.minimum(a, b).mean(),
                     min(a.mean(), b.mean()), (a+b).mean()])


def aggregate(iterator, genes, pairs, singles, minimum=2):
    """Keep the historical roster intact, exposing unsupported cells as NaN."""
    genes, pairs = np.asarray(genes), np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError('Integer canonical pair roster required')
    if np.any(pairs < 0) or np.any(pairs >= len(genes)) or np.any(pairs[:, 0] >= pairs[:, 1]):
        raise ValueError('Canonical pair indices required')
    names = [(str(genes[a]), str(genes[b])) for a, b in pairs]
    if len(set(names)) != len(names) or any(a >= b for a, b in names):
        raise ValueError('Unique canonical pair identities required')
    if minimum < 1:
        raise ValueError('Positive construct support required')
    lookup = {name: i for i, name in enumerate(names)}
    sums = np.zeros((2, 2, len(pairs), 6))
    counts = np.zeros((2, 2, len(pairs)), dtype=np.int32)
    raw_counts = np.zeros_like(counts)
    orientations = np.zeros((2, 2, len(pairs), 2), dtype=np.int32)
    orientation_target = np.zeros_like(orientations, dtype=float)
    tally = Counter()
    for a, b, values in iterator:
        if a['control'] or b['control'] or a['gene'] == b['gene']:
            continue
        pair = tuple(sorted((a['gene'], b['gene'])))
        if pair not in lookup:
            tally['outside_historical_roster_rows'] += 1
            continue
        i = lookup[pair]
        x = singles[:, :, a['index'], 0]
        y = singles[:, :, b['index'], 1]
        valid = np.isfinite(values)
        raw_counts[:, :, i] += valid
        matched = valid & np.isfinite(x) & np.isfinite(y)
        canonical_x, canonical_y = (x, y) if a['gene'] == pair[0] else (y, x)
        orientation = int(a['gene'] != pair[0])
        features = np.stack((values, canonical_x, canonical_y, np.minimum(x, y), x+y,
                            np.abs(x-y)), axis=-1)
        sums[:, :, i] += np.where(matched[:, :, None], features, 0)
        counts[:, :, i] += matched
        orientations[:, :, i, orientation] += matched
        orientation_target[:, :, i, orientation] += np.where(matched, values, 0)
    means = np.divide(sums, counts[..., None], out=np.full_like(sums, np.nan),
                      where=counts[..., None] >= minimum)
    orientation_means = np.divide(orientation_target, orientations,
        out=np.full_like(orientation_target, np.nan), where=orientations > 0)
    # target, endpoint singles, construct-first min, sum, mean absolute difference
    return dict(means=means, counts=counts, raw_counts=raw_counts,
                eligible=counts >= minimum, missing_matched_single_counts=raw_counts-counts,
                orientation_counts=orientations, orientation_means=orientation_means), dict(tally)
