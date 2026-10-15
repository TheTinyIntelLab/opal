"""Validate recovered aggregate axes independently of raw reconstruction."""
import numpy as np


def validate_aggregates(sums, support, totals, mapping):
    if mapping.ndim != 1 or mapping.dtype.kind not in 'iu' or totals.shape != mapping.shape:
        raise ValueError('Original cell-axis membership required')
    if sums.ndim != 3 or support.shape != sums.shape[:2] or sums.shape[1] != 2:
        raise ValueError('Aggregate partition axes mismatch')
    if np.any(mapping < -1) or np.any(mapping >= len(sums)):
        raise ValueError('Cell-to-condition index outside declared axes')
    if not np.isfinite(totals).all() or np.any(totals < 0) or np.any(totals[mapping >= 0] <= 0):
        raise ValueError('Invalid full-library denominators')
    counted = np.zeros_like(support)
    valid = mapping >= 0
    np.add.at(counted, (mapping[valid], np.flatnonzero(valid) % 2), 1)
    if not np.array_equal(counted, support):
        raise ValueError('Retained membership/support mismatch')
