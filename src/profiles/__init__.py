"""Fixed library normalization and matched-stratum population effects."""
import numpy as np
from scipy import sparse


def validate_counts(x, declared_provenance):
    if not declared_provenance or not declared_provenance.strip():
        raise ValueError("Explicit raw-count provenance is required; integer values are not proof")
    values = x.data if sparse.issparse(x) else np.asarray(x)
    if x.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Counts must be a finite nonnegative 2D matrix")
    if not np.allclose(values, np.round(values), atol=1e-7, rtol=0):
        raise ValueError("Declared raw counts contain noninteger values")


def library_normalize(x, target):
    totals = np.asarray(x.sum(axis=1)).ravel()
    if np.any(totals <= 0) or not np.isfinite(target) or target <= 0:
        raise ValueError("Positive cell totals and a positive frozen normalizer are required")
    scale = target / totals
    return sparse.diags(scale) @ x if sparse.issparse(x) else np.asarray(x) * scale[:, None]


def effect(mean, control, scale="log2"):
    if scale == "log2":
        return np.log2(mean + 1) - np.log2(control + 1)
    if scale == "count":
        return mean - control
    raise ValueError("Effect scale must be log2 or count")


def mean_profile(x):
    return np.asarray(x.mean(axis=0)).ravel()


def eligible_genes(reference_counts, reference_labels, minimum_mean=0.1, minimum_detection=0.01):
    """Only controls and singles may be passed by the caller."""
    keep = np.zeros(reference_counts.shape[1], dtype=bool)
    for label in sorted(set(reference_labels)):
        rows = np.flatnonzero(reference_labels == label)
        part = reference_counts[rows]
        detection = np.asarray((part > 0).mean(axis=0)).ravel()
        keep |= (mean_profile(part) >= minimum_mean) & (detection >= minimum_detection)
    return keep
