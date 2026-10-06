"""One error implementation and explicitly conditional uncertainty summaries."""
import numpy as np


def cosine(x, y, tolerance=1e-12):
    den = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return np.divide(np.sum(x * y, axis=-1), den, out=np.full(den.shape, np.nan), where=den > tolerance)


def metrics(prediction, target, additive, tolerance=1e-12):
    error = np.mean((prediction - target) ** 2, axis=1)
    base = np.mean((additive - target) ** 2, axis=1)
    alignment = cosine(prediction - additive, target - additive, tolerance)
    pearson = cosine(prediction - prediction.mean(1, keepdims=True), target - target.mean(1, keepdims=True))
    skill = None if base.sum() <= tolerance else float(1 - error.sum() / base.sum())
    return {"mse": float(error.mean()), "mae": float(np.abs(prediction - target).mean()),
            "skill_additive": skill, "cosine": float(np.nanmean(alignment)) if np.isfinite(alignment).any() else None,
            "cosine_defined_pairs": int(np.isfinite(alignment).sum()),
            "pearson": float(np.nanmean(pearson)) if np.isfinite(pearson).any() else None,
            "residual_norm_error": float(np.mean(np.abs(np.linalg.norm(prediction - additive, axis=1) - np.linalg.norm(target - additive, axis=1)))),
            "pairs": len(target)}, error


def paired_interval(differences, pairs, seed=42, draws=2000):
    rng = np.random.default_rng(seed)
    differences = np.asarray(differences)
    n = len(differences)
    conditional = differences[rng.integers(n, size=(draws, n))].mean(1)
    targets = sorted(set(pairs.ravel().tolist()))
    cluster = []
    for _ in range(draws):
        sampled = rng.choice(targets, len(targets), replace=True)
        counts = {t: int(sum(sampled == t)) for t in targets}
        weight = np.array([counts[a] * counts[b] for a, b in pairs])
        if weight.sum():
            cluster.append(float(np.average(differences, weights=weight)))
    return {"paired_mean_improvement": float(differences.mean()),
            "conditional_pair_bootstrap_95": np.quantile(conditional, [0.025, 0.975]).tolist(),
            "target_product_weight_bootstrap_95": np.quantile(cluster, [0.025, 0.975]).tolist() if cluster else None,
            "draws": draws, "interpretation": "descriptive conditional sensitivity; shared controls and biological replication not captured"}


def component_panel(singles, pairs, genes, top=20, control_sd=None, alpha=0.1):
    scale = np.ones(singles.shape[1]) if control_sd is None else np.asarray(control_sd) + alpha
    score = np.abs(singles) / scale
    rankings = [np.lexsort((genes, -row))[:top] for row in score]
    return [np.union1d(rankings[a], rankings[b]) for a, b in pairs]


def geometry(singles, pairs, response, tau):
    a, b = singles[pairs[:, 0]], singles[pairs[:, 1]]
    additive, residual = a + b, response - a - b
    rms = lambda v: np.sqrt(np.mean(v * v, axis=1))
    magnitude = rms(residual) / (rms(a) + rms(b) + tau)
    reinforcement = np.mean(residual * additive, axis=1) / (rms(additive) ** 2 + tau ** 2)
    orthogonal = []
    for x, y, z in zip(a, b, response):
        u, s, _ = np.linalg.svd(np.stack([x, y], axis=1), full_matrices=False)
        basis = u[:, s > max(1e-10, s.max() * 1e-8)]
        outside = z - basis @ (basis.T @ z)
        orthogonal.append(np.sqrt(np.mean(outside ** 2)) / (np.sqrt(np.mean(z ** 2)) + tau))
    return {"nonadditivity": magnitude, "reinforcement": reinforcement, "orthogonal": np.array(orthogonal)}


def joint_reference_bootstrap(controls, singles, doubles, pairs, seed=42):
    """Resample each shared reference once, then reuse it across its incident pairs.

    Inputs are already normalized cell-level expression in one matched stratum.
    Replicate-level resampling must be performed upstream for biological replicates.
    """
    rng = np.random.default_rng(seed)
    sample = lambda a: a[rng.integers(len(a), size=len(a))].mean(0)
    c = sample(controls)
    s = np.stack([sample(a) - c for a in singles])
    d = np.stack([sample(a) - c for a in doubles])
    return d - s[pairs[:, 0]] - s[pairs[:, 1]], s, c
