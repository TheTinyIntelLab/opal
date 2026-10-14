"""Source-only rank-four coordinates, rank-two modes, fixed target solve."""
import hashlib
import numpy as np

from .contracts import ContractError
from .splits import folds, source_permutation


class Unsupported(ContractError):
    """Complete query population must receive the frozen reference fallback."""


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def rank_check(x, rank=2):
    sv = np.linalg.svd(x, compute_uv=False)
    effective = int(np.sum(sv > sv[0] * 1e-8)) if len(sv) and sv[0] > 0 else 0
    condition = float(sv[0] / sv[rank - 1]) if effective >= rank else None
    record = dict(singular_values=sv.tolist(), effective_rank=effective, condition_number=condition)
    if effective < rank or condition > 1e4:
        raise Unsupported(f"Unsupported numerical rank/condition: {record}")
    return record


def source_weights(ids, strata=None):
    """Equal stratum, equal canonical pair; repeated rows share their weight."""
    ids = np.asarray(ids)
    strata = np.repeat("source", len(ids)) if strata is None else np.asarray(strata)
    if len(strata) != len(ids):
        raise ContractError("Source stratum/identity length mismatch")
    weights = np.zeros(len(ids), dtype=np.float64)
    for st in np.unique(strata):
        pairs = np.unique(ids[strata == st])
        for p in pairs:
            mask = (strata == st) & (ids == p)
            weights[mask] = 1 / (len(np.unique(strata)) * len(pairs) * mask.sum())
    return weights


def fit_encoder(singles, strata=None):
    strata = np.repeat("source", len(singles)) if strata is None else np.asarray(strata)
    blocks = [np.unique(np.asarray(singles[strata == st], dtype=np.float64).reshape(-1, singles.shape[-1]), axis=0)
              for st in np.unique(strata)]
    profiles = np.concatenate(blocks)
    weights = np.concatenate([np.repeat(1 / (len(blocks) * len(b)), len(b)) for b in blocks])
    mean = weights @ profiles
    scale = np.maximum(np.sqrt(weights @ ((profiles - mean) ** 2)), 1e-6)
    standardized = (profiles - mean) / scale
    weighted = standardized * np.sqrt(weights[:, None])
    _, sv, vt = np.linalg.svd(weighted, full_matrices=False)
    rank_check(weighted, 4)
    return dict(mean=mean, scale=scale, basis=vt[:4].T, singular_values=sv,
                scale_floor=1e-6, relative_rank_tolerance=1e-8)


def coordinates(singles, encoder):
    z = ((singles - encoder["mean"]) / encoder["scale"]) @ encoder["basis"]
    u, v = z[:, 0], z[:, 1]
    return np.column_stack([u[:, i] * v[:, i] if i == j else
                            (u[:, i] * v[:, j] + u[:, j] * v[:, i]) / np.sqrt(2)
                            for i in range(4) for j in range(i, 4)])


def ridge(x, y, alpha, weights=None):
    weights = np.repeat(1 / len(x), len(x)) if weights is None else np.asarray(weights) / np.sum(weights)
    xm, ym = weights @ x, weights @ y
    scale = np.maximum(np.sqrt(weights @ ((x - xm) ** 2)), 1e-6)
    xc, yc = (x - xm) / scale, y - ym
    coef = np.linalg.solve(xc.T @ (weights[:, None] * xc) + alpha * np.eye(x.shape[1]), xc.T @ (weights[:, None] * yc))
    return dict(center=xm, scale=scale, coefficients=coef, intercept=ym, alpha=float(alpha))


def ridge_predict(state, x):
    return (x - state["center"]) / state["scale"] @ state["coefficients"] + state["intercept"]


def fit_source_free(source_x, bridge_x, bridge_y, encoder, source_ids=None, strata=None):
    """Gene-averaged loss plus unit Frobenius penalty, fixed source scale."""
    source_phi = coordinates(source_x, encoder)
    weights = np.repeat(1 / len(source_x), len(source_x)) if source_ids is None else source_weights(source_ids, strata)
    center = weights @ source_phi
    scale = np.maximum(np.sqrt(weights @ ((source_phi - center) ** 2)), 1e-6)
    phi = (coordinates(bridge_x, encoder) - center) / scale
    residual = bridge_y - bridge_x.sum(1)
    xm, ym = phi.mean(0), residual.mean(0)
    xc, yc = phi - xm, residual - ym
    n = residual.size
    coefficients = np.linalg.solve(xc.T @ xc / n + np.eye(10), xc.T @ yc / n)
    # Bridge centering fits the same shared gene intercept. Scale and
    # coordinates remain source-only; no target representation is learned.
    return dict(center=center, scale=scale, coefficients=coefficients,
                intercept=ym - xm @ coefficients, alpha=1.,
                objective="mean_gene_bridge_squared_error_plus_frobenius_norm_squared")


NUISANCE_MENU = [("zero", 1), ("mean", 1)] + [("affine", a) for a in (.1, 1, 10)]


def fit_nuisance(x, y, kind, alpha=1, weights=None):
    weights = np.repeat(1 / len(x), len(x)) if weights is None else np.asarray(weights) / np.sum(weights)
    additive = x.sum(1)
    if kind == "zero":
        return dict(kind=kind, gamma=1., intercept=np.zeros(y.shape[1]), alpha=alpha)
    if kind == "mean":
        return dict(kind=kind, gamma=1., intercept=weights @ (y - additive), alpha=alpha)
    am, ym = weights @ additive, weights @ y
    ac, yc = additive - am, y - ym
    gamma = float(np.clip((weights @ np.mean(ac * yc, axis=1) + alpha) / (weights @ np.mean(ac ** 2, axis=1) + alpha), 0, 1.5))
    return dict(kind=kind, gamma=gamma, intercept=ym - gamma * am, alpha=alpha)


def nuisance_predict(state, x):
    return (state["gamma"] - 1) * x.sum(1) + state["intercept"]


def grouped_masks(ids, minimum=12, strata=None):
    assignment = folds(ids)
    for k in range(3):
        val = np.asarray([assignment[p] == k for p in ids])
        groups = np.repeat("source", len(ids)) if strata is None else np.asarray(strata)
        if any(len(set(ids[(groups == st) & ~val])) < minimum or not np.any((groups == st) & val) for st in np.unique(groups)):
            raise Unsupported("Insufficient source-inner fitting support")
        yield ~val, val


def select_nuisance(x, y, ids, strata=None):
    rows = []
    validation_weights = source_weights(ids, strata)
    for kind, alpha in NUISANCE_MENU:
        losses = []
        for train, val in grouped_masks(ids, strata=strata):
            fitted = fit_nuisance(x[train], y[train], kind, alpha, source_weights(ids[train], None if strata is None else strata[train]))
            losses.append(float(validation_weights[val] @ np.mean((x[val].sum(1) + nuisance_predict(fitted, x[val]) - y[val]) ** 2, axis=1)))
        rows.append(dict(kind=kind, alpha=alpha, mse=float(np.sum(losses))))
    best = min(rows, key=lambda r: (r["mse"], r["kind"], r["alpha"]))
    return fit_nuisance(x, y, best["kind"], best["alpha"], source_weights(ids, strata)), rows


def crossfit_nuisance(x, y, ids, strata=None):
    residuals, selections = np.empty_like(y), []
    for train, val in grouped_masks(ids, strata=strata):
        # Nested selection uses only this cross-fit training subset.
        fitted, rows = select_nuisance(x[train], y[train], ids[train], None if strata is None else strata[train])
        residuals[val] = y[val] - x[val].sum(1) - nuisance_predict(fitted, x[val])
        selections.append(dict(selected=fitted["kind"], alpha=fitted["alpha"], attempts=rows))
    return residuals, selections


def shuffled_residuals(residual, ids, strata, seed):
    """Permute response residual blocks, retaining their measured anchors."""
    mapping = source_permutation(ids, seed)
    index = {(str(st), str(p)): i for i, (st, p) in enumerate(zip(strata, ids))}
    if len(index) != len(ids):
        raise Unsupported("Residual shuffle requires unique pair/stratum blocks")
    try:
        order = [index[(str(st), mapping[str(p)])] for st, p in zip(strata, ids)]
    except KeyError as error:
        raise Unsupported("Shared source shuffle requires complete pair support in each stratum") from error
    result = residual[order].copy()
    def digest(values):
        values = np.ascontiguousarray(values, dtype=np.float64)
        return hashlib.sha256(str(values.shape).encode() + values.tobytes()).hexdigest()
    return result, dict(seed=int(seed), mapping=mapping,
                        original_residual_sha256=digest(residual), permuted_residual_sha256=digest(result),
                        fitting_ids=sorted(set(map(str, ids))),
                        interpretation="permute cross-fitted residual pair assignments within strata; measured additive and nuisance anchors frozen")


def fit_source(x, y, ids, alpha, strata=None, shuffle_seed=None):
    weights = source_weights(ids, strata)
    groups = np.repeat("source", len(ids)) if strata is None else np.asarray(strata)
    if any(len(set(ids[groups == st])) < 12 for st in np.unique(groups)):
        raise Unsupported("Insufficient source fitting identities")
    encoder = fit_encoder(x, groups)
    residual, selections = crossfit_nuisance(x, y, ids, groups)
    nuisance, final_selection = select_nuisance(x, y, ids, groups)
    shuffle_metadata = None
    if shuffle_seed is not None:
        residual, shuffle_metadata = shuffled_residuals(residual, ids, groups, shuffle_seed)
        shuffle_metadata["nuisance_selection"] = final_selection
        shuffle_metadata["crossfit_nuisance_selection"] = selections
    weighted_residual = residual * np.sqrt(weights[:, None])
    _, sv, vt = np.linalg.svd(weighted_residual, full_matrices=False)
    rank_check(weighted_residual, 2)
    basis = vt[:2].T
    mapping = ridge(coordinates(x, encoder), residual @ basis, alpha, weights)
    raw = ridge_predict(mapping, coordinates(x, encoder))[:, :, None] * basis.T[None, :, :]
    scales = np.sqrt(weights @ np.mean(raw ** 2, axis=2))
    if np.any(scales <= 1e-8) or not np.isfinite(scales).all():
        raise Unsupported("Unsupported source mode RMS")
    return dict(kind="source_modes", encoder=encoder, nuisance=nuisance, basis=basis,
                mapping=mapping, scales=scales, output_singular_values=sv,
                source_selection=final_selection, crossfit_selection=selections, source_ids=ids.copy(),
                shuffle_metadata=shuffle_metadata)


def mode_features(state, x):
    if state["kind"] == "mean_modes":
        coefficients = (x.sum(1) - state["additive_center"]) @ state["basis"]
    else:
        coefficients = ridge_predict(state["mapping"], coordinates(x, state["encoder"]))
    result = coefficients[:, :, None] * state["basis"].T[None, :, :] / state["scales"][None, :, None]
    if not np.isfinite(result).all():
        raise Unsupported("Nonfinite source extrapolation")
    return result


def fit_mean_modes(source, x, ids=None, strata=None):
    weights = np.repeat(1 / len(x), len(x)) if ids is None else source_weights(ids, strata)
    additive = x.sum(1)
    center = weights @ additive
    weighted = (additive - center) * np.sqrt(weights[:, None])
    _, sv, vt = np.linalg.svd(weighted, full_matrices=False)
    rank_check(weighted, 2)
    basis = vt[:2].T
    raw = ((additive - center) @ basis)[:, :, None] * basis.T[None, :, :]
    scales = np.sqrt(weights @ np.mean(raw ** 2, axis=2))
    if np.any(scales <= 1e-8):
        raise Unsupported("Unsupported mean mode RMS")
    return dict(kind="mean_modes", basis=basis, additive_center=center, scales=scales,
                nuisance=source["nuisance"], output_singular_values=sv)


def solve_target(features, residual):
    fc, ec = features - features.mean(0), residual - residual.mean(0)
    design = fc.transpose(0, 2, 1).reshape(-1, 2)
    diagnostic = rank_check(design)
    n = residual.size  # Four equally weighted bridge identities times G.
    eta = np.linalg.solve(design.T @ design / n + np.eye(2), design.T @ ec.reshape(-1) / n)
    return dict(eta=eta, feature_mean=features.mean(0), intercept=residual.mean(0), diagnostic=diagnostic)


def adapted_predict(source, adapter, x):
    f = mode_features(source, x) - adapter["feature_mean"]
    return x.sum(1) + nuisance_predict(source["nuisance"], x) + adapter["intercept"] + np.einsum("nlg,l->ng", f, adapter["eta"])


def adapt(source, bridge_x, bridge_y, query_x, residual_mapping=None):
    if len(bridge_x) != 4:
        raise ContractError("Exactly four acquired bridges required")
    features = mode_features(source, bridge_x)
    residual = bridge_y - bridge_x.sum(1) - nuisance_predict(source["nuisance"], bridge_x)
    if residual_mapping is not None:
        residual = residual[residual_mapping]
    adapter = solve_target(features, residual)
    query_features = mode_features(source, query_x)
    full = np.einsum("nlg,l->ng", query_features - adapter["feature_mean"], adapter["eta"])
    full_rms = rms(full)
    stability = []
    for leave in range(4):
        mask = np.arange(4) != leave
        loo = solve_target(features[mask], residual[mask])
        correction = np.einsum("nlg,l->ng", query_features - loo["feature_mean"], loo["eta"])
        stability.append(dict(left_out=leave, eta=loo["eta"].tolist(),
                              mode_change_rms=rms(correction - full),
                              intercept_change_rms=rms(loo["intercept"] - adapter["intercept"]),
                              diagnostic=loo["diagnostic"]))
    adapter["stability"] = stability
    adapter["full_mode_rms"] = full_rms
    adapter["zero_mode_branch"] = full_rms < 1e-8
    if full_rms < 1e-8:
        adapter["eta"] = np.zeros(2)
    elif max(r["mode_change_rms"] for r in stability) > full_rms:
        raise Unsupported("Leave-one-bridge mode correction is unstable")
    return adapter


def select_source(x, y, ids, attempt=None, strata=None, shuffle_seed=None):
    rows = []
    validation_weights = source_weights(ids, strata)
    # Outer source-inner fits re-estimate every nuisance, scaler, and mode.
    for alpha in (.1, 1, 10):
        errors, failure = [], None
        for train, val in grouped_masks(ids, strata=strata):
            try:
                source = fit_source(x[train], y[train], ids[train], alpha, None if strata is None else strata[train], shuffle_seed)
                pred = x[val].sum(1) + nuisance_predict(source["nuisance"], x[val])
                pred += np.einsum("nlg,l->ng", mode_features(source, x[val]), source["scales"])
                validation_y = y[val]
                validation_shuffle = None
                if shuffle_seed is not None:
                    anchors = x[val].sum(1) + nuisance_predict(source["nuisance"], x[val])
                    groups = np.repeat("source", val.sum()) if strata is None else strata[val]
                    residual, validation_shuffle = shuffled_residuals(y[val] - anchors, ids[val], groups, shuffle_seed)
                    validation_y = anchors + residual
                errors.append(float(validation_weights[val] @ np.mean((pred - validation_y) ** 2, axis=1)))
                if attempt:
                    attempt(dict(stage="source_penalty", alpha=alpha, success=True,
                                 fitting_ids=ids[train].tolist(), validation_ids=ids[val].tolist(),
                                 shuffle_metadata=source["shuffle_metadata"], validation_shuffle=validation_shuffle))
            except Unsupported as error:
                failure = str(error)
                if attempt:
                    attempt(dict(stage="source_penalty", alpha=alpha, success=False, reason=failure,
                                 fitting_ids=ids[train].tolist(), validation_ids=ids[val].tolist()))
        rows.append(dict(alpha=alpha, mse=float(np.sum(errors)) if failure is None else None, failure=failure))
    valid = [r for r in rows if r["mse"] is not None]
    if not valid:
        raise Unsupported("No supported source penalty fit")
    best = min(valid, key=lambda r: (r["mse"], r["alpha"]))
    try:
        source = fit_source(x, y, ids, best["alpha"], strata, shuffle_seed)
    except Unsupported as error:
        if attempt:
            attempt(dict(stage="source_refit", alpha=best["alpha"], success=False,
                         reason=str(error), fitting_ids=ids.tolist()))
        raise
    if attempt:
        attempt(dict(stage="source_refit", alpha=best["alpha"], success=True, fitting_ids=ids.tolist()))
    return source, rows
