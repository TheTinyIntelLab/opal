"""Source-locked same-information references using existing kernel algebra."""
import json
from pathlib import Path

import numpy as np

from ..replicated.models import ComponentAdditive, ProfileKernel
from ..series_bilinear import (fit_reference as historical_fit_reference,
                              predict_reference as historical_predict_reference,
                              reference_menu)
from .contracts import read_json
from .model import (Unsupported, fit_nuisance, grouped_masks, nuisance_predict,
                    ridge, ridge_predict, source_weights)
from .splits import bridges


FAMILIES = ("no_change", "additive", "frozen_base", "bridge_offset", "bridge_affine",
            "bridge_incidence", "target_profile_ridge", "target_profile_kernel",
            "pooled_profile_kernel", "source_kernel_posterior")


def kernel_fit(x, y, alpha, weights=None):
    return ProfileKernel(dict(kind="global_plus_pair", alpha=alpha, bandwidth=1.)).fit(x, y, weights).__dict__.copy()


def kernel_restore(state):
    model = ProfileKernel(state["config"])
    model.__dict__.update(state)
    return model


def centered_kernel(model, left, right):
    k = model.kernel(left, right)
    return (k - (model.kernel(left, model.references) @ model.weights)[:, None]
            - (model.weights @ model.kernel(model.references, right))[None, :] + model.grand)


def source_base(x, y, ids, strata=None):
    """Freeze a source-only simple/kernel reference using historical algebra.

    This is a new source fit, not a deployment checkpoint assigned a test
    score. The historical model-series selectors and caches are untouched.
    """
    root = Path(__file__).resolve().parents[2]
    menu = reference_menu(read_json(root / "configs/experiments/replicated_fixed_state_v1.json")["menu"])
    attempts = []
    validation_weights = source_weights(ids, strata)
    for config in menu:
        losses = []
        failure = None
        for train, val in grouped_masks(ids, strata=strata):
            try:
                state = fit_base(x[train], y[train], ids[train], config, None if strata is None else strata[train])
                losses.append(float(validation_weights[val] @ np.mean((predict_base(state, x[val], ids[val]) - y[val]) ** 2, axis=1)))
            except ValueError as error:
                failure = str(error)
        attempts.append(dict(config=config, mse=float(np.sum(losses)) if failure is None else None, failure=failure))
    valid = [r for r in attempts if r["mse"] is not None]
    if not valid:
        raise Unsupported("Frozen Base reference menu is unsupported")
    selected = min(valid, key=lambda r: (r["mse"], r["config"]["id"]))
    return fit_base(x, y, ids, selected["config"], strata), attempts


def fit_base(x, y, ids, config, strata=None):
    bank = dict(singles=x, responses=y, pair_ids=ids, culture_ids=np.repeat("source", len(ids)) if strata is None else np.asarray(strata))
    fitted = historical_fit_reference(config, bank)
    if fitted.get("legacy") is not None:
        expert = fitted["legacy"]
        fitted = dict(legacy_state=expert.__dict__.copy(),
                      legacy_type="incidence" if isinstance(expert, ComponentAdditive) else "kernel")
    return dict(config=config, fitted=fitted)


def predict_base(state, x, ids):
    if "config" not in state:
        # Safe restoration of the first engineering attempt's explicit
        # simple-reference states, without refitting or relabeling its scores.
        kind = state["kind"]
        if kind == "additive":
            return x.sum(1)
        if kind == "affine":
            return x.sum(1) + nuisance_predict(state["fitted"], x)
        if kind == "component_additive":
            model = ComponentAdditive(state["alpha"])
            model.__dict__.update(state["fitted"])
            return x.sum(1) + model.predict(ids)
        return kernel_restore(state["fitted"]).predict(x)
    fitted = state["fitted"]
    if "legacy_state" in fitted:
        if fitted["legacy_type"] == "incidence":
            expert = ComponentAdditive(state["config"]["alpha"])
            expert.__dict__.update(fitted["legacy_state"])
        else:
            expert = kernel_restore(fitted["legacy_state"])
        fitted = dict(legacy=expert)
    return historical_predict_reference(state["config"], fitted, dict(singles=x, pair_ids=ids))


def fit_reference(family, alpha, source_x, source_y, source_ids, bridge_x, bridge_y, bridge_ids, base, source_strata=None):
    state = dict(family=family, alpha=alpha)
    genes = source_x.shape[-1]
    if family in ("no_change", "additive"):
        return state
    if family == "frozen_base":
        return dict(state, base=base)
    residual = bridge_y - bridge_x.sum(1)
    if family == "bridge_offset":
        return dict(state, intercept=residual.mean(0))
    if family == "bridge_affine":
        return dict(state, nuisance=fit_nuisance(bridge_x, bridge_y, "affine", alpha))
    if family == "bridge_incidence":
        fitted = ComponentAdditive(alpha).fit(bridge_ids, residual).__dict__.copy()
        return dict(state, fitted=fitted)
    if family == "target_profile_ridge":
        scale = np.maximum(np.unique(source_x.reshape(-1, genes), axis=0).std(0), 1e-6)
        # Profile-ridge references have a gene-vector intercept and a fixed
        # source-reference scale, with no target PCA or outcome-fitted scale.
        return dict(state, scale=scale, fitted=ridge(bridge_x.sum(1) / scale, residual, alpha))
    if family == "target_profile_kernel":
        return dict(state, fitted=kernel_fit(bridge_x, bridge_y, alpha))
    if family == "pooled_profile_kernel":
        return dict(state, fitted=kernel_fit(np.concatenate([source_x, bridge_x]),
                                            np.concatenate([source_y, bridge_y]), alpha,
                                            np.concatenate([source_weights(source_ids, source_strata) * len(source_x), np.ones(len(bridge_x))]) / (len(source_x) + len(bridge_x))))
    if family == "source_kernel_posterior":
        fitted = kernel_fit(source_x, source_y, alpha, source_weights(source_ids, source_strata))
        model = kernel_restore(fitted)
        kss = centered_kernel(model, source_x, source_x) + alpha * np.diag(1 / model.weights)
        kbs = centered_kernel(model, bridge_x, source_x)
        covariance = centered_kernel(model, bridge_x, bridge_x) - kbs @ np.linalg.solve(kss, kbs.T)
        error = bridge_y - model.predict(bridge_x)
        intercept = error.mean(0)
        h = np.eye(len(bridge_x)) - np.ones((len(bridge_x), len(bridge_x))) / len(bridge_x)
        coefficients = np.linalg.solve(h @ covariance @ h + alpha * len(bridge_x) * np.eye(len(bridge_x)), error - intercept)
        return dict(state, fitted=fitted, bridge_x=bridge_x.copy(), posterior_projection=np.linalg.solve(kss, kbs.T),
                    posterior_center=h, coefficients=coefficients, intercept=intercept)
    raise ValueError(f"Unknown reference: {family}")


def predict_reference(state, x, ids):
    family = state["family"]
    if family == "no_change":
        return np.zeros_like(x[:, 0])
    if family == "additive":
        return x.sum(1)
    if family == "frozen_base":
        return predict_base(state["base"], x, ids)
    if family == "bridge_offset":
        return x.sum(1) + state["intercept"]
    if family == "bridge_affine":
        return x.sum(1) + nuisance_predict(state["nuisance"], x)
    if family == "bridge_incidence":
        model = ComponentAdditive(state["alpha"])
        model.__dict__.update(state["fitted"])
        return x.sum(1) + model.predict(ids)
    if family == "target_profile_ridge":
        return x.sum(1) + ridge_predict(state["fitted"], x.sum(1) / state["scale"])
    model = kernel_restore(state["fitted"])
    if family in ("target_profile_kernel", "pooled_profile_kernel"):
        return model.predict(x)
    covariance = centered_kernel(model, x, state["bridge_x"]) - centered_kernel(model, x, model.references) @ state["posterior_projection"]
    return model.predict(x) + state["intercept"] + covariance @ state["posterior_center"] @ state["coefficients"]


def select_references(x, y, ids, attempt=None, strata=None):
    strata = np.repeat("source", len(ids)) if strata is None else np.asarray(strata)
    rows = []
    partitions = list(grouped_masks(ids, strata=strata))
    bases = [source_base(x[train], y[train], ids[train], strata[train])[0] for train, _ in partitions]
    for family in FAMILIES:
        for alpha in ((1,) if family in ("no_change", "additive", "frozen_base", "bridge_offset") else (.1, 1, 10)):
            losses, failure = [], None
            for inner, (train, val) in enumerate(partitions):
                held = sorted(set.intersection(*(set(ids[val & (strata == st)]) for st in np.unique(strata))))
                if len(held) < 8:
                    failure = "G1 source-inner block cannot retain four bridges and four query identities"
                    break
                bridge_ids = bridges(held)
                stratum_losses = []
                for st in np.unique(strata):
                    bmask = val & (strata == st) & np.isin(ids, bridge_ids)
                    qmask = val & (strata == st) & ~np.isin(ids, bridge_ids)
                    try:
                        if len(ids[bmask]) != 4 or len(set(ids[bmask])) != 4 or len(set(ids[qmask])) < 4:
                            raise Unsupported("G1 requires four distinct bridges and four queries per stratum")
                        base = bases[inner]
                        state = fit_reference(family, alpha, x[train], y[train], ids[train], x[bmask], y[bmask], ids[bmask], base, strata[train])
                        pred = predict_reference(state, x[qmask], ids[qmask])
                        stratum_losses.append(float(source_weights(ids[qmask]) @ np.mean((pred - y[qmask]) ** 2, axis=1)))
                    except (Unsupported, np.linalg.LinAlgError) as error:
                        failure = str(error)
                    if attempt:
                        attempt(dict(stage="reference_selection", family=family, alpha=alpha, stratum=str(st),
                                     success=failure is None, reason=failure, source_ids=ids[train].tolist(),
                                     bridge_ids=ids[bmask].tolist(), query_ids=ids[qmask].tolist()))
                if failure is None:
                    losses.append(float(np.mean(stratum_losses)))
            rows.append(dict(family=family, alpha=alpha, mse=float(np.mean(losses)) if failure is None else None,
                             failure=failure))
    best_by_family = {}
    for family in FAMILIES:
        valid = [r for r in rows if r["family"] == family and r["mse"] is not None]
        if valid:
            best_by_family[family] = min(valid, key=lambda r: (r["mse"], r["alpha"]))
    if len(best_by_family) != len(FAMILIES):
        raise Unsupported("G1 unavailable: complete same-information reference selection unsupported")
    selector = min(best_by_family.values(), key=lambda r: (r["mse"], r["family"], r["alpha"]))
    return best_by_family, selector, rows
