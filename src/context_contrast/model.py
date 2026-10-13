"""Fold-local float64 models, exhaustive finite selection and contrast algebra."""
import json
import numpy as np
from ..replicated.models import ProfileKernel, ComponentAdditive
from .contracts import PROTOCOL, Unsupported, ContractError, masks


def menu():
    fixed = [dict(id=f, family=f, alpha=1) for f in ("no_change", "additive", "offset", "base")]
    families = ("affine", "incidence", "phi8", "mean18", "context18", "mean_response18",
        "pooled_kernel", "per_context_profile", "per_context_product", "joint_context", "phi18")
    fixed += [dict(id=f + "-" + str(a), family=f, alpha=a) for f in families for a in PROTOCOL["penalties"]]
    fixed += [dict(id=f + "-" + str(seed) + "-" + str(a), family=f, alpha=a, seed=seed)
        for f in ("alignment_response", "alignment_contrast") for seed in PROTOCOL["shuffle_seeds"]
        for a in PROTOCOL["penalties"]]
    return fixed


def base_menu():
    # The disclosed 24-family Base procedure from series_bilinear.reference_menu.
    result = [dict(id="additive", kind="additive", role="reference", alpha=1)]
    result += [dict(id=f"{k}-{float(a)}", kind=k, role="reference", alpha=a, bandwidth=1.)
        for k in ("shared_offset", "global", "single_sum", "component_additive") for a in PROTOCOL["penalties"]]
    result += [dict(id="null", kind="no_change", role="reference"), dict(id="response-mean", kind="response_mean", role="reference")]
    result += [dict(id=f"affine-{a}", kind="affine", alpha=a, role="reference") for a in PROTOCOL["penalties"]]
    result += [dict(id=f"sum-pca-{r}-{a}", kind="sum_pca", rank=r, alpha=a, role="reference")
        for r in (4, 8) for a in PROTOCOL["penalties"]]
    return result


def rank_check(x, required, name):
    singular = np.linalg.svd(x, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-8)) if len(singular) and singular[0] > 0 else 0
    condition = float(singular[0] / singular[required-1]) if rank >= required else None
    record = dict(name=name, required_rank=required, effective_rank=rank,
        singular_values=singular.tolist(), condition_number=condition, unique_rows=len(np.unique(x, axis=0)))
    if rank < required or condition > 1e4:
        raise Unsupported("Unsupported " + name, record)
    return record


def encoder(x):
    # Count literal singles once per context, rather than degree-weighting pairs.
    profiles = np.concatenate([np.unique(x[:, c].reshape(-1, x.shape[-1]), axis=0) for c in range(2)])
    center = profiles.mean(0)
    centered = profiles - center
    diagnostic = rank_check(centered, 4, "single_PCA")
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return dict(P=vt[:4].T, center=center, diagnostic=diagnostic)


def vech(matrix):
    return np.column_stack([matrix[:, i, j] * (1 if i == j else np.sqrt(2))
        for i in range(4) for j in range(i, 4)])


def sym_product(a, b):
    m = a[:, :, None] * b[:, None, :]
    return (m + m.transpose(0, 2, 1)) / 2


def features(x, enc, family="phi18"):
    z = x @ enc["P"]
    d, bar = z[:, 1]-z[:, 0], (z[:, 1]+z[:, 0])/2
    phi8 = np.column_stack((d.sum(1), np.abs(d[:, 0]-d[:, 1])))
    if family == "phi8":
        return phi8
    if family in ("mean18", "mean_response18", "context18"):
        first = np.column_stack((bar.sum(1), np.abs(bar[:, 0]-bar[:, 1])))
        block = sym_product(bar[:, 0], bar[:, 1])
        if family == "context18":
            # Symmetric nonlinear mean/context control, same 18/36 capacity.
            block += sym_product(d[:, 0], d[:, 1])
        return np.column_stack((first, vech(block)))
    block = sym_product(bar[:, 0], d[:, 1]) + sym_product(bar[:, 1], d[:, 0])
    return np.column_stack((phi8, vech(block)))


def kernel_fit(x, residual, alpha, kind="global_plus_pair"):
    model = ProfileKernel(dict(kind=kind, alpha=alpha, bandwidth=1.))
    # ProfileKernel subtracts additivity internally; this gives a residual model.
    return model.fit(x, residual+x.sum(1)).__dict__.copy()


def kernel_predict(state, x):
    model = ProfileKernel(state["config"])
    model.__dict__.update(state)
    return model.predict(x)-x.sum(1)


def request(recorder, stage, ids, function, spec=None):
    """Persist every dependent fit and failure before returning to its caller."""
    try:
        state = function()
    except (ValueError, np.linalg.LinAlgError) as error:
        if recorder:
            recorder(dict(stage=stage, spec=spec, fitting_ids=list(map(str, ids)), success=False,
                reason=str(error), diagnostic=getattr(error, "diagnostic", None)), None)
        raise
    if recorder:
        recorder(dict(stage=stage, spec=spec, fitting_ids=list(map(str, ids)), success=True,
            diagnostic=state.get("diagnostics", state.get("diagnostic"))), state)
    return state


def select_pooled(x, residual, ids, recorder):
    pooled = x.mean(1)
    scores = []
    for alpha in PROTOCOL["penalties"]:
        loss, failure = [], None
        for train, val in masks(ids):
            try:
                state = request(recorder, "pooled_inner", ids[train],
                    lambda: kernel_fit(pooled[train], residual[train], alpha), dict(alpha=alpha, validation_ids=ids[val].tolist()))
                loss.extend(np.mean((kernel_predict(state, pooled[val])-residual[val])**2, axis=1))
            except (ValueError, np.linalg.LinAlgError) as e:
                failure = str(e)
        scores.append(dict(alpha=alpha, mse=float(np.mean(loss)) if failure is None else None, failure=failure))
    valid = [r for r in scores if r["mse"] is not None]
    if not valid:
        raise Unsupported("Pooled nuisance selection unavailable")
    alpha = min(valid, key=lambda r: (r["mse"], r["alpha"]))["alpha"]
    return request(recorder, "pooled_refit", ids, lambda: kernel_fit(pooled, residual, alpha), dict(alpha=alpha)), scores


def crossfit_nuisance(x, residual, ids, recorder):
    mean, contrast = residual.mean(1), residual[:, 1]-residual[:, 0]
    k, h = np.empty_like(mean), np.empty_like(contrast)
    for train, val in masks(ids):
        fitted, _ = select_pooled(x[train], mean[train], ids[train], recorder)
        k[val] = kernel_predict(fitted, x[val].mean(1))
        h[val] = request(recorder, "contrast_offset_oof", ids[train], lambda: dict(mean=contrast[train].mean(0)))["mean"]
    pooled, selections = select_pooled(x, mean, ids, recorder)
    offset = request(recorder, "contrast_offset_refit", ids, lambda: dict(mean=contrast.mean(0)))["mean"]
    return pooled, offset, mean-k, contrast-h, selections


def head_fit(phi, residual, alpha):
    center = phi.mean(0)
    scale = np.maximum(phi.std(0), 1e-6)
    design = (phi-center)/scale
    diagnostics = [rank_check(design, phi.shape[1], "feature_design"), rank_check(residual, 2, "contrast_output")]
    _, _, vt = np.linalg.svd(residual, full_matrices=False)
    basis = vt[:2].T*np.sqrt(residual.shape[1])
    target = residual @ basis/residual.shape[1]
    coefficient = np.linalg.solve(design.T@design/len(phi)+alpha*np.eye(phi.shape[1]), design.T@target/len(phi))
    return dict(center=center, scale=scale, scale_floor=1e-6, B=basis, W=coefficient.T,
        alpha=alpha, diagnostics=diagnostics, head_coefficients=int(coefficient.size),
        objective="mean_pair_gene_error_plus_lambda_frobenius_W")


def head_predict(state, phi):
    return ((phi-state["center"])/state["scale"]) @ state["W"].T @ state["B"].T


def permutation(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    if np.array_equal(order, np.arange(n)):
        order = np.roll(order, 1)
    return order


def fit_base(x, y, ids, recorder):
    from ..series_bilinear import fit_reference, predict_reference
    from ..replicated.models import ComponentAdditive
    states, selection = [], []
    for c in range(2):
        def bank(mask):
            return dict(singles=x[mask, c], responses=y[mask, c], pair_ids=ids[mask], culture_ids=np.repeat("fixed-context", np.sum(mask)))
        rows = []
        for config in base_menu():
            loss, failures = [], []
            for train, val in masks(ids):
                try:
                    fitted = request(recorder, "base_inner", ids[train],
                        lambda: fit_reference(config, bank(train)), dict(config=config, context=c, validation_ids=ids[val].tolist()))
                    loss.extend(np.mean((predict_reference(config, fitted, bank(val))-y[val, c])**2, axis=1))
                except (ValueError, np.linalg.LinAlgError) as e:
                    failures.append(str(e))
            rows.append(dict(config=config, mse=None if failures else float(np.mean(loss)), failures=failures))
        valid = [r for r in rows if r["mse"] is not None]
        if not valid:
            raise Unsupported("Base refit unsupported")
        config = min(valid, key=lambda r: (r["mse"], r["config"]["id"]))["config"]
        fitted = request(recorder, "base_refit", ids, lambda: fit_reference(config, bank(np.ones(len(ids), bool))), dict(config=config, context=c))
        # Save the existing safe classical state, never a whole-panel checkpoint.
        if fitted.get("legacy") is not None:
            obj = fitted["legacy"]
            fitted = dict(legacy_state=obj.__dict__.copy(), legacy_type="incidence" if isinstance(obj, ComponentAdditive) else "kernel")
        states.append(dict(config=config, fitted=fitted))
        selection.append(rows)
    return dict(states=states, selection=selection)


def base_predict(state, x, ids):
    from ..series_bilinear import predict_reference
    result = []
    for c, stored in enumerate(state["states"]):
        fitted = dict(stored["fitted"])
        if "legacy_state" in fitted:
            obj = ComponentAdditive(stored["config"]["alpha"]) if fitted["legacy_type"] == "incidence" else ProfileKernel(stored["config"])
            obj.__dict__.update(fitted["legacy_state"])
            fitted = dict(legacy=obj)
        result.append(predict_reference(stored["config"], fitted, dict(singles=x[:, c], pair_ids=ids)))
    return np.stack(result, axis=1)


def fit(spec, x, y, ids, recorder=None):
    x, y, ids = np.asarray(x, float), np.asarray(y, float), np.asarray(ids, str)
    if x.shape != (len(ids), 2, 2, y.shape[-1]) or y.shape != (len(ids), 2, x.shape[-1]) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ContractError("Finite pair/context/component/gene schema required")
    if len(set(ids)) != len(ids):
        raise ContractError("Unique aggregate canonical pairs required")
    family, alpha = spec["family"], spec["alpha"]
    state = dict(spec=spec, fitting_ids=ids.copy())
    residual = y-x.sum(2)
    if family in ("no_change", "additive"):
        return state
    if family == "base":
        return dict(state, base=fit_base(x, y, ids, recorder))
    if family == "offset":
        return dict(state, offset=residual.mean(0))
    if family == "affine":
        a = x.sum(2)
        ac, yc = a-a.mean(0), y-y.mean(0)
        gamma = (np.mean(ac*yc, axis=(0, 2))+alpha)/(np.mean(ac**2, axis=(0, 2))+alpha)
        return dict(state, gamma=gamma, offset=y.mean(0)-gamma[:, None]*a.mean(0))
    if family == "incidence":
        return dict(state, incidence=[request(recorder, "incidence_context", ids,
            lambda: ComponentAdditive(alpha).fit(ids, residual[:, c]).__dict__.copy(), dict(context=c, alpha=alpha)) for c in range(2)])
    if family in ("per_context_profile", "per_context_product"):
        kind = "single_sum" if family.endswith("profile") else "pair_product"
        return dict(state, kernels=[request(recorder, "kernel_context", ids,
            lambda: kernel_fit(x[:, c], residual[:, c], alpha, kind), dict(context=c, alpha=alpha)) for c in range(2)])
    if family == "pooled_kernel":
        return dict(state, pooled=request(recorder, "pooled_reference", ids,
            lambda: kernel_fit(x.mean(1), residual.mean(1), alpha), dict(alpha=alpha)),
            h=request(recorder, "pooled_reference_offset", ids, lambda: dict(mean=(residual[:, 1]-residual[:, 0]).mean(0)))["mean"])
    if family == "joint_context":
        # A symmetric pair kernel on concatenated context-specific single profiles.
        joint = np.concatenate((x[:, 0], x[:, 1]), axis=-1)
        return dict(state, joint=request(recorder, "joint_reference", ids,
            lambda: kernel_fit(joint, residual.reshape(len(ids), -1), alpha), dict(alpha=alpha)))
    enc = request(recorder, "single_encoder", ids, lambda: encoder(x))
    pooled, h, mean_residual, t_residual, selections = crossfit_nuisance(x, residual, ids, recorder)
    feature_family = "phi18" if family.startswith("alignment") else family
    phi = features(x, enc, feature_family)
    target = mean_residual if family == "mean_response18" else t_residual
    order = None
    if family.startswith("alignment"):
        order = permutation(len(ids), spec["seed"])
        if family == "alignment_response":
            target = target[order]
        else:
            phi = phi[order]
    head = request(recorder, "residual_head", ids, lambda: head_fit(phi, target, alpha))
    return dict(state, encoder=enc, pooled=pooled, h=h, head=head, pooled_selection=selections,
        alignment_permutation=order, diagnostics=[enc["diagnostic"]]+head["diagnostics"])


def predict(state, x, ids):
    family = state["spec"]["family"]
    additive = x.sum(2)
    if family == "no_change":
        return np.zeros_like(additive)
    if family == "additive":
        return additive
    if family == "base":
        return base_predict(state["base"], x, ids)
    if family == "offset":
        return additive+state["offset"]
    if family == "affine":
        return additive*state["gamma"][None, :, None]+state["offset"]
    if family == "incidence":
        values = []
        for stored in state["incidence"]:
            model = ComponentAdditive(state["spec"]["alpha"])
            model.__dict__.update(stored)
            values.append(model.predict(ids))
        return additive+np.stack(values, axis=1)
    if family.startswith("per_context"):
        return additive+np.stack([kernel_predict(s, x[:, c]) for c, s in enumerate(state["kernels"])], axis=1)
    if family == "joint_context":
        return additive+kernel_predict(state["joint"], np.concatenate((x[:, 0], x[:, 1]), axis=-1)).reshape(additive.shape)
    k = kernel_predict(state["pooled"], x.mean(1))
    t = np.broadcast_to(state["h"], k.shape).copy()
    if "head" in state:
        f = "phi18" if family.startswith("alignment") else family
        phi = features(x, state["encoder"], f)
        correction = head_predict(state["head"], phi)
        if family == "mean_response18":
            k = k+correction
        else:
            t += correction
    return additive+np.stack((k-t/2, k+t/2), axis=1)


def select(x, y, ids, recorder=None):
    rows, states = [], {}
    for spec in menu():
        losses, failures = [], []
        for inner, (train, val) in enumerate(masks(ids)):
            try:
                fitted = request(recorder, "arm_inner", ids[train],
                    lambda: fit(spec, x[train], y[train], ids[train], recorder), dict(spec, inner=inner, validation_ids=ids[val].tolist()))
                losses.extend(np.mean((predict(fitted, x[val], ids[val])-y[val])**2, axis=(1, 2)))
            except (ValueError, np.linalg.LinAlgError) as e:
                failures.append(str(e))
        rows.append(dict(spec=spec, mse=float(np.mean(losses)) if not failures else None, failures=failures))
    best = {}
    for family in sorted({s["family"] for s in menu()}):
        seeds = PROTOCOL["shuffle_seeds"] if family.startswith("alignment") else [None]
        for seed in seeds:
            eligible = [r for r in rows if r["spec"]["family"] == family and r["spec"].get("seed") == seed and r["mse"] is not None]
            arm = family+"-"+str(seed) if seed is not None else family
            if eligible:
                winner = min(eligible, key=lambda r: (r["mse"], r["spec"]["alpha"]))
                best[arm] = winner
            else:
                best[arm] = None
    reference = [r for r in best.values() if r and r["spec"]["family"] != "phi18" and not r["spec"]["family"].startswith("alignment")]
    selector = min(reference, key=lambda r: (r["mse"], r["spec"]["family"], r["spec"]["alpha"]))
    # Selection is final before any outer evaluator outcome is loaded.
    for arm, winner in best.items():
        if winner:
            try:
                states[arm] = request(recorder, "arm_refit", ids, lambda: fit(winner["spec"], x, y, ids, recorder), winner["spec"])
            except (ValueError, np.linalg.LinAlgError):
                states[arm] = None
        else:
            states[arm] = None
    if states[selector["spec"]["family"]] is None:
        raise Unsupported("Locked selector final refit failed; no outcome-driven replacement")
    return states, dict(comparisons=rows, best_by_arm=best, selector=selector,
        complete_reference_support=all(v is not None for v in states.values()))
