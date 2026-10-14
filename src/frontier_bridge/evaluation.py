"""Evaluator-only discrimination and conditional target-incidence sensitivity."""
import json

import numpy as np


def retrieval(prediction, truth):
    n = len(truth)
    if n <= 1:
        return dict(score=None, scores=None, undefined_rows=list(range(n)), candidate_count=n)
    p = prediction - prediction.mean(0)
    y = truth - truth.mean(0)
    pn, yn = np.linalg.norm(p, axis=1), np.linalg.norm(y, axis=1)
    undefined = np.flatnonzero((pn <= 1e-12) | (yn <= 1e-12)).tolist()
    if undefined:
        return dict(score=None, scores=None, undefined_rows=undefined, candidate_count=n)
    distance = 1 - (p / pn[:, None]) @ (y / yn[:, None]).T
    selected = np.diag(distance)
    # Average rank among numerical ties, in the full registered candidate pool.
    equal = np.isclose(distance, selected[:, None], rtol=1e-12, atol=1e-12)
    less = (distance < selected[:, None]) & ~equal
    rank = 1 + less.sum(1) + (equal.sum(1) - 1) / 2
    scores = (1 / rank - 1 / n) / (1 - 1 / n)
    return dict(score=float(scores.mean()), scores=scores.tolist(), ranks=rank.tolist(),
                undefined_rows=[], candidate_count=n)


def diagnostics(pred, truth, additive):
    main = pred - additive
    residual = additive - truth
    pc, yc = pred - pred.mean(0), truth - truth.mean(0)
    denom = np.sqrt((pc ** 2).sum(0) * (yc ** 2).sum(0))
    defined = denom > 1e-12
    correlation = np.divide((pc * yc).sum(0), denom, out=np.zeros_like(denom), where=defined)
    return dict(mse=float(np.mean((pred - truth) ** 2)), mae=float(np.mean(np.abs(pred - truth))),
                query_panel_centered_error=float(np.mean((pc - yc) ** 2)),
                gene_correlation_mean=float(correlation[defined].mean()) if defined.any() else None,
                gene_correlation_undefined=int((~defined).sum()),
                decomposition=dict(main=float(np.mean(main ** 2)), residual=float(np.mean(residual ** 2)),
                                   cross=float(2 * np.mean(main * residual))),
                retrieval=retrieval(pred, truth))


def metric_controls(truth, seed=20261031):
    exact = retrieval(truth.copy(), truth)
    constant = retrieval(np.repeat(truth.mean(0)[None, :], len(truth), axis=0), truth)
    # A seeded nontrivial identity permutation, never passed to prediction.
    order = np.random.default_rng(seed).permutation(len(truth))
    if np.array_equal(order, np.arange(len(truth))):
        order = np.roll(order, 1)
    shuffled = retrieval(truth[order], truth)
    qualified = exact["score"] is not None and shuffled["score"] is not None and exact["score"] > shuffled["score"] and constant["score"] is None
    return dict(exact=exact, constant=constant, identity_shuffled=shuffled,
                identity_permutation=order.tolist(), qualified=qualified, role="evaluator_only")


def bootstrap_weights(pair_ids, draws=2000, seed=20261031):
    parts = [json.loads(p) for p in pair_ids]
    targets = sorted({c for p in parts for c in p})
    index = {c: i for i, c in enumerate(targets)}
    rng = np.random.default_rng(seed)
    multiplicity = np.asarray([np.bincount(rng.integers(len(targets), size=len(targets)), minlength=len(targets))
                               for _ in range(draws)])
    weights = np.column_stack([multiplicity[:, index[a]] * multiplicity[:, index[b]] for a, b in parts]).astype(float)
    return weights, targets, multiplicity


def interval(draws):
    return np.percentile(draws, [2.5, 97.5]).tolist() if len(draws) else None


def evaluate(predictions, truth, singles, pair_ids, strata, fold_ids, cfg):
    """Inputs have one row per pair/stratum; exclusive folds are pooled once."""
    if len(set(zip(pair_ids, strata))) != len(pair_ids):
        raise ValueError("Repeated query identity/stratum evaluation rows")
    weights, targets, multiplicity = bootstrap_weights(pair_ids, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    valid = np.ones(cfg["bootstrap_draws"], dtype=bool)
    group_masks = [strata == st for st in sorted(set(strata))]
    for mask in group_masks:
        valid &= weights[:, mask].sum(1) > 0
    metrics, controls, uncertainty = {}, {}, {}
    controls = {st: metric_controls(truth[strata == st]) for st in sorted(set(strata))}
    for arm, pred in predictions.items():
        by_stratum, bootstrap_loss, bootstrap_retrieval = {}, [], []
        retrieval_available = True
        for st, mask in zip(sorted(set(strata)), group_masks):
            record = diagnostics(pred[mask], truth[mask], singles[mask].sum(1))
            by_stratum[st] = record
            denominator = weights[:, mask].sum(1)
            loss = np.mean((pred[mask] - truth[mask]) ** 2, axis=1)
            bootstrap_loss.append((weights[valid][:, mask] @ loss) / denominator[valid])
            if record["retrieval"]["scores"] is None:
                retrieval_available = False
            else:
                bootstrap_retrieval.append((weights[valid][:, mask] @ record["retrieval"]["scores"]) / denominator[valid])
        metrics[arm] = dict(mse=float(np.mean([v["mse"] for v in by_stratum.values()])),
                           mae=float(np.mean([v["mae"] for v in by_stratum.values()])),
                           retrieval=float(np.mean([v["retrieval"]["score"] for v in by_stratum.values()])) if retrieval_available else None,
                           strata=by_stratum,
                           fold_losses={str(f): float(np.mean([np.mean((pred[(fold_ids == f) & (strata == st)] - truth[(fold_ids == f) & (strata == st)]) ** 2) for st in sorted(set(strata))])) for f in sorted(set(fold_ids))},
                           stratum_fold_losses={str(st): {str(f): float(np.mean((pred[(fold_ids == f) & (strata == st)] - truth[(fold_ids == f) & (strata == st)]) ** 2)) for f in sorted(set(fold_ids))} for st in sorted(set(strata))})
        uncertainty[arm] = dict(mse_draws=np.mean(bootstrap_loss, axis=0),
                                retrieval_draws=np.mean(bootstrap_retrieval, axis=0) if retrieval_available else None)
    comparisons = {}
    for arm in predictions:
        difference = uncertainty[arm]["mse_draws"] - uncertainty["candidate"]["mse_draws"]
        retrieval_difference = None
        if uncertainty[arm]["retrieval_draws"] is not None and uncertainty["candidate"]["retrieval_draws"] is not None:
            retrieval_difference = interval(uncertainty["candidate"]["retrieval_draws"] - uncertainty[arm]["retrieval_draws"])
        reference_mse = metrics[arm]["mse"]
        comparisons[arm] = dict(absolute_mse_improvement_interval=interval(difference),
                                observed_relative_mse_gain=(reference_mse - metrics["candidate"]["mse"]) / reference_mse if reference_mse > 0 else None,
                                retrieval_difference_interval=retrieval_difference)
    effective = {}
    for st, mask in zip(sorted(set(strata)), group_masks):
        w = weights[:, mask]
        effective[st] = np.divide(w.sum(1) ** 2, (w ** 2).sum(1), out=np.zeros(len(w)), where=(w ** 2).sum(1) > 0).tolist()
    sensitivity = dict(scope="conditional target-product-weight sensitivity; fixed exposed labels, references, source selection and target fit; not biological confidence",
                       draws=cfg["bootstrap_draws"], valid_draws=int(valid.sum()),
                       zero_denominator_draw_ids=np.flatnonzero(~valid).tolist(), common_valid_draw_ids=np.flatnonzero(valid).tolist(),
                       comparisons=comparisons, target_identities=targets, effective_pair_counts=effective,
                       effective_target_counts=np.divide(multiplicity.sum(1) ** 2, (multiplicity ** 2).sum(1)).tolist(),
                       procedure_uncertainty="unavailable; not refitted", program_ranking="unavailable; no frozen scientific program",
                       independent_reference_attribution="unavailable; prepared banks retain shared anchors")
    return metrics, controls, sensitivity


def direction_gate(metrics, controls, uncertainty, supported, cfg, controls_supported=True):
    selected = uncertainty["comparisons"]["selector"]
    mse_interval = selected["absolute_mse_improvement_interval"]
    retrieval_interval = selected["retrieval_difference_interval"]
    challengers = ["mean_feature", "source_free", "no_change", "additive"] + [f"{family}_{seed}" for family in ("source_shuffle", "bridge_shuffle") for seed in cfg["shuffle_seeds"]]
    checks = dict(adaptation_supported=bool(supported),
                  control_fits_supported=bool(controls_supported),
                  relative_gain=selected["observed_relative_mse_gain"] is not None and selected["observed_relative_mse_gain"] >= cfg["minimum_relative_mse_gain"],
                  mse_lower_positive=mse_interval is not None and mse_interval[0] > 0,
                  controls_beaten=all(metrics["candidate"]["mse"] < metrics[a]["mse"] for a in challengers),
                  retrieval_qualified=all(c["qualified"] for c in controls.values()),
                  retrieval_point_positive=metrics["candidate"]["retrieval"] is not None and metrics["selector"]["retrieval"] is not None and metrics["candidate"]["retrieval"] > metrics["selector"]["retrieval"],
                  retrieval_lower_positive=retrieval_interval is not None and retrieval_interval[0] > 0)
    return dict(checks=checks, passed=all(checks.values()))
