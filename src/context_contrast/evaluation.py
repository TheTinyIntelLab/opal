"""Evaluator-only aggregate endpoints and five evidence gates."""
import json
import numpy as np
from .contracts import PROTOCOL, ContractError


def retrieval(prediction, truth):
    n = len(truth)
    if n < 2:
        return dict(score=None, ranks=None, undefined=list(range(n)), pool=n)
    p, t = prediction-prediction.mean(0), truth-truth.mean(0)
    pn, tn = np.linalg.norm(p, axis=1), np.linalg.norm(t, axis=1)
    undefined = np.flatnonzero((pn <= 1e-12) | (tn <= 1e-12)).tolist()
    if undefined:
        return dict(score=None, ranks=None, undefined=undefined, pool=n)
    distance = 1-(p/pn[:, None])@(t/tn[:, None]).T
    target = np.diag(distance)
    ties = np.isclose(distance, target[:, None], rtol=1e-12, atol=1e-12)
    rank = 1+((distance < target[:, None]) & ~ties).sum(1)+(ties.sum(1)-1)/2
    return dict(score=float(np.mean((1/rank-1/n)/(1-1/n))), ranks=rank.tolist(), undefined=[], pool=n,
        random_no_tie_expectation=float((np.sum(1/np.arange(1, n+1))-1)/(n-1)))


def retrieval_null(prediction, truth):
    """One identity permutation shared across both contexts, full pools unchanged."""
    rng = np.random.default_rng(PROTOCOL["retrieval_seed"])
    values, orders = [], []
    for _ in range(99):
        order = rng.permutation(len(truth))
        orders.append(order.tolist())
        values.append([retrieval(prediction[order, c], truth[:, c])["score"] for c in range(2)])
    return dict(seed=PROTOCOL["retrieval_seed"], permutations=orders, distribution=values,
        role="evaluator_only", preserves_context_blocks=True, used_for_selection=False)


def metric_qualification(half_a, half_b, training_mean):
    if half_a is None or half_b is None:
        return dict(qualified=False, reason="disjoint_measurement_half_positive_unavailable", biological_confirmation=False)
    negatives = dict(train_mean=np.broadcast_to(training_mean, half_b.shape),
        identity_shuffle=half_a[np.random.default_rng(20261033).permutation(len(half_a))],
        affine=.7*half_a+.2, amplitude=2*half_a)
    exact = [retrieval(half_b[:, c], half_b[:, c]) for c in range(2)]
    realistic = [retrieval(half_a[:, c], half_b[:, c]) for c in range(2)]
    negative = {name: [retrieval(v[:, c], half_b[:, c]) for c in range(2)] for name, v in negatives.items()}
    null = retrieval_null(half_a, half_b)
    passed = []
    for c in range(2):
        distribution = [r[c] for r in null["distribution"] if r[c] is not None]
        # Cosine retrieval cannot reject positive affine/amplitude transformations.
        # Report those invariances and qualify against uninformative controls;
        # primary response MSE independently detects calibration distortions.
        shuffled = negative["identity_shuffle"][c]["score"]
        score = realistic[c]["score"]
        passed.append(score is not None and exact[c]["score"] is not None and distribution and
            score > np.percentile(distribution, 95) and shuffled is not None and score > shuffled)
    return dict(qualified=bool(all(passed)), exact=exact, realistic=realistic, negatives=negative,
        empirical_null=null, calibration_negative_mse={k: np.mean((v-half_b)**2, axis=(0, 2)).tolist() for k, v in negatives.items()},
        scope="disjoint technical halves; cosine is affine/amplitude invariant; no cultures or biological accuracy ceiling",
        biological_confirmation=False)


def endpoint(pred, truth):
    pc, tc = pred-pred.mean(0), truth-truth.mean(0)
    denom = np.sqrt(np.sum(pc**2, axis=0)*np.sum(tc**2, axis=0))
    valid = denom > 1e-12
    correlation = np.sum(pc*tc, axis=0)[valid]/denom[valid]
    return dict(mse=float(np.mean((pred-truth)**2)), mae=float(np.mean(np.abs(pred-truth))),
        across_perturbation_correlation=float(correlation.mean()) if valid.any() else None,
        undefined_correlation_genes=int((~valid).sum()), retrieval=retrieval(pred, truth))


def weights(ids):
    parts = [json.loads(p) for p in ids]
    interventions = sorted({c for p in parts for c in p})
    rng = np.random.default_rng(20261031)
    multiplicity = np.asarray([np.bincount(rng.integers(len(interventions), size=len(interventions)), minlength=len(interventions)) for _ in range(2000)])
    index = {c: i for i, c in enumerate(interventions)}
    w = np.column_stack([multiplicity[:, index[a]]*multiplicity[:, index[b]] for a, b in parts])
    return w.astype(float), interventions, multiplicity


def evaluate(predictions, truth, singles, ids, fallback, fold_ids, program=None, independent_anchor=None):
    if len(set(ids)) != len(ids) or truth.shape != (len(ids), 2, singles.shape[-1]):
        raise ContractError("One globally held aggregate per canonical pair/context required")
    if any(p.shape != truth.shape or not np.isfinite(p).all() for p in predictions.values()):
        raise ContractError("Every arm must score the complete finite population")
    w, interventions, multiplicity = weights(ids)
    denominators = w.sum(1)
    valid = denominators > 0
    losses = {name: np.mean((p-truth)**2, axis=2) for name, p in predictions.items()}
    metrics, uncertainty = {}, {}
    additive = singles.sum(2)
    true_t = (truth-additive)[:, 1]-(truth-additive)[:, 0]
    for name, pred in predictions.items():
        contexts = [endpoint(pred[:, c], truth[:, c]) for c in range(2)]
        pred_t = (pred-additive)[:, 1]-(pred-additive)[:, 0]
        metrics[name] = dict(contexts=contexts, t_contrast_mse=float(np.mean((pred_t-true_t)**2)),
            original_query_pairs=len(ids), scored_query_pairs=len(pred), fallback_pairs=int(np.sum(fallback[name])),
            folds={str(f): np.mean((pred[fold_ids == f]-truth[fold_ids == f])**2, axis=(0, 2)).tolist() for f in np.unique(fold_ids)},
            independent_anchor_departure_mse=None if independent_anchor is None else float(np.mean((pred-additive-independent_anchor)**2)),
            program=None)
        if program is not None:
            vector = np.asarray(program["weights"], float)
            if vector.shape != (truth.shape[-1],) or not program.get("frozen_before_outcomes") or not program.get("provenance"):
                raise ContractError("Program requires a pre-scoring source/prespecified freeze")
            k = program["top_k"]
            if not isinstance(k, int) or not 1 <= k <= len(ids) or program["pool"] != list(ids):
                raise ContractError("Frozen program top-k/full pool mismatch")
            metrics[name]["program"] = dict(mse=np.mean(((pred-truth)@vector)**2, axis=0).tolist(),
                top_k_overlap=[len(set(np.argsort(pred[:, c]@vector)[-k:]) & set(np.argsort(truth[:, c]@vector)[-k:]))/k for c in range(2)])
        delta = losses["selector"]-losses[name]
        draws = (w[valid]@delta)/denominators[valid, None]
        uncertainty[name] = dict(absolute_paired_loss_difference=delta.mean(0).tolist(),
            conditional_bounds=np.percentile(draws, [2.5, 97.5], axis=0).T.tolist(),
            relative_gain=[float(delta[:, c].mean()/losses["selector"][:, c].mean()) if losses["selector"][:, c].mean() > 0 else None for c in range(2)],
            scope="conditional intervention-product sensitivity; exposed labels, references, fitted selection fixed; no biological sampling interval")
    return dict(metrics=metrics, uncertainty=uncertainty,
        retrieval_null={name: retrieval_null(p, truth) for name, p in predictions.items()},
        sensitivity=dict(seed=20261031, requested_draws=2000, retained_draws=int(valid.sum()),
            zero_denominator_draws=np.flatnonzero(~valid).tolist(), interventions=interventions,
            multiplicities=multiplicity.tolist(), shared_references_fixed=True), utility_claim=False if program is None else None)


def development_checks(result, support, metric_ok, references_ok):
    metrics, uncertainty = result["metrics"], result["uncertainty"]
    candidate = metrics["phi18"]
    u = uncertainty["phi18"]
    required = [k for k in metrics if k not in ("selector", "phi18", "no_change", "additive", "base", "pooled_kernel", "per_context_profile", "per_context_product", "joint_context")]
    gains = all(v is not None and v >= .05 for v in u["relative_gain"])
    positive_bound = all(bounds[0] > 0 for bounds in u["conditional_bounds"])
    better_controls = all(all(candidate["contexts"][c]["mse"] < metrics[arm]["contexts"][c]["mse"] for c in range(2)) for arm in required)
    discrimination = all(candidate["contexts"][c]["retrieval"]["score"] is not None and
        metrics["selector"]["contexts"][c]["retrieval"]["score"] is not None and
        candidate["contexts"][c]["retrieval"]["score"] > metrics["selector"]["contexts"][c]["retrieval"]["score"] for c in range(2))
    checks = dict(primary_gain=gains, positive_conditional_bound=positive_bound,
        better_required_controls=better_controls, qualified_discrimination=bool(metric_ok and discrimination),
        no_candidate_fallback=candidate["fallback_pairs"] == 0,
        all_required_supported=bool(support), complete_population=candidate["original_query_pairs"] == candidate["scored_query_pairs"],
        disjoint_references=bool(references_ok))
    return dict(passed=all(checks.values()), checks=checks)


def five_gates(qualification, integrity, primary, swaps, confirmation=None, frontier=None):
    gate1 = qualification.get("qualified") is True and qualification.get("numerical_acceptance_verified") is True
    gate2 = integrity.get("qualified") is True
    gate3 = gate1 and gate2 and primary.get("passed") is True and len(swaps) == 2 and all(s.get("passed") is True for s in swaps)
    # This fixed-context pipeline cannot establish gates 4/5. Reviewed external
    # confirmation results require their separately registered evaluator.
    gates = [dict(gate=1, passed=gate1, status="pass" if gate1 else "access_resource_or_qualification_failure"),
        dict(gate=2, passed=gate2, status="pass" if gate2 else "unqualified"),
        dict(gate=3, passed=gate3, status="pass" if gate3 else "supported_negative" if gate1 and gate2 else "unqualified"),
        dict(gate=4, passed=False, status="separate_confirmation_required"),
        dict(gate=5, passed=False, status="native_and_utility_evidence_required")]
    return dict(experiment_id=PROTOCOL["experiment_id"], gates=gates,
        development_nominated=gate3, independent_biological_confirmation=False, frontier_claim=False,
        status="nominate_for_fresh_data" if gate3 else "retire_finite_phi18" if gates[2]["status"] == "supported_negative" else "unqualified_inconclusive",
        primary=primary, reference_swaps=swaps, confirmation_evidence_supplied=confirmation is not None,
        frontier_evidence_supplied=frontier is not None)
