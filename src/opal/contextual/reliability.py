"""Grouped empirical RMSE radii; no exchangeability or OOD guarantee is inferred."""
import math
import numpy as np


def empirical_radius(predictions, responses, pairs, nominal=.9):
    p, y = np.asarray(predictions), np.asarray(responses)
    pairs = np.asarray(pairs)
    if not 0 < nominal < 1 or p.shape != y.shape or p.ndim != 2 or not p.shape[1] or pairs.shape != (len(p), 2):
        raise ValueError("Invalid radius diagnostic inputs")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Radius inputs must be finite")
    errors = np.sqrt(np.mean((p - y) ** 2, axis=1))
    groups = {}
    for pair, error in zip(pairs, errors):
        key = tuple(sorted(map(int, pair)))
        groups[key] = max(groups.get(key, 0.), float(error))
    scores = sorted(groups.values())
    rank = math.ceil((len(scores) + 1) * nominal)
    finite = bool(scores) and rank <= len(scores)
    return {"status": "empirical_diagnostic" if finite else "finite_radius_unavailable",
            "nominal": nominal, "radius": scores[rank - 1] if finite else None,
            "calibration_groups": len(scores), "calibration_rows": len(errors), "ordered_rank": rank,
            "score_unit": "pair_group_maximum_gene_vector_RMSE_over_source_contexts",
            "group_scores": scores, "row_errors": errors.tolist(), "formal_guarantee": False,
            "limitations": ["shared targets and references", "source to held-context distribution shift",
                            "not per-gene intervals", "not a validated abstention policy"]}


def selective_risk(errors, support_distances):
    errors, distances = np.asarray(errors), np.asarray(support_distances)
    if errors.shape != distances.shape or errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all() or not np.isfinite(distances).all():
        raise ValueError("Invalid selective risk diagnostics")
    ordered = np.sort(distances)
    output = []
    for fraction in [.25, .5, .75, 1.]:
        threshold = ordered[max(0, int(np.ceil(len(errors) * fraction)) - 1)]
        accepted = distances <= threshold
        output.append({"requested_fraction": fraction, "accepted_fraction": float(accepted.mean()),
                       "accepted_conditions": int(accepted.sum()), "distance_threshold": float(threshold),
                       "mean_squared_error": float(errors[accepted].mean()), "ties_included": True})
    return output
