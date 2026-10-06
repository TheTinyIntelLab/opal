"""Separately fitted risk diagnostics; shift coverage is never guaranteed."""
import math
import numpy as np
from scipy.stats import spearmanr


def evaluate_policy(features, errors, groups, *, fit_groups, calibration_groups, evaluation_groups,
                    nominal=.9, maximum_rmse=.25, minimum_accepted_fraction=.5):
    x, y, groups = np.asarray(features, float), np.asarray(errors, float), np.asarray(groups)
    partitions = [set(fit_groups), set(calibration_groups), set(evaluation_groups)]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Risk fitting, calibration and evaluation groups overlap')
    if x.ndim != 2 or y.shape != (len(x),) or groups.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('Risk evaluation needs finite diagnostic features and nonnegative RMSE')
    masks = [np.isin(groups, list(p)) for p in partitions]
    if any(not m.any() for m in masks):
        raise ValueError('Each risk partition needs observed conditions')
    fit, cal, test = masks
    mean, scale = x[fit].mean(0), np.maximum(x[fit].std(0), 1.)
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(z.shape[1]); penalty[0, 0] = 0.
    coefficient = np.linalg.solve(z[fit].T @ z[fit] + penalty, z[fit].T @ y[fit])
    estimate = np.maximum(z @ coefficient, 0.)
    # One conservative score per biological group, not one independent draw per cell.
    scores = sorted(float(np.max(y[(groups == g) & cal] - estimate[(groups == g) & cal])) for g in calibration_groups)
    rank = math.ceil((len(scores) + 1) * nominal)
    adjustment = max(0., scores[rank - 1]) if rank <= len(scores) else None
    predicted_risk, observed = estimate[test], y[test]
    upper = predicted_risk + adjustment if adjustment is not None else None
    # A predeclared risk threshold, without choosing it on evaluation outcomes.
    accepted = predicted_risk <= maximum_rmse
    policies = [{'policy': 'accept_none', 'accepted_fraction': 0., 'accepted_conditions': 0, 'mse': None},
                {'policy': 'accept_all', 'accepted_fraction': 1., 'accepted_conditions': len(observed), 'mse': float(np.mean(observed ** 2))},
                {'policy': 'risk_threshold', 'accepted_fraction': float(accepted.mean()), 'accepted_conditions': int(accepted.sum()),
                 'mse': float(np.mean(observed[accepted] ** 2)) if accepted.any() else None}]
    for fraction in [.1, .25, .5, .75]:
        # Thresholds come from calibration predicted risks. Test ties are included.
        threshold = float(np.quantile(estimate[cal], fraction))
        keep = predicted_risk <= threshold
        policies.append({'policy': 'calibration_quantile', 'requested_fraction': fraction,
                         'threshold': threshold, 'accepted_fraction': float(keep.mean()),
                         'accepted_conditions': int(keep.sum()),
                         'mse': float(np.mean(observed[keep] ** 2)) if keep.any() else None})
    correlation = float(spearmanr(predicted_risk, observed).statistic) if np.std(predicted_risk) > 1e-12 and np.std(observed) > 1e-12 else None
    interval_coverage = float(np.mean(observed <= upper)) if upper is not None else None
    useful = accepted.mean() >= minimum_accepted_fraction and accepted.any() and np.sqrt(np.mean(observed[accepted] ** 2)) <= maximum_rmse
    return {'status': 'development_diagnostic_not_validated', 'uncertainty_target': 'gene-vector RMSE',
        'risk_fit_groups': list(fit_groups), 'calibration_groups': list(calibration_groups),
        'evaluation_groups': list(evaluation_groups), 'nominal': nominal,
        'interval_status': 'finite_group_radius_unavailable' if upper is None else 'experimental_source_radius',
        'interval_adjustment': adjustment, 'interval_coverage': interval_coverage,
        'mean_upper_rmse': float(np.mean(upper)) if upper is not None else None,
        'ranking_spearman': correlation if correlation is None or np.isfinite(correlation) else None,
        'risk_rmse': float(np.sqrt(np.mean((predicted_risk - observed) ** 2))),
        'policies': policies, 'usefulness_gate': bool(useful),
        'reliability_gate': 'failed' if upper is None or interval_coverage < nominal or not useful else 'development_only',
        'by_group': {str(g): {'mse': float(np.mean(y[(groups == g) & test] ** 2)),
                             'accepted_fraction': float(np.mean(estimate[(groups == g) & test] <= maximum_rmse))}
                     for g in evaluation_groups},
        'failure_detection': {'threshold_rmse': maximum_rmse,
            'rejected_bad_fraction': float(np.mean(~accepted[observed > maximum_rmse])) if np.any(observed > maximum_rmse) else None},
        'formal_coverage_guarantee': False, 'fresh_confirmation': False,
        'limits': 'Shared-study development errors; model fitting folds share source labels. Partition isolation does not create independent confirmation.'}
