"""Assumption-labelled paired-culture power sensitivity, never empirical approval."""
import math
from scipy.stats import nct, t


def paired_power(n, standardized_excess, alpha):
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError('At least two independent culture blocks required')
    if not math.isfinite(standardized_excess) or standardized_excess <= 0 or not 0 < alpha < 1:
        raise ValueError('Positive excess-over-margin and valid alpha required')
    return float(nct.sf(t.ppf(1-alpha, n-1), n-1, math.sqrt(n)*standardized_excess))


def minimum_blocks(standardized_excess, target=.9, alpha=.0125, maximum=4096):
    if not 0 < target < 1 or maximum < 2:
        raise ValueError('Valid target and finite sample-size ceiling required')
    # Validate even if the target would be reached at the first sample size.
    paired_power(2, standardized_excess, alpha)
    low, high = 2, 2
    while paired_power(high, standardized_excess, alpha) < target and high < maximum:
        low, high = high+1, min(maximum, high*2)
    if paired_power(high, standardized_excess, alpha) < target:
        return None
    while low < high:
        mid = (low+high)//2
        if paired_power(mid, standardized_excess, alpha) >= target:
            high = mid
        else:
            low = mid+1
    return low


def sensitivity():
    rows = []
    for gap in (.1, .2, .3, .5, 1.):
        for target in (.8, .9):
            n = minimum_blocks(gap, target)
            rows.append(dict(standardized_excess_over_margin=gap, per_context_target_power=target,
                independent_blocks_per_context=n, planning_power=paired_power(n, gap, .0125) if n else None,
                two_context_assay_outcomes=60*n if n else None,
                independently_initiated_cultures=6*n if n else None))
    return dict(schema='opal-prospective-power-sensitivity-v1', rows=rows, alpha_per_context=.0125,
        co_primary_contexts=2, family_one_sided_alpha_upper=.025,
        standardized_gap_definition='(expected culture-level paired MSE improvement - reviewed useful absolute margin) / SD of culture-level paired improvement',
        assumptions=['independent culture/reference blocks within context', 'iid normally distributed culture-level paired loss differences',
                     'one frozen candidate and source-locked comparator', 'no attrition or baseline-selection uncertainty in the planning calculation'],
        joint_power_lower_bound_at_90_percent_each=.8, biological_variance_estimated=False,
        useful_margin_selected=False, sample_size_recommended=False, empirical_power_reviewed=False,
        limitation='Sensitivity only. Pilot covariance, nonnormality, missingness, reference sharing and actual cost must be reviewed before choosing n.')
