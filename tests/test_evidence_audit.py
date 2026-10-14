import numpy as np
import pytest
from opal.evidence_audit import checked_losses, conditional_bounds, measurement_summary


def test_prediction_population_refuses_overlap_missing_shape_and_nonfinite():
    seen = np.zeros(3, bool)
    truth = np.ones((2, 4))
    losses = checked_losses({'a': truth+2}, truth, np.array([0, 2]), 3, seen)
    np.testing.assert_array_equal(losses['a'], [4, 4])
    with pytest.raises(ValueError, match='Duplicated'):
        checked_losses({'a': truth}, truth, np.array([0, 1]), 3, seen)
    with pytest.raises(ValueError, match='Finite complete prediction'):
        checked_losses({'a': np.full((1, 4), np.nan)}, truth[:1], np.array([1]), 3, seen)
    assert not seen[1]


def test_paired_bounds_keep_direction_and_deterministic_draws():
    result = conditional_bounds([1, 2], [3, 4], np.array([[0, 0], [1, 1]]))
    np.testing.assert_array_equal(result, [2, 2])
    with pytest.raises(ValueError):
        conditional_bounds([np.nan], [1], np.array([[0]]))


def test_measurement_accounting_preserves_unsupported_without_assigning_errors():
    row = dict(context='a', drug='x', status='technical_support_qualified', response_half_mse=1,
               residual_half_mse=2, zero_response_mse=3, zero_residual_mse=4)
    missing = dict(context='a', drug='y', status='unsupported', reasons=['support'])
    record = dict(rows=[row, missing], declared_population=2, supported_population=1)
    assert measurement_summary(record)['a']['supported'] == 1
    assert measurement_summary(record)['a']['means']['response_half_mse'] == 1
    record['rows'][1] = row.copy()
    with pytest.raises(ValueError, match='Duplicated'):
        measurement_summary(record)
