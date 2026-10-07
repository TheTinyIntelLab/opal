import numpy as np
import pytest
from tools.run_dose_design import dose_predict, precision_variance


def test_dose_curve_has_no_query_response_argument():
    x = np.array([10., 1000.]); y = np.array([[1., 2.], [3., 4.]])
    p = dose_predict(x, y, 100., dict(kind='ridge', degree=1, alpha=.001))
    np.testing.assert_allclose(p, [2., 3.])
    with pytest.raises(ValueError):
        dose_predict([], np.zeros((0, 2)), 100., dict(kind='mean'))
    with pytest.raises(ValueError):
        dose_predict(x, y, 0., dict(kind='mean'))


def test_more_cells_does_not_remove_biological_dependence():
    sd = dict(state=.1, culture=.06, target=.06, pair=.05, shared_control=.04, cell=.2)
    pairs = [[0, 1], [0, 2], [1, 2]]
    small = precision_variance(sd, 3, 3, pairs, 50)
    large = precision_variance(sd, 3, 3, pairs, 1000000)
    assert large > sd['state'] ** 2 / 3
    assert .99 * small < large < small
    # More states cannot average away the same globally shared target/pair effects.
    floor = sd['target'] ** 2 / 3 + sd['pair'] ** 2 / 3
    assert precision_variance(sd, 1000000, 3, pairs, 200) > floor


def test_target_imbalance_preserves_crossed_covariance():
    sd = dict(state=0., culture=0., target=1., pair=0., shared_control=0., cell=0.)
    balanced = precision_variance(sd, 1, 1, [[0, 1], [2, 3], [4, 5]], 100)
    incident = precision_variance(sd, 1, 1, [[0, 1], [0, 2], [0, 3]], 100)
    assert incident > balanced
