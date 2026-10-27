"""Root counterexamples written independently of the export implementation."""
from types import SimpleNamespace

import numpy as np
import pytest

from opal.trained_growth import TaskModel, train_degree_support


def test_duplicate_orientation_never_creates_a_second_partner():
    pairs = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int64)
    try:
        degree, support = train_degree_support(pairs, np.arange(3), 3)
    except ValueError:
        return  # Rejecting a duplicated canonical TRAIN key is also safe.
    assert degree.tolist() == [1, 1, 0]
    assert not support.any()


class _NeverPredict:
    state = dict(singles=np.zeros(3), support=np.array([True, True, False]))

    def predict(self, pairs):
        raise AssertionError('An unsupported input reached the numerical predictor')


def fixture_model():
    return TaskModel(
        SimpleNamespace(label='root fixture', biological_delivery=False),
        dict(task='fixture', context='fixture-context'), _NeverPredict(),
        ['A', 'B', 'C'], ['DROPPED'], np.array([2, 2, 0], dtype=np.int64))


@pytest.mark.parametrize(('pair', 'reason'), [
    ((' A ', 'B'), 'malformed_name'),
    (('A', ''), 'malformed_name'),
    (('A', None), 'malformed_name'),
    (('A', 'UNKNOWN'), 'unknown_gene'),
    (('A', 'DROPPED'), 'absent_gene'),
    (('A', 'C'), 'unsupported_gene'),
    (('A', 'A'), 'self_pair'),
])
def test_unsupported_inputs_cannot_reach_prediction(pair, reason):
    result = fixture_model().predict([pair])[0]
    assert result['status'] == 'abstain'
    assert result['reason'] == reason
    assert result['value'] is None
    assert result['uncertainty'] == 'unavailable'


def test_context_mismatch_cannot_reach_prediction():
    result = fixture_model().predict([('A', 'B')], context='other-context')[0]
    assert result['reason'] == 'unknown_context'
    assert result['value'] is None


def test_query_cannot_replace_measured_source_singles():
    with pytest.raises(TypeError):
        fixture_model().predict([('A', 'B')], singles=np.ones(3))
