"""No-fit, explicit-state numerical controls for the experimental matrix."""
import numpy as np
import pytest
from opal.flexible_singles import FlexibleSingles
from tools import wave5_numerical_matrix as w


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Numerical tests cannot fit')
    for name in ('fit', 'prepare', 'solve'):
        monkeypatch.setattr(FlexibleSingles, name, forbidden)


def state():
    model = FlexibleSingles(nodes=True, centers=4)
    model.singles = np.array([-.2, .1, .3])
    model.support = np.ones(3, dtype=bool)
    model.input_center = np.zeros(2)
    model.input_scale = np.ones(2)
    model.centers = np.array([[-1., 0.], [0., 1.], [1., 0.], [0., -1.]])
    model.whitener = np.eye(4)
    model.bandwidth = 1.
    model.center = np.zeros(10)
    model.scale = np.ones(10)
    model.coefficient = np.linspace(-.1, .1, 10)
    model.offset = .03
    return model


@pytest.mark.parametrize('variant', w.VARIANTS)
def test_variants_keep_signal_order_and_state(variant):
    model = state()
    query = np.tile([[0, 1], [0, 2], [1, 2]], (7, 1))
    before = {k: np.asarray(v).copy() for k, v in vars(model).items()}
    baseline = w.trace(model, query, variant=variant)['prediction']
    irregular = w.trace(model, query, (1, 3, 7), variant)['prediction']
    restored = w.trace(model, query[::-1], (3,), variant)['prediction'][::-1]
    np.testing.assert_allclose(baseline, irregular, atol=1e-12, rtol=0)
    np.testing.assert_allclose(baseline, restored, atol=1e-12, rtol=0)
    np.testing.assert_allclose(baseline, model.predict(query), atol=1e-12, rtol=0)
    assert np.max(abs(baseline-(model.singles[query].sum(1)+model.offset))) > .01
    for k, v in before.items():
        np.testing.assert_array_equal(v, vars(model)[k])


def test_exact_legacy_trace_and_occurrence_alignment():
    model = state()
    population = np.tile([[0, 1], [0, 2], [1, 2]], (1400, 1))
    ids = np.array([0, 2047, 2048, 2048, 4199])
    trace = w.global_trace(model, population, ids)
    np.testing.assert_array_equal(trace['kernel'], w.stage(model, population[ids])['kernel'])
    np.testing.assert_allclose(trace['prediction'], model.predict(population[ids]), atol=1e-12, rtol=0)


def test_gate_keeps_zero_relative_tolerance_and_zero_reference():
    observed = w.difference(np.array([0., 1e8+1e-6]), np.array([0., 1e8]))
    assert not observed['within_atol']
    assert observed['max_relative_nonzero_reference'] < 1e-12
    assert w.difference(np.array([1e-13]), np.array([0.]))['zero_reference_nonzero_error'] == 1
    with pytest.raises(ValueError, match='nonfinite'):
        w.difference(np.array([np.nan]), np.array([0.]))


def test_fixed_order_is_not_accuracy_oracle():
    terms = np.array([[2.**54, 1., -2.**54]])
    assert w.contract_product(terms, np.ones(3), 'ascending')[0] == 0
    assert (terms[0, 0]+terms[0, 2])+terms[0, 1] == 1


def test_nonfinite_stage_and_no_overwrite(tmp_path):
    model = state()
    model.whitener[0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        w.stage(model, np.array([[0, 1]]))
    output = tmp_path/'receipt.json'
    w.write(output, {'one': 1})
    with pytest.raises(FileExistsError):
        w.write(output, {'two': 2})
