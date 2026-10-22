"""Fixed-state arithmetic controls; never fit or select any model."""
import numpy as np
import pytest
from opal.flexible_singles import FlexibleSingles
from tools import wave4_numerical_diagnosis as w


@pytest.fixture(autouse=True)
def prohibit_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Wave-four arithmetic tests cannot fit, prepare or solve')
    for name in ('fit', 'prepare', 'solve'):
        monkeypatch.setattr(FlexibleSingles, name, forbidden)


def fixed_state(ill_conditioned=False):
    """C's independent explicit-state design, with ordinary signal retained."""
    model = FlexibleSingles(nodes=False, centers=64)
    model.singles = np.array([-.2, .1, .3], dtype=np.float64)
    model.support = np.ones(3, dtype=bool)
    model.input_center = np.zeros(2)
    model.input_scale = np.ones(2)
    model.centers = np.column_stack((np.linspace(-1., 1., 64), np.linspace(.5, -.5, 64)))
    model.bandwidth = 1.
    model.whitener = np.eye(64)
    model.center = np.zeros(67)
    model.scale = np.ones(67)
    model.coefficient = np.zeros(67)
    model.coefficient[4] = .125
    model.offset = .03
    if ill_conditioned:
        model.whitener[:, 0] = np.where(np.arange(64) % 2, -1., 1.)*1e5
        model.scale[3] = 1e-8
        # Keep amplified contribution bounded while retaining cancellation.
        model.coefficient[3] = 1e-10
    return model


def pairs(count=2049):
    return np.tile(np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64), (count//3+1, 1))[:count]


def test_explicit_stage_trace_matches_unchanged_historical_prediction():
    model, query = fixed_state(), pairs()
    trace = w.query_trace(model, query)
    np.testing.assert_array_equal(trace['prediction'], model.predict(query))
    np.testing.assert_array_equal(trace['features'][:, :3],
        np.column_stack((model.singles[query].sum(1), abs(np.diff(model.singles[query], axis=1)).ravel(),
                         model.singles[query].prod(1))))
    # Ordinary nonlinear signal cannot be removed to make consistency pass.
    assert np.max(abs(trace['prediction']-(model.singles[query].sum(1)+model.offset))) > .01


@pytest.mark.parametrize('size', [1, 2, 63, 2047, 2048, 2049])
def test_ordinary_fixed_state_boundary_permutation_and_orientation(size):
    model, query = fixed_state(), pairs()
    reference = model.predict(query)
    current = w.query_trace(model, query, size)['prediction']
    np.testing.assert_allclose(current, reference, rtol=0, atol=1e-12)
    permutation = np.arange(len(query))[::-1]
    np.testing.assert_allclose(model.predict(query[permutation])[np.argsort(permutation)], reference, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model.predict(query[:, ::-1]), reference)


def test_injected_whitened_error_has_prespecified_signed_amplification():
    model, query = fixed_state(), np.array([[0, 1], [0, 2]])
    model.scale[4] = 1e-5
    # Centering the measured feature value here only sets an explicit fixture;
    # it does not estimate any statistic from targets or fit a transform.
    features = w.stages(model, query)['features']
    model.center[4] = features[0, 4]
    changed = features.copy()
    delta = 2.**-40
    changed[0, 4] += delta
    original_prediction = w.projection(model, features, query)
    changed_prediction = w.projection(model, changed, query)
    observed = changed_prediction-original_prediction
    expected = np.array([delta/model.scale[4]*model.coefficient[4], 0.])
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
    assert abs(observed[0]) > 1e-12


def test_global_mapping_keeps_occurrence_alignment_and_kernel_values():
    model, population = fixed_state(), pairs(4101)
    ids = np.array([0, 2047, 2048, 2048, 4096, 4100])
    local = w.query_trace(model, population[ids])
    global_map = w.global_trace(model, population, ids)
    for key in ('ordered', 'coordinates', 'distance', 'kernel'):
        np.testing.assert_array_equal(local[key], global_map[key])
    np.testing.assert_allclose(w.projection(model, global_map['features'], population[ids]),
                               model.predict(population[ids]), rtol=0, atol=1e-12)


def test_adversarial_state_and_layout_controls_are_finite_and_immutable():
    model, query = fixed_state(True), pairs()
    before = {k: v.copy() for k, v in vars(model).items() if isinstance(v, np.ndarray)}
    trace = w.query_trace(model, query)
    for value in trace.values():
        assert np.isfinite(value).all()
    np.testing.assert_array_equal(trace['kernel'], w.stages(model, np.asfortranarray(query))['kernel'])
    # A portable test does not require every BLAS stack to have the same bug.
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(model, key), value)


def test_empty_and_invalid_query_boundaries_use_unchanged_model_guards():
    model = fixed_state()
    assert model.predict(np.empty((0, 2), dtype=np.int64)).shape == (0,)
    for query in (np.array([[0, 0]]), np.array([[0, 3]]), np.array([[0., 1.]])):
        with pytest.raises(ValueError):
            model.predict(query)
    model.support[2] = False
    with pytest.raises(ValueError, match='support'):
        model.predict(np.array([[0, 2]]))


def test_exact_absolute_gate_cannot_be_masked_by_relative_tolerance():
    a = np.array([1., 1e8])
    b = a.copy()
    b[0] += 2e-12
    result = w.difference(a, b)
    assert result['max_abs'] > 1e-12
    assert not result['exact']
    assert result['index'] == (0,)


def test_no_overwrite_refuses_before_saved_state_access(tmp_path, monkeypatch):
    import json
    freeze = tmp_path/'synthetic-freeze.json'
    freeze.write_text(json.dumps(dict(
        numerical=dict(authorized=True, saved_state_only=True, synthetic_no_fit=True,
                       absolute_tolerance=1e-12, relative_tolerance=0),
        successor_authorized=False, scope=dict(biological_fit=False, test_scoring=False),
        files={})), encoding='utf-8')
    output = tmp_path/'existing.json'
    output.write_text('preserve these exact bytes', encoding='utf-8')
    monkeypatch.setattr(w.sys, 'argv', ['wave4_numerical_diagnosis.py',
        '--freeze', str(freeze),
        '--stopped', str(tmp_path/'nonexistent-state'), '--output', str(output)])
    with pytest.raises(FileExistsError):
        w.main()
    assert output.read_text(encoding='utf-8') == 'preserve these exact bytes'
    assert not output.with_suffix('.started.json').exists()
