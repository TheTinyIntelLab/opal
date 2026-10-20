import numpy as np
import pytest
import torch

from opal.block_completion import ScalarCompletion, ScalarReference, checked_pairs, fit_reference_grid
from opal.models.fitness import GrowthReference


def fixture():
    pairs = np.array([(a, b) for a in range(10) for b in range(a+1, 10)])
    rng = np.random.default_rng(17)
    rng.shuffle(pairs)
    singles = np.linspace(-.4, 0, 10)
    return pairs, singles


def test_baselines_hand_calculation_and_roundtrip(tmp_path):
    pairs, singles = fixture()
    y = singles[pairs].sum(1)
    for kind in ('mean', 'zero', 'additive', 'minimum', 'calibration', 'node'):
        model = ScalarReference(kind).fit(singles, pairs[:30], y[:30])
        path = tmp_path/(kind+'.npz')
        model.save(path)
        np.testing.assert_array_equal(model.predict(pairs[30:]), ScalarReference.load(path).predict(pairs[30:]))
        if kind == 'mean':
            np.testing.assert_array_equal(model.predict(pairs[30:]), np.full(15, y[:30].mean()))
        if kind == 'additive':
            np.testing.assert_array_equal(model.predict(pairs[30:]), y[30:])
        if kind == 'minimum':
            np.testing.assert_array_equal(model.predict(pairs[30:]), singles[pairs[30:]].min(1))


def test_scalar_completion_is_deterministic_symmetric_and_serializes(tmp_path):
    torch.set_num_threads(1)
    pairs, _ = fixture()
    y = np.sin(pairs[:, 0])*np.sin(pairs[:, 1])
    args = (pairs[:30], y[:30], 10, 1., (pairs[30:40], y[30:40]))
    a, b = [ScalarCompletion(steps=50).fit(*args) for _ in range(2)]
    np.testing.assert_array_equal(a.predict(pairs[40:]), b.predict(pairs[40:]))
    np.testing.assert_array_equal(a.predict(pairs[40:]), a.predict(pairs[40:, ::-1]))
    a.save(tmp_path/'model.npz')
    np.testing.assert_array_equal(a.predict(pairs[40:]), ScalarCompletion.load(tmp_path/'model.npz').predict(pairs[40:]))
    assert a.parameter_count == 160
    assert a.best_step in (25, 50)


def test_train_validation_leakage_and_bad_inputs_fail():
    pairs, singles = fixture()
    with pytest.raises(ValueError, match='leakage'):
        ScalarCompletion(steps=1).fit(pairs[:30], np.zeros(30), 10, 1., (pairs[:2], np.zeros(2)))
    for query in (np.array([[0, 0]]), np.array([[0, 10]]), np.array([[0., 1.]])):
        with pytest.raises(ValueError):
            checked_pairs(query, 10)
    with pytest.raises(ValueError, match='Duplicate'):
        ScalarReference().fit(singles, np.array([[0, 1], [1, 0]]), np.zeros(2))
    with pytest.raises(ValueError, match='Finite'):
        ScalarReference().fit(singles, pairs[:3], np.array([0., np.nan, 0.]))


def test_fits_never_receive_held_outcome_and_guards_run():
    pairs, singles = fixture()
    y = singles[pairs].sum(1)
    a = ScalarReference('node').fit(singles, pairs[:30], y[:30])
    altered = y.copy()
    altered[30:] = 1e20
    b = ScalarReference('node').fit(singles, pairs[:30], altered[:30])
    np.testing.assert_array_equal(a.predict(pairs[30:]), b.predict(pairs[30:]))
    def stop():
        raise RuntimeError('guard reached')
    with pytest.raises(RuntimeError, match='guard reached'):
        ScalarCompletion(steps=25).fit(pairs[:30], y[:30], 10, 1., (pairs[30:40], y[30:40]), check=stop)


def test_streamed_ridge_matches_existing_dense_reference():
    pairs, singles = fixture()
    y = singles[pairs].sum(1)+np.sin(pairs[:, 0]+pairs[:, 1])*.01
    grid = fit_reference_grid(singles, pairs[:30], y[:30])
    assert len(grid) == 12
    for model in grid:
        if model.kind in {'calibration', 'node'}:
            expected = GrowthReference(model.kind, model.alpha).fit(singles[None], pairs[:30], y[None, :30])
            np.testing.assert_allclose(model.predict(pairs[30:]), expected.predict(pairs[30:])[0], rtol=0, atol=1e-12)


def test_scalar_completion_recovers_simple_low_rank_signal():
    torch.set_num_threads(1)
    pairs, _ = fixture()
    y = np.full(len(pairs), .2)
    model = ScalarCompletion(rank=2, penalty=.0001, steps=300).fit(
        pairs[:30], y[:30], 10, 1., (pairs[30:40], y[30:40]))
    assert np.mean((model.predict(pairs[40:])-y[40:])**2) < .25*np.mean(y[40:]**2)
