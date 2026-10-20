"""Synthetic prerequisites for the frozen cross-infection diagnostic."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from opal.models.fitness import FitnessCompletion, GrowthReference

spec = importlib.util.spec_from_file_location('reset_development', Path(__file__).resolve().parents[1] / 'tools/reset_development.py')
reset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reset)


def fixture():
    pairs = np.array([(a, b) for a in range(6) for b in range(a+1, 6)])
    singles = np.arange(24).reshape(2, 2, 6) / 100
    reps = singles[:, :, pairs].sum(-1)
    d = dict(pairs=pairs, genes=np.array(list('ABCDEF')), singles=singles,
             replicates=reps, responses=reps.mean(1), counts=np.full((2, 2, len(pairs)), 4),
             single_counts=np.full((2, 2, 6), 20))
    # Each train gene has support, all split labels occur.
    split = np.array([0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2])
    return d, split


def test_integrity_rejects_missingness_duplicates_coverage_and_aggregation():
    d, split = fixture()
    reset.validate(d, split)
    for field, mutate in [('replicates', lambda x: x.__setitem__((0, 0, 0), np.nan)),
                          ('pairs', lambda x: x.__setitem__(1, x[0])),
                          ('counts', lambda x: x.__setitem__((0, 0, 0), 3)),
                          ('responses', lambda x: x.__setitem__((0, 0), 999))]:
        bad = {k: v.copy() for k, v in d.items()}
        mutate(bad[field])
        with pytest.raises(ValueError):
            reset.validate(bad, split)
    with pytest.raises(ValueError):
        reset.validate(d, np.zeros_like(split))
    bad_split = split.copy()
    bad_split[np.any(d['pairs'] == 5, axis=1)] = 2
    with pytest.raises(ValueError, match='supported'):
        reset.validate(d, bad_split)


def test_baseline_correctness_metrics_and_source_only_selection():
    d, split = fixture()
    singles, y, pairs = d['singles'][:, 0], d['replicates'][:, 0], d['pairs']
    train, val = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    cfg = dict(references=['zero', 'additive', 'minimum', 'calibration', 'node'], reference_alpha=100.)
    a = reset.selected_reference(singles, pairs, y, train, val, cfg)
    changed = y.copy()
    changed[:, split == 2] = 1e8
    b = reset.selected_reference(singles, pairs, changed, train, val, cfg)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(GrowthReference('additive').fit(singles, pairs[train], y[:, train]).predict(pairs), y)
    np.testing.assert_array_equal(GrowthReference('minimum').fit(singles, pairs[train], y[:, train]).predict(pairs), singles[:, pairs].min(-1))
    assert reset.mse(np.array([1., 3.]), np.array([2., 1.])) == 2.5
    with pytest.raises(ValueError):
        reset.mse(np.array([np.nan]), np.array([0.]))
    with pytest.raises(ValueError):
        reset.mse(np.zeros(2), np.zeros(3))


def test_negative_control_preserves_residual_values_and_context_covariance():
    x = np.arange(60.).reshape(2, 30)
    a = reset.shuffle_residuals(x, 7)
    np.testing.assert_array_equal(a, reset.shuffle_residuals(x, 7))
    np.testing.assert_array_equal(np.sort(a, axis=1), x)
    np.testing.assert_array_equal(a[1]-a[0], np.full(30, 30.))
    assert not np.array_equal(x, a)


def test_dyadic_uncertainty_is_deterministic_and_keeps_constant_differences():
    d, _ = fixture()
    pairs = d['pairs']
    a = reset.node_bootstrap(np.full(len(pairs), -.2), pairs, 6, 100, 17)
    np.testing.assert_allclose(a, [-.2, -.2], atol=1e-15)
    delta = np.arange(len(pairs)) / 100
    assert reset.node_bootstrap(delta, pairs, 6, 100, 17) == reset.node_bootstrap(delta, pairs, 6, 100, 17)


def test_completion_serialization_symmetry_leakage_and_replay(tmp_path):
    torch.set_num_threads(1)
    d, split = fixture()
    pairs = d['pairs']
    train, val, test = [np.flatnonzero(split == i) for i in range(3)]
    y = np.zeros((2, len(train)))
    models = [FitnessCompletion('independent', 8, .01, 25, 20261022).fit(pairs[train], y, 6, np.ones(2),
              validation=(pairs[val], np.zeros((2, len(val))))) for _ in range(2)]
    np.testing.assert_array_equal(models[0].predict(pairs[test]), models[1].predict(pairs[test]))
    path = tmp_path / 'model.npz'
    models[0].save(path)
    np.testing.assert_array_equal(models[0].predict(pairs[test]), FitnessCompletion.load(path).predict(pairs[test]))
    np.testing.assert_array_equal(models[0].predict(pairs[test]), models[0].predict(pairs[test, ::-1]))
    for query in [[[0, 0]], [[0, 6]]]:
        with pytest.raises(ValueError):
            models[0].predict(query)
    with pytest.raises(ValueError, match='leakage'):
        FitnessCompletion('independent', 8, .01, 25).fit(pairs[train], y, 6, np.ones(2), validation=(pairs[train[:1]], np.zeros((2, 1))))


def test_run_requires_ordered_passing_receipt_and_refuses_overwrite(tmp_path):
    from opal.artifacts import write_json
    write_json(tmp_path / 'protocol.json', dict(code=reset.signature(), created_utc='2026-01-01', data_root='missing'))
    write_json(tmp_path / 'prefit_tests.json', dict(code=reset.signature(), returncode=1))
    with pytest.raises(ValueError, match='Passing'):
        reset.run(tmp_path)
    with pytest.raises(FileExistsError):
        reset.freeze(None, None, tmp_path)
