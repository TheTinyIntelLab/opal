import numpy as np
from tools.study_kernel_attribution import kernel_components, normalized, predict_residual, select


def test_normalization_is_fitting_only_and_shuffled_control_preserves_global():
    rng = np.random.default_rng(5)
    singles = rng.normal(size=(6, 9))
    pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    components, permutation = kernel_components(singles, pairs, [.25, 1., 4.], 18)
    train, val = np.arange(4), np.arange(4, 6)
    matrix = components[("global", 1.)].copy()
    _, scale = normalized(matrix, train)
    matrix[np.ix_(val, val)] *= 1000
    assert normalized(matrix, train)[1] == scale
    assert sorted(permutation.tolist()) == list(range(6))
    assert not np.allclose(components[("global_plus_pair", 1.)], components[("shuffled_pair", 1.)])


def test_selection_does_not_use_outer_validation_responses():
    rng = np.random.default_rng(5)
    singles = rng.normal(size=(6, 9))
    pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    components, _ = kernel_components(singles, pairs, [.25, 1., 4.], 18)
    residual = rng.normal(size=(6, 9))
    train = np.arange(4)
    folds = [(np.array([0, 1]), np.array([2, 3])), (np.array([2, 3]), np.array([0, 1]))]
    first, _ = select(components, residual @ residual.T / 9, train, folds, [.1, 1.])
    residual[4:] += 1000
    second, _ = select(components, residual @ residual.T / 9, train, folds, [.1, 1.])
    assert first == second
    predicted, weights, _ = predict_residual(components, first, train, np.array([4, 5]), residual)
    np.testing.assert_allclose(predicted, weights @ residual[train])
