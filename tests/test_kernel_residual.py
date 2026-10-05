import numpy as np
from opal.models.kernel import single_kernel, symmetric_kernel, global_kernel, smoother, prediction_error
from opal.models.kernel import KernelResidual
from opal.training import fit
from opal.inference import export_members, Predictor
import pytest


def test_symmetric_pair_kernel_is_psd_and_order_invariant():
    rng = np.random.default_rng(77)
    singles = rng.normal(size=(6, 12))
    pairs = np.array([(0, 1), (2, 4), (1, 5), (0, 4)])
    single = single_kernel(singles)
    for interaction in [False, True]:
        kernel = symmetric_kernel(single, pairs, pairs, interaction)
        swapped = symmetric_kernel(single, pairs[:, ::-1], pairs, interaction)
        np.testing.assert_allclose(kernel, swapped, atol=1e-12)
        assert np.linalg.eigvalsh(kernel).min() > -1e-12


def test_gram_error_matches_gene_space_and_intercept():
    rng = np.random.default_rng(4)
    singles = rng.normal(size=(6, 12))
    pairs = np.array([(0, 1), (2, 4), (1, 5), (0, 4), (2, 5)])
    kernel = global_kernel(singles, pairs)
    train, query = np.array([0, 1, 2]), np.array([3, 4])
    weights = smoother(kernel, train, query, 0.1)
    np.testing.assert_allclose(weights.sum(1), 1)
    residual = rng.normal(size=(5, 12))
    direct = np.mean((weights @ residual[train] - residual[query]) ** 2)
    efficient = prediction_error(weights, train, query, residual @ residual.T / 12)
    np.testing.assert_allclose(direct, efficient)
    shifted = residual + rng.normal(size=12)
    shifted_mse = np.mean((weights @ shifted[train] - shifted[query]) ** 2)
    np.testing.assert_allclose(direct, shifted_mse)


@pytest.mark.parametrize('kind', ['shared_offset', 'global', 'single_sum', 'pair_product', 'global_plus_pair', 'global_plus_pair_10'])
def test_kernel_fit_matches_smoother_and_safe_export(tmp_path, kind):
    rng = np.random.default_rng(12)
    singles = rng.normal(size=(6, 13))
    pairs = np.array([(0, 1), (2, 4), (1, 5), (0, 4), (2, 5)])
    doubles = rng.normal(size=(5, 13))
    config = {'family': 'kernel', 'kind': kind, 'alpha': 0.1, 'bandwidth': 1}
    model = KernelResidual(singles, config).fit(pairs[:3], doubles[:3])
    residual = doubles - singles[pairs[:, 0]] - singles[pairs[:, 1]]
    weights = smoother(model.kernel(pairs, pairs), np.arange(3), np.arange(3, 5), .1)
    expected = singles[pairs[3:, 0]] + singles[pairs[3:, 1]] + weights @ residual[:3]
    np.testing.assert_allclose(model.predict(pairs[3:]), expected, atol=1e-11)
    np.testing.assert_allclose(model.predict(pairs[3:, ::-1]), expected, atol=1e-11)
    targets = np.array(list('abcdef'))
    data = {'singles': singles, 'targets': targets, 'genes': np.array([f'g{i}' for i in range(13)]),
            'control': np.zeros(13), 'pairs': pairs[:3], 'query_pairs': pairs[3:]}
    member = fit(config, singles, pairs[:3], doubles[:3])
    export_members([member], data, tmp_path, {'context': 'assay', 'scale': 'log2'})
    predictor = Predictor(tmp_path)
    np.testing.assert_allclose(predictor.predict('a', 'e')['effect'], expected[0], atol=1e-11)
    with pytest.raises(ValueError, match='Unsupported context'):
        predictor.predict('a', 'e', context='other-assay')
    with (tmp_path / 'member_0.npz').open('ab') as handle:
        handle.write(b'tampering')
    with pytest.raises(ValueError, match='checksum'):
        Predictor(tmp_path)
