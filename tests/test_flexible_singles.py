import numpy as np
import pytest

from opal.flexible_singles import FlexibleSingles


def toy():
    pairs = np.array([(a, b) for a in range(20) for b in range(a+1, 20)])
    np.random.default_rng(8).shuffle(pairs)
    singles = np.linspace(-1, .1, 20)
    target = singles[pairs].sum(1)+np.sin(singles[pairs].prod(1)*4)
    return singles, pairs, target


@pytest.mark.parametrize('nodes', [False, True])
def test_symmetric_deterministic_roundtrip(nodes, tmp_path):
    s, p, y = toy()
    a, b = [FlexibleSingles(nodes=nodes).fit(s, p[:140], y[:140]) for _ in range(2)]
    np.testing.assert_array_equal(a.predict(p[140:]), b.predict(p[140:]))
    np.testing.assert_array_equal(a.predict(p[140:]), a.predict(p[140:, ::-1]))
    a.save(tmp_path/'model.npz')
    np.testing.assert_array_equal(a.predict(p[140:]), FlexibleSingles.load(tmp_path/'model.npz').predict(p[140:]))
    np.testing.assert_array_equal(a.center_rows, b.center_rows)


def test_training_only_moments_centers_and_normal_equations():
    s, p, y = toy()
    m = FlexibleSingles(nodes=True).fit(s, p[:140], y[:140])
    x = np.sort(s[p[:140]], axis=1)
    np.testing.assert_array_equal(m.input_center, x.mean(0))
    np.testing.assert_array_equal(m.centers, m._coordinates(p[:140])[m.center_rows])
    z = (m._features(p[:140])-m.center)/m.scale
    residual = y[:140]-s[p[:140]].sum(1)-m.offset
    expected = np.linalg.solve(z.T@z+m.alpha*np.eye(z.shape[1]), z.T@residual)
    np.testing.assert_allclose(m.coefficient, expected, rtol=1e-8, atol=1e-9)
    altered = y.copy()
    altered[140:] = 1e15
    other = FlexibleSingles(nodes=True).fit(s, p[:140], altered[:140])
    np.testing.assert_array_equal(m.coefficient, other.coefficient)


def test_fixed_map_penalty_solve_reuse_and_guard():
    s, p, y = toy()
    calls = []
    a = FlexibleSingles().prepare(s, p[:140], y[:140], check=lambda: calls.append(1))
    centers = a.centers.copy()
    for alpha in (1., 10., 100., 1000.):
        a.solve(alpha)
        b = FlexibleSingles(alpha=alpha).fit(s, p[:140], y[:140])
        np.testing.assert_array_equal(a.predict(p[140:]), b.predict(p[140:]))
        np.testing.assert_array_equal(a.centers, centers)
    assert calls


def test_invalid_missing_or_unsupported_inputs(tmp_path):
    s, p, y = toy()
    with pytest.raises(ValueError):
        FlexibleSingles().fit(s, p[:20], y[:20])
    with pytest.raises(ValueError):
        FlexibleSingles().fit(np.zeros(20), p[:140], y[:140])
    with pytest.raises(ValueError):
        FlexibleSingles().fit(s, np.vstack((p[:140], p[:1, ::-1])), np.r_[y[:140], y[:1]])
    broken = y.copy()
    broken[0] = np.nan
    with pytest.raises(ValueError):
        FlexibleSingles().fit(s, p[:140], broken[:140])
    keep = (p < 19).all(1)
    m = FlexibleSingles().fit(s, p[keep], y[keep])
    with pytest.raises(ValueError, match='support'):
        m.predict(np.array([[0, 19]]))
    with pytest.raises(ValueError):
        m.predict(np.array([[0, 0]]))


def test_nonlinear_single_function_is_learnable_without_gene_identity():
    s, p, y = toy()
    m = FlexibleSingles(alpha=1.).fit(s, p[:140], y[:140])
    error = np.mean((m.predict(p[140:])-y[140:])**2)
    additive = np.mean((s[p[140:]].sum(1)-y[140:])**2)
    assert error < additive*.2
