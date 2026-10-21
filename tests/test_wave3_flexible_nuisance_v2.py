"""Synthetic qualification for the nested strongest-reference discriminator."""
import numpy as np
import pytest
from opal.flexible_singles import FlexibleSingles
from tools import wave3_flexible_nuisance_v2 as w


def fixture():
    rng = np.random.default_rng(61)
    pairs = np.array([(i, j) for i in range(16) for j in range(i+1, 16)])
    singles = rng.normal(size=16)*.2
    nuisance = rng.normal(size=(len(pairs), 15))
    y = singles[pairs].sum(1)+.3*nuisance[:, 0]+rng.normal(size=len(pairs))*.01
    tr, va, te = np.arange(80), np.arange(80, 100), np.arange(100, 120)
    return pairs, singles, nuisance, y, tr, va, te


def test_nested_reference_reproduces_established_flexible_nodes():
    p, s, n, y, tr, va, te = fixture()
    transform = FlexibleSingles(nodes=True).fit(s, p[tr], y[tr])
    model, menu, x = w.fit_reference(transform, p, s, None, y, tr, va, lambda: None)
    historical = FlexibleSingles(nodes=True).prepare(s, p[tr], y[tr])
    losses = []
    predictions = []
    for alpha in w.CONFIG['alphas']:
        historical.solve(alpha)
        losses.append(np.mean((historical.predict(p[va])-y[va])**2))
        predictions.append(historical.predict(p[te]))
    assert menu['chosen'] == int(np.argmin(losses))
    np.testing.assert_allclose(w.reference_predict(model, x, p, s, te),
                               predictions[menu['chosen']], rtol=0, atol=1e-12)
    augmented = w.mapped_features(transform, p, n)
    np.testing.assert_array_equal(augmented[:, :x.shape[1]], x)
    np.testing.assert_array_equal(augmented[:, x.shape[1]:], n)


def test_held_outcomes_and_covariates_do_not_change_fitted_state(tmp_path):
    p, s, n, y, tr, va, te = fixture()
    def run(target, nuisance):
        transform = FlexibleSingles(nodes=True).fit(s, p[tr], target[tr])
        model, menu, x = w.fit_reference(transform, p, s, nuisance, target, tr, va, lambda: None)
        return transform, model, menu, x
    transform, model, menu, x = run(y, n)
    changed = y.copy()
    changed[te] = 1e9
    altered = n.copy()
    altered[te] = -1e9
    other, m2, menu2, _ = run(changed, altered)
    assert menu == menu2
    for key in model:
        np.testing.assert_array_equal(model[key], m2[key])
    np.testing.assert_array_equal(transform.centers, other.centers)
    np.savez_compressed(tmp_path/'ridge.npz', **model)
    np.testing.assert_array_equal(w.reference_predict(model, x, p, s, te),
        w.reference_predict(w.load_npz(tmp_path/'ridge.npz'), x, p, s, te))
    transform.save(tmp_path/'map.npz')
    np.testing.assert_array_equal(w.mapped_features(transform, p, n),
        w.mapped_features(FlexibleSingles.load(tmp_path/'map.npz'), p, n))


def test_single_map_is_symmetric_and_nuisance_stays_pair_keyed():
    p, s, n, y, tr, _, _ = fixture()
    transform = FlexibleSingles(nodes=True).fit(s, p[tr], y[tr])
    np.testing.assert_array_equal(w.mapped_features(transform, p, n),
                                  w.mapped_features(transform, p[:, ::-1], n))


def test_nuisance_can_explain_control_generated_residual():
    p, s, n, y, tr, va, te = fixture()
    transform = FlexibleSingles(nodes=True).fit(s, p[tr], y[tr])
    base, _, x = w.fit_reference(transform, p, s, None, y, tr, va, lambda: None)
    augmented, _, z = w.fit_reference(transform, p, s, n, y, tr, va, lambda: None)
    assert np.mean((w.reference_predict(augmented, z, p, s, te)-y[te])**2) < .1*np.mean(
        (w.reference_predict(base, x, p, s, te)-y[te])**2)


def test_augmented_coefficients_match_centered_dense_equations():
    p, s, n, y, tr, va, _ = fixture()
    transform = FlexibleSingles(nodes=True).fit(s, p[tr], y[tr])
    model, menu, x = w.fit_reference(transform, p, s, n, y, tr, va, lambda: None)
    z = (x[tr]-model['center'])/model['scale']
    residual = y[tr]-s[p[tr]].sum(1)
    expected = np.linalg.solve(z.T@z+menu['alpha']*np.eye(z.shape[1]),
                               z.T@(residual-residual.mean()))
    np.testing.assert_allclose(model['coefficient'], expected, atol=1e-9, rtol=1e-8)


def test_gate_requires_both_infections_controls_and_all_references():
    names = w.FAMILIES+('existing_calibrated_main', 'historical_minimum', 'guide_minimum', 'zero')
    stats = {k: {'mse': 1.} for k in names}
    stats['aligned'] = {'mse': .9}
    stats.update({f'shuffled-{seed}': {'mse': 1.} for seed in w.CONFIG['control_seeds']})
    comparison = {'flexible_nuisance': {'relative_gain': .05, 'uncertainty': {'gene_conditional_95': [-.2, -.01]}}}
    infections = [{'aligned': {'mse': .9}, 'flexible_nuisance': {'mse': 1.}} for _ in range(2)]
    assert w.gate(stats, comparison, infections, True)
    assert not w.gate(stats, comparison, infections, False)
    infections[1]['aligned']['mse'] = 1.01
    assert not w.gate(stats, comparison, infections, True)
    infections[1]['aligned']['mse'] = .9
    stats['flexible_nodes']['mse'] = .89
    assert not w.gate(stats, comparison, infections, True)
    stats['flexible_nodes']['mse'] = 1.
    comparison['flexible_nuisance']['relative_gain'] = .04999999
    assert not w.gate(stats, comparison, infections, True)


def test_freeze_refuses_overwrite_and_registered_bytes_are_complete(tmp_path):
    from argparse import Namespace
    with pytest.raises(FileExistsError):
        w.freeze(Namespace(output=tmp_path))
    assert all(len(v) == 64 for v in w.signature().values())
    assert w.CONFIG['rank'] == 8 and w.CONFIG['steps'] == 500
    assert len(w.CONFIG['alphas']) == 4 and w.CONFIG['centers'] == 64
