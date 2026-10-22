import hashlib

import numpy as np
import pytest

from opal.flexible_prediction import CONTRACT, FixedOrderFlexiblePredictor
from opal.flexible_singles import FlexibleSingles
from tools.wave5_numerical_matrix import trace


def payload():
    return dict(format_version=1, nodes=True, alpha=10., n_centers=4, seed=1,
        singles=np.array([-.2, .1, .3]), support=np.ones(3, bool),
        input_center=np.zeros(2), input_scale=np.ones(2), center_rows=np.arange(4),
        centers=np.array([[-1., 0.], [0., 1.], [1., 0.], [0., -1.]]),
        whitener=np.array([[1., 1e5, -1e5, .5], [-.5, -1e5, 1e5, 1.],
                           [2., 1e5, -1e5, -.5], [1., -1e5, 1e5, 2.]]),
        bandwidth=1., center=np.zeros(10), scale=np.full(10, 2.**-20),
        offset=.03, coefficient=np.linspace(-.1, .1, 10))


def legacy(tmp_path, data=None):
    p = tmp_path/'legacy.npz'
    np.savez(p, **(payload() if data is None else data))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def test_fixed_order_matches_measured_candidate_and_preserves_legacy(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No fitting in a prediction contract')
    for method in ('fit', 'prepare', 'solve'):
        monkeypatch.setattr(FlexibleSingles, method, forbidden)
    path, sha = legacy(tmp_path)
    original_bytes = path.read_bytes()
    old = FlexibleSingles.load(path)
    model = FixedOrderFlexiblePredictor.from_legacy_checkpoint(path, sha)
    query = np.tile([[0, 1], [0, 2], [1, 2]], (700, 1))
    actual = model.predict(query)
    expected = trace(old, query, variant='ascending_joint')['prediction']
    np.testing.assert_array_equal(actual, expected)
    assert np.max(abs(actual-old.singles[query].sum(1))) > 1
    rng = np.random.default_rng(41)
    perm = rng.permutation(len(query))
    variants = [model.predict(query[perm])[np.argsort(perm)], model.predict(query[::-1])[::-1],
                model.predict(query[:, ::-1]), model.predict(np.asfortranarray(query)),
                model.predict(np.repeat(query, 2, axis=0)[::2])]
    for size in (1, 2, 63, 2047, 2048, 2049, len(query)):
        variants.append(np.concatenate([model.predict(query[s:s+size]) for s in range(0, len(query), size)]))
    for value in variants:
        np.testing.assert_array_equal(value, actual)
    assert path.read_bytes() == original_bytes
    saved = tmp_path/'fixed.npz'
    model.save(saved)
    with np.load(saved) as z:
        assert str(z['prediction_contract']) == CONTRACT
        assert int(z['format_version']) == 2
    restored = FixedOrderFlexiblePredictor.load(saved, hashlib.sha256(saved.read_bytes()).hexdigest())
    np.testing.assert_array_equal(restored.predict(query), actual)
    with pytest.raises(ValueError, match='format'):
        FlexibleSingles.load(saved)
    with pytest.raises(FileExistsError):
        model.save(saved)
    assert model.predict(np.empty((0, 2), int)).shape == (0,)


@pytest.mark.parametrize('field,value', [
    ('scale', np.zeros(10)), ('input_scale', np.zeros(2)), ('bandwidth', np.nan),
    ('offset', True), ('nodes', 1), ('support', np.ones(3, int)),
    ('coefficient', np.full(10, np.inf)), ('whitener', np.eye(3)),
    ('center_rows', np.array([0, 1, 1, 3])), ('singles', np.ones(3, np.float32)),
    ('seed', -1), ('alpha', np.array([10.])),
])
def test_invalid_frozen_state_refuses(tmp_path, field, value):
    data = payload()
    data[field] = value
    path, sha = legacy(tmp_path, data)
    with pytest.raises(ValueError):
        FixedOrderFlexiblePredictor.from_legacy_checkpoint(path, sha)


def test_hash_contract_support_and_query_refusals(tmp_path):
    data = payload()
    data['support'][2] = False
    path, sha = legacy(tmp_path, data)
    with pytest.raises(ValueError, match='hash'):
        FixedOrderFlexiblePredictor.from_legacy_checkpoint(path, '0'*64)
    with pytest.raises(ValueError, match='SHA256'):
        FixedOrderFlexiblePredictor.from_legacy_checkpoint(path, None)
    model = FixedOrderFlexiblePredictor.from_legacy_checkpoint(path, sha)
    for query in (np.array([[0, 2]]), np.array([[0, 0]]), np.array([[0, 3]]),
                  np.array([[0., 1.]]), np.array([0, 1])):
        with pytest.raises(ValueError):
            model.predict(query)
    with pytest.raises(ValueError):
        FixedOrderFlexiblePredictor.load(path)


def test_extreme_finite_state_refuses_unrepresentable_prediction(tmp_path):
    data = payload()
    data['coefficient'][:] = np.finfo(float).max
    path, sha = legacy(tmp_path, data)
    model = FixedOrderFlexiblePredictor.from_legacy_checkpoint(path, sha)
    with pytest.raises(ValueError, match='representable'):
        model.predict(np.array([[0, 1]]))
