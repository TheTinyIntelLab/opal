import numpy as np
import pytest
from opal.models.composition import Composition


def test_recovers_overlap_and_roundtrips(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 2, 7))
    model = Composition(.1)
    f = model.features(x)
    y = .2 + f @ np.array([.7, -.8, .3])
    model.fit(x[:250], y[:250])
    assert np.mean((model.predict(x[250:]) - y[250:]) ** 2) < 1e-12
    affine = Composition(.1, overlap=False).fit(x[:250], y[:250])
    assert np.mean((affine.predict(x[250:]) - y[250:]) ** 2) > .01
    assert np.array_equal(model.predict(x), model.predict(x[:, ::-1]))
    path = tmp_path / 'model.npz'; model.save(path)
    assert np.array_equal(model.predict(x), Composition.load(path).predict(x))


def test_constant_features_and_refusals():
    x = np.zeros((5, 2, 4)); y = np.ones((5, 4))
    model = Composition().fit(x, y)
    assert np.array_equal(model.predict(x), y)
    with pytest.raises(ValueError): Composition(0)
    with pytest.raises(ValueError): model.predict(np.zeros((5, 2, 3)))
    with pytest.raises(ValueError): model.fit(x, y * np.nan)
