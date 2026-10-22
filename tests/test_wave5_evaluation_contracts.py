import hashlib

import numpy as np
import pytest

from opal.block_completion import ScalarCompletion
from opal.evaluation_contracts import metrics_v2, strict_load_completion_v2
from opal.scalar_screen import metrics


@pytest.mark.parametrize('dtype', [np.uint8, np.int64, np.float32, np.float64])
def test_metrics_use_float64_arithmetic(dtype):
    y, pred = np.array([255, 0], dtype=dtype), np.array([0, 255], dtype=dtype)
    r = metrics_v2(y, pred)
    assert r['mse'] == 65025
    assert r['mae'] == 255
    assert r['bias'] == 0
    assert r == metrics(y.astype(float), pred.astype(float))


def test_signed_integer_overflow_and_ordinary_replay():
    y = np.array([2**62, -2**62], dtype=np.int64)
    pred = -y
    assert metrics_v2(y, pred)['mse'] == float(2**126)
    assert metrics_v2(y, pred)['mae'] == float(2**63)
    y, pred = np.array([1., 2., 4.]), np.array([2., 0., 5.])
    assert metrics_v2(y, pred) == metrics(y, pred)


@pytest.mark.parametrize('y,pred', [([1e308, -1e308], [-1e308, 1e308]),
    ([np.nan], [0]), ([1], [np.inf]), ([], []), ([[1]], [[2]]),
    ([True], [False]), (['1'], ['2']), ([1j], [2j])])
def test_metrics_refuse_invalid_or_unrepresentable_values(y, pred):
    with pytest.raises(ValueError):
        metrics_v2(y, pred)


def completion_payload():
    return dict(rank=2, penalty=.01, steps=25, seed=1, format_version=2,
        normalized_upper=np.arange(6, dtype=np.float32)/10, support=np.ones(3, bool),
        scale=1., best_step=25, parameter_count=12)


@pytest.mark.parametrize('field,value', [('penalty', True), ('penalty', np.array([.01])),
    ('scale', np.array([1.])), ('scale', -1.), ('scale', np.nan), ('rank', 2.5),
    ('best_step', 26), ('parameter_count', 11), ('support', np.ones(3, int)),
    ('normalized_upper', np.zeros(6, np.float64)), ('normalized_upper', np.full(6, np.inf, np.float32)),
    ('seed', -1), ('format_version', 99)])
def test_strict_checkpoint_refusals(tmp_path, field, value):
    payload = completion_payload()
    payload[field] = value
    path = tmp_path/'bad.npz'
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        strict_load_completion_v2(path)


def test_strict_checkpoint_exact_valid_replay_and_hash_tamper(tmp_path):
    path = tmp_path/'valid.npz'
    np.savez(path, **completion_payload())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    old = ScalarCompletion.load(path)
    restored = strict_load_completion_v2(path, sha)
    np.testing.assert_array_equal(restored.predict(np.array([[0, 1], [2, 1]])),
                                  old.predict(np.array([[0, 1], [2, 1]])))
    with path.open('ab') as f:
        f.write(b'tamper')
    with pytest.raises(ValueError, match='hash'):
        strict_load_completion_v2(path, sha)
