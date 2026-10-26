"""Byte planes are a reversible integer permutation, not numeric precision loss."""
import hashlib

import numpy as np
import pytest

from opal.frontier_tournament import Predictor
from opal import frontier_packing as old
from opal import frontier_packing_v2 as new


def test_shuffled_triangle_and_original_formats_roundtrip(tmp_path):
    rng = np.random.default_rng(88103)
    matrix = rng.normal(size=(12, 12))
    matrix = np.triu(matrix)+np.triu(matrix, 1).T
    matrix[0, 1] = matrix[1, 0] = np.nextafter(0., 1.)
    singles = np.linspace(-.2, .2, 12)
    support = np.ones(12, dtype=bool)
    base = Predictor(dict(kind='sum', singles=singles, support=support, mean=0.))
    model = Predictor(dict(kind='completion', singles=singles, support=support, matrix=matrix), base)
    pairs = np.array([[0, 1], [2, 3], [3, 2], [0, 1]])
    for label, writer in (('old', old.save_checkpoint), ('new', new.save_checkpoint)):
        path = tmp_path/(label+'.npz')
        writer(model, path)
        loaded = new.load_checkpoint(path, hashlib.sha256(path.read_bytes()).hexdigest())
        assert loaded.state['matrix'].tobytes() == matrix.tobytes()
        assert np.array_equal(loaded.predict(pairs), model.predict(pairs))
        with pytest.raises(FileExistsError): new.save_checkpoint(model, path)
    path = tmp_path/'base.npz'
    new.save_checkpoint(base, path)
    assert np.array_equal(new.load_checkpoint(path, old.file_sha256(path)).predict(pairs), base.predict(pairs))


def test_bad_byte_plane_shape_refuses(tmp_path):
    path = tmp_path/'bad.npz'
    np.savez_compressed(path, contract=new.PACKING_CONTRACT, model_contract=old.CONTRACT,
        format_version=2, model__kind='completion', model__matrix_n=2,
        model__matrix_upper_bytes=np.zeros((7, 3), dtype=np.uint8), model__matrix_byte_order=np.dtype(np.float64).str,
        model__singles=np.zeros(2), model__support=np.ones(2, dtype=bool))
    with pytest.raises(ValueError, match='byte planes'):
        new.load_checkpoint(path, old.file_sha256(path))
