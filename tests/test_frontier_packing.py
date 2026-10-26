"""Oracle checks for lossless checkpoint packing. No registered profile or fit."""
import hashlib
import zipfile

import numpy as np
import pytest

from opal.frontier_packing import (
    CONTRACT, MAX_PACKED_GENES, PACKING_CONTRACT, Predictor, file_sha256, load_checkpoint,
    save_checkpoint)
import opal.frontier_packing as packing
from tools.frontier_tournament import predictions


def _symmetric(n, rng):
    values = rng.normal(size=(n, n))
    matrix = np.triu(values, 1)
    matrix = matrix + matrix.T
    matrix[np.diag_indices(n)] = rng.normal(size=n)
    return np.array(matrix, dtype=np.float64, copy=True)


def _completion(matrix, singles, support, base=None):
    model = Predictor(dict(kind='completion', singles=singles, support=support, matrix=matrix), base)
    return model


def _singles(n):
    singles = np.linspace(-0.4, 0.7, n, dtype=np.float64)
    singles[0] = np.float64(np.pi)
    support = np.ones(n, dtype=bool)
    support[0] = False
    return singles, support


def _linear_base(singles, support):
    width = 3
    coefficient = np.array([np.float64(np.pi), np.nextafter(np.float64(1.0), np.float64(2.0)), 0.0])
    coefficient[2] = np.copysign(np.float64(0.0), np.float64(-1.0))
    return Predictor(dict(
        kind='linear', singles=singles, support=support, nodes=np.bool_(False),
        center=np.zeros(width), scale=np.ones(width), coefficient=coefficient,
        offset=np.float64(-0.125)))


def _kernel_with_fortran_basis(singles, support):
    landmarks = np.array([[1, 2], [2, 3]], dtype=np.int64)
    basis = np.array([
        [np.float64(np.pi), np.nextafter(np.float64(0.0), np.float64(1.0)), np.float64(-0.5)],
        [np.float64(0.25), np.float64(0.0), np.float64(1.5)],
    ], dtype=np.float64)
    basis[1, 1] = np.copysign(np.float64(0.0), np.float64(-1.0))
    basis = np.asfortranarray(basis)
    width = basis.shape[1]
    model = Predictor(dict(
        kind='kernel', singles=singles, support=support, z=np.linspace(-0.3, 0.3, len(singles)),
        landmarks=landmarks, basis=np.ascontiguousarray(basis),
        center=np.zeros(width), scale=np.ones(width),
        coefficient=np.array([0.2, -0.1, 0.4], dtype=np.float64), offset=np.float64(0.05)))
    model.state['basis'] = np.array(basis, copy=True, order='K')
    return model


def _craft_npy(descr, shape, data, fortran=False):
    meta = {'descr': descr, 'fortran_order': fortran, 'shape': shape}
    text = repr(meta).encode('latin1')
    pad = (64 - ((10 + len(text) + 1) % 64)) % 64
    header = text + b' ' * pad + b'\n'
    return b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header + data


def test_completion_upper_triangle_preserves_scalar_bits_and_base(tmp_path):
    rng = np.random.default_rng(7)
    singles, support = _singles(6)
    matrix = _symmetric(6, rng)
    matrix[1, 1] = np.copysign(np.float64(0.0), np.float64(-1.0))
    matrix[2, 3] = matrix[3, 2] = np.nextafter(np.float64(0.0), np.float64(1.0))
    matrix[4, 5] = matrix[5, 4] = np.float64(np.pi)
    base = _linear_base(singles, support)
    model = _completion(matrix, singles, support, base)
    queries = np.array([[0, 1], [1, 2], [2, 4], [3, 5]], dtype=np.int64)
    before, before_support = predictions(model, queries, singles)
    path = tmp_path / 'completion.npz'
    save_checkpoint(model, path)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    assert 'model__matrix.npy' not in names
    assert 'model__matrix_upper.npy' in names
    _shape, fortran, _data = packing._npy_meta(
        archive_member(path, 'model__matrix_upper.npy'))
    assert fortran is False
    loaded = load_checkpoint(path, file_sha256(path), historical_roots=(tmp_path,))
    after, after_support = predictions(loaded, queries, singles)
    assert np.array_equal(before, after)
    assert np.array_equal(before_support, after_support)
    assert loaded.state['matrix'].tobytes() == matrix.tobytes()
    assert loaded.base.state['coefficient'].tobytes() == base.state['coefficient'].tobytes()
    assert str(loaded.state['kind']) == 'completion'
    with pytest.raises(ValueError, match='Unknown checkpoint contract'):
        Predictor.load(path, file_sha256(path))


def archive_member(path, name):
    with zipfile.ZipFile(path) as archive:
        return archive.read(name)


def test_signed_zero_asymmetry_is_not_repaired(tmp_path):
    singles, support = _singles(4)
    matrix = _symmetric(4, np.random.default_rng(3))
    matrix[1, 2] = np.float64(0.0)
    matrix[2, 1] = np.copysign(np.float64(0.0), np.float64(-1.0))
    assert np.array_equal(matrix, matrix.T)
    model = _completion(matrix, singles, support, _linear_base(singles, support))
    path = tmp_path / 'asymmetric.npz'
    with pytest.raises(ValueError, match='bitwise symmetric'):
        save_checkpoint(model, path)
    assert not path.exists()


def test_noncompletion_stays_on_the_original_contract(tmp_path):
    singles, support = _singles(5)
    support[:] = True
    model = _linear_base(singles, support)
    pairs = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int64)
    original = tmp_path / 'original.npz'
    adapted = tmp_path / 'adapted.npz'
    model.save(original)
    save_checkpoint(model, adapted)
    loaded = load_checkpoint(adapted, file_sha256(adapted))
    direct = Predictor.load(adapted, file_sha256(adapted))
    assert np.array_equal(loaded.predict(pairs), model.predict(pairs))
    assert np.array_equal(direct.predict(pairs), model.predict(pairs))
    assert str(packing._decode_npy(archive_member(adapted, 'contract.npy'))) == CONTRACT


def test_mixed_roster_predictions_match(tmp_path):
    rng = np.random.default_rng(11)
    singles, support = _singles(5)
    completion = _completion(_symmetric(5, rng), singles, support, _linear_base(singles, support))
    linear = _linear_base(singles, support)
    queries = np.array([[0, 1], [1, 3], [2, 4]], dtype=np.int64)
    expected = []
    for index, model in enumerate((completion, linear)):
        path = tmp_path / f'model-{index}.npz'
        save_checkpoint(model, path)
        expected.append((path, file_sha256(path), predictions(model, queries, singles)[0]))
    for path, digest, before in expected:
        loaded = load_checkpoint(path, digest, historical_roots=(tmp_path,))
        assert np.array_equal(predictions(loaded, queries, singles)[0], before)


def test_fortran_base_basis_keeps_its_layout(tmp_path):
    singles, support = _singles(5)
    support[:] = True
    base = _kernel_with_fortran_basis(singles, support)
    assert base.state['basis'].flags.f_contiguous
    assert not base.state['basis'].flags.c_contiguous
    matrix = _symmetric(5, np.random.default_rng(5))
    model = _completion(matrix, singles, support, base)
    pairs = np.array([[1, 2], [2, 3], [1, 4]], dtype=np.int64)
    before = model.predict(pairs)
    path = tmp_path / 'fortran-base.npz'
    save_checkpoint(model, path)
    _shape, fortran, _data = packing._npy_meta(archive_member(path, 'base__basis.npy'))
    assert fortran is True
    loaded = load_checkpoint(path, file_sha256(path), historical_roots=(tmp_path,))
    basis = loaded.base.state['basis']
    assert basis.flags.f_contiguous and not basis.flags.c_contiguous
    assert basis.tobytes(order='K') == base.state['basis'].tobytes(order='K')
    assert np.array_equal(loaded.predict(pairs), before)
    assert loaded.state['matrix'].flags.c_contiguous


def test_malformed_npy_header_refuses_before_allocation(monkeypatch):
    def allocated(*_args, **_kwargs):
        raise AssertionError('array allocation')

    monkeypatch.setattr(np, 'load', allocated)
    monkeypatch.setattr(np, 'empty', allocated)
    monkeypatch.setattr(np, 'zeros', allocated)
    monkeypatch.setattr(np, 'array', allocated)
    monkeypatch.setattr(np, 'ndarray', allocated)
    huge = _craft_npy('<f8', (2 ** 30, 2 ** 30), b'\x00' * 8)
    with pytest.raises(ValueError, match='size cap'):
        packing._decode_npy(huge)
    mismatch = _craft_npy('<f8', (4, 4), b'\x00' * 8)
    with pytest.raises(ValueError, match='does not match'):
        packing._decode_npy(mismatch)
    with pytest.raises(ValueError, match='Unsupported NPY dtype'):
        packing._decode_npy(_craft_npy('|O', (1,), b'\x00' * 8))
    with pytest.raises(ValueError, match='Unsupported NPY dtype'):
        packing._decode_npy(_craft_npy('<c16', (1,), b'\x00' * 16))
    with pytest.raises(ValueError, match='Structured NPY dtype'):
        packing._decode_npy(_craft_npy([('value', '<f8')], (1,), b'\x00' * 8))


def test_exact_c_payload_decodes_and_keeps_order():
    values = np.arange(4, dtype='<f8').reshape(2, 2)
    decoded = packing._decode_npy(_craft_npy('<f8', (2, 2), values.tobytes()))
    assert decoded.flags.c_contiguous and not decoded.flags.f_contiguous
    assert decoded.tobytes() == values.tobytes()


def test_hash_tamper_overwrite_and_gene_cap(tmp_path):
    singles, support = _singles(4)
    support[:] = True
    model = _linear_base(singles, support)
    path = tmp_path / 'once.npz'
    save_checkpoint(model, path)
    digest = file_sha256(path)
    with pytest.raises(FileExistsError):
        save_checkpoint(model, path)
    assert file_sha256(path) == digest
    with pytest.raises(ValueError, match='identity'):
        load_checkpoint(path, '0' * 64)
    flipped = bytearray(path.read_bytes())
    flipped[-1] ^= 1
    tampered = tmp_path / 'tampered.npz'
    tampered.write_bytes(flipped)
    with pytest.raises(ValueError, match='identity'):
        load_checkpoint(tampered, digest)
    wide = _symmetric(MAX_PACKED_GENES + 1, np.random.default_rng(1))
    wide_singles = np.zeros(MAX_PACKED_GENES + 1, dtype=np.float64)
    wide_support = np.ones(MAX_PACKED_GENES + 1, dtype=bool)
    huge = _completion(wide, wide_singles, wide_support)
    refused = tmp_path / 'wide.npz'
    with pytest.raises(ValueError, match='600'):
        save_checkpoint(huge, refused)
    assert not refused.exists()


def test_historical_file_is_not_removed(tmp_path):
    historical = tmp_path / 'history'
    historical.mkdir()
    sentinel = historical / 'keep.txt'
    sentinel.write_text('historical', encoding='utf-8')
    singles, support = _singles(4)
    support[:] = True
    model = _completion(_symmetric(4, np.random.default_rng(2)), singles, support)
    path = historical / 'model.npz'
    save_checkpoint(model, path)
    before = sorted(item.name for item in historical.iterdir())
    load_checkpoint(path, file_sha256(path), historical_roots=(historical,))
    assert sorted(item.name for item in historical.iterdir()) == before
    assert sentinel.read_text(encoding='utf-8') == 'historical'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == file_sha256(path)
