"""Lossless checkpoint packing for completion matrices.

Only a bitwise-symmetric float64 completion matrix is stored, as its canonical
upper triangle including the diagonal. Every stored bit is copied. Values are
not quantized, averaged, symmetrized, or retrained. Any other layer, including
the residual base, is copied in full. Non-completion checkpoints stay on the
original predictor contract. Non-matrix arrays keep their C or Fortran layout.
An NPY member is rejected from its header, before the described array is allocated,
when the dtype is unsupported or the declared shape does not match the payload.

Loading a packed file rebuilds that original contract in a temporary directory
outside the historical tree, then reuses Predictor.load. The temporary files
are removed by the context manager. Historical records are never deleted.
"""
from pathlib import Path
import ast
import hashlib
import io
import re
import tempfile
import zipfile

import numpy as np

from .frontier_tournament import CONTRACT, Predictor


PACKING_CONTRACT = 'opal-frontier-packing-v1'
FORMAT_VERSION = 1
MAX_PACKED_GENES = 600
MAX_FILE_BYTES = 8 * 1024 ** 2
MAX_UNCOMPRESSED_BYTES = 16 * 1024 ** 2
TRANSIENT_BYTES = 32 * 1024 ** 2
MAX_ZIP_MEMBERS = 64
_MEMBER = re.compile(r'^[A-Za-z0-9_]+\.npy$')
_HEX = re.compile(r'^[0-9a-f]{64}$')
_UNICODE = re.compile(r'^<U([1-9][0-9]{0,4})$')
_BYTES = re.compile(r'^(?:<|\|)S([1-9][0-9]{0,5})$')
_FIXED_DTYPES = {
    '|b1': 1, '|i1': 1, '|u1': 1, '?': 1,
    '<i1': 1, '<u1': 1, '<i2': 2, '<u2': 2, '<i4': 4, '<u4': 4, '<i8': 8, '<u8': 8,
    '<f2': 2, '<f4': 4, '<f8': 8,
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(model, path):
    """Write a completion model in the packing contract, or the original format."""
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    if path.exists():
        raise FileExistsError(path)
    if _has_completion(model):
        _write_packed(path, model)
    else:
        model.save(path)
    return path


def load_checkpoint(path, expected_sha, historical_roots=()):
    """Load one original or packed checkpoint. The sha256 is required."""
    path = Path(path)
    _require_sha(path, expected_sha)
    members = _archive_members(path)
    contract = str(_decode_npy(members['contract'])) if 'contract' in members else None
    if contract == CONTRACT:
        return _load_original(path, expected_sha, members)
    if contract == PACKING_CONTRACT:
        return _load_packed(path, historical_roots, members)
    raise ValueError('Unknown checkpoint contract')


class MixedLoader:
    """Self-describing loader for a roster that mixes both checkpoint contracts."""

    def __init__(self, historical_roots=()):
        self.historical_roots = tuple(historical_roots)

    def load(self, path, expected_sha256=None):
        return load_checkpoint(path, expected_sha256, historical_roots=self.historical_roots)


def _has_completion(model):
    layers = [model]
    if model.base is not None:
        if model.base.base is not None:
            raise ValueError('Only one residual layer supported')
        layers.append(model.base)
    return any(str(layer.state.get('kind')) == 'completion' for layer in layers)


def _layout_copy(array):
    """Copy values without converting a C or Fortran matrix to the other order."""
    array = np.asarray(array)
    if array.dtype.hasobject or array.dtype.names is not None or array.dtype.kind in 'cOVmM':
        raise ValueError('Unsupported checkpoint dtype')
    if array.ndim >= 2 and not (array.flags.c_contiguous or array.flags.f_contiguous):
        raise ValueError('Noncontiguous checkpoint layout is not supported')
    copied = np.array(array, copy=True, order='K')
    if array.ndim >= 2 and (
            copied.flags.c_contiguous != array.flags.c_contiguous
            or copied.flags.f_contiguous != array.flags.f_contiguous):
        raise ValueError('Checkpoint copy changed memory layout')
    return copied


def _snapshot_state(state):
    copied = {}
    for key, value in state.items():
        copied[str(key)] = _layout_copy(value)
    return copied


def _upper_triangle(matrix):
    """Copy the upper triangle only when the matrix is already bitwise symmetric."""
    matrix = np.asarray(matrix)
    if matrix.dtype != np.dtype(np.float64) or not matrix.dtype.isnative:
        raise ValueError('Completion matrix must be native float64')
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Completion matrix must be square')
    n = int(matrix.shape[0])
    if n < 1 or n > MAX_PACKED_GENES:
        raise ValueError(
            f'Packed completion genes {n} outside 1..{MAX_PACKED_GENES}')
    canonical = np.ascontiguousarray(matrix)
    mirrored = np.ascontiguousarray(matrix.T)
    if not np.array_equal(canonical.view(np.uint64), mirrored.view(np.uint64)):
        raise ValueError('Completion matrix is not bitwise symmetric')
    upper = np.array(canonical[np.triu_indices(n)], copy=True, order='C')
    if upper.dtype != np.dtype(np.float64) or upper.shape != (n * (n + 1) // 2,):
        raise ValueError('Upper triangle layout is not canonical')
    if not np.isfinite(upper).all():
        raise ValueError('Nonfinite completion upper triangle')
    if not np.array_equal(upper.view(np.uint64), canonical[np.triu_indices(n)].view(np.uint64)):
        raise ValueError('Upper triangle bits changed')
    return upper, n


def _layer_payload(state):
    copied = _snapshot_state(state)
    kind = str(copied.get('kind'))
    if kind != 'completion':
        if 'matrix_upper' in copied or 'matrix_n' in copied:
            raise ValueError('Noncanonical packing field on a non-completion layer')
        return copied, False
    if 'matrix' not in copied:
        raise ValueError('Completion matrix is missing')
    upper, n = _upper_triangle(copied['matrix'])
    singles = copied.get('singles')
    support = copied.get('support')
    if singles is None or singles.ndim != 1 or len(singles) != n or singles.dtype.kind not in 'iuf':
        raise ValueError('Completion singles do not match the matrix')
    if singles.dtype.kind == 'f' and not np.isfinite(singles).all():
        raise ValueError('Nonfinite singles')
    if support is None or support.shape != singles.shape or support.dtype != np.dtype(bool):
        raise ValueError('Completion support is not a boolean gene mask')
    payload = {key: value for key, value in copied.items() if key != 'matrix'}
    payload['matrix_upper'] = upper
    payload['matrix_n'] = np.int64(n)
    return payload, True


def _write_packed(path, model):
    payload = {
        'contract': PACKING_CONTRACT,
        'model_contract': CONTRACT,
        'format_version': np.int64(FORMAT_VERSION),
    }
    layers = [('model__', model.state)]
    if model.base is not None:
        layers.append(('base__', model.base.state))
    packed_any = False
    for prefix, state in layers:
        arrays, packed = _layer_payload(state)
        packed_any = packed_any or packed
        for key, value in arrays.items():
            name = prefix + key
            if name in payload:
                raise ValueError('Duplicate checkpoint field: ' + name)
            payload[name] = value
    if not packed_any:
        raise ValueError('Packed writer requires a completion matrix')
    _reject_payload_size(payload)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **payload)
    if path.stat().st_size > MAX_FILE_BYTES:
        path.unlink()
        raise ValueError('Packed checkpoint file exceeds the size cap')


def _reject_payload_size(payload):
    total = 0
    for value in payload.values():
        total += int(np.asarray(value).nbytes)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError('Checkpoint arrays exceed the uncompressed cap')


def _require_sha(path, expected):
    if not isinstance(expected, str) or _HEX.fullmatch(expected) is None:
        raise ValueError('Checkpoint hash must be a lowercase sha256 hex digest')
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0 or size > MAX_FILE_BYTES:
        raise ValueError(f'Checkpoint file size is outside the cap: {size}')
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError('Checkpoint identity mismatch')


def _archive_members(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ZIP_MEMBERS:
            raise ValueError('Checkpoint member count is not canonical')
        names = []
        total = 0
        members = {}
        for info in infos:
            if info.flag_bits & 0x1:
                raise ValueError('Encrypted checkpoint member')
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError('Unsupported checkpoint compression')
            if not _MEMBER.fullmatch(info.filename):
                raise ValueError('Noncanonical checkpoint member name')
            if info.file_size <= 0 or info.file_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError('Uncompressed checkpoint member exceeds the cap')
            total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError('Uncompressed checkpoint exceeds the cap')
            with archive.open(info) as raw:
                payload = raw.read(info.file_size + 1)
            if len(payload) != info.file_size:
                raise ValueError('Zip member length is not canonical')
            names.append(info.filename)
            members[info.filename[:-4]] = payload
        if len(names) != len(set(names)):
            raise ValueError('Duplicate checkpoint member')
        return members


def _reject_archive(path):
    _archive_members(path)


def _descr_itemsize(descr):
    if not isinstance(descr, str):
        raise ValueError('Structured NPY dtype is rejected')
    if descr in _FIXED_DTYPES:
        return _FIXED_DTYPES[descr]
    unicode = _UNICODE.fullmatch(descr)
    if unicode:
        return int(unicode.group(1)) * 4
    raw = _BYTES.fullmatch(descr)
    if raw:
        return int(raw.group(1))
    raise ValueError('Unsupported NPY dtype')


def _declared_nbytes(shape, itemsize):
    if not isinstance(shape, tuple):
        raise ValueError('NPY shape is not canonical')
    if itemsize <= 0 or itemsize > MAX_UNCOMPRESSED_BYTES:
        raise ValueError('NPY dtype size exceeds the cap')
    total = 1
    for dim in shape:
        if type(dim) is not int or dim < 0 or dim > MAX_UNCOMPRESSED_BYTES:
            raise ValueError('NPY shape exceeds the size cap')
        total *= dim
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError('NPY shape exceeds the size cap')
    nbytes = total * itemsize
    if nbytes > MAX_UNCOMPRESSED_BYTES:
        raise ValueError('NPY shape exceeds the size cap')
    return nbytes


def _npy_meta(payload):
    """Read an NPY header without constructing the array it describes."""
    if len(payload) < 10 or payload[:6] != b'\x93NUMPY':
        raise ValueError('NPY header is not canonical')
    major = payload[6]
    if major == 1:
        header_len = int.from_bytes(payload[8:10], 'little')
        header_start = 10
    elif major == 2:
        if len(payload) < 12:
            raise ValueError('NPY header is not canonical')
        header_len = int.from_bytes(payload[8:12], 'little')
        header_start = 12
    else:
        raise ValueError('Unsupported NPY header')
    header_end = header_start + header_len
    if header_len <= 0 or header_end > len(payload):
        raise ValueError('NPY header is not canonical')
    try:
        meta = ast.literal_eval(payload[header_start:header_end].decode('latin1').strip())
    except (SyntaxError, ValueError, MemoryError, UnicodeError) as error:
        raise ValueError('NPY header is not canonical') from error
    if not isinstance(meta, dict) or not {'descr', 'fortran_order', 'shape'} <= set(meta):
        raise ValueError('NPY header is not canonical')
    if type(meta['fortran_order']) is not bool:
        raise ValueError('NPY header is not canonical')
    itemsize = _descr_itemsize(meta['descr'])
    declared = _declared_nbytes(meta['shape'], itemsize)
    data = payload[header_end:]
    if declared != len(data):
        raise ValueError('NPY payload length does not match the declared shape')
    return meta['shape'], meta['fortran_order'], data


def _decode_npy(payload):
    shape, fortran, _data = _npy_meta(payload)
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if array.shape != shape or array.dtype.hasobject or array.dtype.names is not None:
        raise ValueError('NPY array does not match its header')
    if array.dtype.kind in 'cOVmM':
        raise ValueError('Unsupported NPY dtype')
    if array.ndim >= 2 and fortran and not array.flags.f_contiguous:
        raise ValueError('NPY fortran flag does not match the array')
    if array.ndim >= 2 and not fortran and not array.flags.c_contiguous:
        raise ValueError('NPY C layout flag does not match the array')
    return _layout_copy(array)


def _arrays_from_members(members):
    if 'contract' not in members:
        raise ValueError('Checkpoint contract is missing')
    return {name: _decode_npy(payload) for name, payload in members.items()}


def _expand_upper(upper, n):
    expected = n * (n + 1) // 2
    upper = np.asarray(upper)
    if upper.dtype != np.dtype(np.float64) or not upper.dtype.isnative:
        raise ValueError('Upper triangle must be native float64')
    if upper.ndim != 1 or upper.shape != (expected,):
        raise ValueError('Upper triangle length is not canonical')
    if not np.isfinite(upper).all():
        raise ValueError('Nonfinite completion upper triangle')
    upper = np.array(upper, dtype=np.float64, copy=True, order='C')
    matrix = np.empty((n, n), dtype=np.float64)
    rows, cols = np.triu_indices(n)
    matrix[rows, cols] = upper
    off = rows != cols
    matrix[cols[off], rows[off]] = upper[off]
    if not np.array_equal(matrix[rows, cols].view(np.uint64), upper.view(np.uint64)):
        raise ValueError('Expanded upper triangle bits changed')
    if not np.array_equal(np.ascontiguousarray(matrix).view(np.uint64),
                          np.ascontiguousarray(matrix.T).view(np.uint64)):
        raise ValueError('Reconstructed matrix lost bitwise symmetry')
    return matrix


def _rehydrate_layer(fields):
    kind = str(fields.get('kind'))
    if kind != 'completion':
        if 'matrix_upper' in fields or 'matrix_n' in fields:
            raise ValueError('Noncanonical packing of a non-completion layer')
        return fields
    allowed_extra = {'matrix_upper', 'matrix_n'}
    if 'matrix' in fields or not allowed_extra <= set(fields):
        raise ValueError('Noncanonical completion packing')
    n_value = fields['matrix_n']
    if n_value.shape != () or n_value.dtype.kind not in 'iu':
        raise ValueError('Completion gene count is not a scalar integer')
    n = int(n_value)
    if n < 1 or n > MAX_PACKED_GENES:
        raise ValueError(f'Packed completion genes {n} outside 1..{MAX_PACKED_GENES}')
    singles = fields.get('singles')
    support = fields.get('support')
    if singles is None or singles.ndim != 1 or len(singles) != n or singles.dtype.kind not in 'iuf':
        raise ValueError('Completion singles do not match the matrix')
    if singles.dtype.kind == 'f' and not np.isfinite(singles).all():
        raise ValueError('Nonfinite singles')
    if support is None or support.dtype != np.dtype(bool) or support.shape != singles.shape:
        raise ValueError('Completion support is not a boolean gene mask')
    matrix = _expand_upper(fields['matrix_upper'], n)
    state = {key: value for key, value in fields.items() if key not in allowed_extra}
    state['matrix'] = matrix
    return state


def _group_layers(arrays, reserved):
    grouped = {'model__': {}, 'base__': {}}
    for name, value in arrays.items():
        if name in reserved:
            continue
        matched = False
        for prefix in grouped:
            if name.startswith(prefix):
                grouped[prefix][name[len(prefix):]] = value
                matched = True
                break
        if not matched:
            raise ValueError('Unknown checkpoint field: ' + name)
    if 'kind' not in grouped['model__']:
        raise ValueError('Checkpoint model layer is missing')
    if not grouped['base__']:
        grouped['base__'] = None
    return grouped['model__'], grouped['base__']


def _split_layers(arrays):
    if str(arrays.get('contract')) != PACKING_CONTRACT:
        raise ValueError('Unknown packing contract')
    if str(arrays.get('model_contract')) != CONTRACT:
        raise ValueError('Packed model contract is not the original predictor contract')
    version = arrays.get('format_version')
    if version is None or version.shape != () or version.dtype.kind not in 'iu' or int(version) != FORMAT_VERSION:
        raise ValueError('Unsupported packing format')
    model_fields, base_fields = _group_layers(
        arrays, {'contract', 'model_contract', 'format_version'})
    return _rehydrate_layer(model_fields), (
        None if base_fields is None else _rehydrate_layer(base_fields))


def _state_bits_equal(loaded, expected):
    if set(loaded) != set(expected):
        raise ValueError('Loaded checkpoint fields differ from the reconstruction')
    for key, expected_value in expected.items():
        got = loaded[key]
        if got.shape != expected_value.shape or got.dtype != expected_value.dtype:
            raise ValueError('Loaded checkpoint schema changed: ' + key)
        if np.ascontiguousarray(got).tobytes() != np.ascontiguousarray(expected_value).tobytes():
            raise ValueError('Loaded checkpoint bits changed: ' + key)


def _reject_historical_overlap(directory, forbidden):
    resolved = Path(directory).resolve()
    for root in forbidden:
        if root is None:
            continue
        other = Path(root).resolve()
        if resolved == other or resolved.is_relative_to(other) or other.is_relative_to(resolved):
            raise ValueError('Transient working directory overlaps a historical path')


def _directory_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob('*') if item.is_file())


def _load_through_original(model_state, base_state, forbidden):
    payload = {'contract': CONTRACT}
    for prefix, state in (('model__', model_state), ('base__', base_state)):
        if state is None:
            continue
        for key, value in state.items():
            payload[prefix + key] = value
    total = sum(int(np.asarray(value).nbytes) for value in payload.values())
    if total > TRANSIENT_BYTES:
        raise ValueError('Transient reconstruction exceeds 32MiB')
    with tempfile.TemporaryDirectory(prefix='opal-frontier-pack-') as temporary:
        directory = Path(temporary).resolve()
        _reject_historical_overlap(directory, forbidden)
        target = directory / 'original-contract.npz'
        with target.open('xb') as stream:
            np.savez_compressed(stream, **payload)
        if target.stat().st_size > TRANSIENT_BYTES or _directory_bytes(directory) > TRANSIENT_BYTES:
            raise ValueError('Transient working file exceeds 32MiB')
        loaded = Predictor.load(target)
        _state_bits_equal(loaded.state, model_state)
        if base_state is None:
            if loaded.base is not None:
                raise ValueError('Original loader invented a base layer')
        else:
            if loaded.base is None:
                raise ValueError('Original loader dropped the base layer')
            _state_bits_equal(loaded.base.state, base_state)
        _install_layout(loaded, model_state, base_state)
        return loaded


def _install_layout(model, model_state, base_state):
    """Keep the validated values and the layout captured before Predictor.load."""
    model.state = {key: _layout_copy(value) for key, value in model_state.items()}
    if base_state is not None:
        model.base.state = {key: _layout_copy(value) for key, value in base_state.items()}


def _load_original(path, expected_sha, members):
    arrays = _arrays_from_members(members)
    if str(arrays.get('contract')) != CONTRACT:
        raise ValueError('Unknown checkpoint contract')
    model_state, base_state = _group_layers(arrays, {'contract'})
    model = Predictor.load(path, expected_sha)
    _state_bits_equal(model.state, model_state)
    if base_state is None:
        if model.base is not None:
            raise ValueError('Original loader invented a base layer')
    else:
        if model.base is None:
            raise ValueError('Original loader dropped the base layer')
        _state_bits_equal(model.base.state, base_state)
    _install_layout(model, model_state, base_state)
    return model


def _load_packed(path, historical_roots, members):
    arrays = _arrays_from_members(members)
    model_state, base_state = _split_layers(arrays)
    forbidden = [Path(path).resolve().parent, *historical_roots]
    return _load_through_original(model_state, base_state, forbidden)
