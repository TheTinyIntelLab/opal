"""Lossless byte-shuffled successor to completion packing v1.

The upper triangle's eight byte positions are stored in separate planes before
DEFLATE. This changes compression only. Reconstruction delegates all original
shape, finite-state and prediction-contract validation to packing v1.
"""
from pathlib import Path

import numpy as np

from . import frontier_packing as legacy

PACKING_CONTRACT = 'opal-frontier-packing-v2'
FORMAT_VERSION = 2


def save_checkpoint(model, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not legacy._has_completion(model):
        return legacy.save_checkpoint(model, path)
    payload = dict(contract=PACKING_CONTRACT, model_contract=legacy.CONTRACT,
                   format_version=np.int64(FORMAT_VERSION))
    for prefix, state in (('model__', model.state),
                          ('base__', None if model.base is None else model.base.state)):
        if state is None:
            continue
        arrays, packed = legacy._layer_payload(state)
        if packed:
            upper = arrays.pop('matrix_upper')
            arrays['matrix_upper_bytes'] = np.ascontiguousarray(upper.view(np.uint8).reshape(-1, 8).T)
            arrays['matrix_byte_order'] = np.array(np.dtype(np.float64).str)
        payload.update({prefix+key: value for key, value in arrays.items()})
    legacy._reject_payload_size(payload)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **payload)
    if path.stat().st_size > legacy.MAX_FILE_BYTES:
        raise ValueError('Packed file exceeds the size cap; preserve failed output')
    return path


def load_checkpoint(path, expected_sha, historical_roots=()):
    path = Path(path)
    legacy._require_sha(path, expected_sha)
    arrays = legacy._arrays_from_members(legacy._archive_members(path))
    contract = str(arrays.get('contract'))
    if contract in (legacy.CONTRACT, legacy.PACKING_CONTRACT):
        return legacy.load_checkpoint(path, expected_sha, historical_roots)
    if contract != PACKING_CONTRACT:
        raise ValueError('Unknown checkpoint contract')
    version = arrays.get('format_version')
    if version is None or version.shape != () or version.dtype.kind not in 'iu' or int(version) != FORMAT_VERSION:
        raise ValueError('Unknown byte-shuffle format')
    for prefix in ('model__', 'base__'):
        if str(arrays.get(prefix+'kind')) != 'completion':
            continue
        number = arrays.get(prefix+'matrix_n')
        if number is None or number.shape != () or number.dtype.kind not in 'iu':
            raise ValueError('Completion gene count must be a scalar integer')
        n = int(number)
        if not 1 <= n <= legacy.MAX_PACKED_GENES:
            raise ValueError('Completion gene count exceeds the bound')
        encoded = arrays.get(prefix+'matrix_upper_bytes')
        order = arrays.get(prefix+'matrix_byte_order')
        if (encoded is None or encoded.dtype != np.uint8 or
                encoded.shape != (8, n*(n+1)//2) or not encoded.flags.c_contiguous):
            raise ValueError('Noncanonical upper-triangle byte planes')
        if order is None or order.shape != () or str(order) != np.dtype(np.float64).str:
            raise ValueError('Unsupported matrix byte order')
        arrays[prefix+'matrix_upper'] = np.ascontiguousarray(encoded.T).reshape(-1).view(np.float64)
        del arrays[prefix+'matrix_upper_bytes'], arrays[prefix+'matrix_byte_order']
    arrays['contract'] = np.array(legacy.PACKING_CONTRACT)
    arrays['format_version'] = np.int64(legacy.FORMAT_VERSION)
    state, base = legacy._split_layers(arrays)
    return legacy._load_through_original(state, base, (path.resolve().parent, *historical_roots))


class MixedLoader:
    def __init__(self, historical_roots=()):
        self.historical_roots = tuple(historical_roots)

    def load(self, path, expected_sha256=None):
        return load_checkpoint(path, expected_sha256, self.historical_roots)
