"""Prospective fixed-order prediction, explicitly incompatible with legacy replay."""
import hashlib
import io
from pathlib import Path

import numpy as np

from .block_completion import checked_pairs
from .models.fitness import design

CONTRACT = 'opal-flexible-fixed-order-f64-v1'
FIELDS = {'singles', 'support', 'input_center', 'input_scale', 'center_rows', 'centers',
          'whitener', 'bandwidth', 'center', 'scale', 'offset', 'coefficient',
          'nodes', 'alpha', 'n_centers', 'seed', 'format_version'}


class FixedOrderFlexiblePredictor:
    """No-fit snapshot with ascending float64 products and reductions.

    Partition/order/layout invariance is prospective and stack-scoped. This
    predictor does not reproduce the registered historical validation vector at
    its 1e-12 gate and must not replace a historical scientific evaluator.
    """

    @classmethod
    def from_legacy_checkpoint(cls, path, expected_sha256):
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError('Expected legacy checkpoint SHA256 required')
        return cls._read(path, expected_sha256, legacy=True)

    @classmethod
    def load(cls, path, expected_sha256=None):
        return cls._read(path, expected_sha256, legacy=False)

    @classmethod
    def _read(cls, path, expected, legacy):
        raw = Path(path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if expected is not None and sha != expected:
            raise ValueError('Checkpoint hash mismatch')
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            fields = FIELDS if legacy else FIELDS | {'prediction_contract', 'source_checkpoint_sha256'}
            if len(z.files) != len(fields) or set(z.files) != fields:
                raise ValueError('Unexpected flexible checkpoint fields')
            values = {name: z[name].copy() for name in fields}
        for name in ('format_version', 'n_centers', 'seed'):
            if values[name].shape != () or values[name].dtype.kind not in 'iu':
                raise ValueError('Integer scalar metadata required')
        if int(values['format_version']) != (1 if legacy else 2):
            raise ValueError('Unsupported prediction checkpoint format')
        if not legacy:
            if values['prediction_contract'].shape != () or str(values['prediction_contract']) != CONTRACT:
                raise ValueError('Unknown prediction contract')
            source = values['source_checkpoint_sha256']
            if source.shape != () or source.dtype.kind != 'U' or len(str(source)) != 64 or any(c not in '0123456789abcdef' for c in str(source)):
                raise ValueError('Source checkpoint digest required')
        if values['nodes'].shape != () or values['nodes'].dtype != np.bool_:
            raise ValueError('Boolean nodes scalar required')
        for name in ('alpha', 'bandwidth', 'offset'):
            value = values[name]
            if value.shape != () or value.dtype.kind not in 'iuf' or not np.isfinite(value):
                raise ValueError('Finite numeric scalar required')
        if float(values['alpha']) <= 0 or float(values['bandwidth']) <= 0:
            raise ValueError('Positive alpha and bandwidth required')
        singles, support = values['singles'], values['support']
        if singles.ndim != 1 or len(singles) < 2 or support.shape != singles.shape or support.dtype != np.bool_:
            raise ValueError('Aligned singles and Boolean support required')
        k, n = int(values['n_centers']), len(singles)
        if k < 2 or int(values['seed']) < 0:
            raise ValueError('Invalid map configuration')
        width = 3+(n if bool(values['nodes']) else 0)+k
        shapes = dict(singles=(n,), input_center=(2,), input_scale=(2,), centers=(k, 2),
                      whitener=(k, k), center=(width,), scale=(width,), coefficient=(width,))
        for name, shape in shapes.items():
            value = values[name]
            if value.shape != shape or value.dtype != np.float64:
                raise ValueError('Native float64 map arrays with exact shapes required')
            finite = value[support] if name == 'singles' else value
            if not np.isfinite(finite).all():
                raise ValueError('Finite supported state required')
        if np.any(values['input_scale'] <= 0) or np.any(values['scale'] <= 0):
            raise ValueError('Positive map scales required')
        rows = values['center_rows']
        if rows.shape != (k,) or rows.dtype.kind not in 'iu' or np.any(rows < 0) or len(np.unique(rows)) != k:
            raise ValueError('Distinct nonnegative center row identities required')
        values['format_version'] = np.array(2)
        values['prediction_contract'] = np.array(CONTRACT)
        values['source_checkpoint_sha256'] = np.array(sha if legacy else str(values['source_checkpoint_sha256']))
        for value in values.values():
            value.flags.writeable = False
        obj = cls()
        obj._state = values
        return obj

    def save(self, path):
        """Write format2 exclusively; existing checkpoints are never overwritten."""
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, **self._state)

    def predict(self, pairs):
        s = self._state
        p = checked_pairs(pairs, len(s['singles']))
        if not s['support'][p].all():
            raise ValueError('Query genes need training-pair support')
        output = np.empty(len(p), dtype=np.float64)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                for start in range(0, len(p), 2048):
                    rows = p[start:start+2048]
                    ordered = np.sort(s['singles'][rows], axis=1)
                    coordinates = (ordered-s['input_center'])/s['input_scale']
                    distance = np.square(coordinates[:, None, :]-s['centers'][None, :, :]).sum(2)
                    kernel = np.exp(-distance/(2*float(s['bandwidth'])**2))
                    white = np.zeros((len(rows), int(s['n_centers'])), dtype=np.float64)
                    for j in range(kernel.shape[1]):
                        white = np.add(white, np.multiply(kernel[:, j:j+1], s['whitener'][j], dtype=np.float64), dtype=np.float64)
                    features = np.concatenate((design(s['singles'], rows, bool(s['nodes'])), white), axis=1)
                    products = np.multiply((features-s['center'])/s['scale'], s['coefficient'], dtype=np.float64)
                    dot = np.add.accumulate(products, axis=1, dtype=np.float64)[:, -1]
                    output[start:start+len(rows)] = (s['singles'][rows].sum(1)+float(s['offset']))+dot
        except (FloatingPointError, OverflowError) as error:
            raise ValueError('Prediction arithmetic is not representable in float64') from error
        if not np.isfinite(output).all():
            raise ValueError('Finite prediction required')
        return output
