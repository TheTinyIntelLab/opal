"""Opt-in scalar evaluation contracts; historical evaluators remain unchanged."""
import hashlib
import io

import numpy as np

from .block_completion import ScalarCompletion
from .scalar_screen import metrics

METRICS_CONTRACT = 'opal-scalar-metrics-v2'
CHECKPOINT_CONTRACT = 'opal-scalar-checkpoint-strict-v2'


def metrics_v2(target, prediction):
    """Float64 arithmetic, with refusal when any reported metric is unrepresentable.

    Integer values are rounded to float64 before subtraction, rather than using
    integer modular arithmetic. This does not promise exact integer arithmetic
    beyond binary64's representable range.
    """
    arrays = [np.asarray(value) for value in (target, prediction)]
    if any(value.dtype.kind not in 'iuf' for value in arrays):
        raise ValueError('Real numeric vectors required')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            result = metrics(*(value.astype(np.float64) for value in arrays))
    except FloatingPointError as error:
        raise ValueError('Metrics are not representable in float64') from error
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise ValueError('Metrics are not finite')
    return result


def strict_load_completion_v2(path, expected_sha256=None):
    """Validate one immutable byte snapshot before invoking the historical loader."""
    with open(path, 'rb') as stream:
        raw = stream.read()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Checkpoint hash mismatch')
    required = {'rank', 'penalty', 'steps', 'seed', 'format_version', 'normalized_upper',
                'support', 'scale', 'best_step', 'parameter_count'}
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        if len(z.files) != len(required) or set(z.files) != required:
            raise ValueError('Unexpected checkpoint fields')
        for name in ('rank', 'steps', 'seed', 'format_version', 'best_step', 'parameter_count'):
            if z[name].shape != () or z[name].dtype.kind not in 'iu':
                raise ValueError('Integer scalar metadata required')
        for name in ('penalty', 'scale'):
            value = z[name]
            if value.shape != () or value.dtype.kind not in 'iuf' or not np.isfinite(value) or value <= 0:
                raise ValueError('Positive finite numeric scalar required')
        support = z['support']
        if support.ndim != 1 or support.dtype != np.bool_ or len(support) < 2:
            raise ValueError('Boolean support vector required')
        rank, steps = int(z['rank']), int(z['steps'])
        if rank < 2 or rank % 2 or steps < 1 or int(z['seed']) < 0 or int(z['format_version']) != 2:
            raise ValueError('Invalid checkpoint configuration')
        if not 1 <= int(z['best_step']) <= steps or int(z['parameter_count']) != 2*len(support)*rank:
            raise ValueError('Invalid checkpoint selection metadata')
        upper = z['normalized_upper']
        if upper.shape != (len(support)*(len(support)+1)//2,) or upper.dtype != np.float32 or not np.isfinite(upper).all():
            raise ValueError('Finite float32 symmetric operator required')
    with np.errstate(over='raise', invalid='raise'):
        try:
            model = ScalarCompletion.load(io.BytesIO(raw))
        except FloatingPointError as error:
            raise ValueError('Reconstructed operator is not representable') from error
    if not np.isfinite(model.matrix).all():
        raise ValueError('Finite reconstructed operator required')
    return model
