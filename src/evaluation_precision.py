"""Opt-in range safeguards; the historical and v2 contracts stay unchanged."""
import numpy as np

from .evaluation_contracts import metrics_v2, strict_load_completion_v2

METRICS_CONTRACT = 'opal-scalar-metrics-v3-underflow-refusal'
CHECKPOINT_CONTRACT = 'opal-scalar-checkpoint-strict-v3-underflow-refusal'


def metrics_v3(target, prediction):
    """Apply v2 checks and refuse IEEE underflow instead of reporting lost loss.

    This conservative contract does not rescale arithmetic or promise exact
    integers beyond float64 precision. It can refuse inputs whose aggregate
    metrics a differently scaled algorithm could represent.
    """
    with np.errstate(under='raise'):
        return metrics_v2(target, prediction)


def strict_load_completion_v3(path, expected_sha256=None):
    """Apply v2 snapshot checks and refuse underflow in operator reconstruction.

    This is byte-identical replay for accepted states, not a checkpoint format
    change or permission to replace an existing scientific evaluation contract.
    """
    with np.errstate(under='raise'):
        return strict_load_completion_v2(path, expected_sha256)
