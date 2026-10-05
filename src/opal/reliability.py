"""Finite-sample error-radius diagnostics; guarantees require exchangeability."""
import math
import numpy as np


def error_radius(calibration_errors,coverage=.9):
    errors=np.asarray(calibration_errors,np.float64)
    if errors.ndim!=1 or not len(errors) or not np.isfinite(errors).all() or np.any(errors<0):raise ValueError('Invalid calibration errors')
    if not np.isfinite(coverage) or not 0<coverage<1:raise ValueError('Invalid requested coverage')
    rank=math.ceil((len(errors)+1)*coverage)
    radius=None if rank>len(errors) else float(np.sort(errors)[rank-1])
    return {'radius':radius,'coverage_requested':coverage,'calibration_pairs':len(errors),'quantile_rank':rank,
        'finite_radius':radius is not None,'score':'per-condition gene-averaged RMSE',
        'interpretation':'split conformal rank under exchangeability; dependent biological pairs do not establish guaranteed coverage'}
