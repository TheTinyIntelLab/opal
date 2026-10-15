"""Integer-coordinate replay and fixed-panel measurement reconstruction."""
import gzip
import itertools
import numpy as np


def coordinates(path, shape, check, chunk_lines=250000):
    """Read the frozen headerless integer format, preserving coordinate order."""
    if chunk_lines <= 0 or len(shape) != 2 or min(shape) <= 0:
        raise ValueError('Positive chunk/annotation dimensions required')
    with gzip.open(path, 'rb') as stream:
        while lines := list(itertools.islice(stream, chunk_lines)):
            block = np.fromstring(b''.join(lines), sep=' ', dtype=np.int64)
            if block.size != len(lines)*3:
                raise ValueError('Exact integer triplets required')
            block = block.reshape(-1, 3)
            row, col, value = block[:, 0]-1, block[:, 1]-1, block[:, 2]
            if np.any(row < 0) or np.any(row >= shape[0]) or np.any(col < 0) or np.any(col >= shape[1]) or np.any(value < 0):
                raise ValueError('Coordinate bounds or nonnegative count violation')
            check()
            yield row, col, value


def reconstruct_metrics(sums, support, keys, contexts, drugs, minimum=10):
    sums, support = np.asarray(sums), np.asarray(support)
    if sums.ndim != 3 or sums.shape[:2] != support.shape or len(keys) != len(sums) or sums.shape[1] != 2:
        raise ValueError('Condition/partition axes mismatch')
    if not np.isfinite(sums).all() or np.any(sums < 0) or np.any(support < 0):
        raise ValueError('Finite nonnegative measurements required')
    keys = [tuple(k) for k in keys]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate condition identity')
    index = {k: i for i, k in enumerate(keys)}
    logged = np.log2(1+sums/np.maximum(support[:, :, None], 1))
    results = []
    for context in contexts:
        for drug in drugs:
            rows, reasons = [], []
            for i, k in enumerate(keys):
                if k[0] != context or k[3] != drug or k[4:] != (1., 1.):
                    continue
                refs = [(context, k[1], k[2], 'vehicle', 0., 0.),
                        (context, k[1], k[2], drug, 1., 0.),
                        (context, k[1], k[2], 'vehicle', 0., 1.)]
                positions = [index.get(r) for r in refs]
                if min(support[i]) < minimum or any(j is None or min(support[j]) < minimum for j in positions):
                    reasons.append(dict(group=list(k), reason='matched partition support below fixed minimum'))
                    continue
                control, single, anchor = positions
                rows.append((logged[i]-logged[control], logged[i]-logged[single]-logged[anchor]+logged[control]))
            if rows and not reasons:
                response = np.mean([r[0] for r in rows], axis=0)
                residual = np.mean([r[1] for r in rows], axis=0)
                results.append(dict(context=context, drug=drug, status='technical_support_qualified', strata=len(rows),
                    response_half_mse=float(np.mean((response[0]-response[1])**2)),
                    residual_half_mse=float(np.mean((residual[0]-residual[1])**2)),
                    zero_response_mse=float(np.mean(response[1]**2)), zero_residual_mse=float(np.mean(residual[1]**2))))
            else:
                results.append(dict(context=context, drug=drug, status='unsupported', reasons=reasons or ['no measured pair stratum']))
    return dict(rows=results, declared_population=len(contexts)*len(drugs),
        supported_population=sum(r['status'] == 'technical_support_qualified' for r in results),
        uncertainty='technical cell partition agreement only; no culture covariance or biological interval')
