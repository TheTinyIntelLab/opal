"""Small internal added-information head, with no change to public model names."""
import json
from pathlib import Path

import numpy as np

from .artifacts import array_hash, read_json, sha256, write_json
from .series import Unavailable
from .series_bilinear import ridge


def reference_projection(control_cells, rank):
    x = np.asarray(control_cells, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Invalid control cells')
    center = x.mean(0)
    _, s, vt = np.linalg.svd(x - center, full_matrices=False)
    available = np.sum(s > max(x.shape) * np.finfo(float).eps * max(s[0], 1))
    if available < rank:
        raise Unavailable('Control projection rank unsupported')
    return center, vt[:rank].T


def projected_variance(single_cells, basis):
    x = np.asarray(single_cells, dtype=np.float64)
    if len(x) < 2 or not np.isfinite(x).all():
        raise Unavailable('Single variance requires finite repeated cells')
    return np.var(x @ basis, axis=0, ddof=1)


def stratified_shuffle(values, counts, seed, bins=(10, 25, 50)):
    """One source/context/stratum per call; singleton strata remain missing."""
    values = np.asarray(values)
    out = np.full_like(values, np.nan)
    groups = np.digitize(counts, bins)
    rng = np.random.default_rng(seed)
    records = []
    for group in sorted(set(groups)):
        ids = np.flatnonzero(groups == group)
        records.append({'bin': int(group), 'size': len(ids), 'eligible': len(ids) >= 2})
        if len(ids) >= 2:
            # Fixed random cyclic permutation avoids unchanged target assignments.
            order = rng.permutation(ids)
            out[order] = values[np.roll(order, 1)]
    return out, records


def split_masks(pair_ids, kind, fold, nfolds=3):
    pairs = [json.loads(p) for p in pair_ids]
    if kind == 'global_pair':
        assignment = {p: i % nfolds for i, p in enumerate(sorted(set(pair_ids)))}
        test = np.asarray([assignment[p] == fold for p in pair_ids])
        return ~test, test
    if kind != 'partner':
        raise ValueError('Unknown split')
    targets = sorted({t for p in pairs for t in p})
    groups = {t: i % nfolds for i, t in enumerate(targets)}
    train = np.asarray([all(groups[t] != fold for t in p) for p in pairs])
    test = np.asarray([min(groups[t] for t in p) == fold for p in pairs])
    return train, test


def features(means, variances, rank, arm):
    z = means[..., :rank]
    mean = np.column_stack((z.sum(1), np.abs(z[:, 0] - z[:, 1])))
    if arm == 'mean':
        return mean
    if arm == 'capacity_mean':
        return np.column_stack((mean, mean ** 2))
    if arm not in ('variance', 'shuffle'):
        raise ValueError('Unknown information arm')
    v = variances[..., :rank]
    return np.column_stack((mean, v.sum(1), np.abs(v[:, 0] - v[:, 1])))


def fit_head(means, variances, residual, w, config, arm):
    x = features(means, variances, config['input_rank'], arm)
    valid = np.isfinite(x).all(1)
    if not valid.all():
        raise Unavailable('Missing summaries in training; zero-correction fallback')
    if len(x) < 12:
        raise Unavailable('Head requires twelve training pairs')
    center = w @ x
    scale = np.maximum(np.sqrt(w @ (x - center) ** 2), 1e-6)
    _, s, vt = np.linalg.svd(np.sqrt(w)[:, None] * (residual - w @ residual), full_matrices=False)
    rank = config['output_rank']
    if np.sum(s > max(residual.shape) * np.finfo(float).eps * max(s[0], 1)) < rank:
        raise Unavailable('Output rank unsupported')
    basis = vt[:rank].T
    fitted = ridge((x - center) / scale, residual @ basis, w, config['alpha'])
    return dict(fitted, basis=basis, center=center, scale=scale, config=config, arm=arm)


def predict_head(head, means, variances, output_genes):
    correction = np.zeros((len(means), output_genes))
    if head is None:
        return correction
    x = features(means, variances, head['config']['input_rank'], head['arm'])
    valid = np.isfinite(x).all(1)
    correction[valid] = (((x[valid] - head['center']) / head['scale']) @ head['coefficients'] + head['intercept']) @ head['basis'].T
    return correction


def save_head(head, directory, contract, role='development_evaluation'):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    arrays = {} if head is None else {k: v for k, v in head.items() if isinstance(v, np.ndarray)}
    metadata = {'schema': 'opal-information-head-1', 'role': role, 'contract': contract,
                'head': None if head is None else {k: v for k, v in head.items() if not isinstance(v, np.ndarray)}}
    np.savez_compressed(path / 'parameters.npz', **arrays)
    metadata['parameters_sha256'] = sha256(path / 'parameters.npz')
    metadata['numeric_values'] = sum(a.size for a in arrays.values())
    write_json(path / 'model.json', metadata)
    write_json(path / 'integrity.json', {'model_sha256': sha256(path / 'model.json')})


def load_head(directory, contract, role='development_evaluation'):
    path = Path(directory)
    metadata = read_json(path / 'model.json')
    if sha256(path / 'model.json') != read_json(path / 'integrity.json')['model_sha256']:
        raise ValueError('Head manifest hash mismatch')
    if metadata['schema'] != 'opal-information-head-1' or metadata['role'] != role or metadata['contract'] != contract:
        raise ValueError('Head source/scale/gene/reference/role mismatch')
    if sha256(path / 'parameters.npz') != metadata['parameters_sha256']:
        raise ValueError('Head parameters hash mismatch')
    with np.load(path / 'parameters.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if any(a.dtype.kind not in 'fiu' or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Unsafe or nonfinite head arrays')
    return None if metadata['head'] is None else dict(metadata['head'], **arrays)


def require_branch(name, evidence):
    gates = {
        'distribution': ['independent_distribution_repeatability'],
        'context': ['qualified_crossed_contexts'],
        'uncertainty': ['stable_procedure', 'independent_calibration'],
        'acquisition': ['useful_prediction', 'assay_partner'],
        'scaling': ['information_controls', 'capacity_benefit', 'fresh_confirmation', 'approved_resources'],
        'confirmation': ['fresh_source', 'matched_measurements', 'custody', 'algorithm_freeze'],
    }
    missing = [g for g in gates[name] if evidence.get(g) is not True]
    if missing:
        raise Unavailable(name + ' requires ' + ', '.join(missing))
