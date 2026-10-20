"""Frozen exploratory original-test support audit; no model fitting or selection."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import load_npz, read_json, sha256, write_json
from opal.models.fitness_inference import FitnessPredictor

COHORTS = ('environment1', 'environment2', 'parp2022', 'parp2023')


def support_groups(pairs, split, genes, novel_genes):
    """Metadata-only groups with fixed tertiles of active-gene training degrees."""
    pairs, split, genes = np.asarray(pairs), np.asarray(split), np.asarray(genes)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or split.shape != (len(pairs),):
        raise ValueError('Invalid pair or assignment schema')
    if np.any(pairs < 0) or np.any(pairs >= len(genes)) or not set(np.unique(split)) <= {0, 1, 2}:
        raise ValueError('Invalid pair identities or split labels')
    if np.any(pairs[:, 0] >= pairs[:, 1]) or len(set(map(tuple, pairs))) != len(pairs):
        raise ValueError('Distinct canonical pairs required')
    degree = np.bincount(pairs[split == 0].ravel(), minlength=len(genes))
    active = degree > 0
    if not active.any() or not active[pairs].all():
        raise ValueError('Every query endpoint requires training support')
    cutpoints = np.quantile(degree[active], [1 / 3, 2 / 3], method='linear')
    tertile = np.searchsorted(cutpoints, degree, side='right')
    test = pairs[split == 2]
    endpoint = np.sort(tertile[test], axis=1)
    members = np.isin(genes, list(novel_genes))[test].sum(1)
    groups = {'all': np.ones(len(test), dtype=bool)}
    for k in range(3):
        groups[f'min_endpoint_train_degree_tertile_{k}'] = endpoint[:, 0] == k
        groups[f'novel_gene_endpoints_{k}'] = members == k
    for a in range(3):
        for b in range(a, 3):
            groups[f'endpoint_train_degree_tertiles_{a}_{b}'] = (endpoint[:, 0] == a) & (endpoint[:, 1] == b)
    return groups, {'active_genes': int(active.sum()), 'cutpoints': cutpoints.tolist(),
                    'active_gene_tertile_counts': np.bincount(tertile[active], minlength=3).tolist(),
                    'novel_genes': sorted(novel_genes), 'novel_gene_count': len(novel_genes),
                    'degree_definition': 'canonical TRAIN pair incidence, shared original split',
                    'tertile_definition': 'active-gene 1/3,2/3 linear quantiles; ties go to higher bin',
                    'group_pair_counts': {k: int(v.sum()) for k, v in groups.items()}}


def paired_gene_interval(names, difference, draws=2000, seed=20261021):
    """Conditional endpoint-multiplicity bootstrap, never a culture interval."""
    _, edges = np.unique(np.asarray(names), return_inverse=True)
    edges = edges.reshape(-1, 2)
    g = int(edges.max()) + 1
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        counts = np.bincount(rng.integers(g, size=g), minlength=g)
        weights = counts[edges[:, 0]] * counts[edges[:, 1]]
        if weights.sum():
            values.append(float(weights @ difference / weights.sum()))
    return {'conditional_gene_loss_difference_95': np.quantile(values, [.025, .975]).tolist(),
            'requested_draws': draws, 'nonempty_draws': len(values), 'seed': seed}


def subgroup_score(names, target, completion, main_effect, mask):
    n = int(mask.sum())
    ng = len(np.unique(names[mask]))
    result = {'pairs': n, 'genes': ng, 'qualifies_descriptive_interval': n >= 50 and ng >= 10,
              'completion_mse': None, 'main_effect_mse': None, 'relative_gain': None,
              'conditional_gene_loss_difference_95': None}
    if n:
        a = (completion[mask] - target[mask]) ** 2
        b = (main_effect[mask] - target[mask]) ** 2
        result.update(completion_mse=float(a.mean()), main_effect_mse=float(b.mean()),
                      relative_gain=float(1 - a.mean() / b.mean()) if b.mean() else None)
        if result['qualifies_descriptive_interval']:
            result.update(paired_gene_interval(names[mask], a - b))
    return result


def freeze(original, external, output):
    if output.exists():
        raise FileExistsError('Preserve prior support registrations')
    paths = [ROOT/'tools/reset_support.py', ROOT/'tests/test_reset_support.py',
             ROOT/'docs/research_reset/support_protocol.md', ROOT/'src/models/fitness.py',
             ROOT/'src/models/fitness_inference.py', original/'prepared/profiles.npz',
             original/'fitness-v1/assignment.npz', original/'fitness-v1/predictions.npz',
             original/'fitness-v1/selection.json', original/'fitness-reference-model/manifest.json']
    manifest = read_json(original/'fitness-reference-model/manifest.json')
    paths += [original/'fitness-reference-model'/name for name in manifest['files']]
    paths += [external/'evaluation-v1'/f'{cohort}_predictions.npz' for cohort in COHORTS]
    files = {str(p.resolve()): sha256(p) for p in paths}
    novel = set()
    for cohort in COHORTS:
        f = load_npz(external/'evaluation-v1'/f'{cohort}_predictions.npz')
        novel.update(f['pair_names'][f['strata'] == 'absent_original_eligible_bank'].ravel().tolist())
    d = load_npz(original/'prepared/profiles.npz')
    split = load_npz(original/'fitness-v1/assignment.npz')['split']
    _, metadata = support_groups(d['pairs'], split, d['genes'], novel)
    output.mkdir(parents=True)
    write_json(output/'protocol.json', {'status': 'frozen_before_new_subgroup_losses',
        'label': 'exploratory exposed original test support audit; no fitting or promotion',
        'original': str(original.resolve()), 'external': str(external.resolve()), 'files': files,
        'metadata': metadata, 'budget': {'workers': 1, 'wall_seconds': 600, 'peak_rss_bytes': 2147483648},
        'replay_tolerance': 1e-12, 'bootstrap_draws': 2000, 'bootstrap_seed': 20261021})
    print({'status': 'frozen', 'groups': metadata['group_pair_counts'], 'cutpoints': metadata['cutpoints']})


def run(output):
    p = read_json(output/'protocol.json')
    if (output/'results.json').exists():
        raise FileExistsError('Preserve completed support results')
    for name, expected in p['files'].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f'Frozen file changed: {name}')
    start = time.perf_counter()
    original = Path(p['original'])
    d = load_npz(original/'prepared/profiles.npz')
    assignment = load_npz(original/'fitness-v1/assignment.npz')['split']
    ids = np.flatnonzero(assignment == 2)
    historical = load_npz(original/'fitness-v1/predictions.npz')
    if not np.array_equal(ids, historical['test_indices']):
        raise ValueError('Historical test ordering differs')
    predictor = FitnessPredictor(original/'fitness-reference-model')
    pairs = d['pairs'][ids]
    names = d['genes'][pairs]
    groups, metadata = support_groups(d['pairs'], assignment, d['genes'], p['metadata']['novel_genes'])
    if metadata != p['metadata']:
        raise ValueError('Frozen metadata changed')
    contexts, replay = {}, {}
    for c, context in enumerate(d['contexts']):
        completion = np.concatenate([predictor.predict(names[i:i+1024], str(context))
                                     for i in range(0, len(names), 1024)])
        expected = historical['baseline'][c]
        error = float(np.max(np.abs(completion - expected)))
        replay[str(context)] = error
        if error > p['replay_tolerance']:
            raise ValueError('Retained predictor differs from historical independent baseline')
        residual = predictor.models[str(context)].predict(pairs)[c]
        main_effect = completion - residual
        contexts[str(context)] = {label: subgroup_score(names, d['responses'][c, ids], completion, main_effect, mask)
                                  for label, mask in groups.items()}
        if time.perf_counter() - start > p['budget']['wall_seconds']:
            raise RuntimeError('Support audit wall limit reached')
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', info.rss)
    if peak > p['budget']['peak_rss_bytes']:
        raise RuntimeError('Support audit memory limit reached')
    write_json(output/'results.json', {'label': p['label'], 'protocol_sha256': sha256(output/'protocol.json'),
        'metadata': metadata, 'historical_replay_max_abs': replay, 'contexts': contexts,
        'resources': {'wall_seconds': time.perf_counter() - start, 'peak_rss_bytes': peak},
        'limits': 'Overlapping descriptive subgroups; no multiplicity-adjusted claims. Random original held pairs do not simulate entirely block-held missing-clique transfer. Intervals condition on measured genes and cultures.'})
    print({'status': 'completed', 'historical_replay_max_abs': replay,
           'wall_seconds': time.perf_counter() - start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'run'])
    parser.add_argument('--original', type=Path)
    parser.add_argument('--external', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        freeze(args.original, args.external, args.output) if args.action == 'freeze' else run(args.output)
