"""Frozen descriptive reuse of exposed external growth cohorts; no model fitting."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import environment, load_npz, sha256, write_json

COHORTS = ('environment1', 'environment2', 'parp2022', 'parp2023')
NOVEL = 'absent_original_eligible_bank'
REGISTERED = ('tools/reset_diagnostics.py', 'tests/test_reset_diagnostics.py',
              'docs/research_reset/diagnostic_protocol.md')


def verify(files):
    for path, digest in files.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f'Frozen input differs: {path}')


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def metrics(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 1 or not len(target):
        raise ValueError('Aligned nonempty vectors required')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Finite vectors required')
    error = prediction-target
    return dict(mse=float(np.mean(error**2)), mae=float(np.mean(abs(error))),
                bias=float(error.mean()), pearson=correlation(prediction, target))


def summary(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), std=float(a.std()),
                quantiles=np.quantile(a, [0, .05, .25, .5, .75, .95, 1]).tolist())


def names_valid(names):
    if names.ndim != 2 or names.shape[1] != 2 or not len(names):
        raise ValueError('Nonempty pair schema required')
    if np.any(names[:, 0] >= names[:, 1]) or len(set(map(tuple, names))) != len(names):
        raise ValueError('Unique canonical different-gene pairs required')


def panel(data, frozen):
    names_valid(data['pair_names'])
    names_valid(frozen['pair_names'])
    rows = frozen['rows']
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(data['pair_names'])):
        raise ValueError('Invalid frozen row assignment')
    if not np.array_equal(data['pair_names'][rows], frozen['pair_names']):
        raise ValueError('Frozen identities misaligned')
    if len(data['single_genes']) != len(set(data['single_genes'])):
        raise ValueError('Duplicate single identity')
    if data['singles'].shape != (len(data['single_genes']), 2):
        raise ValueError('Two infection singles required')
    if data['gamma'].shape != (len(data['pair_names']), 2) or data['counts'].shape != data['gamma'].shape:
        raise ValueError('Two infection targets/counts required')
    for key in ('gamma', 'singles', 'counts'):
        if not np.isfinite(data[key]).all():
            raise ValueError('Nonfinite measurements')
    if np.any(data['counts'] < 2) or np.any(data['counts'] != data['counts'].astype(int)):
        raise ValueError('Invalid construct counts')
    lookup = {str(g): s for g, s in zip(data['single_genes'], data['singles'], strict=True)}
    mask = frozen['strata'] == NOVEL
    available = np.array([a in lookup and b in lookup for a, b in frozen['pair_names']])
    include = mask & available
    names = frozen['pair_names'][include]
    names_valid(names)
    singles = np.array([[lookup[a], lookup[b]] for a, b in names])
    result = dict(names=names, target=data['gamma'][rows[include]],
                  counts=data['counts'][rows[include]], minimum=singles.min(axis=1),
                  additive=singles.sum(axis=1), strict_novel_total=int(mask.sum()),
                  missing_single_pairs=int(np.sum(mask & ~available)))
    for key in ('completion', 'main_effect', 'minimum', 'additive', 'zero'):
        if frozen[key].shape != (len(rows),) or not np.isfinite(frozen[key]).all():
            raise ValueError('Invalid historical predictions')
        result['original_'+key] = frozen[key][include]
    return result


def conditional_interval(names, difference, draws=2000):
    names_valid(names)
    if difference.shape != (len(names),) or not np.isfinite(difference).all():
        raise ValueError('Aligned finite differences required')
    _, inverse = np.unique(names, return_inverse=True)
    edges = inverse.reshape(-1, 2)
    g = int(edges.max())+1
    rng = np.random.default_rng(20261022)
    values = []
    for _ in range(draws):
        count = np.bincount(rng.integers(g, size=g), minlength=g)
        w = count[edges[:, 0]]*count[edges[:, 1]]
        if w.sum():
            values.append(float(w @ difference / w.sum()))
    return dict(mean=float(difference.mean()), gene_conditional_95=np.quantile(values, [.025, .975]).tolist(),
                requested_draws=draws, nonempty_draws=len(values))


def cross_infection(minimum, target):
    if minimum.shape != target.shape or minimum.ndim != 2 or minimum.shape[1] != 2:
        raise ValueError('Aligned pair by two-infection arrays required')
    return dict(infection1_singles_to_infection2=metrics(minimum[:, 0], target[:, 1]),
                infection2_singles_to_infection1=metrics(minimum[:, 1], target[:, 0]),
                raw_target_pearson=correlation(target[:, 0], target[:, 1]),
                residual_pearson=correlation(target[:, 0]-minimum[:, 0], target[:, 1]-minimum[:, 1]),
                target_disagreement_mse=float(np.mean((target[:, 0]-target[:, 1])**2)),
                single_minimum_disagreement_mse=float(np.mean((minimum[:, 0]-minimum[:, 1])**2)))


def analyze(panels):
    scores, transfers = {}, {}
    for name, p in panels.items():
        target = p['target'].mean(1)
        record = dict(strict_novel_total=p['strict_novel_total'],
                      missing_single_pairs=p['missing_single_pairs'],
                      target=summary(target), cross_infection=cross_infection(p['minimum'], p['target']),
                      coverage={})
        for threshold in (2, 4, 8):
            keep = np.all(p['counts'] >= threshold, axis=1)
            n, g = int(keep.sum()), len(np.unique(p['names'][keep]))
            s = dict(pairs=n, genes=g, small_panel=n < 50 or g < 10, methods={})
            if n:
                predictions = {k: p[k] for k in ('original_completion', 'original_main_effect', 'original_minimum')}
                predictions.update(local_minimum=p['minimum'].mean(1), local_additive=p['additive'].mean(1))
                s['methods'] = {k: metrics(v[keep], target[keep]) for k, v in predictions.items()}
                s['target'] = summary(target[keep])
                difference = (p['original_completion'][keep]-target[keep])**2-(p['original_main_effect'][keep]-target[keep])**2
                s['completion_minus_main'] = conditional_interval(p['names'][keep], difference)
            record['coverage'][str(threshold)] = s
        scores[name] = record
    for a, b in itertools.combinations(COHORTS, 2):
        pa, pb = panels[a], panels[b]
        la = {tuple(p): i for i, p in enumerate(pa['names'])}
        lb = {tuple(p): i for i, p in enumerate(pb['names'])}
        common = sorted(la.keys() & lb.keys())
        if not common:
            transfers[a+'__'+b] = dict(pairs=0, status='unavailable')
            continue
        ids = {a: np.array([la[p] for p in common]), b: np.array([lb[p] for p in common])}
        truth = {k: panels[k]['target'][ids[k]].mean(1) for k in (a, b)}
        reference = {k: panels[k]['minimum'][ids[k]].mean(1) for k in (a, b)}
        residual = {k: truth[k]-reference[k] for k in (a, b)}
        record = dict(pairs=len(common), genes=len(np.unique(common)),
                      raw_pearson=correlation(truth[a], truth[b]),
                      residual_pearson=correlation(residual[a], residual[b]), directions={})
        for source, target in ((a, b), (b, a)):
            base = metrics(reference[target], truth[target])
            transferred = metrics(reference[target]+residual[source], truth[target])
            delta = (reference[target]+residual[source]-truth[target])**2-(reference[target]-truth[target])**2
            rng = np.random.default_rng(20261022)
            shuffled = [metrics(reference[target]+rng.permutation(residual[source]), truth[target])['mse'] for _ in range(20)]
            record['directions'][source+'_to_'+target] = dict(local_minimum=base,
                fixed_residual_transfer=transferred,
                relative_mse_gain=1-transferred['mse']/base['mse'] if base['mse'] else None,
                transfer_minus_minimum=conditional_interval(np.array(common), delta),
                shuffled_edge_mse=summary(shuffled))
        transfers[a+'__'+b] = record
    return dict(label='exploratory diagnostic reuse; no fitting or external promotion',
                cohorts=scores, matched_strict_novel_transfers=transfers,
                uncertainty='conditional on these measured infections and genes; overlapping studies/panels are not independent replicates')


def freeze(args):
    if args.output.exists():
        raise FileExistsError('Preserve prior registrations')
    historical = json.loads((ROOT/'results/fitness-external-20261020/protocol.json').read_text())
    verify(historical['files'])
    files = dict(historical['files'])
    for relative in REGISTERED:
        path = ROOT/relative
        files[str(path)] = sha256(path)
    for cohort in COHORTS:
        for path in (args.prepared/f'{cohort}.npz', args.frozen/f'{cohort}_predictions.npz'):
            files[str(path.resolve())] = sha256(path)
    for path in (args.original/'prepared/profiles.npz', ROOT/'results/fitness-external-20261020/results.json'):
        files[str(path.resolve())] = sha256(path)
    args.output.mkdir(parents=True)
    write_json(args.output/'protocol.json', dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        label='registered after historical outcome exposure, before new diagnostic computation',
        files=files, prepared=str(args.prepared.resolve()), frozen=str(args.frozen.resolve()),
        environment=environment(), parent_commit='11b495e3e106ae6e6ccc3bfe07dc2f805c15d316'))
    print('frozen exploratory diagnostic protocol')


def execute(output, replay=False):
    protocol = json.loads((output/'protocol.json').read_text())
    verify(protocol['files'])
    if not replay and (output/'results.json').exists():
        raise FileExistsError('Preserve completed results')
    start = time.perf_counter()
    panels = {c: panel(load_npz(Path(protocol['prepared'])/f'{c}.npz'),
                       load_npz(Path(protocol['frozen'])/f'{c}_predictions.npz')) for c in COHORTS}
    result = analyze(panels)
    elapsed = time.perf_counter()-start
    memory = psutil.Process().memory_info()
    peak = getattr(memory, 'peak_wset', memory.rss)
    if elapsed > 600 or peak > 2*1024**3:
        raise RuntimeError('Registered resource cap exceeded')
    if replay:
        if result != json.loads((output/'results.json').read_text()):
            raise RuntimeError('Diagnostic replay differs')
        write_json(output/'replay.json', dict(status='exact scientific JSON replay', results_sha256=sha256(output/'results.json')))
    else:
        write_json(output/'results.json', result)
        write_json(output/'resources.json', dict(wall_seconds=elapsed, peak_rss_bytes=peak,
            protocol_sha256=sha256(output/'protocol.json'), results_sha256=sha256(output/'results.json')))
    if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 16*1024**2:
        raise RuntimeError('Artifact cap exceeded')
    print(json.dumps(dict(status='replayed' if replay else 'completed', wall_seconds=elapsed, peak_rss_bytes=peak)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'replay'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--frozen', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        freeze(args) if args.action == 'freeze' else execute(args.output, args.action == 'replay')
