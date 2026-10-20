"""Registered, unchanged-model reference-growth source transfer evaluation."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import sha256, write_json, load_npz
from opal.models.fitness_inference import FitnessPredictor

COHORTS = ('environment1', 'environment2', 'parp2022', 'parp2023')
STRATA = ('original_train', 'original_validation', 'original_test')


def verify_files(files):
    for path, digest in files.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f'Frozen input differs: {path}')


def strata_for(names, original, split, genes):
    lookup = {tuple(sorted((str(genes[a]), str(genes[b])))): STRATA[int(s)]
              for (a, b), s in zip(original, split, strict=True)}
    return np.array([lookup.get(tuple(sorted(p)), 'absent_original_eligible_bank') for p in names])


def metrics(prediction, target):
    error = prediction - target
    correlation = None
    if len(target) > 1 and np.std(prediction) > 0 and np.std(target) > 0:
        correlation = float(np.corrcoef(prediction, target)[0, 1])
    return {'mse': float(np.mean(error**2)), 'mae': float(np.mean(abs(error))),
            'bias': float(np.mean(error)), 'pearson': correlation}


def uncertainty(names, difference, seed=20261020, draws=2000):
    """Pair and gene-resampling intervals conditional on these measured cultures."""
    rng = np.random.default_rng(seed)
    _, edges = np.unique(names, return_inverse=True)
    edges = edges.reshape(-1, 2)
    n, g = len(names), int(edges.max()) + 1
    pair_values, gene_values = [], []
    for _ in range(draws):
        pair_values.append(float(np.mean(difference[rng.integers(n, size=n)])))
        counts = np.bincount(rng.integers(g, size=g), minlength=g)
        weights = counts[edges[:, 0]] * counts[edges[:, 1]]
        if weights.sum():
            gene_values.append(float(weights @ difference / weights.sum()))
    return {'pair_conditional_95': np.quantile(pair_values, [.025, .975]).tolist(),
            'gene_resampling_conditional_95': np.quantile(gene_values, [.025, .975]).tolist(),
            'draws': draws, 'nonempty_gene_draws': len(gene_values)}


def freeze(args):
    output = args.output
    if output.exists():
        raise FileExistsError('Preserve prior registrations')
    model_freeze = json.loads((ROOT/'results/fitness-external-20261020/model_freeze.json').read_text())
    verify_files(model_freeze['files'])
    source = load_npz(args.original/'prepared/profiles.npz')
    split = load_npz(args.original/'fitness-v1/assignment.npz')['split']
    predictor = FitnessPredictor(args.original/'fitness-reference-model')
    supported = {g for g, flag in zip(predictor.genes, predictor.models['K562'].support) if flag}
    output.mkdir(parents=True)
    panels = {}
    for cohort in COHORTS:
        d = load_npz(args.prepared/f'{cohort}.npz')
        names = d['pair_names']
        include = np.array([a != b and a in supported and b in supported for a, b in names])
        include &= np.all(d['counts'] >= 2, axis=1)
        names = names[include]
        if not len(names) or len(set(map(tuple, names))) != len(names):
            raise ValueError('Empty or duplicate panel')
        strata = strata_for(names, source['pairs'], split, source['genes'])
        # Freeze predictions without using external outcome values.
        predictions = np.concatenate([predictor.predict(names[i:i+1024], 'K562')
                                      for i in range(0, len(names), 1024)])
        ids = np.array([[predictor.index[a], predictor.index[b]] for a, b in names])
        residual = predictor.models['K562'].predict(ids)[0]
        singles = predictor.arrays['singles'][0][ids]
        np.savez_compressed(output/f'{cohort}_predictions.npz', pair_names=names,
                            rows=np.flatnonzero(include), strata=strata, completion=predictions,
                            main_effect=predictions-residual, additive=singles.sum(1),
                            minimum=singles.min(1), zero=np.zeros(len(names)))
        panels[cohort] = {s: {'pairs': int(np.sum(strata == s)),
                              'genes': len(np.unique(names[strata == s]))} for s in np.unique(strata)}
    cfg = json.loads((ROOT/'configs/experiments/fitness_external_20261020.json').read_text())
    files = dict(model_freeze['files'])
    for path in [args.prepared/'preparation.json', ROOT/'tools/prepare_external_fitness.py',
                 ROOT/'tools/evaluate_external_fitness.py', ROOT/'configs/experiments/fitness_external_20261020.json']:
        files[str(path.resolve())] = sha256(path)
    for c in COHORTS:
        for path in [args.prepared/f'{c}.npz', output/f'{c}_predictions.npz']:
            files[str(path.resolve())] = sha256(path)
    write_json(output/'protocol.json', {'config': cfg, 'files': files, 'panels': panels,
                                      'prepared': str(args.prepared.resolve()), 'freeze': model_freeze})
    print(json.dumps({'status': 'frozen_before_scoring', 'panels': panels}))


def run(output):
    p = json.loads((output/'protocol.json').read_text())
    verify_files(p['files'])
    if (output/'results.json').exists():
        raise FileExistsError('Completed evaluation is immutable')
    start, cpu = time.perf_counter(), time.process_time()
    results, passes = {}, []
    for cohort in COHORTS:
        d = load_npz(Path(p['prepared'])/f'{cohort}.npz')
        frozen = load_npz(output/f'{cohort}_predictions.npz')
        names, rows = frozen['pair_names'], frozen['rows']
        y = d['gamma'][rows]
        if y.shape != (len(names), 2) or not np.isfinite(y).all():
            raise ValueError('Invalid infection targets')
        cohortscores = {}
        for stratum in np.unique(frozen['strata']):
            mask = frozen['strata'] == stratum
            truth = y[mask]
            score = {'pairs': int(mask.sum()), 'genes': len(np.unique(names[mask])),
                     'replicate_disagreement_mse': float(np.mean(np.diff(truth, axis=1)**2)),
                     'methods': {}}
            for method in ('completion', 'main_effect', 'additive', 'minimum', 'zero'):
                pred = frozen[method][mask]
                score['methods'][method] = {'mean_target': metrics(pred, truth.mean(1)),
                                             'infections': [metrics(pred, truth[:, r]) for r in range(2)]}
            candidate, reference = frozen['completion'][mask], frozen['main_effect'][mask]
            difference = (candidate-truth.mean(1))**2-(reference-truth.mean(1))**2
            score['completion_minus_main_effect'] = uncertainty(names[mask], difference)
            cm, rm = [score['methods'][m]['mean_target']['mse'] for m in ('completion', 'main_effect')]
            score['relative_mse_gain'] = float(1-cm/rm)
            score['qualifies_primary'] = bool(stratum == 'absent_original_eligible_bank' and
                                             mask.sum() >= 50 and score['genes'] >= 10)
            score['passes'] = bool(score['relative_mse_gain'] >= .02 and
                score['completion_minus_main_effect']['gene_resampling_conditional_95'][1] < 0 and
                all(score['methods']['completion']['infections'][r]['mse'] <
                    score['methods']['main_effect']['infections'][r]['mse'] for r in range(2)) and
                all(cm <= score['methods'][m]['mean_target']['mse'] for m in ('additive', 'minimum', 'zero')))
            if score['qualifies_primary']:
                passes.append(score['passes'])
            # Extra-information diagnostic: external single responses, no fitted coefficients.
            si = {g: s for g, s in zip(d['single_genes'], d['singles'], strict=True)}
            available = np.array([a in si and b in si for a, b in names[mask]])
            score['external_single_diagnostic'] = {'pairs': int(available.sum())}
            if available.any():
                ss = np.array([[si[a], si[b]] for a, b in names[mask][available]])
                for method, pred in [('additive', ss.sum(1)), ('minimum', ss.min(1))]:
                    score['external_single_diagnostic'][method] = metrics(pred.mean(1), truth[available].mean(1))
            cohortscores[stratum] = score
        results[cohort] = cohortscores
    peak = psutil.Process().memory_info()
    write_json(output/'results.json', {'cohorts': results, 'qualifying_cohorts': len(passes),
        'outcome': 'PROMISING' if len(passes) == 4 and all(passes) else
                   'FAILED HYPOTHESIS' if passes else 'INSUFFICIENT EVIDENCE',
        'scope': 'unchanged K562 reference-growth source transfer; two infections per cohort; no new laboratory or Jurkat claim',
        'resources': {'wall_seconds': time.perf_counter()-start, 'cpu_seconds': time.process_time()-cpu,
                      'peak_rss_bytes': getattr(peak, 'peak_wset', peak.rss)}})
    print(json.dumps({'status': 'completed', 'qualifying_cohorts': len(passes), 'passes': passes}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--prepared', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        freeze(args) if args.action == 'freeze' else run(args.output)
