"""Freeze, then run a bounded nested development test of gene-local composition."""
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
from opal.artifacts import environment, load_npz, read_json, sha256, write_json
from opal.models.composition import Composition
from opal.models.kernel import smoother
from tools.improve_residual_model import candidates, fold_sets, select


def signature():
    return {p: sha256(ROOT / p) for p in ['tools/study_composition.py',
        'src/models/composition.py', 'src/models/kernel.py', 'src/splits/__init__.py',
        'tools/improve_residual_model.py', 'configs/experiments/composition_20261020.json']}


def freeze(config, output):
    if output.exists():
        raise FileExistsError('Use a new output; registered attempts are immutable')
    output.mkdir(parents=True)
    protocol = {'config': config, 'code': signature(), 'environment': environment(), 'datasets': {}}
    for name in config['datasets']:
        path = ROOT / 'data/prepared' / name / 'profiles.npz'
        data = load_npz(path)
        pairs = data['pairs']
        if len({tuple(sorted(p)) for p in pairs.tolist()}) != len(pairs):
            raise ValueError('Duplicate canonical pairs')
        record = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                  'pairs': len(pairs), 'genes': len(data['genes']), 'splits': {}}
        for kind in config['splits']:
            outer = fold_sets(pairs, kind, config['seed'], config['outer_folds'])
            if len(outer) != config['outer_folds']:
                raise ValueError('Insufficient feasible outer folds; no redraw')
            rows = []
            for f, (train, test) in enumerate(outer):
                inner = fold_sets(pairs[train], kind, config['seed'] + f + 1, config['inner_folds'])
                if len(inner) < 2:
                    raise ValueError('Insufficient inner folds; no redraw')
                rows.append({'train': train.tolist(), 'test': test.tolist(),
                             'inner': [{'train': train[a].tolist(), 'test': train[b].tolist()} for a, b in inner]})
            record['splits'][kind] = rows
        protocol['datasets'][name] = record
    protocol['baseline_menu'] = {
        'kernel': 'unchanged candidates() from tools/improve_residual_model.py; inner-selected all families',
        'simple': ['additive', 'no_change', 'shared_residual_mean'],
        'composition_ablation': 'pooled overlap and hierarchical affine; same intercept and penalties',
        'reduced_rank': 'linear residual ridge with fitted-output rank [4,8,16] and alpha [.1,1,10]',
        'selection': 'equal-pair all-gene inner MSE; ties by menu order; no outer-score choice'}
    write_json(output / 'protocol.json', protocol)
    print(json.dumps({'status': 'frozen', 'sha256': sha256(output / 'protocol.json')}), flush=True)


def local_menu(config):
    return [(family, alpha) for family in ['composition', 'affine'] for alpha in config['alphas']] + [('pooled', 1.)]


def fit_local(key, x, y, train, test):
    family, alpha = key
    model = Composition(alpha, overlap=family != 'affine', pooled=family == 'pooled').fit(x[train], y[train])
    return model, model.predict(x[test])


def rrr(kernel, residual, train, test, alpha, rank):
    # Truncate fitted training outputs, not validation outcomes.
    wt = smoother(kernel, train, train, alpha)
    mean = residual[train].mean(0)
    fitted = wt @ residual[train] - mean
    eig, u = np.linalg.eigh(fitted @ fitted.T)
    indices = np.flatnonzero(eig > max(float(eig.max()), 1.) * 1e-12)[-rank:]
    basis = fitted.T @ (u[:, indices] / np.sqrt(eig[indices]))
    pred = smoother(kernel, train, test, alpha) @ residual[train] - mean
    return mean + (pred @ basis) @ basis.T


def bootstrap(delta, seed, count):
    rng = np.random.default_rng(seed)
    means = np.mean(delta[rng.integers(len(delta), size=(count, len(delta)))], axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def run(output):
    protocol = read_json(output / 'protocol.json'); config = protocol['config']
    if protocol['code'] != signature():
        raise ValueError('Code differs from frozen protocol')
    if (output / 'summary.json').exists():
        raise FileExistsError('Completed experiment is immutable')
    started = time.perf_counter(); peak = 0; results = {}
    for name, source in protocol['datasets'].items():
        path = ROOT / source['path']
        if sha256(path) != source['sha256']:
            raise ValueError('Data changed after freeze')
        data = load_npz(path); singles, pairs, y = data['singles'], data['pairs'], data['doubles']
        x = singles[pairs]; additive = x.sum(1); residual = y - additive
        kernels = candidates(singles, pairs); gram = residual @ residual.T / y.shape[1]
        linear = next(k for c, k in kernels if c['kind'] == 'global')
        result = {}; local = local_menu(config)
        for kind, folds in source['splits'].items():
            labels = ['composition', 'affine', 'pooled', 'kernel', 'reduced_rank',
                      'additive', 'no_change', 'shared_residual_mean', 'selected_baseline']
            predictions = {label: np.zeros_like(y, dtype=float) for label in labels}
            coverage = np.zeros(len(y), dtype=int); details = []; weighted = {label: 0. for label in labels}; weight_count = 0
            for number, fold in enumerate(folds):
                train, test = np.array(fold['train']), np.array(fold['test'])
                inner = [(np.array(f['train']), np.array(f['test'])) for f in fold['inner']]
                if set(train) & set(test) or any(set(a) & set(test) or set(b) & set(test) for a,b in inner):
                    raise ValueError('Outer label leakage')
                scores = []
                for key in local:
                    error, count = 0., 0
                    for a, b in inner:
                        _, pred = fit_local(key, x, y, a, b)
                        error += float(np.sum(np.mean((pred - y[b]) ** 2, axis=1))); count += len(b)
                    scores.append({'family': key[0], 'alpha': key[1], 'mse': error / count})
                chosen = {family: min([s for s in scores if s['family'] == family], key=lambda s:s['mse'])
                          for family in ['composition', 'affine', 'pooled']}
                relative = [(np.searchsorted(train, a), np.searchsorted(train, b)) for a,b in inner]
                best_kernel, kernel_search = select(kernels, gram, train, relative)
                rank_search = []
                for alpha in [.1, 1., 10.]:
                    for rank in [4, 8, 16]:
                        error, count = 0., 0
                        for a,b in inner:
                            pred = additive[b] + rrr(linear, residual, a, b, alpha, rank)
                            error += float(np.sum(np.mean((pred-y[b]) ** 2, axis=1))); count += len(b)
                        rank_search.append({'alpha': alpha, 'rank': rank, 'mse': error/count})
                best_rank = min(rank_search, key=lambda s:s['mse'])
                pred = {family: fit_local((family, s['alpha']), x, y, train, test)[1] for family,s in chosen.items()}
                kernel_config, kernel = kernels[best_kernel['index']]
                pred['kernel'] = additive[test] + smoother(kernel, train, test, kernel_config['alpha']) @ residual[train]
                pred['reduced_rank'] = additive[test] + rrr(linear, residual, train, test, best_rank['alpha'], best_rank['rank'])
                pred['additive'], pred['no_change'] = additive[test], np.zeros_like(y[test])
                pred['shared_residual_mean'] = additive[test] + residual[train].mean(0)
                baseline_scores = {f:s['mse'] for f,s in chosen.items() if f != 'composition'}
                baseline_scores.update(kernel=best_kernel['mse'], reduced_rank=best_rank['mse'])
                for label in ['additive','no_change','shared_residual_mean']:
                    error, count = 0., 0
                    for a,b in inner:
                        p = additive[b] if label == 'additive' else np.zeros_like(y[b]) if label == 'no_change' else additive[b]+residual[a].mean(0)
                        error += float(np.sum(np.mean((p-y[b])**2, axis=1))); count += len(b)
                    baseline_scores[label] = error/count
                selected = min(baseline_scores, key=baseline_scores.get)
                pred['selected_baseline'] = pred[selected]
                gene_weights = np.var(y[train], axis=0)
                gene_weights = gene_weights / max(float(gene_weights.mean()), 1e-12)
                for label,p in pred.items():
                    predictions[label][test] += p
                    weighted[label] += float(np.sum(np.mean(gene_weights*(p-y[test])**2, axis=1)))
                weight_count += len(test); coverage[test] += 1
                details.append({'fold': number, 'train_pairs': len(train), 'test_pairs': len(test),
                    'local_search': scores, 'kernel_selected': best_kernel, 'rank_search': rank_search,
                    'baseline_selection': baseline_scores, 'selected_baseline': selected,
                    'mse': {label:float(np.mean((p-y[test])**2)) for label,p in pred.items()}})
                memory = psutil.Process().memory_info()
                peak = max(peak, getattr(memory, 'peak_wset', memory.rss))
                if time.perf_counter()-started > config['compute']['wall_seconds'] or peak > config['compute']['peak_rss_bytes']:
                    raise RuntimeError('Registered resource cap reached; retain partial attempt')
                write_json(output / f'{name}-{kind}-progress.json', details)
                print(name, kind, number, details[-1]['mse']['composition'], selected, flush=True)
            included = coverage > 0
            per_pair = {}
            for label in labels:
                predictions[label][included] /= coverage[included,None]
                per_pair[label] = np.mean((predictions[label][included]-y[included])**2, axis=1)
            delta = per_pair['composition']-per_pair['selected_baseline']
            result[kind] = {'covered_pairs': int(included.sum()), 'coverage': coverage.tolist(),
                'mse': {k:float(v.mean()) for k,v in per_pair.items()},
                'effect_weighted_mse_per_occurrence': {k:v/weight_count for k,v in weighted.items()},
                'conditional_pair_bootstrap_loss_difference_95': bootstrap(delta,config['seed'],config['bootstrap_draws']),
                'uncertainty_scope': 'conditional pair sampling only; shared singles/control and dependent folds, not biological calibration',
                'folds': details}
            # Retain only the hypothesis and its selected reference at gene level;
            # all baseline scalar scores and selection histories remain in JSON.
            np.savez_compressed(output / f'{name}-{kind}-predictions.npz',
                                composition=predictions['composition'],
                                selected_baseline=predictions['selected_baseline'], coverage=coverage)
        # Final development-only refit selected by CV, never outer scoring.
        all_indices = np.arange(len(y)); cv = fold_sets(pairs,'random',config['seed'],config['inner_folds'])
        final_scores = []
        for alpha in config['alphas']:
            losses = [(float(np.mean((fit_local(('composition',alpha),x,y,a,b)[1]-y[b])**2)),len(b)) for a,b in cv]
            final_scores.append((float(np.average([a for a,b in losses],weights=[b for a,b in losses])),alpha))
        alpha = min(final_scores)[1]
        model = Composition(alpha).fit(x,y); checkpoint = output/f'{name}-deployment.npz'; model.save(checkpoint)
        replay = float(np.max(np.abs(model.predict(x)-Composition.load(checkpoint).predict(x))))
        result['deployment'] = {'alpha':alpha, 'training_pairs':len(y), 'checkpoint_sha256':sha256(checkpoint),
            'gene_order_sha256': __import__('hashlib').sha256(data['genes'].tobytes()).hexdigest(),
            'replay_max_abs':replay, 'parameters':int(model.coefficient.size+model.ymean.size), 'score':None}
        write_json(output/f'{name}-gene_schema.json', {'genes':data['genes'].tolist(), 'source_sha256':source['sha256']})
        results[name] = result
    elapsed = time.perf_counter()-started
    memory = psutil.Process().memory_info()
    peak = max(peak, getattr(memory, 'peak_wset', memory.rss))
    if elapsed > config['compute']['wall_seconds'] or peak > config['compute']['peak_rss_bytes']:
        raise RuntimeError('Final resource cap reached; retain partial attempt')
    summary = {'status':'completed', 'classification':'INSUFFICIENT EVIDENCE', 'protocol_sha256':sha256(output/'protocol.json'),
               'datasets':results, 'wall_seconds':elapsed, 'peak_rss_bytes':peak,
               'scope':config['scope'], 'code':signature()}
    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) + len(json.dumps(summary).encode()) * 2
    if size > config['compute']['artifact_bytes']:
        raise RuntimeError('Artifact cap exceeded')
    summary['artifact_bytes'] = size; write_json(output/'summary.json',summary)
    print(json.dumps({'status':'completed','wall_seconds':elapsed,'peak_rss_bytes':peak,'artifact_bytes':size}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze','run'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/experiments/composition_20261020.json')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        freeze(read_json(args.config),args.output) if args.action=='freeze' else run(args.output)
