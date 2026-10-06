"""Nested development experiments; never read historical evaluator responses."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
import time
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from opal.artifacts import load_npz, write_json, sha256, source_hash, fingerprint
from opal.models.kernel import single_kernel, symmetric_kernel, global_kernel, smoother, prediction_error
from opal.splits import partner_splits


def fold_sets(pairs, kind, seed, count):
    if kind == 'partner':
        folds = partner_splits(pairs.astype(str), seed, count, minimum_development=3, minimum_test=1)
        return [(np.array(f['development']), np.array(f['test'])) for f in folds if f['feasible']]
    order = np.random.default_rng(seed).permutation(len(pairs))
    assignments = np.empty(len(pairs), int)
    assignments[order] = np.arange(len(pairs)) % count
    return [(np.flatnonzero(assignments != i), np.flatnonzero(assignments == i)) for i in range(count)]


def candidates(singles, pairs):
    zero = np.zeros((len(pairs), len(pairs)))
    global_gram = global_kernel(singles, pairs)
    output = [({'kind': 'shared_offset', 'alpha': 1.0}, zero)]
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
        output.append(({'kind': 'global', 'alpha': alpha}, global_gram))
    for bandwidth in [0.25, 1, 4]:
        gram = single_kernel(singles, bandwidth=bandwidth)
        summed = symmetric_kernel(gram, pairs, pairs)
        product = symmetric_kernel(gram, pairs, pairs, interaction=True)
        for kind, kernel in [('single_sum', summed), ('pair_product', product),
                             ('global_plus_pair', global_gram + product),
                             ('global_plus_pair_10', global_gram + 10 * product)]:
            for alpha in [0.001, 0.01, 0.1, 1, 10]:
                output.append(({'kind': kind, 'bandwidth': bandwidth, 'alpha': alpha}, kernel))
    return output


def select(configs, gram, indices, folds, allowed=None):
    records = []
    for i, (config, kernel) in enumerate(configs):
        if allowed is not None and config['kind'] not in allowed:
            continue
        errors, sizes = [], []
        for train, val in folds:
            train, val = indices[train], indices[val]
            weights = smoother(kernel, train, val, config['alpha'])
            errors.append(prediction_error(weights, train, val, gram))
            sizes.append(len(val))
        records.append({'index': i, 'config': config, 'mse': float(np.average(errors, weights=sizes))})
    return min(records, key=lambda r: (r['mse'], r['index'])), records


def study(dataset, output, seed=20261005):
    if output.exists():
        raise FileExistsError('Preserve completed studies; choose a new output directory')
    output.mkdir(parents=True)
    started = time.perf_counter()
    data = load_npz(dataset / 'profiles.npz')
    singles, pairs, doubles = (np.asarray(data[k], dtype=np.float64 if k != 'pairs' else int)
                               for k in ['singles', 'pairs', 'doubles'])
    residual = doubles - singles[pairs[:, 0]] - singles[pairs[:, 1]]
    gram = residual @ residual.T / residual.shape[1]
    configs = candidates(singles, pairs)
    write_json(output / 'protocol.json', {'seed': seed, 'outer_folds': 5, 'inner_folds': 3,
        'candidate_count': len(configs), 'configurations': [c for c, _ in configs],
        'source_hash': source_hash(), 'script_sha256': sha256(__file__),
        'profiles_sha256': sha256(dataset / 'profiles.npz'),
        'scope': 'Exploratory nested development only; historical test outcomes previously known.',
        'evaluator_access': False, 'endpoint': 'equal-pair all-gene MSE',
        'negative_result_policy': 'Record both random and partner protocols and all candidate choices.'})
    with zipfile.ZipFile(output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__, 'tools/improve_residual_model.py')
        for file in sorted(Path('src').rglob('*.py')):
            archive.write(file, file.as_posix())
    results = {}
    for kind in ['random', 'partner']:
        folds = fold_sets(pairs, kind, seed, 5)
        numerator = {'adaptive': 0., 'shared_global': 0., 'additive': 0.}
        predictions = {key: np.zeros_like(residual) for key in numerator}
        coverage = np.zeros(len(pairs), int)
        records = []
        for f, (train, val) in enumerate(folds):
            inner = fold_sets(pairs[train], kind, seed + f + 1, 3)
            if len(inner) < 2:
                raise ValueError('Insufficient feasible inner folds; no redraw')
            winner, search = select(configs, gram, train, inner)
            baseline, _ = select(configs, gram, train, inner, {'global', 'shared_offset'})
            record = {'fold': f, 'train': train.tolist(), 'validation': val.tolist(),
                      'inner_splits': [{'train': train[t].tolist(), 'validation': train[v].tolist()} for t, v in inner],
                      'adaptive': winner, 'shared_global': baseline, 'search': search}
            for key, chosen in [('adaptive', winner), ('shared_global', baseline)]:
                config, kernel = configs[chosen['index']]
                weights = smoother(kernel, train, val, config['alpha'])
                predicted = weights @ residual[train]
                predictions[key][val] += predicted
                record[key + '_mse'] = float(np.mean((predicted - residual[val]) ** 2))
                numerator[key] += record[key + '_mse'] * len(val)
            record['additive_mse'] = float(np.mean(residual[val] ** 2))
            numerator['additive'] += record['additive_mse'] * len(val)
            coverage[val] += 1
            records.append(record)
            print(dataset.name, kind, f, winner['config'], 'MSE', record['adaptive_mse'], flush=True)
        eligible = coverage > 0
        for key in predictions:
            predictions[key][eligible] /= coverage[eligible, None]
        mse = {key: value / sum(len(val) for _, val in folds) for key, value in numerator.items()}
        # Partner folds overlap, so report equal-condition averaged predictions too.
        averaged = {key: float(np.mean((value[eligible] - residual[eligible]) ** 2))
                    for key, value in predictions.items()}
        results[kind] = {'mse_per_holdout_occurrence': mse, 'mse_averaged_predictions': averaged,
            'skill': 1 - mse['adaptive'] / mse['additive'],
            'gain_over_shared_global': mse['shared_global'] - mse['adaptive'],
            'covered_pairs': int(eligible.sum()), 'records': records,
            'caveat': 'Dependent folds; no independent replication or frontier claim.'}
        np.savez_compressed(output / f'{kind}_nested_predictions.npz', **predictions,
                            residual=residual, coverage=coverage, pair_indices=data['pair_indices'])
    # Freeze deployment candidate from all development CV, never choose by outer scores.
    indices = np.arange(len(pairs))
    winner, search = select(configs, gram, indices, fold_sets(pairs, 'random', seed, 5))
    config, _ = configs[winner['index']]
    all_pairs = np.concatenate([pairs, data['query_pairs']])
    full_configs = candidates(singles, all_pairs)
    kernel = full_configs[winner['index']][1]
    weights = smoother(kernel, indices, np.arange(len(pairs), len(all_pairs)), config['alpha'])
    query = data['query_pairs']
    predictions = singles[query[:, 0]] + singles[query[:, 1]] + weights @ residual
    np.savez_compressed(output / 'unscored_query_predictions.npz', predictions=predictions, pairs=query)
    write_json(output / 'selection.json', {'winner': winner, 'search': search, 'evaluator_access': False,
        'prediction_sha256': sha256(output / 'unscored_query_predictions.npz')})
    results['seconds'] = time.perf_counter() - started
    write_json(output / 'results.json', results)
    return {kind: {k: v for k, v in record.items() if k != 'records'}
            for kind, record in results.items() if isinstance(record, dict)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/improvements/residual-kernel-v1'))
    parser.add_argument('--seed', type=int, default=20261005)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summary = {name: study(Path('data/prepared') / name, args.output / name, args.seed)
                   for name in ['norman-log2', 'wessels-log2']}
    write_json(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
