"""Stronger nested linear controls on the exact saved improvement folds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import argparse
from threadpoolctl import threadpool_limits
from opal.artifacts import read_json, write_json, load_npz, sha256
from opal.baselines import fit_baseline, predict_baseline
from opal.representations import fit_representation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-root', type=Path, default=Path('results/improvements/residual-kernel-v1'))
    args = parser.parse_args()
    root = args.study_root
    output = root / 'expanded_linear_controls.json'
    if output.exists():
        raise FileExistsError(output)
    configs = [{'family': 'mean_residual'}]
    configs += [{'family': family, 'alpha': alpha, 'rank': 4}
                for family in ['global', 'ridge', 'reduced_rank']
                for alpha in [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]]
    summary = {'configurations': configs, 'evaluator_access': False,
               'script_sha256': sha256(__file__), 'datasets': {}}
    with threadpool_limits(limits=1):
        for name in ['norman-log2', 'wessels-log2']:
            data = load_npz(Path('data/prepared') / name / 'profiles.npz')
            singles = np.asarray(data['singles'], np.float64)
            pairs, doubles = data['pairs'], np.asarray(data['doubles'], np.float64)
            embedding = fit_representation(singles, 16)['embedding']
            reference = read_json(root / name / 'results.json')
            result = {}
            for kind in ['random', 'partner']:
                numerator, total = 0., 0
                oof, coverage = np.zeros_like(doubles), np.zeros(len(pairs), int)
                records = []
                for fold in reference[kind]['records']:
                    search = []
                    for config in configs:
                        errors, sizes = [], []
                        for inner in fold['inner_splits']:
                            train, val = np.array(inner['train']), np.array(inner['validation'])
                            model = fit_baseline(config, embedding, pairs[train], doubles[train], singles)
                            pred = predict_baseline(model, pairs[val])
                            errors.append(float(np.mean((pred - doubles[val]) ** 2)))
                            sizes.append(len(val))
                        search.append({'config': config, 'mse': float(np.average(errors, weights=sizes))})
                    chosen = min(search, key=lambda r: r['mse'])
                    train, val = np.array(fold['train']), np.array(fold['validation'])
                    model = fit_baseline(chosen['config'], embedding, pairs[train], doubles[train], singles)
                    pred = predict_baseline(model, pairs[val])
                    mse = float(np.mean((pred - doubles[val]) ** 2))
                    numerator += mse * len(val)
                    total += len(val)
                    oof[val] += pred
                    coverage[val] += 1
                    records.append({'fold': fold['fold'], 'chosen': chosen, 'search': search, 'outer_mse': mse})
                eligible = coverage > 0
                oof[eligible] /= coverage[eligible, None]
                mse = numerator / total
                kernel_mse = reference[kind]['mse_per_holdout_occurrence']['adaptive']
                result[kind] = {'nested_linear_mse': mse, 'kernel_mse': kernel_mse,
                    'kernel_relative_mse_reduction': 1 - kernel_mse / mse,
                    'averaged_linear_predictions_mse': float(np.mean((oof[eligible] - doubles[eligible]) ** 2)),
                    'records': records}
                np.savez_compressed(root / name / f'{kind}_strong_control_predictions.npz',
                                    predictions=oof, coverage=coverage, pair_indices=data['pair_indices'])
                print(name, kind, mse, 'kernel relative gain', 1 - kernel_mse / mse, flush=True)
            summary['datasets'][name] = result
    write_json(output, summary)


if __name__ == '__main__':
    main()
