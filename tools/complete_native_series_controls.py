"""Matched linear control for native adaptations' extra observed-single supervision."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, sha256, write_json
from run_model_series_bilinear import source_banks, metrics, resource_check


def fit(x, y, alpha):
    center, mean = x.mean(0), y.mean(0)
    xc, yc = x - center, y - mean
    coefficients = solve(xc.T @ xc / len(x) + alpha * np.eye(x.shape[1]), xc.T @ yc / len(x), assume_a='pos')
    return mean - center @ coefficients, coefficients


def main():
    output = ROOT / 'results/opal_model_series_v2/native-extra-single-controls'
    local = Path('C:/Users/USER/Desktop/opal-model-series-20261010-v2/native-extra-single-controls')
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'scope_before_fit.json', {'classification': 'linear extra-information reference, not an Opal variant',
        'penalties': settings['penalties'], 'additional_inputs': 'same all-source-stratum observed singles as GEARS/CPA adaptations',
        'pair_budget': 'full, acquisition seed 11; unchanged outer assignment', 'seed': settings['training_seed'],
        'presence_coefficients': 'categorical indicators, no physical-dose inference', 'extra_pair_labels': 0,
        'primary_scale_unchanged': True, 'cell_level_training': False, 'runner_sha256': sha256(Path(__file__))})
    with threadpool_limits(limits=1):
        for source, (path, original) in source_banks().items():
            if source == 'combi-seq':
                continue
            with np.load(path, allow_pickle=False) as z:
                bank = {k: z[k] for k in ['singles', 'responses', 'pair_ids', 'culture_ids']}
            rows = read_json(ROOT / 'results/opal_model_series_v2/primary' / source / 'conditions.json')
            mapping = read_json(original / 'scope_before_fit.json')['global_fold_assignment']
            for row in [r for r in rows if r['budget'] == 1 and r['order'] == 'uniform'
                        and r['acquisition_seed'] == 11 and r['training_seed'] == 20261017]:
                resource_check(local, output, settings)
                contract = read_json(Path(row['checkpoints'][0]['path']).parent.parent / 'contract.json')
                stratum = bank['culture_ids'] == row['stratum']
                dev = stratum & np.isin(bank['pair_ids'], contract['acquired_pair_ids'])
                test = stratum & np.isin(bank['pair_ids'], contract['held_pair_ids'])
                targets = sorted({t for p in bank['pair_ids'][stratum] for t in json.loads(p)})
                lookup = {t: i for i, t in enumerate(targets)}
                profiles = np.empty((len(targets), bank['responses'].shape[1]))
                for p, values in zip(bank['pair_ids'][stratum], bank['singles'][stratum]):
                    for t, value in zip(json.loads(p), values):
                        profiles[lookup[t]] = value
                def design(ids):
                    x = np.zeros((len(ids), len(targets)))
                    for i, pair in enumerate(ids):
                        for target in json.loads(pair):
                            x[i, lookup[target]] = 1
                    return x
                x, y = design(bank['pair_ids'][dev]), bank['responses'][dev]
                val = np.asarray([mapping[p] == (row['outer'] + 1) % 3 for p in bank['pair_ids'][dev]])
                references_x = np.vstack((np.zeros((1, len(targets))), np.eye(len(targets))))
                references_y = np.vstack((np.zeros((1, y.shape[1])), profiles))
                started, cpu = time.monotonic(), time.process_time()
                attempts = []
                for alpha in settings['penalties']:
                    intercept, coefficients = fit(np.vstack((references_x, x[~val])), np.vstack((references_y, y[~val])), alpha)
                    score = float(np.mean((intercept + x[val] @ coefficients - y[val]) ** 2))
                    attempts.append({'alpha': alpha, 'development_mse': score})
                chosen = min(attempts, key=lambda r: (r['development_mse'], r['alpha']))['alpha']
                intercept, coefficients = fit(np.vstack((references_x, x)), np.vstack((references_y, y)), chosen)
                prediction = intercept + design(bank['pair_ids'][test]) @ coefficients
                identity = f'{source}-{row["stratum"]}-{row["outer"]}'
                work = local / identity
                work.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(work / 'parameters.npz', intercept=intercept, coefficients=coefficients, targets=np.asarray(targets))
                with np.load(work / 'parameters.npz', allow_pickle=False) as z:
                    replay = z['intercept'] + design(bank['pair_ids'][test]) @ z['coefficients']
                if not np.array_equal(replay, prediction):
                    raise ValueError('Extra-single linear replay failed')
                write_json(output / (identity + '.json'), {'source': source, 'stratum': row['stratum'], 'outer': row['outer'],
                    'held_rows': int(test.sum()), 'metrics': metrics(prediction, bank['responses'][test], np.zeros_like(prediction)),
                    'acquired_pair_labels': int(dev.sum()), 'extra_single_labels': len(targets), 'control_labels': 1,
                    'attempts': attempts, 'selected_alpha': chosen, 'parameters_sha256': sha256(work / 'parameters.npz'),
                    'replay_max_abs': 0, 'cpu_seconds': time.process_time() - cpu, 'wall_seconds': time.monotonic() - started})
    print('24 extra-single linear controls completed')


if __name__ == '__main__':
    main()
