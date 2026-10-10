"""Execute the frozen primary ablations on coherent retrospective development banks."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import fingerprint, read_json, sha256, write_json
from opal.series import OpalBase, subset
from run_model_series_bilinear import ablations, metrics, resource_check


def run(args):
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = {'config_sha256': sha256(ROOT / 'configs/experiments/opal_model_series_v2.json'),
              'runner_sha256': sha256(Path(__file__)), 'ablations_sha256': sha256(ROOT / 'tools/run_model_series_bilinear.py'),
              'scope': 'Same frozen full-label ablations; retrospective coherent development banks only',
              'protected_evaluator_labels': 0, 'changed_primary_endpoint': False}
    path = args.output / 'scope_before_fit.json'
    if path.exists() and read_json(path) != freeze:
        raise ValueError('Retrospective ablation freeze changed')
    write_json(path, freeze)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for source in ['norman-log2', 'wessels-log2', 'norman-count', 'wessels-count']:
            original = ROOT / 'results/opal_model_series_v2/retrospective' / source
            scope = read_json(original / 'scope_before_fit.json')
            prepared = ROOT / 'data/prepared' / source / 'profiles.npz'
            if sha256(prepared) != scope['profiles_sha256']:
                raise ValueError('Changed retrospective source')
            with np.load(prepared, allow_pickle=False) as z:
                profiles, pairs, responses, targets = [z[k] for k in ['singles', 'pairs', 'doubles', 'targets']]
            singles = profiles[pairs].copy()
            for i, pair in enumerate(pairs):
                singles[i] = singles[i, np.argsort([str(targets[t]) for t in pair])]
            ids = np.asarray([json.dumps(sorted(str(targets[t]) for t in p), separators=(',', ':')) for p in pairs])
            bank = dict(singles=singles, responses=responses.astype(float), pair_ids=ids,
                        culture_ids=np.repeat(source, len(ids)))
            assignment = np.asarray([scope['fold_assignment'][p] for p in ids])
            for outer in range(3):
                destination = args.output / f'{source}-{outer}.json'
                if destination.exists():
                    continue
                resource_check(args.local, args.output, settings)
                started, cpu = time.monotonic(), time.process_time()
                row = read_json(original / f'fold-{outer}.json')
                cp = next(c for c in row['checkpoints'] if c['model'] == 'Opal-Guard')
                model = OpalBase.load(cp['path'])
                train, held = assignment != outer, assignment == outer
                development = subset(bank, train)
                development['partitions'] = np.where(assignment[train] == (outer + 1) % 3, 'validation', 'train')
                query = {k: v[held] for k, v in bank.items() if k != 'responses'}
                base = model._base(query)
                predictions, refusals = ablations(model, development, query)
                # Outcomes are accessed by metrics only after all predictions exist.
                report = dict(source=source, outer=outer, held_rows=int(held.sum()), unique_labels=int(train.sum()),
                    metrics={k: metrics(p, bank['responses'][held], base) for k, p in predictions.items()},
                    unavailable=refusals, source_sha256=scope['profiles_sha256'],
                    prediction_hashes={k: fingerprint(p.tolist()) for k, p in predictions.items()},
                    cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - started,
                    process_peak_rss_bytes=psutil.Process().memory_info().peak_wset)
                work = args.local / f'{source}-{outer}'
                work.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(work / 'per_pair_losses.npz', **{
                    k: np.mean((p - bank['responses'][held]) ** 2, axis=1) for k, p in predictions.items()})
                write_json(destination, report)
                print(source, outer, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    run(parser.parse_args())
