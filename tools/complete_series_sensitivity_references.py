"""Fit the declared simple/historical controls under every sensitivity information tier."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, sha256, write_json
from opal.series import OpalBase, subset
from opal.series_bilinear import weights
from opal.replicated.models import select, fit_candidate, predict_candidate
from opal.replicated.retrospective import pair_folds
from run_model_series_bilinear import source_banks, metrics, resource_check


def main():
    output = ROOT / 'results/opal_model_series_v2/sensitivity-references'
    local = Path('C:/Users/USER/Desktop/opal-model-series-20261010-v2/sensitivity-references')
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    menu = read_json(ROOT / 'configs/experiments/replicated_fixed_state_v1.json')['menu']
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'scope_before_fit.json', {'menus_unchanged': True, 'additional_pair_labels': 0,
        'config_sha256': sha256(ROOT / 'configs/experiments/opal_model_series_v2.json'),
        'role': 'Complete missing historical/simple references for the already frozen 51 sensitivity populations',
        'original_sensitivity_records_replaced': False, 'runner_sha256': sha256(Path(__file__))})
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for source, (path, _) in source_banks().items():
            with np.load(path, allow_pickle=False) as z:
                bank = {k: z[k] for k in ['singles', 'responses', 'pair_ids', 'culture_ids']}
            for record_path in sorted((ROOT / 'results/opal_model_series_v2/sensitivities').glob(source + '-*.json')):
                resource_check(local, output, settings)
                record = read_json(record_path)
                checkpoint = Path(record['checkpoint']['path'])
                contract = read_json(checkpoint.parent / 'contract.json')
                dev = np.isin(bank['pair_ids'], contract['acquired_pair_ids'])
                test = np.isin(bank['pair_ids'], contract['held_pair_ids'])
                if record['kind'] == 'transport':
                    dev &= bank['culture_ids'] != record['query_stratum']
                    test &= bank['culture_ids'] == record['query_stratum']
                if int(dev.sum()) != record['acquired_pair_stratum_labels'] or int(test.sum()) != record['outer_rows']:
                    raise ValueError('Sensitivity reference population changed')
                development = subset(bank, dev)
                inner = pair_folds(development['pair_ids'], settings['training_seed'])
                development['partitions'] = np.where(inner == 0, 'validation', 'train')
                query = {k: v[test] for k, v in bank.items() if k != 'responses'}
                started, cpu = time.monotonic(), time.process_time()
                selected = select(development, menu)
                predictions = {'training-mean': np.repeat((weights(development) @ development['responses'])[None], int(test.sum()), axis=0)}
                for role in ['reference', 'candidate']:
                    fitted = fit_candidate(selected[role], development, np.ones(len(development['pair_ids']), bool))
                    predictions['historical-' + role] = predict_candidate(selected[role], fitted, query)
                base = OpalBase.load(checkpoint)._base(query)
                report = {'source': source, 'kind': record['kind'], 'outer': record['outer'],
                    'query_stratum': record['query_stratum'], 'acquired_unique_pairs': record['acquired_unique_pairs'],
                    'acquired_pair_stratum_labels': record['acquired_pair_stratum_labels'], 'additional_pair_labels': 0,
                    'original_population_sha256': sha256(record_path), 'selected': selected, 'per_stratum': {}}
                for culture in sorted(set(query['culture_ids'])):
                    mask = query['culture_ids'] == culture
                    report['per_stratum'][str(culture)] = {'rows': int(mask.sum()), 'metrics': {
                        name: metrics(p[mask], bank['responses'][test][mask], base[mask]) for name, p in predictions.items()}}
                work = local / record_path.stem
                work.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(work / 'per_pair_losses.npz', **{
                    name: np.mean((p - bank['responses'][test]) ** 2, axis=1) for name, p in predictions.items()})
                report.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - started)
                write_json(output / record_path.name, report)
                print(record_path.stem, flush=True)


if __name__ == '__main__':
    main()
