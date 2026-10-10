"""Produce safe all-development refits only after the finite evaluation is scored."""
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
from opal.artifacts import array_hash, fingerprint, read_json, sha256, write_json
from opal.series import OpalBase, OpalGuard
from opal.replicated.retrospective import pair_folds
from run_model_series_bilinear import source_banks, variants, resource_check


def sources():
    for name, (path, _) in source_banks().items():
        with np.load(path, allow_pickle=False) as z:
            bank = {k: z[k] for k in ['singles', 'responses', 'pair_ids', 'culture_ids']}
        row = next(r for r in read_json(ROOT / 'results/opal_model_series_v2/primary' / name / 'conditions.json')
                   if r['checkpoints'])
        contract = read_json(Path(row['checkpoints'][0]['path']).parent.parent / 'contract.json')
        yield name, path, bank, contract, 'all source strata, equal-stratum acquired pair aggregates'
    for name in ['norman-log2', 'wessels-log2', 'norman-count', 'wessels-count']:
        path = ROOT / 'data/prepared' / name / 'profiles.npz'
        with np.load(path, allow_pickle=False) as z:
            profiles, pairs, responses, targets = [z[k] for k in ['singles', 'pairs', 'doubles', 'targets']]
        singles = profiles[pairs].copy()
        for i, p in enumerate(pairs):
            singles[i] = singles[i, np.argsort([str(targets[t]) for t in p])]
        bank = dict(singles=singles, responses=responses.astype(float),
                    pair_ids=np.asarray([json.dumps(sorted(str(targets[t]) for t in p), separators=(',', ':')) for p in pairs]),
                    culture_ids=np.repeat(name, len(pairs)))
        row = read_json(ROOT / 'results/opal_model_series_v2/retrospective' / name / 'fold-0.json')
        contract = read_json(Path(row['checkpoints'][0]['path']).parent.parent / 'contract.json')
        yield name, path, bank, contract, 'historical equal-stratum development aggregate; protected query bank excluded'


def run(args):
    evidence = ROOT / 'results/opal_model_series_v2'
    for name in ['norman-log2', 'wessels-log2', 'norman-count', 'wessels-count']:
        if read_json(evidence / 'retrospective-budget-completion' / name / 'completion.json')['conditions'] != 78:
            raise ValueError('Deployment refused before retrospective budgets are complete')
    if len(list((evidence / 'retrospective-ablations').glob('*-*.json'))) != 12:
        raise ValueError('Deployment refused before retrospective ablations are complete')
    if read_json(evidence / 'verification/native-replay.json')['count'] != 72:
        raise ValueError('Deployment refused before native checkpoint replay')
    if len(list((evidence / 'sensitivity-references').glob('*.json'))) != 52:
        raise ValueError('Deployment refused before all sensitivity references')
    if len(list((evidence / 'native-extra-single-controls').glob('*.json'))) != 25:
        raise ValueError('Deployment refused before extra-single controls')
    if len(list((evidence / 'normalization-diagnostics').glob('*.json'))) != 3:
        raise ValueError('Deployment refused before normalization diagnostics')
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    settings['checkpoint_array_store'] = str(args.local / 'array_store')
    menu = read_json(ROOT / 'configs/experiments/replicated_fixed_state_v1.json')['menu']
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = {'role': 'all-development deployment after evaluation; no evaluation score assigned',
              'config_sha256': sha256(ROOT / 'configs/experiments/opal_model_series_v2.json'),
              'runner_sha256': sha256(Path(__file__)), 'source_banks': {name: sha256(path) for name, path, _, _, _ in sources()},
              'protected_original_genetic_query_labels': 0, 'inner_split': 'three canonical pair folds; fold one validates',
              'scientific_menus_changed': False, 'primary_contexts_pooled': 'within each source/TNF contract only'}
    freeze_path = args.output / 'scope_before_fit.json'
    if freeze_path.exists() and read_json(freeze_path) != freeze:
        raise ValueError('Deployment freeze changed')
    write_json(freeze_path, freeze)
    torch.set_num_threads(1)
    records = []
    with threadpool_limits(limits=1):
        for name, source, bank, original_contract, tier in sources():
            destination = args.output / (name + '.json')
            if destination.exists():
                records.append(read_json(destination))
                continue
            resource_check(args.local, args.output, settings)
            work = args.local / name
            work.mkdir(parents=True, exist_ok=False)
            contract = dict(original_contract, acquired_pair_ids=sorted(set(bank['pair_ids'])), held_pair_ids=[],
                checkpoint_role='all-development deployment refit after scoring', information_tier=tier,
                split_lineage=fingerprint({'source': sha256(source), 'deployment_freeze': fingerprint(freeze)}))
            write_json(work / 'contract.json', contract)
            bank['partitions'] = np.where(pair_folds(bank['pair_ids'], settings['outer_seed']) == 1, 'validation', 'train')
            started, cpu = time.monotonic(), time.process_time()
            guard = OpalGuard(settings, menu).fit(bank, contract)
            # In-sample queries verify serialization only, never deployment performance.
            query = {k: v[:8] for k, v in bank.items() if k in ['singles', 'pair_ids', 'culture_ids']}
            records_for_source = []
            for label, model in variants(guard).items():
                checkpoint = model.save(work / label / 'checkpoint')
                predicted = model.predict(query, contract)
                restored = OpalBase.load(checkpoint).predict(query, contract)
                difference = float(np.max(np.abs(predicted['predictions'] - restored['predictions'])))
                if difference > settings['replay_atol']:
                    raise ValueError('Deployment safe-state replay failed')
                metadata = read_json(checkpoint / 'model.json')
                records_for_source.append(dict(model=label, checkpoint=str(checkpoint), replay_max_abs=difference,
                    model_sha256=sha256(checkpoint / 'model.json'), parameters_sha256=sha256(checkpoint / 'parameters.npz'),
                    parameter_counts=metadata['parameter_counts'], expert_available=model.fitted['expert_mask'].tolist(),
                    logical_numeric_values=metadata.get('logical_numeric_values', metadata['stored_numeric_values']),
                    checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.iterdir() if p.is_file()),
                    array_store=str(args.local / 'array_store'), query_prediction_sha256=array_hash(predicted['predictions']),
                    evaluation_score=None))
            report = dict(source=name, source_sha256=sha256(source), information_tier=tier,
                unique_pair_labels=len(set(bank['pair_ids'])), aggregate_pair_rows=len(bank['pair_ids']),
                all_label_roles=['selection', 'fit', 'PCA/SVD', 'OOF experts', 'router validation', 'stopping'],
                protected_original_query_labels=0, checkpoints=records_for_source,
                wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu,
                process_peak_rss_bytes=psutil.Process().memory_info().peak_wset,
                recommendation='Opal-Base; Interaction/Guard experimental', evaluation_score=None,
                replay_query_role='in-sample serialization check, no performance claim')
            write_json(destination, report)
            records.append(report)
            print(name, len(records_for_source), flush=True)
    write_json(args.output / 'manifest.json', {'records': records, 'models': 21, 'evaluated_deployment_scores': False,
        'local_weights_only': True, 'redistribution_authorized': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
