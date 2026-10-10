"""Frozen full-label pooled-stratum, transport and withheld-target diagnostics."""
import argparse
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
from opal.series import OpalGuard, OpalBase, subset, Unavailable
from opal.replicated.retrospective import pair_folds
from run_model_series_bilinear import source_banks, resource_check, variants, metrics


def run(args):
    config = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    menu = read_json(ROOT / 'configs/experiments/replicated_fixed_state_v1.json')['menu']
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = {'config_sha256': sha256(ROOT / 'configs/experiments/opal_model_series_v2.json'),
              'kinds': ['pooled', 'transport', 'leave-target-group-out'], 'budget': 1,
              'seed': config['training_seed'], 'target_groups': 'sorted identities, index modulo three',
              'transport': 'leave one stratum out and globally exclude all outer held pairs',
              'primary_populations_unchanged': True, 'selection_uses_held_outcomes': False,
              'scope': 'separately scoped exposed retrospective sensitivity; no primary endpoint replacement'}
    path = args.output / 'scope_before_fit.json'
    if path.exists() and read_json(path) != freeze:
        raise ValueError('Sensitivity freeze changed')
    write_json(path, freeze)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for source, (bank_path, historical) in source_banks().items():
            with np.load(bank_path, allow_pickle=False) as archive:
                bank = {k: archive[k] for k in ('singles','responses','pair_ids','culture_ids')}
            if source == 'combi-seq':
                assignment = pair_folds(bank['pair_ids'], config['outer_seed'])
            else:
                mapping = read_json(historical / 'scope_before_fit.json')['global_fold_assignment']
                assignment = np.asarray([mapping[p] for p in bank['pair_ids']])
            targets = sorted({t for p in bank['pair_ids'] for t in json.loads(p)})
            strata = sorted(set(bank['culture_ids']))
            base_contract = read_json(Path(next(r for r in read_json(ROOT / 'results/opal_model_series_v2/primary' / source / 'conditions.json') if r['checkpoints'])['checkpoints'][0]['path']).parent.parent / 'contract.json')
            for kind in freeze['kinds']:
                for outer in range(3):
                    held_targets = set(targets[outer::3])
                    target_test = np.asarray([bool(set(json.loads(p)) & held_targets) for p in bank['pair_ids']])
                    for query_stratum in (strata if kind == 'transport' else ['all']):
                        identity = f'{source}-{kind}-{outer}-{query_stratum}'
                        record_path = args.output / (identity + '.json')
                        if record_path.exists():
                            continue
                        test = target_test if kind == 'leave-target-group-out' else assignment == outer
                        dev = ~test
                        if kind == 'transport':
                            dev &= bank['culture_ids'] != query_stratum
                            test &= bank['culture_ids'] == query_stratum
                        held = sorted(set(bank['pair_ids'][test]))
                        dev &= ~np.isin(bank['pair_ids'], held)
                        acquired = sorted(set(bank['pair_ids'][dev]))
                        development = subset(bank, dev)
                        inner = pair_folds(development['pair_ids'], config['training_seed'])
                        development['partitions'] = np.where(inner == 0, 'validation', 'train')
                        query = {k: bank[k][test] for k in ('singles','pair_ids','culture_ids')}
                        contract = dict(base_contract, held_pair_ids=held, acquired_pair_ids=acquired,
                                        information_tier=kind, checkpoint_role='retrospective_sensitivity')
                        if kind == 'transport':
                            contract['transport_mode'] = 'same-context'
                        work = args.local / identity
                        resource_check(args.local, ROOT / 'results/opal_model_series_v2', config)
                        if work.exists():
                            attempt = 1
                            while work.with_name(identity + '-retry-' + str(attempt)).exists():
                                attempt += 1
                            work = work.with_name(identity + '-retry-' + str(attempt))
                        work.mkdir(parents=True)
                        write_json(work / 'contract.json', contract)
                        started, cpu = time.monotonic(), time.process_time()
                        record = {'source':source,'kind':kind,'outer':outer,'query_stratum':query_stratum,
                                  'acquired_unique_pairs':len(acquired),'acquired_pair_stratum_labels':int(dev.sum()),
                                  'outer_rows':int(test.sum()),'held_unique_pairs':len(held),
                                  'per_stratum':{},'unavailable':{},'global_pair_exclusion':True}
                        predictions = {'no-change':np.zeros_like(bank['responses'][test]),'additivity':query['singles'].sum(1)}
                        try:
                            model = OpalGuard(config, menu).fit(development, contract)
                            checkpoint = model.save(work / 'checkpoint')
                            restored = OpalBase.load(checkpoint)
                            for label, variant in variants(model).items():
                                prediction = variant.predict(query, contract)
                                replay = variants(restored)[label].predict(query, contract)
                                difference = float(np.max(np.abs(prediction['predictions']-replay['predictions'])))
                                if difference > config['replay_atol']:
                                    raise ValueError('Sensitivity replay failed')
                                predictions[label] = prediction['predictions']
                                record.setdefault('support',{})[label] = {'unsupported_fraction':float(np.mean(~prediction['support'])), 'replay_max_abs':difference}
                            record['checkpoint'] = {'path':str(checkpoint),'model_sha256':sha256(checkpoint / 'model.json'), 'parameters_sha256':sha256(checkpoint / 'parameters.npz'),
                                                    'derived_variants':'Base and Interaction extracted from saved fitted Guard state, without refitting'}
                        except Unavailable as error:
                            record['unavailable'] = {k:str(error) for k in ['Opal-Base','Opal-Interaction','Opal-Guard']}
                        for stratum in sorted(set(query['culture_ids'])):
                            mask = query['culture_ids'] == stratum
                            base = predictions.get('Opal-Base', predictions['no-change'])[mask]
                            record['per_stratum'][stratum] = {'rows':int(mask.sum()),'metrics':{k:metrics(v[mask],bank['responses'][test][mask],base) for k,v in predictions.items()}}
                        record.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-started)
                        write_json(record_path, record)
                        print(identity,record['acquired_unique_pairs'],record['acquired_pair_stratum_labels'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local',type=Path,default=Path('C:/Users/USER/Desktop/opal-model-series-20261010-v2/sensitivities'))
    parser.add_argument('--output',type=Path,default=ROOT / 'results/opal_model_series_v2/sensitivities')
    run(parser.parse_args())
