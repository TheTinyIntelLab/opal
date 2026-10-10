"""Complete finite acquisition budgets on coherent genetic development banks."""
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
from opal.artifacts import array_hash, fingerprint, read_json, sha256, write_json
from opal.series import OpalBase, OpalGuard, Unavailable, acquisition_order, subset
from opal.replicated.retrospective import pair_folds
from opal.replicated.models import select, fit_candidate, predict_candidate
from run_model_series_bilinear import metrics, variants, resource_check


def run(args):
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    writer_volumes = {path.resolve().drive.rstrip(':') for path in [args.local, args.output]}
    settings.update(checkpoint_array_store=str(args.local / 'array_store'),
                    reserve_bytes={drive: 1073741824 for drive in writer_volumes if drive})
    menu = read_json(ROOT / 'configs/experiments/replicated_fixed_state_v1.json')['menu']
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for source_name in ['norman-log2','wessels-log2','norman-count','wessels-count']:
            source = ROOT / 'data/prepared' / source_name
            with np.load(source / 'profiles.npz', allow_pickle=False) as z:
                profiles, pairs, responses, targets, genes = [z[k] for k in ['singles','pairs','doubles','targets','genes']]
            singles = profiles[pairs].copy()
            for i, p in enumerate(pairs):
                singles[i] = singles[i, np.argsort([str(targets[t]) for t in p])]
            bank = {'singles': singles, 'responses': responses.astype(float),
                    'pair_ids':np.asarray([json.dumps(sorted([str(targets[t]) for t in p]), separators=(',', ':')) for p in pairs]),
                    'culture_ids':np.repeat(source_name, len(pairs))}
            assignment = pair_folds(bank['pair_ids'], settings['outer_seed'])
            old = ROOT / 'results/opal_model_series_v2/retrospective' / source_name
            scope = read_json(old / 'scope_before_fit.json')
            if sha256(source / 'profiles.npz') != scope['profiles_sha256']:
                raise ValueError('Retrospective source changed after freeze')
            directory = args.output / source_name
            directory.mkdir(exist_ok=True)
            freeze = dict(scope, orders=settings['orders'], budgets=settings['budgets'], acquisition_seeds=settings['sampling_seeds'],
                          stability_seeds=settings['stability_seeds'], protected_evaluator_access=0,
                          array_storage='content-addressed numeric rows; no executable deserialization',
                          single_order_amendment='Explicitly align single vectors with sorted canonical components; symmetric score replay verifies earlier source adapter.')
            freeze_path = directory / 'scope_before_fit.json'
            if freeze_path.exists() and read_json(freeze_path) != freeze:
                raise ValueError('Retrospective budget freeze changed')
            write_json(freeze_path, freeze)
            records = []
            for outer in range(3):
                test, dev = assignment == outer, assignment != outer
                query = {k:v[test] for k,v in bank.items() if k != 'responses'}
                population = sorted(set(bank['pair_ids'][dev]))
                # Verify old results, never overwrite their arrays/checkpoints.
                old_row = read_json(old / f'fold-{outer}.json')
                old_contract = read_json(Path(old_row['checkpoints'][0]['path']).parent.parent / 'contract.json')
                alignment_replay = {}
                for cp in old_row['checkpoints']:
                    loaded = OpalBase.load(cp['path'])
                    proper = loaded.predict(query, old_contract)['predictions']
                    with np.load(Path(cp['path']).parent / 'freeze/predictions.npz', allow_pickle=False) as z:
                        difference = float(np.max(np.abs(proper-z['predictions'])))
                    if difference > settings['replay_atol']:
                        raise ValueError('Single-order amendment changes old predictions beyond declared tolerance')
                    alignment_replay[cp['model']] = difference
                for order in settings['orders']:
                    for seed in settings['sampling_seeds']:
                        prefix = acquisition_order(population, order, seed)
                        for budget in settings['budgets']:
                            acquired = sorted(prefix[:int(budget*len(prefix))])
                            seeds = [settings['training_seed']]
                            if budget == 1 and order == 'uniform' and seed == 11:
                                seeds += settings['stability_seeds']
                            for training_seed in seeds:
                                identity = fingerprint({'source':source_name,'outer':outer,'acquired':acquired,'training_seed':training_seed})
                                work = args.local / source_name / identity
                                receipt_path = work / 'receipt.json'
                                # A completed retry is the retained fit for this identity.
                                # Never duplicate it because the original attempt is partial.
                                if not receipt_path.exists():
                                    completed = sorted(work.parent.glob(identity + '-retry-*/receipt.json'))
                                    for prior in args.reuse_local:
                                        previous = prior / source_name / identity
                                        if (previous / 'receipt.json').exists():
                                            completed.append(previous / 'receipt.json')
                                        completed.extend(sorted(previous.parent.glob(identity + '-retry-*/receipt.json')))
                                    if completed:
                                        receipt_path = completed[0]
                                        work = receipt_path.parent
                                condition = {'outer':outer,'order':order,'acquisition_seed':seed,'budget':budget,'training_seed':training_seed,
                                             'source':source_name,'unique_labels':len(acquired),'aggregate_pair_labels':len(acquired),
                                             'held_rows':int(test.sum()),'fit_id':identity,'checkpoint_directory':str(work),
                                             'information_tier':scope['information_tier'],'protected_evaluator_labels':0}
                                if receipt_path.exists():
                                    receipt = read_json(receipt_path)
                                else:
                                    retained = resource_check(args.local, args.output, settings)
                                    if retained > settings['new_artifact_bytes_cap'] - 5000000:
                                        write_json(args.output / 'storage_refusal.json', {'condition':condition,'retained_bytes':retained,
                                                   'reason':'Insufficient artifact headroom to guarantee next full-gene checkpoint; cap unchanged',
                                                   'remaining_work':'uncompleted conditions, deployment refits and final build remain open'})
                                        raise RuntimeError('Full-gene checkpoint artifact headroom exhausted')
                                    if work.exists():
                                        attempt = 1
                                        while work.with_name(identity + f'-retry-{attempt}').exists():
                                            attempt += 1
                                        work = work.with_name(identity + f'-retry-{attempt}')
                                        receipt_path = work / 'receipt.json'
                                        condition['checkpoint_directory'] = str(work)
                                    work.mkdir(parents=True, exist_ok=False)
                                    started, cpu = time.monotonic(), time.process_time()
                                    contract = dict(old_contract, acquired_pair_ids=acquired, held_pair_ids=bank['pair_ids'][test].tolist(),
                                                    checkpoint_role='frozen_retrospective_budget_evaluation')
                                    write_json(work / 'contract.json', contract)
                                    mask = dev & np.isin(bank['pair_ids'], acquired)
                                    development = subset(bank, mask)
                                    development['partitions'] = np.where(assignment[mask] == (outer+1)%3, 'validation', 'train')
                                    config = dict(settings, training_seed=training_seed)
                                    predictions = {'no-change':np.zeros_like(bank['responses'][test]),'additivity':query['singles'].sum(1)}
                                    receipt = {'checkpoints':[],'unavailable':{},'alignment_replay_max_abs':alignment_replay,
                                               'source_profiles_sha256':scope['profiles_sha256'],'fallbacks':{}}
                                    models = {}
                                    if acquired:
                                        predictions['training-mean'] = np.repeat(development['responses'].mean(0,keepdims=True),int(test.sum()),axis=0)
                                        for role in ['reference','candidate']:
                                            try:
                                                chosen = select(development, menu)[role]
                                                fitted = fit_candidate(chosen, development, np.ones(len(development['pair_ids']),bool))
                                                predictions['historical-'+role] = predict_candidate(chosen,fitted,query)
                                            except (ValueError, np.linalg.LinAlgError) as error:
                                                receipt['unavailable']['historical-'+role] = str(error)
                                    if budget == 1 and training_seed == settings['training_seed']:
                                        for cp in old_row['checkpoints']:
                                            model = OpalBase.load(cp['path'])
                                            predictions[cp['model']] = model.predict(query, old_contract)['predictions']
                                        receipt['checkpoints'] = old_row['checkpoints']
                                        receipt['preserved_full_label_refit'] = True
                                    else:
                                        try:
                                            guard = OpalGuard(config, menu).fit(development, contract)
                                            models = variants(guard)
                                        except Unavailable as error:
                                            receipt['unavailable'].update({name:str(error) for name in ['Opal-Base','Opal-Interaction','Opal-Guard']})
                                        for label, model in models.items():
                                            checkpoint = model.save(work / label / 'checkpoint')
                                            result = model.predict(query, contract)
                                            replay = OpalBase.load(checkpoint).predict(query, contract)['predictions']
                                            difference = float(np.max(np.abs(replay-result['predictions'])))
                                            if difference > settings['replay_atol']:
                                                raise ValueError('Retrospective budget replay mismatch')
                                            predictions[label] = result['predictions']
                                            receipt['checkpoints'].append({'model':label,'path':str(checkpoint),'model_sha256':sha256(checkpoint/'model.json'),
                                                                          'parameters_sha256':sha256(checkpoint/'parameters.npz'),'replay_max_abs':difference})
                                            receipt['fallbacks'][label] = float(np.mean(~result['support']))
                                    base = predictions.get('Opal-Base', predictions['no-change'])
                                    receipt['metrics'] = {label:metrics(p,bank['responses'][test],base) for label,p in predictions.items()}
                                    receipt['prediction_hashes'] = {label:array_hash(p) for label,p in predictions.items()}
                                    np.savez_compressed(work / 'per_pair_losses.npz', **{label:np.mean((p-bank['responses'][test])**2,axis=1) for label,p in predictions.items()})
                                    receipt.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-started,
                                                   raw_prediction_storage='Regenerate exactly from safe checkpoints and hashed original bank; per-pair losses retained locally')
                                    write_json(receipt_path, receipt)
                                condition.update(metrics=receipt['metrics'], unavailable=receipt['unavailable'], checkpoints=receipt['checkpoints'])
                                records.append(condition)
                                write_json(directory / 'conditions.json', records)
                                print(source_name,outer,order,seed,budget,training_seed,flush=True)
            write_json(directory / 'completion.json', {'conditions':len(records),'status':'finite acquisition prefixes and full-label stability seeds completed'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local',type=Path,default=Path('D:/opal-model-series-20261010-v2-continuation/retrospective-budgets'))
    parser.add_argument('--output',type=Path,default=ROOT/'results/opal_model_series_v2/retrospective-budgets')
    parser.add_argument('--reuse-local', type=Path, action='append', default=[],
                        help='Read-only prior receipt roots; completed fits are reused without moving files')
    run(parser.parse_args())
