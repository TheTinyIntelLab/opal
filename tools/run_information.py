"""Finite exposed-data development; freeze predictions before evaluator access."""
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import array_hash, read_json, sha256, write_json
from opal.information import fit_head, load_head, predict_head, save_head, split_masks
from opal.series import OpalBase, Unavailable
from opal.series_bilinear import FIT_CACHE, select_reference, predict_reference, weights, pca, encode, main_design, cross_features
from prepare_information import OUT, LOCAL, resource_check
from run_model_series_bilinear import metrics


def sliced(bank, mask):
    return {k: v[mask] for k, v in bank.items()}


def diagnostics(bank, config, reference):
    w = weights(bank)
    design = main_design(config, reference, bank)
    record = {'pairs': len(bank['pair_ids']), 'reference_kind': config['kind'], 'exact_main_design_available': design is not None}
    if design is None:
        return record
    a = np.sqrt(w)[:, None] * design
    sv = np.linalg.svd(a, compute_uv=False)
    record.update(main_columns=a.shape[1], main_rank=int(np.linalg.matrix_rank(a)), main_singular_values=sv.tolist(),
                  main_condition=float(sv[0]/sv[-1]) if sv[-1] > 1e-15 else None)
    record['interaction'] = []
    for rank in (4, 8):
        try:
            encoder = pca(bank['singles'], rank)
            f = cross_features(encode(bank['singles'], encoder))
            before = np.sqrt(w)[:, None]*f
            after = before - a @ np.linalg.lstsq(a, before, rcond=None)[0]
            energy = float(np.sum(after**2)/np.sum(before**2)) if np.sum(before**2) else None
            record['interaction'].append({'rank': rank, 'before_rank': int(np.linalg.matrix_rank(before)),
                'after_rank': int(np.linalg.matrix_rank(after)), 'retained_energy': energy,
                'before_singular_values': np.linalg.svd(before, compute_uv=False).tolist(),
                'after_singular_values': np.linalg.svd(after, compute_uv=False).tolist()})
        except Unavailable as error:
            record['interaction'].append({'rank': rank, 'unavailable': str(error)})
    return record


def run():
    cfg = read_json(ROOT / 'configs/experiments/opal_information_v1.json')
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    menu = read_json(ROOT / 'configs/experiments/replicated_fixed_state_v1.json')['menu']
    sources = read_json(OUT / 'measurement/summaries.json')
    freeze = {'configuration_sha256': sha256(ROOT / 'configs/experiments/opal_information_v1.json'),
        'protocol_sha256': sha256(ROOT / 'docs/next_cycle/opal_information_protocol.md'),
        'code': {str(p): sha256(ROOT/p) for p in ['src/information.py', 'src/series.py', 'src/series_bilinear.py', 'tools/run_information.py']},
        'sources': sources, 'role': 'exposed development, not sealed confirmation', 'utc_order': 'written before first fit/score'}
    frozen = OUT / 'development/execution_freeze.json'
    if frozen.exists():
        raise FileExistsError('Do not reopen completed or partial study without preserved amendment')
    write_json(frozen, freeze)
    configs = [None]+[dict(input_rank=i, output_rank=o, alpha=a) for i,o,a in itertools.product(cfg['input_ranks'], cfg['output_ranks'], cfg['penalties'])]
    records, ranks = [], []
    with threadpool_limits(limits=1):
        for source in sources:
            if sha256(source['bank']) != source['bank_sha256'] or sha256(source['summary']) != source['summary_sha256']:
                raise ValueError('Source changed after preparation')
            with np.load(source['bank'], allow_pickle=False) as z:
                original = {k: z[k] for k in ['singles', 'responses', 'pair_ids', 'culture_ids']}
            with np.load(source['summary'], allow_pickle=False) as z:
                summary = {k: z[k] for k in z.files}
            contract = {'source': source['source'], 'genes': source['genes'], 'units': source['units'],
                'assay': 'PertSpectra single-cell genetic aggregate', 'reference_provenance': source['summary_sha256'],
                'split_lineage': sha256(frozen), 'context_contract': source['source']}
            for precision in (False, True):
                bank = dict(original)
                if precision:
                    bank['singles'] = summary['precision_singles']
                means = summary['precision_projected' if precision else 'means_projected']
                for split in cfg['splits']:
                    for fold in range(3):
                        trainmask, testmask = split_masks(bank['pair_ids'], split, fold)
                        for stratum in sorted(set(bank['culture_ids'])):
                            trainids = np.flatnonzero(trainmask & (bank['culture_ids'] == stratum))
                            testids = np.flatnonzero(testmask & (bank['culture_ids'] == stratum))
                            if not len(testids):
                                continue
                            identity = f"{source['source']}-{split}-{fold}-{stratum}-{'precision' if precision else 'original'}"
                            destination = LOCAL / 'evaluation' / identity
                            resource_check(2000000)
                            destination.mkdir(parents=True, exist_ok=False)
                            train, query = sliced(bank, trainids), sliced(bank, testids)
                            # Query object never contains outcomes during prediction.
                            truth = query.pop('responses')
                            started = time.monotonic()
                            FIT_CACHE.clear()
                            base = OpalBase(settings, menu).fit(train, contract)
                            base.save(destination/'base')
                            basepred = predict_reference(base.config, base.fitted['reference'], query)
                            # Reload complete safe Base, including its shared/reference state.
                            from opal import load_model
                            reloaded = load_model(destination/'base')
                            if not np.allclose(predict_reference(reloaded.config, reloaded.fitted['reference'], query), basepred, atol=1e-10, rtol=0):
                                raise ValueError('Base replay failed')
                            if not precision:
                                ranks.append(dict(id=identity, **diagnostics(train, base.config, base.fitted['reference'])))
                            inner = []
                            for innerfold in range(3):
                                it, iv = split_masks(train['pair_ids'], split, innerfold)
                                ib, vb = sliced(train,it), sliced(train,iv)
                                try:
                                    if len(ib['pair_ids']) < 12 or len(vb['pair_ids']) < 4:
                                        raise Unavailable('Inner split insufficient support')
                                    bc, br, bs = select_reference(ib, settings, menu)
                                    inner.append((it,iv, ib['responses']-predict_reference(bc,br,ib),predict_reference(bc,br,vb),None,bs))
                                except Unavailable as error:
                                    inner.append((it,iv,None,None,str(error),None))
                            predictions = {'Base': basepred}
                            armrecords = {}
                            for arm in cfg['arms']:
                                variance_key = ('precision_' if precision else '') + ('shuffle' if arm == 'shuffle' else 'variance')
                                variance = summary[variance_key]
                                attempts = []
                                for candidate in configs:
                                    losses, failures = [], []
                                    for it,iv,residual,bp,failed,_ in inner:
                                        if failed:
                                            failures.append(failed)
                                            continue
                                        head = None
                                        try:
                                            if candidate is not None:
                                                head = fit_head(means[trainids][it],variance[trainids][it],residual,weights(sliced(train,it)),candidate,arm)
                                        except Unavailable as error:
                                            failures.append(str(error))
                                        # Unsupported configs remain zero corrections on the same rows.
                                        pred = bp + predict_head(head, means[trainids][iv],variance[trainids][iv],truth.shape[1])
                                        losses.append(float(weights(sliced(train,iv)) @ np.mean((pred-train['responses'][iv])**2,axis=1)))
                                    attempts.append({'config':candidate,'validation_mse':float(np.mean(losses)) if losses else None,'failures':failures})
                                eligible = [a for a in attempts if a['validation_mse'] is not None and not a['failures']]
                                chosen = min(eligible,key=lambda a:(a['validation_mse'],a['config'] is not None,json.dumps(a['config'],sort_keys=True)))['config'] if eligible else None
                                head, refusal = None, None
                                try:
                                    if chosen is not None:
                                        head=fit_head(means[trainids],variance[trainids],train['responses']-predict_reference(base.config,base.fitted['reference'],train),weights(train),chosen,arm)
                                except Unavailable as error:
                                    refusal=str(error)
                                correction=predict_head(head,means[testids],variance[testids],truth.shape[1])
                                save_head(head,destination/arm,contract)
                                replay=predict_head(load_head(destination/arm,contract),means[testids],variance[testids],truth.shape[1])
                                if not np.allclose(replay,correction,atol=1e-10,rtol=0):
                                    raise ValueError('Head replay failed')
                                predictions[arm]=basepred+correction
                                armrecords[arm]={'attempts':attempts,'chosen':chosen,'refusal':refusal,
                                    'numeric_values':read_json(destination/arm/'model.json')['numeric_values'],
                                    'checkpoint_bytes':sum(p.stat().st_size for p in (destination/arm).iterdir()),
                                    'fallback_rows':int(np.sum(~np.isfinite(variance[testids]).all((1,2)))) if arm in ('variance','shuffle') else 0,
                                    'replay_max_abs':float(np.max(np.abs(replay-correction)))}
                            # Freeze all predictions and their order before evaluation.
                            np.savez_compressed(destination/'predictions.npz',**predictions,pair_ids=query['pair_ids'])
                            predhash=sha256(destination/'predictions.npz')
                            write_json(destination/'freeze.json',{'prediction_sha256':predhash,'contract':contract,'train_pairs':train['pair_ids'].tolist(),'held_pairs':query['pair_ids'].tolist(),'role':'development_evaluation'})
                            results={arm:metrics(pred,truth,basepred) for arm,pred in predictions.items()}
                            error=basepred-truth
                            correction=predictions['variance']-basepred
                            attribution={'base_error_mse':float(np.mean(error**2)),'correction_mse':float(np.mean(correction**2)),
                                'twice_error_correction':float(2*np.mean(error*correction)),
                                'variance_error_mse':float(np.mean((error+correction)**2)),
                                'same_anchor_residual_mse_is_response_mse':True}
                            calibration={name:metrics(p,truth,basepred) for name,p in {'exact':truth,'constant':np.broadcast_to(train['responses'].mean(0),truth.shape),
                                'identity_shuffle':truth[np.random.default_rng(cfg['shuffle_seed']).permutation(len(truth))]}.items()}
                            record={'id':identity,'source':source['source'],'split':split,'fold':fold,'stratum':str(stratum),'precision':precision,
                                'training_pairs':len(trainids),'held_pairs':len(testids),'train_pair_ids':train['pair_ids'].tolist(),
                                'held_pair_ids':query['pair_ids'].tolist(),'pair_representation_labels':0,
                                'validation_labels_in_budget':True,'inner_train_validation_counts':[[int(it.sum()),int(iv.sum())] for it,iv,*_ in inner],
                                'base_selection':base.selection,'inner_base_selections':[entry[-1] for entry in inner],
                                'arms':armrecords,'metrics':results,'error_attribution':attribution,'metric_controls':calibration,
                                'prediction_sha256':predhash,'checkpoint':str(destination),'wall_seconds':time.monotonic()-started,
                                'checkpoint_total_bytes':sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),'independent_cultures':False}
                            write_json(OUT/'development/folds'/ (identity+'.json'),record)
                            records.append(record)
                            print(identity,results['variance']['mse'],flush=True)
    aggregates=[]
    for source,split,precision in itertools.product(cfg['sources'],cfg['splits'],[False,True]):
        rows=[r for r in records if r['source']==source and r['split']==split and r['precision']==precision]
        values={arm:float(np.mean([sum(r['held_pairs']*r['metrics'][arm]['mse'] for r in rows if r['stratum']==s)/sum(r['held_pairs'] for r in rows if r['stratum']==s) for s in sorted({r['stratum'] for r in rows})])) for arm in ['Base']+cfg['arms']}
        gains={arm:100*(1-values['variance']/values[arm]) for arm in ['Base','mean','capacity_mean','shuffle']}
        aggregates.append({'source':source,'split':split,'precision':precision,'mse':values,'variance_gain_percent':gains,
            'triage_pass':all(g>=2 for g in gains.values()),'profiles':sum(r['held_pairs'] for r in rows)})
    gate=all(r['triage_pass'] for r in aggregates if r['split']=='partner')
    write_json(OUT/'development/summary.json',{'aggregates':aggregates,'information_gate_passed':gate,
        'selected_procedure':'variance frozen procedure' if gate else 'retain existing Base reference',
        'scientific_verdict':'development triage positive; confirmation required' if gate else 'complete_negative: bounded variance representation failed administrative triage',
        'confirmation_executed':False,'fold_records':len(records),'candidate_configs_per_arm':13,
        'new_independent_cultures':0,'unique_pairs':{'TNFA-positive':108,'TNFA-negative':52}})
    write_json(OUT/'diagnostics/ranks.json',ranks)


if __name__=='__main__':
    run()
