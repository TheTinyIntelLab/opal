"""Freeze and execute the new Horlbeck measured-growth pair experiment."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import environment,load_npz,read_json,sha256,write_json
from opal.models.fitness import FitnessCompletion,GrowthReference


def signature():
    return {name:sha256(ROOT/name) for name in ['src/models/fitness.py','tools/study_fitness.py',
        'tools/prepare_fitness.py','configs/experiments/fitness_completion_20261020.json']}


def assignment(size,seed):
    order=np.random.default_rng(seed).permutation(size)
    labels=np.empty(size,dtype=np.int8)
    labels[order[:int(size*.6)]]=0
    labels[order[int(size*.6):int(size*.8)]]=1
    labels[order[int(size*.8):]]=2
    return labels


def freeze(data,output):
    if output.exists():raise FileExistsError('Preserve prior registrations')
    cfg=read_json(ROOT/'configs/experiments/fitness_completion_20261020.json')
    if cfg['learning_rate']!=.03 or cfg['validation_interval']!=25:
        raise ValueError('Implementation fixes learning rate .03 and validation every 25 steps')
    d=load_npz(data/'profiles.npz');pairs=d['pairs']
    if len({tuple(sorted(p)) for p in pairs.tolist()})!=len(pairs):raise ValueError('Canonical duplicate pairs')
    split=assignment(len(pairs),cfg['seed'])
    support=np.bincount(pairs[split==0].ravel(),minlength=len(d['genes']))>0
    if not np.all(support[pairs]):raise ValueError('Unseen genes would invalidate supported-gene task')
    output.mkdir(parents=True)
    np.savez_compressed(output/'assignment.npz',split=split)
    protocol={'config':cfg,'code':signature(),'data_root':str(data.resolve()),
        'profiles_sha256':sha256(data/'profiles.npz'),'preparation':read_json(data/'preparation.json'),
        'assignment_sha256':sha256(output/'assignment.npz'),'split_counts':{name:int(np.sum(split==i)) for i,name in enumerate(['train','validation','test'])},
        'environment':environment(),'access':'test responses used only in final scoring after all validation selections are frozen',
        'information':'matched singles, train pair growth in both contexts, gene identities; no GI scores, GI correlations, knowledge graphs or expression',
        'scope':'within-source held-pair prediction; new endpoint, not transcriptomic model improvement or independent biological confirmation'}
    write_json(output/'protocol.json',protocol)
    print(json.dumps({'status':'frozen','split_counts':protocol['split_counts'],'hash':sha256(output/'protocol.json')}),flush=True)


def run(output):
    p=read_json(output/'protocol.json');cfg=p['config']
    if signature()!=p['code']:raise ValueError('Frozen code differs')
    if (output/'results.json').exists():raise FileExistsError('Completed experiment is immutable')
    data=Path(p['data_root'])
    if sha256(data/'profiles.npz')!=p['profiles_sha256'] or sha256(output/'assignment.npz')!=p['assignment_sha256']:raise ValueError('Data/assignment drift')
    d=load_npz(data/'profiles.npz');pairs=d['pairs'];y=d['responses'];singles=d['singles'].mean(1)
    split=load_npz(output/'assignment.npz')['split'];train,val,test=[np.flatnonzero(split==i) for i in range(3)]
    scales=np.maximum(y[:,train].std(1),1e-6)
    started=time.perf_counter();cpu=time.process_time();fits=[];peak=0
    def guard():
        nonlocal peak
        mem=psutil.Process().memory_info();peak=max(peak,getattr(mem,'peak_wset',mem.rss))
        if time.perf_counter()-started>cfg['resource_caps']['wall_seconds'] or peak>cfg['resource_caps']['rss_bytes']:
            raise RuntimeError('Registered compute cap reached; retain partial attempt')
    ref_configs=[(kind,1.) for kind in ['zero','additive','minimum']]
    ref_configs += [(kind,alpha) for kind in ['calibration','node'] for alpha in cfg['reference_alphas']]
    refs=[];ref_scores=[]
    for kind,alpha in ref_configs:
        model=GrowthReference(kind,alpha).fit(singles,pairs[train],y[:,train])
        prediction=model.predict(pairs[val])
        ref_scores.append(np.mean(((prediction-y[:,val])/scales[:,None])**2,axis=1));refs.append(model)
        guard()
    ref_scores=np.asarray(ref_scores)
    chosen=np.argmin(ref_scores,axis=0)
    # Freeze one validation-selected measured-single reference per context.
    base=np.asarray([refs[i].predict(pairs)[c] for c,i in enumerate(chosen)])
    reference_selection=[{'context':str(d['contexts'][c]),'kind':ref_configs[i][0],'alpha':ref_configs[i][1]} for c,i in enumerate(chosen)]
    candidates=[]
    for mode in ['independent','pooled','joint']:
        for rank in cfg['ranks']:
            for penalty in cfg['penalties']:
                model=FitnessCompletion(mode,rank,penalty,cfg['steps'],cfg['seed']).fit(
                    pairs[train],y[:,train]-base[:,train],len(d['genes']),scales,
                    validation=(pairs[val],y[:,val]-base[:,val]))
                validation=base[:,val]+model.predict(pairs[val])
                scores=np.mean(((validation-y[:,val])/scales[:,None])**2,axis=1)
                path=output/f'{mode}-r{rank}-p{penalty}.npz';model.save(path)
                record={'mode':mode,'rank':rank,'penalty':penalty,'best_step':model.best_step,
                    'validation_normalized_mse':scores.tolist(),'parameters':model.parameter_count,
                    'checkpoint':path.name,'checkpoint_sha256':sha256(path),'history':model.history}
                candidates.append((model,record))
                guard();fits.append(record)
                write_json(output/'fit_progress.json',{'fits':fits,'peak_rss_bytes':peak,'elapsed_seconds':time.perf_counter()-started})
                print(mode,rank,penalty,'validation',scores.tolist(),'seconds',round(time.perf_counter()-started,2),flush=True)
    joint_indices=[i for i,(_,r) in enumerate(candidates) if r['mode']=='joint']
    winner=min(joint_indices,key=lambda i:np.mean(candidates[i][1]['validation_normalized_mse']))
    baseline_predictions=[ref.predict(pairs[test]) for ref in refs]
    baseline_labels=[{'family':kind,'alpha':alpha} for kind,alpha in ref_configs]
    baseline_val=list(ref_scores)
    for model,r in candidates:
        if r['mode']=='joint':continue
        baseline_predictions.append(base[:,test]+model.predict(pairs[test]))
        baseline_labels.append({k:r[k] for k in ['mode','rank','penalty','best_step','checkpoint']})
        baseline_val.append(r['validation_normalized_mse'])
    best_baselines=np.argmin(np.asarray(baseline_val),axis=0)
    selection={'reference':reference_selection,'candidate':candidates[winner][1],
               'baseline_per_context':[baseline_labels[i] for i in best_baselines],
               'baseline_validation_normalized_mse':np.asarray(baseline_val).tolist()}
    # No test metric was computed above. Persist model choices BEFORE scoring.
    write_json(output/'selection.json',selection)
    predicted=base[:,test]+candidates[winner][0].predict(pairs[test])
    baseline=np.asarray([baseline_predictions[i][c] for c,i in enumerate(best_baselines)])
    rows=[];rng=np.random.default_rng(cfg['seed']);promote=[]
    for c,context in enumerate(d['contexts']):
        a=(predicted[c]-y[c,test])**2;b=(baseline[c]-y[c,test])**2
        delta=a-b
        # Bounded-memory conditional pair bootstrap, not culture uncertainty.
        boot=np.asarray([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(cfg['bootstrap_draws'])])
        ci=np.quantile(boot,[.025,.975]).tolist();gain=1-float(a.mean()/b.mean())
        promote.append(gain>=.02 and ci[1]<0)
        rows.append({'context':str(context),'candidate_mse':float(a.mean()),'baseline_mse':float(b.mean()),'relative_gain':gain,
            'loss_difference_conditional_95':ci,'candidate_pearson':float(np.corrcoef(predicted[c],y[c,test])[0,1]),
            'baseline_pearson':float(np.corrcoef(baseline[c],y[c,test])[0,1]),
            'replicate_mse':float(np.mean((d['replicates'][c,0,test]-d['replicates'][c,1,test])**2)),
            'all_reference_mse':[{**label,'mse':float(np.mean((prediction[c]-y[c,test])**2))} for label,prediction in zip(baseline_labels,baseline_predictions)]})
    model=candidates[winner][0];reloaded=FitnessCompletion.load(output/candidates[winner][1]['checkpoint'])
    replay=float(np.max(np.abs(model.predict(pairs[test])-reloaded.predict(pairs[test]))))
    if replay!=0:raise RuntimeError('Checkpoint replay differs')
    np.savez_compressed(output/'predictions.npz',candidate=predicted,baseline=baseline,test_indices=test)
    # Save the actual reference predictions needed to reproduce the residual
    # checkpoint's full growth output, with context/gene schema, outside git.
    np.savez_compressed(output/'reference_and_schema.npz',reference_matrix=base,genes=d['genes'],contexts=d['contexts'],pairs=pairs,singles=singles)
    guard()
    size=sum(f.stat().st_size for f in output.rglob('*') if f.is_file())
    if size>cfg['resource_caps']['experiment_bytes']-100000:raise RuntimeError('Artifact cap reached')
    result={'classification':'PROMISING' if all(promote) else 'BASELINE-LEVEL' if np.mean([r['relative_gain'] for r in rows])>=0 else 'FAILED HYPOTHESIS',
        'protocol_sha256':sha256(output/'protocol.json'),'selection':selection,'contexts':rows,
        'primary_candidate_normalized_mse':float(np.mean([r['candidate_mse']/s**2 for r,s in zip(rows,scales)])),
        'primary_baseline_normalized_mse':float(np.mean([r['baseline_mse']/s**2 for r,s in zip(rows,scales)])),
        'training_target_scales':scales.tolist(),'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
        'peak_rss_bytes':peak,'artifact_bytes_before_receipt':size,'checkpoint_replay_max_abs':replay,
        'all_fits':fits,'scope':p['scope'],'uncertainty':'conditional on deposited gene-pair population and shared references, not independent-culture calibration'}
    write_json(output/'results.json',result)
    print(json.dumps({k:result[k] for k in ['classification','contexts','wall_seconds','peak_rss_bytes']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--data',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        freeze(args.data,args.output) if args.action=='freeze' else run(args.output)
