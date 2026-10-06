"""Nested, development-only comparisons of separately registered architectures."""
from pathlib import Path
import argparse
import sys
import time
import zipfile
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import read_json,write_json,load_npz,sha256,source_hash,fingerprint
from opal.models.advanced import AdvancedResidual, signal_weights
from opal.models.kernel import KernelResidual,prediction_error,smoother
from opal.inference import export_members,Predictor
from tools.improve_residual_model import fold_sets,candidates,select


def configurations(epochs):
    common={'epochs':epochs,'width':32,'rank':8,'alpha':10,'expert':'global','decay':.01}
    variants=[('response_mlp',{}),('noise_only',{'noise':True}),
        ('prior_attention',{'priors':True}),('prior_mean_pool',{'priors':True,'attention':False}),
        ('prior_noise',{'priors':True,'noise':True}),
        ('structured',{'priors':True,'noise':True,'structured':True}),
        ('signal_loss',{'priors':True,'noise':True,'signal_loss':1}),
        ('full',{'priors':True,'noise':True,'structured':True,'signal_loss':1}),
        ('shuffled_prior',{'priors':True,'noise':True,'structured':True,'signal_loss':1,'shuffle':True}),
        ('larger_global_rank',{'rank':16}),('stronger_shrinkage',{'priors':True,'noise':True,'structured':True,'signal_loss':1,'coefficient_penalty':.1}),
        ('stronger_expert',{'priors':True,'noise':True,'structured':True,'signal_loss':1,'alpha':100})]
    return [{'label':label,**common,**extra} for label,extra in variants]


def resource_variant(resources,config,seed):
    result=resources
    if config.get('shuffle'):
        result={**resources,'bank':resources['bank'].copy(),'source_mask':resources['source_mask'].copy()}
        order=np.random.default_rng(seed).permutation(len(resources['bank']))
        result['bank'][:,1:]=resources['bank'][order,1:]
        result['source_mask'][:,1:]=resources['source_mask'][order,1:]
    return result


def prior_ridge(singles,resources,pairs,doubles,train,val,alpha):
    bank=resources['bank'].reshape(len(singles),-1).astype(np.float64)
    # Same target information as fusion, with a linear symmetric pair map.
    a,b=bank[pairs[:,0]],bank[pairs[:,1]]
    features=np.concatenate([a+b,np.abs(a-b),a*b],axis=1)
    center=features[train].mean(0);scale=np.maximum(features[train].std(0),1e-5)
    x=(features-center)/scale
    additive=singles[pairs[:,0]]+singles[pairs[:,1]]
    residual=doubles-additive;mean=residual[train].mean(0)
    kernel=x@x.T/x.shape[1]
    weights=solve(kernel[np.ix_(train,train)]+alpha*np.eye(len(train)),residual[train]-mean,assume_a='pos')
    return additive[val]+mean+kernel[np.ix_(val,train)]@weights


def score(prediction,truth,variance):
    weights=signal_weights(truth,variance,1)
    return {'mse':float(np.mean((prediction-truth)**2)),
            'bounded_signal_mse':float(np.mean(weights*(prediction-truth)**2))}


def study(name,root,seed,epochs,outer_count,inner_count):
    output=root/name
    if (output/'results.json').exists():
        return read_json(output/'results.json')
    if output.exists():raise FileExistsError('Incomplete immutable study exists: '+str(output))
    output.mkdir(parents=True)
    started=time.perf_counter()
    dataset=ROOT/'data/prepared'/name
    data=load_npz(dataset/'profiles.npz')
    resources=load_npz(ROOT/'data/architecture_resources'/name/'resources.npz')
    if not np.array_equal(data['targets'],resources['targets']) or not np.array_equal(data['genes'],resources['genes']):
        raise ValueError('Prior/measurement resource order mismatch')
    singles=data['singles'];pairs=data['pairs'];doubles=data['doubles']
    residual=doubles-singles[pairs[:,0]]-singles[pairs[:,1]]
    gram=residual@residual.T/residual.shape[1]
    configs=configurations(epochs)
    kernels=candidates(singles,pairs)
    protocol={'seed':seed,'outer_folds':outer_count,'inner_folds':inner_count,'configurations':configs,
        'kernel_candidate_count':len(kernels),'prior_ridge_penalties':[.1,1,10,100],
        'mixtures':[.25,1.],'scope':'Exploratory nested development only; historical test outcomes known',
        'endpoint':'equal-pair all-gene MSE; bounded-signal metric secondary, evaluation truth used only after prediction',
        'resource_sha256':sha256(ROOT/'data/architecture_resources'/name/'resources.npz'),
        'profiles_sha256':sha256(dataset/'profiles.npz'),'source_hash':source_hash(),
        'runner_sha256':sha256(__file__),'evaluator_access':False,
        'negative_result_policy':'All ablations and both split protocols recorded; no model family forced'}
    write_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted((ROOT/'src').rglob('*.py')):archive.write(file,str(file.relative_to(ROOT)))
        for file in ['tools/study_architecture.py','tools/improve_residual_model.py','tools/prepare_architecture.py']:
            archive.write(ROOT/file,file)
    all_results={}
    for kind in ['random','partner']:
        folds=fold_sets(pairs,kind,seed,outer_count)
        records=[]; predictions={};counts=np.zeros(len(pairs),int); totals={};weighted={}
        for f,(train,val) in enumerate(folds):
            inner=fold_sets(pairs[train],kind,seed+f+1,inner_count)
            if len(inner)<2:raise ValueError('Insufficient feasible inner folds; no redraw')
            winner,kernel_search=select(kernels,gram,train,inner)
            kc=kernels[winner['index']][0]
            kernel=KernelResidual(singles,kc).fit(pairs[train],doubles[train])
            family_predictions={'kernel':kernel.predict(pairs[val]),'additive':singles[pairs[val,0]]+singles[pairs[val,1]]}
            search=[]
            for index,config in enumerate(configs):
                errors={.25:[],1.:[]};sizes=[]
                variant=resource_variant(resources,config,seed+719)
                for j,(ti,vi) in enumerate(inner):
                    t,v=train[ti],train[vi]
                    model=AdvancedResidual(singles,variant,config).fit(pairs[t],doubles[t],resources['double_variance'][t],seed+index*97+j)
                    simple=model.predict(pairs[v],0);complex_=model.predict(pairs[v],1)
                    for mixture in errors:
                        prediction=simple+mixture*(complex_-simple)
                        errors[mixture].append(float(np.mean((prediction-doubles[v])**2)))
                    sizes.append(len(v))
                for mixture,error in errors.items():
                    search.append({'config':{**config,'mixture':mixture},'mse':float(np.average(error,weights=sizes))})
            ridge_search=[]
            for alpha in [.1,1,10,100]:
                errors=[];sizes=[]
                for ti,vi in inner:
                    t,v=train[ti],train[vi]
                    pred=prior_ridge(singles,resources,pairs,doubles,t,v,alpha)
                    errors.append(float(np.mean((pred-doubles[v])**2)));sizes.append(len(v))
                ridge_search.append({'alpha':alpha,'mse':float(np.average(errors,weights=sizes))})
            ridge=min(ridge_search,key=lambda r:r['mse'])
            family_predictions['prior_ridge']=prior_ridge(singles,resources,pairs,doubles,train,val,ridge['alpha'])
            best=min(search,key=lambda r:r['mse'])
            # Report each predeclared ablation with its mixture tuned on inner folds.
            chosen_by_label={label:min([r for r in search if r['config']['label']==label],key=lambda r:r['mse'])
                             for label in [c['label'] for c in configs]}
            fitted={}
            for index,(label,chosen) in enumerate(chosen_by_label.items()):
                config=chosen['config']
                model=AdvancedResidual(singles,resource_variant(resources,config,seed+719),config).fit(
                    pairs[train],doubles[train],resources['double_variance'][train],seed+index*97+997)
                fitted[label]=model
                family_predictions[label]=model.predict(pairs[val],config['mixture'])
            family_predictions['adaptive_neural']=family_predictions[best['config']['label']]
            eligible=[{'family':'kernel','mse':winner['mse']},{'family':'prior_ridge','mse':ridge['mse']},
                      {'family':'adaptive_neural','mse':best['mse']}]
            competitive=min(eligible,key=lambda r:r['mse'])
            family_predictions['competitive']=family_predictions[competitive['family']]
            metrics={}
            for key,pred in family_predictions.items():
                metrics[key]=score(pred,doubles[val],resources['double_variance'][val])
                totals[key]=totals.get(key,0)+metrics[key]['mse']*len(val)
                weighted[key]=weighted.get(key,0)+metrics[key]['bounded_signal_mse']*len(val)
                predictions.setdefault(key,np.zeros_like(doubles,dtype=np.float64))[val]+=pred
            counts[val]+=1
            record={'fold':f,'train':train.tolist(),'validation':val.tolist(),
                'inner_splits':[{'train':train[t].tolist(),'validation':train[v].tolist()} for t,v in inner],
                'kernel_selection':winner,'kernel_search':kernel_search,'prior_ridge_search':ridge_search,
                'neural_selection':best,'neural_search':search,'competitive_selection':competitive,
                'metrics':metrics,'training_logs':{label:model.training_log for label,model in fitted.items()}}
            records.append(record)
            write_json(output/f'{kind}_fold_{f}.json',record)
            print(name,kind,f,'competitive',competitive['family'],'MSE',metrics['competitive']['mse'],flush=True)
        covered=counts>0;occurrences=sum(len(v) for _,v in folds)
        for pred in predictions.values():pred[covered]/=counts[covered,None]
        metrics={key:{'mse_per_occurrence':totals[key]/occurrences,
                      'bounded_signal_mse_per_occurrence':weighted[key]/occurrences,
                      'mse_averaged_predictions':float(np.mean((pred[covered]-doubles[covered])**2))}
                 for key,pred in predictions.items()}
        np.savez_compressed(output/f'{kind}_predictions.npz',**predictions,truth=doubles,coverage=counts,
                            pair_indices=data['pair_indices'],pairs=pairs,genes=data['genes'])
        all_results[kind]={'metrics':metrics,'covered_pairs':int(covered.sum()),'holdout_occurrences':occurrences,
                           'records':records,'dependent_partner_folds':kind=='partner'}
    # Freeze a deployment candidate using the inner folds of the first random split,
    # never outer scores. This rule is conservative and declared before the study.
    selection_record=all_results['random']['records'][0]
    chosen=selection_record['competitive_selection'];indices=np.arange(len(pairs))
    if chosen['family']=='kernel':
        config=selection_record['kernel_selection']['config'];model=KernelResidual(singles,config).fit(pairs,doubles)
        member={'type':'kernel','model':model,'config':config,'seed':seed}
    elif chosen['family']=='adaptive_neural':
        config=selection_record['neural_selection']['config']
        model=AdvancedResidual(singles,resource_variant(resources,config,seed+719),config).fit(pairs,doubles,resources['double_variance'],seed)
        member={'type':'advanced','model':model,'config':config,'seed':seed}
    else:
        from opal.models.experts import FeatureRidgeResidual
        config={'alpha':min(selection_record['prior_ridge_search'],key=lambda r:r['mse'])['alpha']}
        model=FeatureRidgeResidual(singles,resources['bank'],config['alpha']).fit(pairs,doubles)
        member={'type':'feature_ridge','model':model,'config':config,'seed':seed}
    write_json(output/'selection.json',{'rule':'first random fold inner-CV procedure; no outer-score selection',
        'family':chosen['family'],'config':config,'historical_evaluator_access':False})
    if member is not None:
        context=read_json(dataset/'source_manifest.json')['context']
        export_members([member],data,output/'checkpoint',{'context':context,'scale':'log2','study':'architecture-v1',
            'status':'development-only exploratory candidate','uncertainty':'uncalibrated'})
        query=data['query_pairs'];pred=model.predict(query,config.get('mixture',1)) if member['type']=='advanced' else model.predict(query)
        np.savez_compressed(output/'unscored_query_predictions.npz',predictions=pred,pairs=query)
        exported=Predictor(output/'checkpoint')
        reconstructed=np.stack([exported.predict(str(data['targets'][a]),str(data['targets'][b]))['effect'] for a,b in query])
        discrepancy=float(np.max(np.abs(pred-reconstructed)))
        if discrepancy>2e-5:raise ValueError('Export disagrees with fitted predictions')
        write_json(output/'export_verification.json',{'maximum_absolute_difference':discrepancy,'query_predictions_scored':False})
    all_results['seconds']=time.perf_counter()-started
    all_results['evaluator_access']=False
    all_results['protocol_sha256']=sha256(output/'protocol.json')
    write_json(output/'results.json',all_results)
    return all_results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/architecture/v1'))
    parser.add_argument('--datasets',nargs='+',default=['norman-log2','wessels-log2','wessels-expanded-log2'])
    parser.add_argument('--seed',type=int,default=20261005)
    parser.add_argument('--epochs',type=int,default=120)
    parser.add_argument('--outer-folds',type=int,default=5)
    parser.add_argument('--inner-folds',type=int,default=3)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        summary={name:study(name,args.output,args.seed,args.epochs,args.outer_folds,args.inner_folds) for name in args.datasets}
    write_json(args.output/'summary.json',{name:{kind:record['metrics'] for kind,record in results.items() if kind in ['random','partner']} for name,results in summary.items()})


if __name__=='__main__':main()
