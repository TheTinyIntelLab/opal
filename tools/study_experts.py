"""Development-only GP, retrieval and optional independent-transfer ablations."""
from pathlib import Path
import argparse,sys,time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import load_npz,write_json,read_json,sha256,source_hash
from opal.models.experts import MeasurementGP,retrieve_residual
from opal.models.advanced import AdvancedResidual
from tools.improve_residual_model import fold_sets,candidates,select
from opal.models.kernel import KernelResidual


def study(name,output,transfer,seed,bounded=False):
    folder=output/name
    if (folder/'results.json').exists():return read_json(folder/'results.json')
    if folder.exists():raise FileExistsError('Incomplete expert study exists')
    folder.mkdir(parents=True)
    data=load_npz(ROOT/'data/prepared'/name/'profiles.npz')
    prefix='enriched-' if transfer else ''
    resources=load_npz(ROOT/'data/architecture_resources'/(prefix+name)/'resources.npz')
    singles,pairs,doubles=data['singles'],data['pairs'],data['doubles']
    residual=doubles-singles[pairs[:,0]]-singles[pairs[:,1]];gram=residual@residual.T/residual.shape[1]
    configs=[{'alpha':a,'bandwidth':b} for a in [.1,1,10] for b in [.25,1,4]]
    common={'epochs':120,'width':32,'rank':8,'alpha':10,'priors':True,'noise':True,'structured':True,'signal_loss':1}
    neural=[{'label':'no_transfer',**common},{'label':'transfer_encoder',**common,'transfer_encoder':True},
            {'label':'transfer_decoder',**common,'transfer_decoder':True},{'label':'transfer_both',**common,'transfer':True},
            {'label':'linear_mse',**common,'linear':True,'signal_loss':0},
            {'label':'linear_weighted',**common,'linear':True}]
    if bounded:neural=[{**c,'linear_clip':8,'label':c['label']+'_bounded'} for c in neural[-2:]]
    from zipfile import ZipFile,ZIP_DEFLATED
    with ZipFile(folder/'source_snapshot.zip','w',ZIP_DEFLATED) as archive:
        for path in (ROOT/'src').rglob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
        archive.write(__file__,'tools/study_experts.py')
    write_json(folder/'protocol.json',{'seed':seed,'outer_folds':5,'inner_folds':3,'GP':configs,'retrieval_neighbors':[1,3,5],
        'transfer_configs':neural if transfer else [],'scope':'exploratory development only; historical outcomes known',
        'evaluator_access':False,'resources_sha256':sha256(ROOT/'data/architecture_resources'/(prefix+name)/'resources.npz'),
        'source_hash':source_hash(),'script_sha256':sha256(__file__),
        'uncertainty':'conditional fixed-kernel latent GP variance; measurement and model selection effects not fully propagated'})
    kernels=candidates(singles,pairs);result={};started=time.perf_counter()
    for kind in ['random','partner']:
        totals={};records=[];counts=np.zeros(len(pairs),int);predictions={};std_predictions=np.zeros_like(doubles)
        for f,(train,val) in enumerate(fold_sets(pairs,kind,seed,5)):
            inner=fold_sets(pairs[train],kind,seed+f+1,3)
            if len(inner)<2:raise ValueError('Insufficient inner folds')
            kernel_selection,_=select(kernels,gram,train,inner)
            kernel=KernelResidual(singles,kernel_selection['config']).fit(pairs[train],doubles[train])
            outer={'kernel':kernel.predict(pairs[val])}
            search=[]
            for config in configs:
                errors=[];sizes=[]
                for ti,vi in inner:
                    t,v=train[ti],train[vi]
                    model=MeasurementGP(singles,resources['bank'],**config).fit(pairs[t],doubles[t],resources['residual_variance'][t])
                    errors.append(float(np.mean((model.predict(pairs[v])-doubles[v])**2)));sizes.append(len(v))
                search.append({'config':config,'mse':float(np.average(errors,weights=sizes))})
            best=min(search,key=lambda x:x['mse'])
            gp=MeasurementGP(singles,resources['bank'],**best['config']).fit(pairs[train],doubles[train],resources['residual_variance'][train])
            outer['GP'],std=gp.predict(pairs[val],return_std=True)
            retrieval=[]
            for neighbors in [1,3,5]:
                errors=[];sizes=[]
                for ti,vi in inner:
                    t,v=train[ti],train[vi]
                    pred=retrieve_residual(singles,resources['bank'],pairs,doubles,t,v,neighbors)
                    errors.append(float(np.mean((pred-doubles[v])**2)));sizes.append(len(v))
                retrieval.append({'neighbors':neighbors,'mse':float(np.average(errors,weights=sizes))})
            neighbor=min(retrieval,key=lambda x:x['mse'])
            outer['retrieval']=retrieve_residual(singles,resources['bank'],pairs,doubles,train,val,neighbor['neighbors'])
            transfer_search=[]
            if transfer:
                for index,config in enumerate(neural):
                    errors=[];sizes=[]
                    for j,(ti,vi) in enumerate(inner):
                        t,v=train[ti],train[vi]
                        model=AdvancedResidual(singles,resources,config).fit(pairs[t],doubles[t],resources['double_variance'][t],seed+97*j)
                        errors.append(float(np.mean((model.predict(pairs[v])-doubles[v])**2)));sizes.append(len(v))
                    transfer_search.append({'config':config,'mse':float(np.average(errors,weights=sizes))})
                    model=AdvancedResidual(singles,resources,config).fit(pairs[train],doubles[train],resources['double_variance'][train],seed+997)
                    outer[config['label']]=model.predict(pairs[val])
                selected=min(transfer_search,key=lambda x:x['mse'])
                outer['selected_transfer']=outer[selected['config']['label']]
            metrics={key:float(np.mean((pred-doubles[val])**2)) for key,pred in outer.items()}
            for key,pred in outer.items():
                totals[key]=totals.get(key,0)+metrics[key]*len(val)
                predictions.setdefault(key,np.zeros_like(doubles))[val]+=pred
            std_predictions[val]+=std;counts[val]+=1
            interval_coverage=float(np.mean(np.abs(outer['GP']-doubles[val])<=1.96*std))
            records.append({'fold':f,'train':train.tolist(),'validation':val.tolist(),'GP_search':search,
                'GP_selection':best,'retrieval_search':retrieval,'metrics':metrics,'transfer_search':transfer_search,
                'conditional_GP_interval_empirical_coverage':interval_coverage,
                'interval_interpretation':'diagnostic only; excludes query measurement variance and selection uncertainty'})
            write_json(folder/f'{kind}_fold_{f}.json',records[-1])
            print(name,kind,f,metrics,flush=True)
        covered=counts>0;total=sum(len(r['validation']) for r in records)
        for pred in predictions.values():pred[covered]/=counts[covered,None]
        std_predictions[covered]/=counts[covered,None]
        np.savez_compressed(folder/f'{kind}_predictions.npz',**predictions,GP_conditional_std=std_predictions,coverage=counts,pair_indices=data['pair_indices'])
        result[kind]={'mse_per_occurrence':{key:value/total for key,value in totals.items()},
            'mse_averaged_predictions':{key:float(np.mean((pred[covered]-doubles[covered])**2)) for key,pred in predictions.items()},
            'covered_pairs':int(covered.sum()),'records':records}
    result['seconds']=time.perf_counter()-started
    write_json(folder/'results.json',result)
    return result


def limited_study(name,output,transfer,bounded=False):
    with threadpool_limits(limits=1):return study(name,output,transfer,20261005,bounded)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--transfer',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('results/architecture/experts-v1'))
    parser.add_argument('--bounded-linear',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--datasets',nargs='+',default=['norman-log2','wessels-log2','wessels-expanded-log2'])
    args=parser.parse_args()
    if args.workers>1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(args.workers,len(args.datasets))) as executor:
            futures={name:executor.submit(limited_study,name,args.output,args.transfer,args.bounded_linear) for name in args.datasets}
            results={name:future.result() for name,future in futures.items()}
    else:
        with threadpool_limits(limits=1):
            results={name:study(name,args.output,args.transfer,20261005,args.bounded_linear) for name in args.datasets}
    write_json(args.output/'summary.json',results)


if __name__=='__main__':main()
