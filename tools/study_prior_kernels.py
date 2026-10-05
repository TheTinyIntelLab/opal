"""Real and shuffled external-prior searches on identical nested folds."""
from pathlib import Path
import sys,argparse,time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import load_npz,write_json,read_json,sha256
from opal.models.prior_kernel import PriorKernelResidual
from opal.models.kernel import smoother,KernelResidual
from opal.inference import export_members,Predictor
from tools.improve_residual_model import candidates,fold_sets,select


def prior_candidates(singles,resources,pairs):
    sources=resources['bank'].shape[1]
    groups=[[i] for i in range(1,sources)]+[[0,i] for i in range(1,sources)]
    groups += [list(range(sources)),list(range(1,sources)),list(range(3,sources))] if sources>3 else [list(range(sources))]
    groups=[g for g in groups if g]
    result=[]
    for indices in groups:
        for bandwidth in [.25,1,4]:
            for weight in [.1,1]:
                config={'kind':'prior_kernel','sources':indices,'bandwidth':bandwidth,'prior_weight':weight,'alpha':1}
                model=PriorKernelResidual(singles,resources['bank'],resources['source_mask'],config)
                kernel=model.kernel(pairs,pairs)
                for alpha in [.01,.1,1,10,100]:result.append(({**config,'alpha':alpha},kernel))
    return result


def study(name,root,prefix,seed):
    output=root/name
    if (output/'results.json').exists():return read_json(output/'results.json')
    if output.exists():raise FileExistsError('Incomplete prior-kernel study exists')
    output.mkdir(parents=True)
    data=load_npz(ROOT/'data/prepared'/name/'profiles.npz')
    source=ROOT/'data/architecture_resources'/(prefix+name)/'resources.npz'
    resources=load_npz(source);singles=data['singles'];pairs=data['pairs'];doubles=data['doubles']
    residual=doubles-singles[pairs[:,0]]-singles[pairs[:,1]];gram=residual@residual.T/residual.shape[1]
    base=candidates(singles,pairs)
    shuffled={**resources,'bank':resources['bank'].copy(),'source_mask':resources['source_mask'].copy()}
    order=np.random.default_rng(seed+671).permutation(len(singles))
    shuffled['bank'][:,1:]=resources['bank'][order,1:];shuffled['source_mask'][:,1:]=resources['source_mask'][order,1:]
    variants={'measured_only':base,'real_priors':base+prior_candidates(singles,resources,pairs),
              'shuffled_priors':base+prior_candidates(singles,shuffled,pairs)}
    write_json(output/'protocol.json',{'seed':seed,'outer_folds':5,'inner_folds':3,'source_names':resources['source_names'].tolist(),
        'configurations':{key:[c for c,k in values] for key,values in variants.items()},
        'source_sha256':sha256(source),'scope':'exploratory development only; extra prior-search budget explicitly unequal to baseline',
        'evaluator_access':False,'matched_control':'real and shuffled searches have identical configurations and folds',
        'export_selection':'first random fold inner-CV only; no outer-score selection','script_sha256':sha256(__file__)})
    results={};started=time.perf_counter()
    for kind in ['random','partner']:
        records=[];totals={};counts=np.zeros(len(pairs),int);predictions={}
        for f,(train,val) in enumerate(fold_sets(pairs,kind,seed,5)):
            inner=fold_sets(pairs[train],kind,seed+f+1,3);record={'fold':f,'train':train.tolist(),'validation':val.tolist(),'choices':{},'mse':{}}
            for key,configs in variants.items():
                winner,search=select(configs,gram,train,inner)
                config,kernel=configs[winner['index']]
                pred=singles[pairs[val,0]]+singles[pairs[val,1]]+smoother(kernel,train,val,config['alpha'])@residual[train]
                mse=float(np.mean((pred-doubles[val])**2))
                record['choices'][key]=winner;record['mse'][key]=mse
                record.setdefault('search',{})[key]=search
                totals[key]=totals.get(key,0)+mse*len(val)
                predictions.setdefault(key,np.zeros_like(doubles))[val]+=pred
            counts[val]+=1;records.append(record)
            write_json(output/f'{kind}_fold_{f}.json',record)
            print(name,kind,f,record['mse'],flush=True)
        covered=counts>0;total=sum(len(r['validation']) for r in records)
        for pred in predictions.values():pred[covered]/=counts[covered,None]
        np.savez_compressed(output/f'{kind}_predictions.npz',**predictions,coverage=counts,pair_indices=data['pair_indices'])
        results[kind]={'mse_per_occurrence':{key:value/total for key,value in totals.items()},
            'mse_averaged_predictions':{key:float(np.mean((pred[covered]-doubles[covered])**2)) for key,pred in predictions.items()},'records':records}
    selected=results['random']['records'][0]['choices']['real_priors']['config']
    if selected['kind']=='prior_kernel':
        model=PriorKernelResidual(singles,resources['bank'],resources['source_mask'],selected).fit(pairs,doubles)
        member={'type':'prior_kernel','model':model,'config':selected,'seed':seed}
    else:
        model=KernelResidual(singles,selected).fit(pairs,doubles);member={'type':'kernel','model':model,'config':selected,'seed':seed}
    context=read_json(ROOT/'data/prepared'/name/'source_manifest.json')['context']
    export_members([member],data,output/'checkpoint',{'context':context,'scale':'log2','status':'development-only research candidate'})
    pred=model.predict(data['query_pairs']);np.savez_compressed(output/'unscored_query_predictions.npz',predictions=pred,pairs=data['query_pairs'])
    loaded=Predictor(output/'checkpoint');observed=np.stack([loaded.predict(str(data['targets'][a]),str(data['targets'][b]))['effect'] for a,b in data['query_pairs']])
    if not np.allclose(pred,observed,atol=1e-9):raise ValueError('Prior-kernel export disagreement')
    write_json(output/'selection.json',{'config':selected,'rule':'first random fold inner CV, fixed before outer scores','evaluator_access':False})
    results['seconds']=time.perf_counter()-started;write_json(output/'results.json',results)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('results/architecture/prior-kernels-v1'))
    parser.add_argument('--prefix',default='enriched-');parser.add_argument('--seed',type=int,default=20261005)
    parser.add_argument('--datasets',nargs='+',default=['norman-log2','wessels-log2','wessels-expanded-protected-log2'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):results={name:study(name,args.output,args.prefix,args.seed) for name in args.datasets}
    write_json(args.output/'summary.json',results)


if __name__=='__main__':main()
