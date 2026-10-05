"""Training-variance-aware kernels versus identically tuned variance placebos."""
from pathlib import Path
import argparse,sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import load_npz,write_json,read_json,sha256
from opal.models.noise_kernel import measurement_smoother,NoiseAwareKernel
from opal.models.kernel import prediction_error
from tools.improve_residual_model import candidates,fold_sets
from opal.inference import export_members


def selection(configs,gram,variance,indices,folds,placebo,seed):
    search=[]
    for index,(config,kernel) in enumerate(configs):
        errors=[];sizes=[]
        for j,(ti,vi) in enumerate(folds):
            train,val=indices[ti],indices[vi]
            noise=variance[train]
            if placebo:noise=noise[np.random.default_rng(seed+j).permutation(len(train))]
            weights=measurement_smoother(kernel,train,val,config['alpha'],noise,config['noise_strength'])
            errors.append(prediction_error(weights,train,val,gram));sizes.append(len(val))
        search.append({'index':index,'config':config,'mse':float(np.average(errors,weights=sizes))})
    return min(search,key=lambda r:r['mse']),search


def study(name,root,seed):
    folder=root/name
    if (folder/'results.json').exists():return read_json(folder/'results.json')
    if folder.exists():raise FileExistsError('Incomplete noise study exists')
    folder.mkdir(parents=True)
    data=load_npz(ROOT/'data/prepared'/name/'profiles.npz');resources=load_npz(ROOT/'data/architecture_resources'/name/'resources.npz')
    singles,pairs,doubles=data['singles'],data['pairs'],data['doubles']
    additive=singles[pairs[:,0]]+singles[pairs[:,1]];residual=doubles-additive;gram=residual@residual.T/residual.shape[1]
    variance=resources['residual_variance'].mean(1)
    base=candidates(singles,pairs);configs=[({**c,'noise_strength':strength},k) for c,k in base for strength in [0,.5,1]]
    write_json(folder/'protocol.json',{'seed':seed,'outer_folds':5,'inner_folds':3,'configurations':[c for c,k in configs],
        'placebo':'variance values permuted within each training fold only; no validation variance used in fitting',
        'scope':'exploratory development-only','evaluator_access':False,'resource_sha256':sha256(ROOT/'data/architecture_resources'/name/'resources.npz'),
        'search_budget':'201 heteroscedastic configurations versus 67 unweighted; placebo search has matching 201',
        'script_sha256':sha256(__file__)})
    result={}
    for kind in ['random','partner']:
        records=[];totals={};count=np.zeros(len(pairs),int);predictions={}
        for f,(train,val) in enumerate(fold_sets(pairs,kind,seed,5)):
            inner=fold_sets(pairs[train],kind,seed+f+1,3);record={'fold':f,'train':train.tolist(),'validation':val.tolist(),'choices':{},'mse':{}}
            for key,placebo,allowed in [('unweighted',False,False),('measurement',False,True),('permuted_variance',True,True)]:
                choices=configs if allowed else [(c,k) for c,k in configs if c['noise_strength']==0]
                winner,search=selection(choices,gram,variance,train,inner,placebo,seed+733)
                config,kernel=choices[winner['index']];noise=variance[train]
                if placebo:noise=noise[np.random.default_rng(seed+997).permutation(len(train))]
                weights=measurement_smoother(kernel,train,val,config['alpha'],noise,config['noise_strength'])
                pred=additive[val]+weights@residual[train];mse=float(np.mean((pred-doubles[val])**2))
                record['choices'][key]=winner;record['mse'][key]=mse;record.setdefault('search',{})[key]=search
                totals[key]=totals.get(key,0)+mse*len(val);predictions.setdefault(key,np.zeros_like(doubles))[val]+=pred
            records.append(record);count[val]+=1;write_json(folder/f'{kind}_fold_{f}.json',record)
            print(name,kind,f,record['mse'],flush=True)
        covered=count>0;total=sum(len(r['validation']) for r in records)
        for pred in predictions.values():pred[covered]/=count[covered,None]
        np.savez_compressed(folder/f'{kind}_predictions.npz',**predictions,coverage=count,pair_indices=data['pair_indices'])
        result[kind]={'mse_per_occurrence':{k:v/total for k,v in totals.items()},
            'mse_averaged_predictions':{k:float(np.mean((p[covered]-doubles[covered])**2)) for k,p in predictions.items()},'records':records}
    config=result['random']['records'][0]['choices']['measurement']['config']
    model=NoiseAwareKernel(singles,config).fit(pairs,doubles,variance)
    export_members([{'type':'noise_kernel','model':model,'config':config,'seed':seed}],data,folder/'checkpoint',
        {'context':read_json(ROOT/'data/prepared'/name/'source_manifest.json')['context'],'scale':'log2','status':'development-only research candidate'})
    write_json(folder/'selection.json',{'config':config,'selection':'first random fold inner-CV; no outer-score selection'})
    write_json(folder/'results.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('results/architecture/noise-kernels-v1'))
    parser.add_argument('--seed',type=int,default=20261005)
    parser.add_argument('--datasets',nargs='+',default=['norman-log2','wessels-log2','wessels-expanded-protected-log2'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):result={name:study(name,args.output,args.seed) for name in args.datasets}
    write_json(args.output/'summary.json',result)


if __name__=='__main__':main()
