"""Nested covariance experts and fitting-aware technical reference sensitivity."""
from pathlib import Path
import argparse,sys,zipfile
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import load_npz,read_json,write_json,sha256,source_hash
from opal.models.covariance_kernel import noise_covariance,covariance_smoother,CovarianceKernel
from opal.models.kernel import prediction_error,KernelResidual
from opal.inference import export_members
from tools.improve_residual_model import candidates,fold_sets


def variants(base,family):
    settings=[(0,0)]
    if family!='unweighted':settings +=[(s,c) for s in [.5,1] for c in ([0] if family=='diagonal' else [0,.5,1])]
    return [({**config,'noise_strength':s,'correlation':c},kernel) for config,kernel in base for s,c in settings]


def regularizers(covariance,indices,configs,placebo,seed):
    block=covariance[np.ix_(indices,indices)]
    if placebo:
        permutation=np.random.default_rng(seed).permutation(len(indices));block=block[np.ix_(permutation,permutation)]
    settings={(c['noise_strength'],c['correlation']) for c,k in configs}
    return {(s,c):noise_covariance(block,s,c) for s,c in settings}


def select(configs,gram,covariance,indices,folds,placebo=False,seed=0):
    regs=[regularizers(covariance,indices[t],configs,placebo,seed+j) for j,(t,v) in enumerate(folds)]
    errors=[]
    for index,(config,kernel) in enumerate(configs):
        total=0.;size=0
        for j,(ti,vi) in enumerate(folds):
            t,v=indices[ti],indices[vi]
            weights=covariance_smoother(kernel,t,v,config['alpha'],regs[j][config['noise_strength'],config['correlation']])
            total+=len(v)*prediction_error(weights,t,v,gram);size+=len(v)
        errors.append({'index':index,'config':config,'mse':total/size})
    return min(errors,key=lambda r:(r['mse'],r['index'])),errors


def fitting_bootstrap(data,draws,train,val,config):
    # Refit reference-derived kernels and response coefficients for every draw.
    predictions=[]
    covariance=draws['residual_covariance'][np.ix_(train,train)]
    for singles,doubles in zip(draws['single_draws'],draws['double_draws']):
        model=CovarianceKernel(singles,config).fit(data['pairs'][train],doubles[train],covariance)
        predictions.append(model.predict(data['pairs'][val]))
    predictions=np.stack(predictions)
    point=data['doubles'][val]
    error=np.mean((predictions-point)**2,axis=(1,2))
    return {'refitted_bootstrap_mse_percentiles':np.percentile(error,[5,50,95]).tolist(),
        'mean_gene_prediction_sd':float(predictions.std(0,ddof=1).mean()),
        'draws':len(predictions),'scope':'fitting and input-reference sampling, conditional on selected configuration, covariance, QC and normalizer; no biological or selection uncertainty'}


def study(name,root,seed,bootstrap=True):
    folder=root/name
    if (folder/'results.json').exists():
        protocol=read_json(folder/'protocol.json')
        if protocol['seed']!=seed:raise ValueError('Completed study seed differs; use a new output folder')
        for key,path in [('profiles_sha256',ROOT/'data/prepared'/name/'profiles.npz'),('draws_sha256',ROOT/'data/architecture_resources'/('joint-'+name)/'draws.npz')]:
            if protocol[key]!=sha256(path):raise ValueError('Completed study input changed: '+key)
        return read_json(folder/'results.json')
    if folder.exists():raise FileExistsError('Incomplete joint study; preserve and choose a new output')
    folder.mkdir(parents=True)
    data=load_npz(ROOT/'data/prepared'/name/'profiles.npz');draws=load_npz(ROOT/'data/architecture_resources'/('joint-'+name)/'draws.npz')
    pairs,singles,doubles=data['pairs'],data['singles'],data['doubles']
    np.testing.assert_array_equal(draws['pair_indices'],data['pair_indices']);np.testing.assert_array_equal(draws['genes'],data['genes'])
    residual=doubles-singles[pairs[:,0]]-singles[pairs[:,1]];gram=residual@residual.T/residual.shape[1]
    covariance=draws['residual_covariance'];base=candidates(singles,pairs)
    families={key:variants(base,key) for key in ['unweighted','diagonal','correlated','permuted_covariance']}
    write_json(folder/'protocol.json',{'seed':seed,'outer_folds':5,'inner_folds':3,'evaluator_access':False,
        'source_hash':source_hash(),'runner_sha256':sha256(__file__),'profiles_sha256':sha256(ROOT/'data/prepared'/name/'profiles.npz'),
        'draws_sha256':sha256(ROOT/'data/architecture_resources'/('joint-'+name)/'draws.npz'),
        'candidate_counts':{k:len(v) for k,v in families.items()},'configurations':{k:[c for c,g in v] for k,v in families.items()},
        'placebo':'training covariance rows and columns jointly permuted; identical correlated search budget',
        'selection':'inner-only; train covariance blocks only; first random outer fold inner selection freezes export',
        'scope':'late exploratory development sensitivity; historical outcomes known'})
    with zipfile.ZipFile(folder/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__,'tools/study_joint_uncertainty.py')
        for p in (ROOT/'src').rglob('*.py'):archive.write(p,str(p.relative_to(ROOT)))
    result={};first=None
    for kind in ['random','partner']:
        records=[];totals={};size=0
        for f,(train,val) in enumerate(fold_sets(pairs,kind,seed,5)):
            inner=fold_sets(pairs[train],kind,seed+f+1,3)
            if len(inner)<2:raise ValueError('Insufficient registered inner folds')
            record={'fold':f,'train':train.tolist(),'validation':val.tolist(),'choices':{},'mse':{}}
            for family,configs in families.items():
                placebo=family=='permuted_covariance'
                winner,search=select(configs,gram,covariance,train,inner,placebo,seed+73)
                config,kernel=configs[winner['index']]
                reg=regularizers(covariance,train,[(config,kernel)],placebo,seed+97)[config['noise_strength'],config['correlation']]
                weights=covariance_smoother(kernel,train,val,config['alpha'],reg)
                error=prediction_error(weights,train,val,gram)
                record['mse'][family]=error;record['choices'][family]=winner
                totals[family]=totals.get(family,0)+error*len(val)
                if family=='correlated':
                    if first is None:first=config
                    if bootstrap:record['technical_refitting']=fitting_bootstrap(data,draws,train,val,config)
            records.append(record);size+=len(val)
            print(name,kind,f,record['mse'],flush=True)
        result[kind]={'mse_per_occurrence':{k:v/size for k,v in totals.items()},'records':records,'dependent_folds':True}
    model=CovarianceKernel(singles,first).fit(pairs,doubles,covariance)
    export_members([{'type':'covariance_kernel','model':model,'config':first,'seed':seed}],data,folder/'checkpoint',
        {'context':read_json(ROOT/'data/prepared'/name/'source_manifest.json')['context'],'scale':'log2','status':'development-only covariance candidate'})
    predictor=__import__('opal.inference',fromlist=['Predictor']).Predictor(folder/'checkpoint')
    a,b=map(str,data['targets'][data['query_pairs'][0]])
    discrepancy=float(np.max(np.abs(predictor.predict(a,b)['effect']-model.predict(data['query_pairs'][:1])[0])))
    if discrepancy>1e-6:raise ValueError('Covariance export mismatch')
    result['export_verification']={'maximum_absolute_difference':discrepancy,'query_scored':False}
    write_json(folder/'selection.json',{'config':first,'rule':'first random fold inner CV; never outer scores'})
    write_json(folder/'results.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/architecture/joint-uncertainty-v1'));p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--datasets',nargs='+',default=['norman-log2','wessels-log2','wessels-expanded-protected-log2']);p.add_argument('--skip-bootstrap',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=1):results={name:study(name,args.output,args.seed,not args.skip_bootstrap) for name in args.datasets}
    write_json(args.output/'summary.json',results)


if __name__=='__main__':main()
