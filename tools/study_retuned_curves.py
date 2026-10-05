"""Subset-retuned learning curves and isolated split error-radius diagnostics."""
from pathlib import Path
import argparse,sys,zipfile
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import load_npz,read_json,write_json,sha256,source_hash
from opal.models.covariance_kernel import CovarianceKernel,noise_covariance,covariance_smoother
from opal.models.kernel import prediction_error
from opal.reliability import error_radius
from tools.improve_residual_model import candidates,fold_sets
from tools.study_joint_uncertainty import variants,select


def score(config,kernel,train,val,gram,covariance):
    regularizer=noise_covariance(covariance[np.ix_(train,train)],config['noise_strength'],config['correlation'])
    weights=covariance_smoother(kernel,train,val,config['alpha'],regularizer)
    return prediction_error(weights,train,val,gram)


def risk_diagnostic(data,covariance,base,train,val,kind,seed):
    calibration_splits=fold_sets(data['pairs'][train],kind,seed,5)
    if not calibration_splits:return {'status':'infeasible','reason':'no feasible calibration split'}
    ti,ci=calibration_splits[0];fit,cal=train[ti],train[ci]
    inner=fold_sets(data['pairs'][fit],kind,seed+1,3)
    if len(inner)<2:return {'status':'infeasible','reason':'insufficient fitting-only inner splits'}
    residual=data['doubles']-data['singles'][data['pairs'][:,0]]-data['singles'][data['pairs'][:,1]]
    gram=residual@residual.T/residual.shape[1]
    configs=variants(base,'diagonal');winner,_=select(configs,gram,covariance,fit,inner,False,seed)
    config=configs[winner['index']][0]
    model=CovarianceKernel(data['singles'],config).fit(data['pairs'][fit],data['doubles'][fit],covariance[np.ix_(fit,fit)])
    cal_error=np.sqrt(np.mean((model.predict(data['pairs'][cal])-data['doubles'][cal])**2,axis=1))
    val_error=np.sqrt(np.mean((model.predict(data['pairs'][val])-data['doubles'][val])**2,axis=1))
    radii=[]
    for coverage in [.8,.9]:
        result=error_radius(cal_error,coverage)
        result['observed_outer_coverage']=float(np.mean(val_error<=result['radius'])) if result['finite_radius'] else None
        radii.append(result)
    return {'status':'complete','fit':fit.tolist(),'calibration':cal.tolist(),'outer_validation':val.tolist(),
        'selected_config':config,'radii':radii,'calibration_rmse':cal_error.tolist(),'outer_rmse':val_error.tolist(),
        'selection_uses_calibration_responses':False,'scope':'empirical error-radius sensitivity only; no calibrated abstention or independent confirmation'}


def study(name,root,seed):
    folder=root/name
    if (folder/'results.json').exists():
        protocol=read_json(folder/'protocol.json')
        if protocol['seed']!=seed:raise ValueError('Completed study seed differs; use a new output folder')
        for key,path in [('profiles_sha256',ROOT/'data/prepared'/name/'profiles.npz'),('draws_sha256',ROOT/'data/architecture_resources'/('joint-'+name)/'draws.npz')]:
            if protocol[key]!=sha256(path):raise ValueError('Completed study input changed: '+key)
        return read_json(folder/'results.json')
    if folder.exists():raise FileExistsError('Incomplete retuned study exists')
    folder.mkdir(parents=True)
    data=load_npz(ROOT/'data/prepared'/name/'profiles.npz');draws=load_npz(ROOT/'data/architecture_resources'/('joint-'+name)/'draws.npz')
    pairs,singles,doubles=data['pairs'],data['singles'],data['doubles'];covariance=draws['residual_covariance']
    residual=doubles-singles[pairs[:,0]]-singles[pairs[:,1]];gram=residual@residual.T/residual.shape[1]
    base=candidates(singles,pairs);families={k:variants(base,k) for k in ['unweighted','diagonal','correlated']}
    write_json(folder/'protocol.json',{'seed':seed,'outer_folds':5,'inner_folds':3,'subset_orders':3,'fractions':[.25,.5,.75,1],
        'subset_rule':'three nested, seed-fixed training orders; full size fitted once in canonical order with identical inner folds to frozen comparator; no validation-driven redraw',
        'profiles_sha256':sha256(ROOT/'data/prepared'/name/'profiles.npz'),'draws_sha256':sha256(ROOT/'data/architecture_resources'/('joint-'+name)/'draws.npz'),
        'source_hash':source_hash(),'runner_sha256':sha256(__file__),'evaluator_access':False,
        'risk_rule':'first fitting/calibration split; model selection fits on fitting subset only; no outer or calibration outcomes select hyperparameters',
        'scope':'exploratory correlated resamplings; differing search budgets; no independence or frontier claim'})
    with zipfile.ZipFile(folder/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__,'tools/study_retuned_curves.py')
        for p in (ROOT/'src').rglob('*.py'):archive.write(p,str(p.relative_to(ROOT)))
    results={}
    for kind in ['random','partner']:
        records=[];risks=[]
        for f,(train,val) in enumerate(fold_sets(pairs,kind,seed,5)):
            full_inner=fold_sets(pairs[train],kind,seed+f+1,3)
            full_winner,_=select(families['unweighted'],gram,covariance,train,full_inner)
            frozen_config,frozen_kernel=families['unweighted'][full_winner['index']]
            for order in range(3):
                permutation=np.random.default_rng(seed+1000*f+order).permutation(len(train))
                for fraction in [.25,.5,.75,1]:
                    if fraction==1 and order>0:continue
                    n=max(6,int(np.floor(len(train)*fraction)));subset=train if fraction==1 else train[permutation[:n]]
                    inner=fold_sets(pairs[subset],kind,seed+f+order+1,3)
                    if len(inner)<2:
                        records.append({'fold':f,'order':order,'fraction':fraction,'status':'infeasible','training_pairs':len(subset)});continue
                    metrics={'additive':float(np.mean(residual[val]**2)),
                        'frozen_full_size_kernel':score(frozen_config,frozen_kernel,subset,val,gram,covariance)};choices={}
                    for family,configs in families.items():
                        winner,_=select(configs,gram,covariance,subset,inner)
                        config,kernel=configs[winner['index']]
                        metrics['retuned_'+family]=score(config,kernel,subset,val,gram,covariance);choices[family]=winner
                    records.append({'fold':f,'order':order,'fraction':fraction,'status':'complete','training_pairs':len(subset),
                        'validation_pairs':len(val),'train':subset.tolist(),'validation':val.tolist(),'mse':metrics,'choices':choices})
            risks.append(risk_diagnostic(data,covariance,base,train,val,kind,seed+700+f))
            print(name,kind,f,'retuned subsets and isolated risk diagnostics complete',flush=True)
        summary={}
        for fraction in [.25,.5,.75,1]:
            group=[r for r in records if r['fraction']==fraction and r['status']=='complete']
            if group:summary[str(fraction)]={'mse_per_occurrence':{k:float(np.average([r['mse'][k] for r in group],weights=[r['validation_pairs'] for r in group])) for k in group[0]['mse']},
                'fits':len(group),'training_pairs_range':[min(r['training_pairs'] for r in group),max(r['training_pairs'] for r in group)]}
        results[kind]={'learning_curves':summary,'records':records,'error_radius_diagnostics':risks}
    write_json(folder/'results.json',results);return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/architecture/retuned-curves-v2'));p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--datasets',nargs='+',default=['norman-log2','wessels-log2','wessels-expanded-protected-log2']);args=p.parse_args()
    with threadpool_limits(limits=1):results={name:study(name,args.output,args.seed) for name in args.datasets}
    write_json(args.output/'summary.json',results)


if __name__=='__main__':main()
