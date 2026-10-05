"""Common-condition coverage comparisons with historical query identities protected."""
from pathlib import Path
import sys,argparse
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import load_npz,write_json,sha256
from opal.models.noise_kernel import measurement_smoother
from opal.models.kernel import prediction_error
from tools.improve_residual_model import candidates,fold_sets,select


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('results/architecture/common-coverage-v1'))
    parser.add_argument('--seed',type=int,default=20261005)
    parser.add_argument('--partner-policy',choices=['conservative','held'],default='held')
    args=parser.parse_args();output=args.output
    if output.exists():raise FileExistsError('Preserve coverage comparison')
    output.mkdir(parents=True)
    old=load_npz(ROOT/'data/prepared/wessels-log2/profiles.npz')
    expanded=load_npz(ROOT/'data/prepared/wessels-expanded-protected-log2/profiles.npz')
    resources=load_npz(ROOT/'data/architecture_resources/wessels-expanded-protected-log2/resources.npz')
    if not np.array_equal(old['genes'],expanded['genes']):raise ValueError('Common comparison gene panel changed')
    ids=lambda d,pairs:[tuple(sorted(map(str,p))) for p in d['targets'][pairs]]
    old_ids=ids(old,old['pairs']);new_ids=ids(expanded,expanded['pairs']);lookup={p:i for i,p in enumerate(new_ids)}
    mapping=np.array([lookup[p] for p in old_ids]);extra=np.array([i for i,p in enumerate(new_ids) if p not in set(old_ids)])
    if set(new_ids)&set(ids(old,old['query_pairs'])):raise ValueError('Historical holdout identity entered expanded development')
    np.testing.assert_allclose(old['doubles'],expanded['doubles'][mapping],atol=1e-10)
    old_res=old['doubles']-old['singles'][old['pairs'][:,0]]-old['singles'][old['pairs'][:,1]]
    new_res=expanded['doubles']-expanded['singles'][expanded['pairs'][:,0]]-expanded['singles'][expanded['pairs'][:,1]]
    old_gram=old_res@old_res.T/old_res.shape[1];new_gram=new_res@new_res.T/new_res.shape[1]
    old_configs=candidates(old['singles'],old['pairs']);new_configs=candidates(expanded['singles'],expanded['pairs'])
    write_json(output/'protocol.json',{'seed':args.seed,'validation_population':'same original forty development pairs',
        'gene_panel_identical':True,'original_holdouts_excluded_from_all_training':True,'additional_training_pairs':len(extra),
        'partner_policy':args.partner_policy+'; extra pairs touching inner or outer excluded targets removed',
        'scope':'exploratory, prior broader-track responses known; not independent confirmation',
        'script_sha256':sha256(__file__)})
    result={}
    with threadpool_limits(limits=1):
        for kind in ['random','partner']:
            records=[];totals={};count=0
            for f,(train,val) in enumerate(fold_sets(old['pairs'],kind,args.seed,5)):
                inner=fold_sets(old['pairs'][train],kind,args.seed+1+f,3)
                chosen,_=select(old_configs,old_gram,train,inner)
                config,kernel=old_configs[chosen['index']]
                from opal.models.kernel import smoother
                original_error=prediction_error(smoother(kernel,train,val,config['alpha']),train,val,old_gram)
                original_pred=old['singles'][old['pairs'][val,0]]+old['singles'][old['pairs'][val,1]]+smoother(kernel,train,val,config['alpha'])@old_res[train]
                variants={'original':float(np.mean((original_pred-old['doubles'][val])**2))}
                selections={}
                outer_held=set(old['targets'][old['pairs'][val]].ravel()) if kind=='partner' else set()
                if kind=='partner' and args.partner_policy=='held':
                    outer_held-=set(old['targets'][old['pairs'][train]].ravel())
                outer_extra=np.array([i for i in extra if not outer_held.intersection(new_ids[i])],int)
                for label,use_extra,use_noise in [('extra_singles',False,False),('extra_pairs',True,False),('extra_pairs_noise',True,True)]:
                    full_train=np.concatenate([mapping[train],outer_extra]) if use_extra else mapping[train]
                    search=[]
                    for index,(config,kernel) in enumerate(new_configs):
                        for strength in ([0,.5,1] if use_noise else [0]):
                            errors=[];sizes=[]
                            for ti,vi in inner:
                                inner_val=mapping[train[vi]]
                                held=set(old['targets'][old['pairs'][train[vi]]].ravel()) if kind=='partner' else set()
                                if kind=='partner' and args.partner_policy=='held':
                                    held-=set(old['targets'][old['pairs'][train[ti]]].ravel())
                                available=np.array([i for i in outer_extra if not held.intersection(new_ids[i])],int)
                                inner_train=np.concatenate([mapping[train[ti]],available]) if use_extra else mapping[train[ti]]
                                weights=measurement_smoother(kernel,inner_train,inner_val,config['alpha'],resources['residual_variance'][inner_train],strength)
                                errors.append(prediction_error(weights,inner_train,inner_val,new_gram));sizes.append(len(inner_val))
                            search.append({'index':index,'strength':strength,'mse':float(np.average(errors,weights=sizes))})
                    winner=min(search,key=lambda r:r['mse']);config,kernel=new_configs[winner['index']]
                    weights=measurement_smoother(kernel,full_train,mapping[val],config['alpha'],resources['residual_variance'][full_train],winner['strength'])
                    variants[label]=prediction_error(weights,full_train,mapping[val],new_gram)
                    selections[label]={'config':config,**winner,'training_pairs':len(full_train)}
                record={'fold':f,'old_train':train.tolist(),'old_validation':val.tolist(),'mse':variants,'selections':selections}
                records.append(record);count+=len(val)
                for k,v in variants.items():totals[k]=totals.get(k,0)+v*len(val)
                print(kind,f,variants,flush=True)
            result[kind]={'mse_per_occurrence':{k:v/count for k,v in totals.items()},'records':records}
    write_json(output/'results.json',result)


if __name__=='__main__':main()
