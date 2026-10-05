"""Frozen TabICL comparator with strictly training-fold output PCA."""
from pathlib import Path
import sys,os,argparse,time,json,urllib.request
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HOME']=str(ROOT/'tools/vendor/hf-cache')
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools/vendor/tabicl_runtime'))
from opal.artifacts import load_npz,write_json,read_json,sha256
from tools.improve_residual_model import fold_sets
from tools.download_architecture_resources import fetch
from tools.study_architecture import prior_ridge


def checkpoint():
    folder=ROOT/'data/architecture_resources/tabicl';folder.mkdir(exist_ok=True)
    metadata=folder/'publisher.json'
    if not metadata.exists():
        fetch('https://huggingface.co/api/models/jingang/TabICL?blobs=true',metadata)
    info=read_json(metadata);filename='tabicl-regressor-v2-20260212.ckpt'
    file=next(x for x in info['siblings'] if x['rfilename']==filename)
    path=fetch(f"https://huggingface.co/jingang/TabICL/resolve/{info['sha']}/{filename}",folder/filename,
               file.get('size',file.get('lfs',{}).get('size')))
    expected=file.get('lfs',{}).get('sha256')
    if expected and sha256(path)!=expected:raise ValueError('TabICL publisher checksum mismatch')
    write_json(folder/'manifest.json',{'repository':'jingang/TabICL','revision':info['sha'],
        'checkpoint':filename,'sha256':sha256(path),'package':'tabicl==2.2.0','loading':'torch weights_only=True'})
    return path


def study(name,output,model_path,seed,rank,estimators):
    from tabicl import TabICLRegressor
    folder=output/name
    if (folder/'results.json').exists():return read_json(folder/'results.json')
    if folder.exists():raise FileExistsError('Incomplete tabular study exists')
    folder.mkdir(parents=True)
    data=load_npz(ROOT/'data/prepared'/name/'profiles.npz')
    resources=load_npz(ROOT/'data/architecture_resources'/name/'resources.npz')
    bank=resources['bank'].reshape(len(data['singles']),-1)
    pairs=data['pairs'];a,b=bank[pairs[:,0]],bank[pairs[:,1]]
    features=np.concatenate([a+b,np.abs(a-b),a*b],axis=1)
    additive=data['singles'][pairs[:,0]]+data['singles'][pairs[:,1]]
    residual=data['doubles']-additive
    write_json(folder/'protocol.json',{'seed':seed,'outer_folds':5,'output_rank':rank,'n_estimators':estimators,
        'model':'tabicl==2.2.0, frozen regressor v2-20260212','checkpoint_sha256':sha256(model_path),
        'selection':'fixed configuration, no outer-score tuning','output_basis':'training residual PCA only',
        'inputs':'same measured-single, GO and Hallmark features as prior ridge',
        'evaluator_access':False,'scope':'exploratory development comparison','script_sha256':sha256(__file__)})
    result={};started=time.perf_counter()
    model=TabICLRegressor(n_estimators=estimators,model_path=model_path,allow_auto_download=False,
        device='cpu',use_amp=False,use_fa3=False,offload_mode=False,n_jobs=1,random_state=seed,verbose=False)
    for kind in ['random','partner']:
        folds=fold_sets(pairs,kind,seed,5);records=[];total=0.;linear_total=0.;size=0
        oof=np.zeros_like(residual);coverage=np.zeros(len(pairs),int)
        for f,(train,val) in enumerate(folds):
            center=features[train].mean(0);scale=np.maximum(features[train].std(0),1e-5)
            x=(features-center)/scale
            mean=residual[train].mean(0)
            _,_,vt=np.linalg.svd(residual[train]-mean,full_matrices=False)
            basis=vt[:min(rank,len(train)-1)].T
            targets=(residual[train]-mean)@basis
            predicted=[]
            for component in range(targets.shape[1]):
                model.fit(x[train],targets[:,component])
                predicted.append(model.predict(x[val]))
            prediction=additive[val]+mean+np.stack(predicted,axis=1)@basis.T
            linear=prior_ridge(data['singles'],resources,pairs,data['doubles'],train,val,10)
            mse=float(np.mean((prediction-data['doubles'][val])**2))
            linear_mse=float(np.mean((linear-data['doubles'][val])**2))
            records.append({'fold':f,'train':train.tolist(),'validation':val.tolist(),'mse':mse,'prior_ridge_alpha10_mse':linear_mse})
            total+=mse*len(val);linear_total+=linear_mse*len(val);size+=len(val)
            oof[val]+=prediction;coverage[val]+=1
            write_json(folder/f'{kind}_fold_{f}.json',records[-1])
            print(name,kind,f,'TabICL MSE',mse,flush=True)
        covered=coverage>0;oof[covered]/=coverage[covered,None]
        np.savez_compressed(folder/f'{kind}_predictions.npz',predictions=oof,coverage=coverage,pair_indices=data['pair_indices'])
        result[kind]={'mse_per_occurrence':total/size,'prior_ridge_alpha10_mse':linear_total/size,
            'mse_averaged_predictions':float(np.mean((oof[covered]-data['doubles'][covered])**2)),
            'covered_pairs':int(covered.sum()),'records':records}
    result['seconds']=time.perf_counter()-started
    write_json(folder/'results.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('results/architecture/tabular-v1'))
    parser.add_argument('--rank',type=int,default=8);parser.add_argument('--estimators',type=int,default=4)
    parser.add_argument('--datasets',nargs='+',default=['norman-log2','wessels-log2','wessels-expanded-log2'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        path=checkpoint()
        results={name:study(name,args.output,path,20261005,args.rank,args.estimators) for name in args.datasets}
    write_json(args.output/'summary.json',results)


if __name__=='__main__':main()
