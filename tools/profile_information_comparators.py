"""New profiling of one frozen exposed-data adaptation, never historical RAM."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_native_series_aggregate import gears_fit,cpa_fit

OUT=ROOT/'results/opal_information_v1/comparators'
LOCAL=Path('C:/Users/USER/Desktop/opal-information-20261011/profiling')


def run(method,stage):
    identity='TNFA-positive-3-0'
    old=ROOT/'results/opal_model_series_v2/native-completed'/f'{method}-aggregate-amended-results'/f'{identity}.json'
    record=json.loads(old.read_text())
    cfg=json.loads((ROOT/'results/opal_model_series_v1/pertspectra/access-v4/prepared.json').read_text())['contexts']['TNFA+']
    with np.load(Path(cfg['local'])/'canonical_bank.npz',allow_pickle=False) as z:
        bank={k:z[k] for k in ['singles','responses','pair_ids','culture_ids']}
    primary=json.loads((ROOT/'results/opal_model_series_v2/primary/TNFA-positive/conditions.json').read_text())
    row=next(r for r in primary if r['stratum']=='3' and r['outer']==0 and r['budget']==1 and r['order']=='uniform' and r['acquisition_seed']==11 and r['training_seed']==20261017)
    contract=json.loads((Path(row['checkpoints'][0]['path']).parent.parent/'contract.json').read_text())
    dev=(bank['culture_ids']=='3')&np.isin(bank['pair_ids'],contract['acquired_pair_ids'])
    held=(bank['culture_ids']=='3')&np.isin(bank['pair_ids'],contract['held_pair_ids'])
    targets=sorted({t for p in bank['pair_ids'][dev|held] for t in json.loads(p)})
    lookup={t:i for i,t in enumerate(targets)}
    profiles=np.empty((len(targets),bank['singles'].shape[-1]))
    for p,s in zip(bank['pair_ids'][bank['culture_ids']=='3'],bank['singles'][bank['culture_ids']=='3']):
        for t,v in zip(json.loads(p),s): profiles[lookup[t]]=v
    x=np.asarray([[lookup[t] for t in json.loads(p)] for p in bank['pair_ids'][dev]])
    query=np.asarray([[lookup[t] for t in json.loads(p)] for p in bank['pair_ids'][held]])
    mapping=json.loads((ROOT/'results/opal_model_series_v1/pertspectra/TNFA-positive/scope_before_fit.json').read_text())['global_fold_assignment']
    validation=np.asarray([mapping[p]==1 for p in bank['pair_ids'][dev]])
    with gzip.open('D:/opal-model-series-20261010-v2-continuation/native-assets/gene2go.json.gz','rt') as f: go=json.load(f)
    destination=LOCAL/method
    destination.mkdir(parents=True,exist_ok=True)
    restored=None
    if stage=='replay':
        with np.load(destination/'parameters.npz',allow_pickle=False) as z: restored={k:z[k] for k in z.files}
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    done=threading.Event()
    peak=[0]
    def sample():
        while not done.wait(.01): peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    sampler=threading.Thread(target=sample); sampler.start()
    start,cpu=time.monotonic(),time.process_time()
    with threadpool_limits(limits=1):
        if method=='GEARS':
            pred,state,history,count=gears_fit(x,bank['responses'][dev],query,profiles,{'hidden_size':record['selected_width']},20261017,go,targets,restored_state=restored)
        else:
            pred,state,history,count=cpa_fit(x,bank['responses'][dev],query,profiles,{'width':record['selected_width'],'splits':np.where(validation,'valid','train')},20261017,restored_state=restored)
    elapsed=time.monotonic()-start
    done.set();sampler.join()
    gpu={'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),'device':torch.cuda.get_device_name(0),'gpu_time_seconds':None}
    if stage=='training':
        if (destination/'parameters.npz').exists(): raise FileExistsError('Preserve prior profiling checkpoint')
        np.savez_compressed(destination/'parameters.npz',**state)
        np.savez_compressed(destination/'predictions.npz',prediction=pred)
        replay_difference=None
    else:
        with np.load(destination/'predictions.npz',allow_pickle=False) as z: replay_difference=float(np.max(np.abs(pred-z['prediction'])))
        if replay_difference>1e-6: raise ValueError('Profiling replay parity failed')
    result={'method':method,'stage':stage,'role':'new profiling, exposed-data protocol adaptation','fully_eligible':False,
        'configuration_source':str(old),'width':record['selected_width'],'seed':20261017,'history':history,
        'mse':float(np.mean((pred-bank['responses'][held])**2)),'training_pairs':int(dev.sum()),'held_pairs':int(held.sum()),
        'single_profiles':len(profiles),'genes':profiles.shape[1],'precision':'float32','batch':32 if method=='GEARS' else 64,
        'parameter_count':count,'wall_seconds_including_fit_graph_predict':elapsed,'cpu_seconds':time.process_time()-cpu,
        'peak_stage_rss_sampled_bytes':peak[0],'sampling_seconds':.01,'gpu':gpu,'checkpoint':str(destination),
        'checkpoint_bytes':(destination/'parameters.npz').stat().st_size,'reference_bytes':profiles.nbytes,
        'historical_training_ram_bytes':None,'replay_max_abs':replay_difference,'torch':torch.__version__,'python':sys.version,
        'preprocessing':'same frozen mean aggregate/GO adaptation; no native author data contract','latency_ms_per_pair_pipeline':1000*elapsed/len(query) if stage=='replay' else None}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{method}-{stage}-profile.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['method','stage','mse','wall_seconds_including_fit_graph_predict','peak_stage_rss_sampled_bytes','replay_max_abs']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['GEARS','CPA'],required=True);p.add_argument('--stage',choices=['training','replay'],required=True)
    args=p.parse_args();run(args.method,args.stage)
