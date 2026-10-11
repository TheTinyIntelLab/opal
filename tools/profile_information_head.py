"""Fresh-process Base/head fit and replay profiling on a frozen development fold."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal import load_model
from opal.artifacts import environment,read_json,sha256,write_json
from opal.information import fit_head,load_head,predict_head,save_head
from opal.series import OpalBase
from opal.series_bilinear import weights,predict_reference
from prepare_information import LOCAL,OUT,resource_check


def run(stage):
    identity='TNFA-positive-global_pair-0-3-original'
    row=read_json(OUT/'development/folds'/f'{identity}.json')
    src=read_json(OUT/'measurement/TNFA-positive.json')
    with np.load(src['bank'],allow_pickle=False) as z:bank={k:z[k] for k in ['singles','responses','pair_ids','culture_ids']}
    with np.load(src['summary'],allow_pickle=False) as z:summaries={k:z[k] for k in z.files}
    train=(bank['culture_ids']=='3')&np.isin(bank['pair_ids'],row['train_pair_ids'])
    held=(bank['culture_ids']=='3')&np.isin(bank['pair_ids'],row['held_pair_ids'])
    development={k:v[train] for k,v in bank.items()}
    query={k:v[held] for k,v in bank.items() if k!='responses'}
    contract=read_json(Path(row['checkpoint'])/'freeze.json')['contract']
    destination=LOCAL/'profiling/Opal';destination.mkdir(parents=True,exist_ok=True)
    resource_check(2000000)
    start,cpu=time.monotonic(),time.process_time()
    settings=read_json(ROOT/'configs/experiments/opal_model_series_v2.json')
    menu=read_json(ROOT/'configs/experiments/replicated_fixed_state_v1.json')['menu']
    with threadpool_limits(limits=1):
        if stage=='training':
            base=OpalBase(settings,menu).fit(development,contract)
            base.save(destination/'base')
            residual=development['responses']-predict_reference(base.config,base.fitted['reference'],development)
            chosen=row['arms']['variance']['chosen']
            head=None if chosen is None else fit_head(summaries['means_projected'][train],summaries['variance'][train],residual,weights(development),chosen,'variance')
            save_head(head,destination/'variance',contract,'profiling')
        else:
            base=load_model(destination/'base');head=load_head(destination/'variance',contract,'profiling')
        training_or_load=time.monotonic()-start
        # Warm batch timing excludes source parsing and load, all separately measured.
        batches=[]
        for _ in range(20):
            t=time.perf_counter()
            bp=predict_reference(base.config,base.fitted['reference'],query)
            prediction=bp+predict_head(head,summaries['means_projected'][held],summaries['variance'][held],512)
            batches.append(time.perf_counter()-t)
        one=[]
        for i in range(len(query['pair_ids'])):
            t=time.perf_counter()
            q={k:v[i:i+1] for k,v in query.items()}
            predict_reference(base.config,base.fitted['reference'],q)+predict_head(head,summaries['means_projected'][held][i:i+1],summaries['variance'][held][i:i+1],512)
            one.append(time.perf_counter()-t)
    if stage=='training':np.savez_compressed(destination/'predictions.npz',prediction=prediction)
    else:
        with np.load(destination/'predictions.npz',allow_pickle=False) as z:
            if not np.allclose(prediction,z['prediction'],atol=1e-10,rtol=0):raise ValueError('Profiling replay failed')
    result={'stage':stage,'role':'new profiling only','source':'TNFA-positive','split':'new lexical global pair fold0 stratum3, differs from historical native fold0',
        'environment':environment(),'precision':'float64','threads':1,'training_pairs':int(train.sum()),'held_pairs':int(held.sum()),'genes':512,
        'warm_batch_rows':int(held.sum()),'batch_repeats':20,'training_or_load_wall_seconds':training_or_load,
        'cpu_seconds':time.process_time()-cpu,'warm_batch_median_seconds':float(np.median(batches)),
        'warm_single_median_seconds':float(np.median(one)),'warm_pairs_per_second':int(held.sum())/float(np.median(batches)),
        'checkpoint':str(destination),'checkpoint_bytes':sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),
        'reference_summary_asset_bytes':Path(src['summary']).stat().st_size,'standalone_projection_values':4*512*8,
        'shared_projection_and_summary_storage':'source summary NPZ, not included in head coefficient count',
        'head_numeric_values':read_json(destination/'variance/model.json')['numeric_values'],
        'base_numeric_values':read_json(destination/'base/model.json')['stored_numeric_values'],
        'mse':float(np.mean((prediction-bank['responses'][held])**2)),'replay_max_abs':0 if stage=='replay' else None,
        'host_memory':'fresh-process receipt separately reports sampled tree peak including imports/source preparation',
        'eligible_cross_hardware_pareto_proof':False,'gpu_time_seconds':None}
    write_json(OUT/f'profiling/Opal-{stage}.json',result)
    print(result['mse'],training_or_load)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['training','replay'],required=True);run(p.parse_args().stage)
