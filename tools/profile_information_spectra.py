"""Fresh no-graph PertSpectra training/replay memory on one preserved fold."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

import h5py
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_native_pertspectra_adapter import SPECTRA,column,label,csc


def run(stage):
    torch.set_num_threads(1);threadpool_limits(limits=1)
    destination=Path('C:/Users/USER/Desktop/opal-information-20261011/profiling/PertSpectra')
    destination.mkdir(parents=True,exist_ok=True)
    old=ROOT/'results/opal_model_series_v2/native-completed/native-results/TNFA-positive-3-0.json'
    record=json.loads(old.read_text())
    info=json.loads((ROOT/'results/opal_model_series_v1/pertspectra/access-v4/prepared.json').read_text())['contexts']['TNFA+']
    with np.load(Path(info['local'])/'canonical_bank.npz',allow_pickle=False) as z: bank={k:z[k] for k in ['pair_ids','culture_ids','genes','responses']}
    rows=json.loads((ROOT/'results/opal_model_series_v2/primary/TNFA-positive/conditions.json').read_text())
    row=next(r for r in rows if r['stratum']=='3' and r['outer']==0 and r['budget']==1 and r['order']=='uniform' and r['acquisition_seed']==11 and r['training_seed']==20261017)
    contract=json.loads((Path(row['checkpoints'][0]['path']).parent.parent/'contract.json').read_text())
    acquired=set(contract['acquired_pair_ids']);held=set(contract['held_pair_ids'])
    targets=sorted({t for p in acquired|held for t in json.loads(p)})
    allowed=acquired|{json.dumps([t],separators=(',',':')) for t in targets}|{'[]'}
    preparation_start=time.monotonic()
    with h5py.File(ROOT/'data/raw/pertspectra-model-series/inhouse.h5ad','r') as f:
        ids,contexts,raw,guides=[column(f['obs'][k]) for k in ['SampleIndex','Treatment','target_gene_name','feature_call']]
        labels=np.asarray([label(t) if len(str(t).split('|'))==len(str(g).split('|')) else '' for t,g in zip(raw,guides)])
        selected=(ids=='3')&(contexts=='TNFA+')&np.isin(labels,list(allowed))
        names=column(f['var'][f['var'].attrs['_index']]);matrix=f['layers/counts']
        lib=np.zeros(int(selected.sum()))
        for start in range(0,len(names),128): lib+=np.asarray(csc(matrix,start,min(start+128,len(names)),len(ids))[selected].sum(1)).ravel()
        panel=[int(np.flatnonzero(names==g)[0]) for g in bank['genes']]
        X=np.column_stack([csc(matrix,g,g+1,len(ids))[selected].toarray().ravel() for g in panel]).astype(np.float32)
        X*=10000/np.maximum(lib[:,None],1);X=np.log2(1+X)
    slabels=labels[selected];ordered=['__control__']+targets;lookup={t:i for i,t in enumerate(ordered)}
    def design(names):
        D=np.zeros((len(names),len(ordered)),np.float32)
        for i,p in enumerate(names):
            for t in json.loads(p) or ['__control__']:D[i,lookup[t]]=1
        return torch.from_numpy(D)
    counts=dict(zip(*np.unique(slabels,return_counts=True)));w=np.asarray([1/counts[p] for p in slabels],np.float32);w=torch.from_numpy(w/w.sum())
    torch.manual_seed(20261017)
    model=SPECTRA(X=X,labels=None,adj_matrix=np.array([]),L=record['selected_factors'],pert_idx=np.repeat(-1,len(ordered)),pert_labels=ordered,use_cell_types=False,device=torch.device('cpu'))
    if stage=='replay':
        with np.load(destination/'parameters.npz',allow_pickle=False) as z:model.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files},strict=True)
    preparation_wall=time.monotonic()-preparation_start
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
    done=threading.Event();peak=[0]
    def sample():
        while not done.wait(.01):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    sampler=threading.Thread(target=sample);sampler.start();started,cpu=time.monotonic(),time.process_time()
    history=[]
    if stage=='training':
        xt,dt=torch.from_numpy(X),design(slabels)
        for epoch in range(100):
            optimizer.zero_grad();loss=model.loss_no_cell_types(xt,w,dt)[0];loss.backward();optimizer.step();history.append(float(loss.detach()))
        if (destination/'parameters.npz').exists():raise FileExistsError('Preserve profiling weights')
        np.savez_compressed(destination/'parameters.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    # Use the unchanged aggregate reference, separately from native per-cell fitting.
    with np.load(Path(info['local'])/'canonical_bank.npz',allow_pickle=False) as z:
        # Control is not included in the bank; reconstruct aggregate normalized count control.
        pass
    control=np.log2(1+np.mean(np.exp2(X[slabels=='[]']).astype(np.float64)-1,axis=0))
    test=(bank['culture_ids']=='3')&np.isin(bank['pair_ids'],list(held));td=design(bank['pair_ids'][test])
    with torch.no_grad():prediction=model.loss_no_cell_types(torch.zeros((len(td),512)),torch.ones(len(td)),td,forward=True)[1].numpy()-control
    elapsed=time.monotonic()-started;done.set();sampler.join()
    difference=None
    if stage=='training':np.savez_compressed(destination/'predictions.npz',prediction=prediction)
    else:
        with np.load(destination/'predictions.npz',allow_pickle=False) as z:difference=float(np.max(np.abs(prediction-z['prediction'])))
        if difference>1e-6:raise ValueError('Replay failed')
    result={'method':'PertSpectra','stage':stage,'fully_eligible':False,'classification':'new no-graph protocol-adaptation profile',
        'configuration_source':str(old),'factors':record['selected_factors'],'epochs':100 if stage=='training' else 0,'seed':20261017,
        'precision':'float32','device':'CPU','history':history,'training_pairs':len(acquired),'held_pairs':int(test.sum()),
        'acquired_pair_cells':int(np.isin(slabels,list(acquired)).sum()),'reference_cells':int((~np.isin(slabels,list(acquired))).sum()),
        'held_cells_used':0,'preprocessing_wall_seconds':preparation_wall,'wall_seconds':elapsed,'cpu_seconds':time.process_time()-cpu,
        'peak_stage_rss_sampled_bytes':peak[0],'historical_training_ram_bytes':None,'gpu_seconds':None,
        'mse':float(np.mean((prediction-bank['responses'][test])**2)),'replay_max_abs':difference,'checkpoint':str(destination),
        'checkpoint_bytes':(destination/'parameters.npz').stat().st_size,'reference_bytes':control.nbytes,'parameter_count':sum(p.numel() for p in model.parameters()),
        'latency_ms_per_pair_pipeline':1000*elapsed/len(td) if stage=='replay' else None,
        'limitations':'Reference reconstruction from float32 logged cells is a new profiling adaptation; historical scores unchanged. No native parity claim.'}
    (ROOT/f'results/opal_information_v1/comparators/PertSpectra-{stage}-profile.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['stage','mse','wall_seconds','peak_stage_rss_sampled_bytes','replay_max_abs']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['training','replay'],required=True);run(p.parse_args().stage)
