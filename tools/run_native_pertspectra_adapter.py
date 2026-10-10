"""Bounded upstream PertSpectra CPU core with a declared no-graph adaptation.

Per-cell inputs are extra information. Scores are protocol-adapted native-core
scores, never the authors' native graph configuration or an Opal module.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,'D:/opal-model-series-20261010-v2/native/PertSpectra/src')
from Spectra.Spectra_Pert import SPECTRA


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')


def column(node):
    if isinstance(node,h5py.Group):
        categories=column(node['categories']); return categories[node['codes'][:]]
    return np.asarray([v.decode() if isinstance(v,bytes) else str(v) for v in node[:]])


def label(raw):
    parts=str(raw).split('|')
    if len(parts)>2 or any(not p or p=='nan' or p.startswith('INTERGENIC') for p in parts): return None
    parts=sorted(p for p in parts if not p.startswith('CTRL00'))
    if len(parts)!=len(set(parts)): return None
    return json.dumps(parts,separators=(',',':'))


def csc(matrix,start,stop,n):
    ptr=matrix['indptr'][start:stop+1]; left,right=int(ptr[0]),int(ptr[-1])
    return sparse.csc_matrix((matrix['data'][left:right],matrix['indices'][left:right],ptr-left),shape=(n,stop-start))


def run(args):
    torch.set_num_threads(1)
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    scope={'revision':'9c4fbdacdadcb8ca559226170e93e6b61436819c','implementation':'unchanged upstream Spectra.Spectra_Pert.SPECTRA CPU core',
           'classification':'protocol-adapted native core','graph':'none; deposited h5ad has no sparse_gene_network and original graph is unavailable',
           'factors':[20,40],'epochs':100,'optimizer':'AdamW, lr .01, weight_decay .001','seed':20261017,
           'scope':'full-budget uniform primary acquisition seed 11; original primary genes/pairs/outer folds; extra observed per-cell information',
           'endpoint':'unchanged primary log2_library_10000_change MSE','native_author_configuration':False,
           'frontier_native_eligibility':False,'held_pair_outcomes_used_in_selection':False,
           'model_input':'log2(1+library-normalized per-cell counts); controls and observed singles permitted; pair cells acquired only',
           'condition_weights':'equal-condition mean loss, inverse cell count per acquired condition',
           'normalization_order_mismatch':'native fits per-cell log; primary responses log aggregate normalized counts, retained and reported explicitly'}
    freeze=output/'scope_before_fit.json'
    if freeze.exists() and json.loads(freeze.read_text())!=scope: raise ValueError('Native adaptation freeze changed')
    write(freeze,scope)
    prepared=json.loads((ROOT/'results/opal_model_series_v1/pertspectra/access-v4/prepared.json').read_text())
    with h5py.File(ROOT/'data/raw/pertspectra-model-series/inhouse.h5ad','r') as f:
        ids,treatment,raw=[column(f['obs'][k]) for k in ['SampleIndex','Treatment','target_gene_name']]
        names=column(f['var'][f['var'].attrs.get('_index','_index')])
        guides=column(f['obs']['feature_call'])
        labels=np.asarray([label(r) if len(str(r).split('|'))==len(str(g).split('|')) else None for r,g in zip(raw,guides)],dtype=object)
        n=len(ids); matrix=f['layers/counts']; libraries=np.zeros(n)
        for start in range(0,len(names),128): libraries+=np.asarray(csc(matrix,start,min(start+128,len(names)),n).sum(1)).ravel()
        for context,info in prepared['contexts'].items():
            source='TNFA-positive' if context=='TNFA+' else 'TNFA-negative'
            with np.load(Path(info['local'])/'canonical_bank.npz',allow_pickle=False) as z:
                bank={k:z[k] for k in ['singles','responses','pair_ids','culture_ids','genes']}
            primary=json.loads((ROOT/'results/opal_model_series_v2/primary'/source/'conditions.json').read_text())
            panel=np.asarray([int(np.flatnonzero(names==gene)[0]) for gene in bank['genes']])
            # Read only the frozen panel after full library sums; never densify all genes.
            columns=[]
            for gene in panel: columns.append(csc(matrix,int(gene),int(gene)+1,n))
            panel_counts=sparse.hstack(columns,format='csr')
            for stratum in sorted(set(bank['culture_ids'])):
                candidates=[r for r in primary if r['stratum']==stratum and r['budget']==1 and r['order']=='uniform' and r['acquisition_seed']==11 and r['training_seed']==20261017]
                for condition in candidates:
                    identity=f'{source}-{stratum}-{condition["outer"]}'
                    destination=output/(identity+'.json')
                    if destination.exists(): continue
                    if __import__('shutil').disk_usage('C:/').free<1073741824: raise RuntimeError('C reserve reached')
                    contract_path=Path(condition['checkpoints'][0]['path']).parent.parent/'contract.json'
                    contract=json.loads(contract_path.read_text())
                    held=set(contract['held_pair_ids']); acquired=set(contract['acquired_pair_ids'])
                    fold_map=json.loads((ROOT/'results/opal_model_series_v1/pertspectra'/source/'scope_before_fit.json').read_text())['global_fold_assignment']
                    validation={p for p in acquired if fold_map[p]==(condition['outer']+1)%3}
                    train_pairs=acquired-validation
                    targets=sorted({t for p in acquired|held for t in json.loads(p)})
                    singles={json.dumps([t],separators=(',',':')) for t in targets}
                    mask=(ids==stratum)&(treatment==context)&(libraries>0)
                    tr=mask & np.asarray([p in train_pairs|singles|{'[]'} for p in labels])
                    va=mask & np.asarray([p in validation for p in labels])
                    selected=tr|va
                    X=panel_counts[selected].toarray().astype(np.float32)
                    X*=10000/libraries[selected,None]; X=np.log2(1+X)
                    selected_labels=labels[selected]; trainmask=tr[selected]; valmask=va[selected]
                    alltargets=['__control__']+targets
                    lookup={t:i for i,t in enumerate(alltargets)}
                    D=np.zeros((selected.sum(),len(alltargets)),np.float32)
                    for i,p in enumerate(selected_labels):
                        for t in json.loads(p) or ['__control__']: D[i,lookup[t]]=1
                    def target_design(pair_ids):
                        design=np.zeros((len(pair_ids),len(alltargets)),np.float32)
                        for i,p in enumerate(pair_ids):
                            for t in json.loads(p) or ['__control__']: design[i,lookup[t]]=1
                        return torch.from_numpy(design)
                    Xt,Dt=torch.from_numpy(X[trainmask]),torch.from_numpy(D[trainmask])
                    counts=dict(zip(*np.unique(selected_labels[trainmask],return_counts=True)))
                    w=np.asarray([1/counts[p] for p in selected_labels[trainmask]],np.float32)
                    wt=torch.from_numpy(w/w.sum())
                    # Validation truth is acquired, never outer truth. Use the frozen aggregate endpoint.
                    bmask=(bank['culture_ids']==stratum)&np.isin(bank['pair_ids'],list(validation))
                    control_rows=mask&(labels=='[]')
                    control=np.log2(1+np.asarray(panel_counts[control_rows].multiply(10000/libraries[control_rows,None]).mean(0)).ravel())
                    records=[]; best=None; started=time.monotonic(); cpu=time.process_time()
                    for factors in scope['factors']:
                        torch.manual_seed(scope['seed'])
                        model=SPECTRA(X=X[trainmask],labels=None,adj_matrix=np.array([]),L=factors,
                              pert_idx=np.repeat(-1,len(alltargets)),pert_labels=alltargets,use_cell_types=False,device=torch.device('cpu'))
                        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
                        history=[]
                        for epoch in range(scope['epochs']):
                            optimizer.zero_grad(); loss=model.loss_no_cell_types(Xt,wt,Dt)[0]; loss.backward(); optimizer.step()
                            history.append(float(loss.detach()))
                        with torch.no_grad():
                            vd=target_design(bank['pair_ids'][bmask]); vx=torch.zeros((len(vd),len(panel)))
                            prediction=model.loss_no_cell_types(vx,torch.ones(len(vd)),vd,forward=True)[1].numpy()-control
                        score=float(np.mean((prediction-bank['responses'][bmask])**2))
                        records.append({'factors':factors,'development_mse':score,'history':history})
                        if best is None or (score,factors)<best[:2]: best=(score,factors,model)
                    # Frozen selected configuration is refitted on ALL acquired cells; outer pairs stay excluded.
                    factors=best[1]; torch.manual_seed(scope['seed'])
                    model=SPECTRA(X=X,labels=None,adj_matrix=np.array([]),L=factors,pert_idx=np.repeat(-1,len(alltargets)),pert_labels=alltargets,use_cell_types=False,device=torch.device('cpu'))
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
                    counts=dict(zip(*np.unique(selected_labels,return_counts=True)))
                    w=np.asarray([1/counts[p] for p in selected_labels],np.float32); w=torch.from_numpy(w/w.sum())
                    xt,dt=torch.from_numpy(X),torch.from_numpy(D)
                    for epoch in range(scope['epochs']):
                        optimizer.zero_grad(); loss=model.loss_no_cell_types(xt,w,dt)[0]; loss.backward(); optimizer.step()
                    test=(bank['culture_ids']==stratum)&np.isin(bank['pair_ids'],list(held))
                    with torch.no_grad():
                        td=target_design(bank['pair_ids'][test]); tx=torch.zeros((len(td),len(panel)))
                        prediction=model.loss_no_cell_types(tx,torch.ones(len(td)),td,forward=True)[1].numpy()-control
                    work=args.local/identity; work.mkdir(parents=True,exist_ok=False)
                    np.savez_compressed(work/'native_parameters.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
                    np.savez_compressed(work/'evaluator_arrays.npz',prediction=prediction,truth=bank['responses'][test])
                    result={'source':source,'stratum':stratum,'outer':condition['outer'],'classification':scope['classification'],
                            'mse':float(np.mean((prediction-bank['responses'][test])**2)),'held_rows':int(test.sum()),
                            'selected_factors':factors,'attempts':records,'acquired_unique_pairs':len(acquired),'pair_cells_accessed':int((selected & np.isin(labels,list(acquired))).sum()),
                            'observed_single_control_cells':int((selected & ~np.isin(labels,list(acquired))).sum()),
                            'held_cells_used':0,'parameter_count':sum(p.numel() for p in model.parameters()),'checkpoint':str(work),
                            'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu}
                    write(destination,result); print(identity,result['mse'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/opal_model_series_v2/native/pertspectra-adapted')
    parser.add_argument('--local',type=Path,default=Path('C:/Users/USER/Desktop/opal-model-series-20261010-v2/native-adapted'))
    run(parser.parse_args())
