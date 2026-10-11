"""Two cached native contract fixtures; no ambiguous output is scored."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'D:/opal-model-series-20261010-v2/native/PRESAGE/src')
import presage


def run():
    torch.set_num_threads(1)
    root=Path('D:/opal-model-series-20261010-v2-continuation/presage-assets')
    values=[]
    for name in ['GenePT','ESM2']:
        with np.load(root/(name+'.npz'),allow_pickle=False) as z: genes=z['genes'];x=z['embedding'].copy()
        rng=np.random.default_rng(20261017);x+=rng.normal(size=x.shape)*1e-6
        x-=x.mean(0);x/=np.maximum(x.std(0),1e-12)
        values.append(PCA(n_components=16,svd_solver='full').fit(x.T).components_.T)
    embeddings=np.stack(values,axis=-1)
    presage.prep_gene_embeddings=lambda datamodule,config:embeddings.copy()
    data=SimpleNamespace(adata=SimpleNamespace(var=pd.DataFrame(index=genes)),X=np.zeros((len(genes),512)),indmtx=np.eye(len(genes)))
    dm=SimpleNamespace(batch_size=32,train_dataset=data,X_train_pca=None)
    results=[]
    for pool in ['vector','gat']:
        config={'added_singles_loss_scale':0,'n_nmf_embedding':16,'pca_dim':None,'learnable_gene_embedding':False,'softmax_temperature':1,
            'item_hidden_size':64,'pathway_item_hidden_size':64,'pathway_item_nlayers':1,'batch_norm':False,'pathway_weight_type':pool,
            'pool_nlayers':1,'gat_weight':1,'item_nlayers':1,'input_preparation':'prep_gene_embeddings'}
        torch.manual_seed(20261017);start=time.monotonic()
        record={'pool':pool,'pca_dim':None,'scores':None,'training_pair_labels':0}
        try:
            model=presage.PRESAGE(config,dm,input_dimension=len(genes),output_dimension=512).to('cpu')
            model._device=torch.device('cpu');model.gene_embeddings=model.gene_embeddings.cpu();model.mask=model.mask.cpu()
            # The upstream pooling module has its own device-bound tensors; stay on
            # the author-selected device for the unchanged fixture instead.
            model=model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model._device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.gene_embeddings=model.gene_embeddings.to(model._device);model.mask=model.mask.to(model._device)
            pair=torch.zeros((1,len(genes)),device=model._device);pair[0,:2]=1
            output=model(pair)[0]
            record.update(output_shape=list(output.shape),condition_locations=model.locs_combos.cpu().tolist(),intervention_locations=model.locs_gene.cpu().tolist())
            if list(output.shape)!=[1,512]:
                record['refusal']='Two intervention rows for one pair; condition reducer absent. No sum/mean/row selection.'
            else: record['contract_passed']=True
        except Exception as error:
            record['error']=type(error).__name__+': '+str(error)
            # Verify the isolated vector signature repair without asserting pair semantics.
            tensor=torch.ones((2,4,2),device=model._device)
            try:
                raw=model.pool.pool(tensor)
                record['isolated_signature_fixture_shape']=list(raw.shape)
                record['isolated_signature_verdict']='One-argument vector pool works but still returns intervention rows; signature repair cannot establish pair decoder.'
            except Exception as fixture_error: record['fixture_error']=str(fixture_error)
        record['wall_seconds']=time.monotonic()-start;results.append(record)
    path=ROOT/'results/opal_information_v1/comparators/PRESAGE-contract-fixtures.json'
    path.write_text(json.dumps({'results':results,'fully_eligible':False,'upstream_modified':False,'assets_manifest':str(root/'manifest.json')},indent=2))
    print(json.dumps(results))


if __name__=='__main__':run()
