"""Native PRESAGE input/row compatibility attempt using exact cached priors."""
import sys,json,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from sklearn.decomposition import PCA
ROOT=Path('C:/Users/USER/Desktop/code/thetinyintellab/opal'); LOCAL=Path('D:/opal-model-series-20261010-v2-continuation')
sys.path.insert(0,'D:/opal-model-series-20261010-v2/native/PRESAGE/src')
import presage
output=LOCAL/'presage-native-attempts'; output.mkdir(exist_ok=False)
embeddings=[]; targets=None
for name in ['GenePT','ESM2']:
 with np.load(LOCAL/'presage-assets'/(name+'.npz')) as z:
  targets=z['genes']; values=z['embedding'].copy()
 # Author preparation: feature centering/scaling and PCA across genes.
 rng=np.random.default_rng(20261017); values+=rng.normal(size=values.shape)*1e-6; values-=values.mean(0); values/=np.maximum(values.std(0),1e-12)
 embeddings.append(PCA(n_components=16,svd_solver='full').fit(values.T).components_.T)
embedding=np.stack(embeddings,axis=-1)
presage.prep_gene_embeddings=lambda datamodule,config:embedding.copy()
data=SimpleNamespace(adata=SimpleNamespace(var=pd.DataFrame(index=targets)),X=np.zeros((len(targets),512)),indmtx=np.eye(len(targets)))
datamodule=SimpleNamespace(batch_size=32,train_dataset=data,X_train_pca=None)
scope={'revision':'2c7b231c60cf110c4aab0acf8ecd2c2b3268fffb','native_code_modified':False,'input_preparation_adapter':'exact cached GenePT/ESM2 selected by literal gene, author feature scaling and PCA; no Opal random projection','sources':'CC BY-NC 4.0 local noncommercial research, no redistribution','source_manifest':str(LOCAL/'presage-assets/manifest.json'),'maximum_configs':2,'seed':20261017,'pair_outcomes_accessed':0,'scores':None,'counterexample':'one two-target condition must return one 512-gene profile'}
(output/'scope_before_attempt.json').write_text(json.dumps(scope,indent=2))
records=[]; torch.set_num_threads(1)
for pc in [16,None]:
 config={'added_singles_loss_scale':0,'n_nmf_embedding':16,'pca_dim':pc,'learnable_gene_embedding':False,'softmax_temperature':1,'item_hidden_size':64,'pathway_item_hidden_size':64,'pathway_item_nlayers':1,'batch_norm':False,'pathway_weight_type':'vector','item_nlayers':1,'input_preparation':'prep_gene_embeddings'}
 torch.manual_seed(20261017); start=time.monotonic()
 try:
  model=presage.PRESAGE(config,datamodule,input_dimension=len(targets),output_dimension=512)
  pair=torch.zeros((1,len(targets)),dtype=torch.float32,device=model._device); pair[0,:2]=1
  model=model.to(model._device); prediction=model(pair)[0]
  shape=list(prediction.shape)
  if shape!=[1,512]: raise ValueError('Native pair output shape '+str(shape)+'; no author aggregation returns one profile per pair. Refuse implicit broadcasting or invented sum/mean adapter.')
  record={'status':'pair_contract_supported','output_shape':shape}
 except Exception as error:
  record={'status':'native_pair_contract_failed','error':type(error).__name__+': '+str(error),'traceback':traceback.format_exc()}
 record.update(pca_dimension=pc,wall_seconds=time.monotonic()-start,scores=None,fit_pair_labels=0); records.append(record); print(record['status'],record.get('error',''),flush=True)
(output/'attempts.json').write_text(json.dumps(records,indent=2))
