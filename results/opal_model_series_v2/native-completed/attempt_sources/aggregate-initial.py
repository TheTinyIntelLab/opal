"""Fresh upstream GEARS/CPA aggregate adaptations on frozen primary folds."""
import argparse,copy,gzip,importlib.util,json,os,sys,time,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path('C:/Users/USER/Desktop/code/thetinyintellab/opal')
LOCAL=Path('D:/opal-model-series-20261010-v2-continuation')
sys.path.insert(0,str(ROOT/'tools/external/GEARS'))
sys.path.insert(0,str(ROOT/'tools/external/CPA'))

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def graph(matrix,threshold=.4,neighbors=20):
 x=np.asarray(matrix,float); x=x-x.mean(1,keepdims=True); x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 similarity=x@x.T; edges=[]; weights=[]
 for i,row in enumerate(similarity):
  selected=np.argsort(-row,kind='stable')[:min(neighbors+1,len(row))]
  for j in selected:
   if row[j]>threshold: edges.append((j,i)); weights.append(row[j])
 return np.asarray(edges,dtype=np.int64).T,np.asarray(weights,np.float32)

def gears_fit(x,y,pairs,profiles,configs,seed,go,targets):
 native=module('fresh_series_gears_model',ROOT/'tools/external/GEARS/gears/model.py')
 utils=module('fresh_series_gears_utils',ROOT/'tools/external/GEARS/gears/utils.py')
 g=profiles.shape[1]; count=len(profiles)
 similarity=np.zeros((count,count))
 for i,a in enumerate(targets):
  for j,b in enumerate(targets):
   aa,bb=set(go[a]),set(go[b]); union=aa|bb; similarity[i,j]=len(aa&bb)/len(union) if union else 0
 edges=[]; weights=[]
 for i,row in enumerate(similarity):
  for j in np.argsort(-row,kind='stable')[:min(21,count)]:
   if row[j]>.1: edges.append((j,i)); weights.append(row[j])
 coexpr=graph(profiles.T)
 config=dict(configs,num_genes=g,num_perts=count,device='cuda',uncertainty=False,no_perturb=False,decoder_hidden_size=16,num_go_gnn_layers=1,num_gene_gnn_layers=1,
 G_go=torch.tensor(np.asarray(edges,dtype=np.int64).T),G_go_weight=torch.tensor(weights,dtype=torch.float32),G_coexpress=torch.tensor(coexpr[0]),G_coexpress_weight=torch.tensor(coexpr[1]))
 torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 model=native.GEARS_Model(config).cuda(); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
 interventions=[[i,-1] for i in range(count)]+x.tolist(); values=np.concatenate([profiles,y]); names=[str(i)+'+ctrl' for i in range(count)]+['+'.join(map(str,p)) for p in x]
 # Every measured gene is retained in the loss; no outcome-selected DE mask.
 filters={name:np.arange(g) for name in names}; control=torch.zeros(g,device='cuda')
 def batch(p):
  n=len(p); return SimpleNamespace(x=control.repeat(n).reshape(-1,1),batch=torch.arange(n,device='cuda').repeat_interleave(g),pert_idx=p)
 history=[]; rng=np.random.default_rng(seed)
 for epoch in range(20):
  model.train(); losses=[]
  for ids in np.array_split(rng.permutation(len(values)),max(1,int(np.ceil(len(values)/32)))):
   prediction=model(batch([interventions[i] for i in ids])); loss=utils.loss_fct(prediction,torch.tensor(values[ids],dtype=torch.float32,device='cuda'),[names[i] for i in ids],ctrl=control,direction_lambda=.1,dict_filter=filters)
   optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_value_(model.parameters(),1.); optimizer.step(); losses.append(float(loss.detach()))
  history.append(float(np.mean(losses)))
 model.eval()
 with torch.no_grad(): prediction=np.concatenate([model(batch(pairs[i:i+32].tolist())).cpu().numpy() for i in range(0,len(pairs),32)])
 state={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}; parameter_count=sum(v.numel() for v in model.parameters())
 del model,optimizer; torch.cuda.empty_cache()
 return prediction,state,history,parameter_count

def cpa_fit(x,y,pairs,profiles,config,seed):
 import pandas as pd,anndata as ad,cpa
 records=[{'perturbation':'control','dose':'0.0','context':'source','split':'train'}]; values=[np.zeros(profiles.shape[1])]
 for i,p in enumerate(profiles): records.append({'perturbation':str(i),'dose':'1.0','context':'source','split':'train'}); values.append(p)
 # A fixed acquired partition is supplied by caller; the tuple includes
 # its train/validation flag without consulting outer labels.
 for p,response,split in zip(x,y,config['splits']):
  ordered=sorted(map(str,p)); records.append({'perturbation':'+'.join(ordered),'dose':'1.0+1.0','context':'source','split':split}); values.append(response)
 start=len(values)
 for p in pairs:
  ordered=sorted(map(str,p)); records.append({'perturbation':'+'.join(ordered),'dose':'1.0+1.0','context':'source','split':'ood'}); values.append(np.zeros(profiles.shape[1]))
 data=ad.AnnData(np.asarray(values,np.float32),obs=pd.DataFrame(records)); data.var_names=[f'g{i}' for i in range(profiles.shape[1])]
 cpa.CPA.pert_encoder=None; cpa.CPA.covars_encoder=None; cpa.CPA.pert_smiles_map=None
 cpa.CPA.setup_anndata(data,perturbation_key='perturbation',control_group='control',dosage_key='dose',categorical_covariate_keys=['context'],is_count_data=False)
 model=cpa.CPA(data,split_key='split',train_split='train',valid_split='valid',test_split='ood',recon_loss='gauss',n_latent=32,n_hidden_encoder=config['width'],n_hidden_decoder=config['width'],n_layers_encoder=2,n_layers_decoder=2,use_batch_norm_encoder=False,use_batch_norm_decoder=False,seed=seed)
 model.train(max_epochs=120,use_gpu=True,batch_size=64,save_path=False,early_stopping_patience=121,check_val_every_n_epoch=10,plan_kwargs={'n_epochs_pretrain_ae':0,'n_epochs_adv_warmup':5,'n_hidden_adv':32,'n_layers_adv':2,'use_batch_norm_adv':False,'n_epochs_verbose':1000},enable_progress_bar=False,logger=False,enable_checkpointing=False,num_sanity_val_steps=0)
 query=data[np.arange(start,len(values))].copy(); model.predict(query,batch_size=64,n_samples=1)
 prediction=query.obsm['CPA_pred']; state={k:v.detach().cpu().numpy() for k,v in model.module.state_dict().items()}; n=sum(v.numel() for v in model.module.parameters())
 del model,data; torch.cuda.empty_cache(); return prediction,state,[],n

def run(args):
 output=LOCAL/(args.method+'-aggregate-results'); output.mkdir(exist_ok=True)
 scope={'method':args.method,'classification':'protocol-adapted upstream architecture/trainer','scale':'unchanged aggregate change; zero reference, not raw counts','presence_coefficients':'CPA 1.0 encodes categorical presence only; no physical-dose assertion','seed':20261017,'configs':[64,128],'epochs':20 if args.method=='GEARS' else 120,'primary_budget':'full only; uniform acquisition seed 11','gene_panel':'unchanged primary panel','outer_labels_in_selection':False,'graph':'GEARS public GO Jaccard plus acquired-single-only coexpression; CPA none','native_author_configuration':False,'frontier_eligible':False,'stopping':'fixed epoch cap; inner grouped validation config selection','source':'both separate TNF contexts','prior_asset':'native-assets/manifest.json'}
 freeze=output/'scope_before_fit.json'
 if freeze.exists() and json.loads(freeze.read_text())!=scope: raise ValueError('Changed native adaptation scope')
 write(freeze,scope)
 with gzip.open(LOCAL/'native-assets/gene2go.json.gz','rt',encoding='utf-8') as f: go=json.load(f)
 torch.set_num_threads(1); threadpool_limits(limits=1)
 prepared=json.loads((ROOT/'results/opal_model_series_v1/pertspectra/access-v4/prepared.json').read_text())
 for context,info in prepared['contexts'].items():
  source='TNFA-positive' if context=='TNFA+' else 'TNFA-negative'
  with np.load(Path(info['local'])/'canonical_bank.npz',allow_pickle=False) as z: bank={k:z[k] for k in ['singles','responses','pair_ids','culture_ids']}
  rows=json.loads((ROOT/'results/opal_model_series_v2/primary'/source/'conditions.json').read_text())
  mapping=json.loads((ROOT/'results/opal_model_series_v1/pertspectra'/source/'scope_before_fit.json').read_text())['global_fold_assignment']
  for row in [r for r in rows if r['budget']==1 and r['order']=='uniform' and r['acquisition_seed']==11 and r['training_seed']==20261017]:
   identity=f'{source}-{row["stratum"]}-{row["outer"]}'; dest=output/(identity+'.json')
   if dest.exists(): continue
   roots=[Path('D:/opal-model-series-20261010-v2'),Path('C:/Users/USER/Desktop/opal-model-series-20261010-v2'),LOCAL]
   retained=sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())
   if retained>2000000000 or shutil.disk_usage('D:/').free<1073741824: raise RuntimeError('Artifact cap or D reserve reached')
   contract=json.loads((Path(row['checkpoints'][0]['path']).parent.parent/'contract.json').read_text())
   dev=(bank['culture_ids']==row['stratum'])&np.isin(bank['pair_ids'],contract['acquired_pair_ids']); test=(bank['culture_ids']==row['stratum'])&np.isin(bank['pair_ids'],contract['held_pair_ids'])
   targets=sorted({t for p in bank['pair_ids'][dev|test] for t in json.loads(p)}); index={t:i for i,t in enumerate(targets)}
   profiles=np.empty((len(targets),bank['singles'].shape[-1])); found=set()
   for p,s in zip(bank['pair_ids'][dev],bank['singles'][dev]):
    for t,v in zip(json.loads(p),s): profiles[index[t]]=v; found.add(t)
   if found!=set(targets) or (args.method=='GEARS' and any(t not in go for t in targets)):
    write(dest,{'status':'ineligible_full_population','missing_training_single_or_GO':[t for t in targets if t not in found or args.method=='GEARS' and t not in go],'scores':None}); continue
   pairs=np.asarray([[index[t] for t in json.loads(p)] for p in bank['pair_ids'][dev]]); query=np.asarray([[index[t] for t in json.loads(p)] for p in bank['pair_ids'][test]])
   validation=np.asarray([mapping[p]==(row['outer']+1)%3 for p in bank['pair_ids'][dev]])
   start=time.monotonic(); cpu=time.process_time(); attempts=[]; best=None
   for width in scope['configs']:
    train=~validation
    if args.method=='GEARS': pred,state,h,n=gears_fit(pairs[train],bank['responses'][dev][train],pairs[validation],profiles,{'hidden_size':width},scope['seed'],go,targets)
    else: pred,state,h,n=cpa_fit(pairs,bank['responses'][dev],pairs[validation],profiles,{'width':width,'splits':np.where(validation,'valid','train')},scope['seed'])
    score=float(np.mean((pred-bank['responses'][dev][validation])**2)); attempts.append({'width':width,'development_mse':score,'history':h})
    if best is None or (score,width)<best: best=(score,width)
   width=best[1]
   if args.method=='GEARS': pred,state,h,n=gears_fit(pairs,bank['responses'][dev],query,profiles,{'hidden_size':width},scope['seed'],go,targets)
   else: pred,state,h,n=cpa_fit(pairs,bank['responses'][dev],query,profiles,{'width':width,'splits':np.where(validation,'valid','train')},scope['seed'])
   work=LOCAL/(args.method+'-aggregate-weights')/identity; work.mkdir(parents=True,exist_ok=False)
   np.savez_compressed(work/'parameters.npz',**state); np.savez_compressed(work/'evaluator_arrays.npz',prediction=pred,truth=bank['responses'][test])
   write(dest,{'source':source,'stratum':row['stratum'],'outer':row['outer'],'method':args.method,'classification':scope['classification'],'mse':float(np.mean((pred-bank['responses'][test])**2)),'held_rows':int(test.sum()),'acquired_unique_pairs':int(dev.sum()),'single_profiles':len(targets),'selected_width':width,'attempts':attempts,'parameter_count':n,'checkpoint':str(work),'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'fixed_epoch_training_history':h})
   print(args.method,identity,float(np.mean((pred-bank['responses'][test])**2)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--method',choices=['GEARS','CPA'],required=True); run(p.parse_args())
