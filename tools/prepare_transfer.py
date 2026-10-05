"""Verified public single-screen transfer; no benchmark combination data loaded."""
from pathlib import Path
import argparse,sys,tarfile,pickle,importlib,hashlib
import numpy as np
import pandas as pd
import anndata
import torch
from torch import nn
from scipy import sparse
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import read_json,write_json,load_npz,sha256


class DataOnlyUnpickler(pickle.Unpickler):
    """Allow only declared NumPy/pandas reconstruction primitives, never arbitrary globals."""
    allowed={('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','_reconstruct'),
        ('numpy._core.multiarray','_reconstruct'),('numpy.core.multiarray','scalar'),
        ('numpy._core.multiarray','scalar'),('numpy.core.numeric','_frombuffer'),
        ('numpy._core.numeric','_frombuffer'),('pandas.core.frame','DataFrame'),
        ('pandas.core.series','Series'),('pandas.core.internals.managers','BlockManager'),
        ('pandas._libs.internals','_unpickle_block'),('pandas.core.indexes.base','_new_Index'),
        ('pandas.core.indexes.base','Index'),('pandas.core.indexes.range','RangeIndex'),
        ('collections','OrderedDict'),('builtins','slice'),('builtins','set')}
    def find_class(self,module,name):
        if (module,name) not in self.allowed:raise pickle.UnpicklingError('Unsupported data global: '+module+'.'+name)
        return getattr(importlib.import_module(module),name)


def extract():
    folder=ROOT/'data/architecture_resources/presage_selected';folder.mkdir(exist_ok=True)
    archive=ROOT/'data/architecture_resources/presage_cache.tar.gz'
    manifest=read_json(ROOT/'data/architecture_resources/presage.json')
    entry=next(f for f in manifest['files'] if f['key']=='cache.tar.gz')
    digest=hashlib.md5()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=entry['checksum'].split(':')[-1]:raise ValueError('PRESAGE archive checksum mismatch')
    inventory=[];extracted=[]
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            inventory.append({'name':member.name,'size':member.size,'regular_file':member.isfile()})
            name=member.name.lower()
            is_screen='replogle' in name and 'k562' in name and name.endswith('.h5ad') and ('gw' in name or 'essential' in name)
            is_summary=member.name=='cache/data/replogle_k562_essential_unfiltered/degs/merged.degs.json.gz'
            is_prior=member.name in ['cache/other_embeddings/GenePT_ada.pkl','cache/other_embeddings/esm_emb_gene2esm.pkl']
            if not (is_screen or is_prior or is_summary):continue
            if not member.isfile() or member.size>12*1024**3:raise ValueError('Unexpected archive member')
            # Flat names derived from validated paths avoid any archive path traversal.
            path=folder/member.name.replace('/','__')
            if not path.exists():
                with tar.extractfile(member) as source,path.open('wb') as target:
                    for block in iter(lambda:source.read(1024*1024),b''):target.write(block)
            if path.stat().st_size!=member.size:raise ValueError('Extracted member size mismatch')
            extracted.append({'member':member.name,'local':str(path.relative_to(ROOT)),'sha256':sha256(path)})
            print('Extracted',member.name,flush=True)
    write_json(folder/'inventory.json',inventory)
    write_json(folder/'manifest.json',{'archive_md5':digest.hexdigest(),'source':'https://zenodo.org/records/15587986',
        'extracted':extracted,'benchmark_combination_sources_extracted':False,
        'policy':'only independent Replogle K562 screens and semantic/protein priors; no Norman or Wessels data'})
    return folder


def graph_bank():
    genes=np.loadtxt(ROOT/'OneDrive_1_10-5-2026/scfoundation_genes.txt',dtype=str)
    graph=sparse.load_npz(ROOT/'OneDrive_1_10-5-2026/go_mask_19264.npz')
    random=np.random.default_rng(4147).normal(size=(len(genes),16))/4
    embedding=np.asarray(graph@random)/np.sqrt(np.maximum(np.asarray(graph.sum(1)),1))
    embedding=(embedding-embedding.mean(0))/np.maximum(embedding.std(0),1e-6)
    return genes,embedding.astype(np.float32)


def aggregate_screen(path):
    a=anndata.read_h5ad(path,backed='r')
    field='perturbation' if 'perturbation' in a.obs else 'condition'
    labels=a.obs[field].astype(str).str.replace('+','_',regex=False).str.replace('ctrl','',regex=False).str.strip('_')
    labels=labels.replace('','control').to_numpy()
    unique=np.array(sorted(set(labels)));lookup={label:i for i,label in enumerate(unique)}
    indices=np.array([lookup[l] for l in labels])
    sums=np.zeros((len(unique),a.n_vars),np.float64);counts=np.bincount(indices,minlength=len(unique))
    for start in range(0,a.n_obs,4096):
        stop=min(a.n_obs,start+4096)
        values=a.X[start:stop]
        indicator=sparse.csr_matrix((np.ones(stop-start),(indices[start:stop],np.arange(stop-start))),shape=(len(unique),stop-start))
        part=indicator@values
        sums+=part.toarray() if sparse.issparse(part) else np.asarray(part)
    means=sums/np.maximum(counts[:,None],1)
    control=lookup.get('control')
    if control is None:raise ValueError('External screen has no explicitly harmonized control')
    select=np.array([l!='control' and '_' not in l for l in unique])
    genes=a.var['gene_name'].astype(str).to_numpy() if 'gene_name' in a.var else a.var_names.to_numpy().astype(str)
    result={'targets':unique[select],'effects':(means[select]-means[control]).astype(np.float32),
            'genes':np.asarray(genes,dtype=str),'counts':counts[select]}
    metadata={'input_sha256':sha256(path),'cells':a.n_obs,'output_genes':a.n_vars,'conditions':int(select.sum()),
        'estimator':'mean author-preprocessed X per condition minus mean control X; not Opal raw-count pseudobulk scale',
        'field':field,'control':'control','context':'K562/Replogle2022/CRISPRi',
        'role':'independent representation/decoder pretraining only, never a direct same-scale benchmark'}
    a.file.close()
    return result,metadata


def pretrain(folder):
    output=ROOT/'data/architecture_resources/transfer';output.mkdir(exist_ok=True)
    if (output/'manifest.json').exists():return
    files=[p for p in folder.glob('*.h5ad') if 'k562_gw' in p.name.lower()]
    if not files:files=[p for p in folder.glob('*.h5ad') if 'k562' in p.name.lower()]
    if not files:
        independent=ROOT/'data/architecture_resources/ReplogleWeissman2022_K562_essential.h5ad'
        if independent.exists():files=[independent]
    if not files:raise FileNotFoundError('Fetch the pinned independent Replogle expression screen with tools/download_architecture_resources.py --replogle')
    path=sorted(files,key=lambda p:p.stat().st_size)[0]
    data,metadata=aggregate_screen(path)
    graph_genes,graph=graph_bank();lookup={g:i for i,g in enumerate(graph_genes)}
    select=np.array([t in lookup for t in data['targets']]);targets=data['targets'][select]
    x=graph[[lookup[t] for t in targets]];effects=data['effects'][select]
    if len(targets)<100:raise ValueError('Independent transfer screen too small')
    rng=np.random.default_rng(63829);order=rng.permutation(len(targets));train=order[:int(.8*len(order))];val=order[int(.8*len(order)):]
    mean=effects[train].mean(0)
    # Training-target PCA; randomized SVD is used only on this independent training matrix.
    from sklearn.utils.extmath import randomized_svd
    _,_,vt=randomized_svd(effects[train]-mean,n_components=16,random_state=63829)
    y=(effects-mean)@vt.T;scale=np.maximum(y[train].std(0),1e-6);y=y/scale
    torch.manual_seed(63829);torch.set_num_threads(1)
    network=nn.Sequential(nn.Linear(16,32),nn.Tanh(),nn.Linear(32,16))
    optimizer=torch.optim.AdamW(network.parameters(),lr=.003,weight_decay=.02)
    tx=torch.tensor(x);ty=torch.tensor(y)
    for epoch in range(200):
        batch=rng.choice(train,size=min(256,len(train)),replace=False)
        loss=(network(tx[batch])-ty[batch]).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():error=float((network(tx[val])-ty[val]).square().mean())
    ridge=solve_ridge(x[train],y[train],x[val],10)
    np.savez_compressed(output/'pretrained.npz',weight=network[0].weight.detach().numpy(),bias=network[0].bias.detach().numpy(),
        output_basis=vt.T,genes=data['genes'],targets=targets,train_indices=train,validation_indices=val,
        graph_genes=graph_genes,graph_embedding=graph)
    write_json(output/'manifest.json',{'source':metadata,'epochs':200,'seed':63829,'training_targets':len(train),
        'validation_targets':len(val),'validation_normalized_pc_mse':error,
        'ridge_alpha10_validation_mse':float(np.mean((ridge-y[val])**2)),
        'prior_mean_validation_mse':float(np.mean(y[val]**2)),'pretrained_sha256':sha256(output/'pretrained.npz'),
        'scope':'GO encoder and response directions transferred from a distinct single-perturbation screen',
        'combination_labels_used':False,'context_adapter':'separate pair decoder; external output directions only, no direct CRISPRi-to-Cas13 effect substitution'})
    print('Independent transfer targets',len(targets),'validation PC MSE',error,flush=True)


def solve_ridge(x,y,query,alpha):
    from scipy.linalg import solve
    mean=x.mean(0);x=x-mean
    return (query-mean)@solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@y,assume_a='pos')


def semantic_resources(folder,names,with_transfer=True):
    manifests=read_json(folder/'manifest.json')['extracted']
    selected=[item for item in manifests if item['member'].endswith('.pkl')]
    for item in selected:
        path=ROOT/item['local'];label='GenePT' if 'GenePT' in path.name else 'ESM2'
        output=ROOT/'data/architecture_resources'/('semantic-'+label)
        output.mkdir(exist_ok=True)
        if (output/'manifest.json').exists():continue
        with path.open('rb') as f:data=DataOnlyUnpickler(f).load()
        random=np.random.default_rng(95831)
        if isinstance(data,pd.DataFrame):
            genes=data.index.astype(str).to_numpy().astype(str)
            values=data.to_numpy()
            if values.dtype.kind not in 'fiu':raise ValueError('Semantic DataFrame is not numeric')
            projection=random.normal(size=(values.shape[1],16))/np.sqrt(values.shape[1])
            embedding=(values@projection).astype(np.float32)
        elif isinstance(data,dict):
            genes=np.array([str(k) for k in data])
            first=np.asarray(next(iter(data.values()))).reshape(-1)
            if first.dtype.kind not in 'fiu':raise ValueError('Semantic mapping is not numeric')
            projection=random.normal(size=(len(first),16))/np.sqrt(len(first))
            embedding=np.empty((len(genes),16),np.float32)
            for start in range(0,len(genes),1024):
                chunk=np.stack([np.asarray(data[g]).reshape(-1) for g in genes[start:start+1024]])
                embedding[start:start+len(chunk)]=chunk@projection
        else:raise ValueError('Unknown declared semantic embedding format')
        if not np.isfinite(embedding).all():raise ValueError('Nonfinite external embeddings')
        embedding=(embedding-embedding.mean(0))/np.maximum(embedding.std(0),1e-6)
        np.savez_compressed(output/'embedding.npz',genes=np.asarray(genes,dtype=str),embedding=embedding)
        write_json(output/'manifest.json',{'source_sha256':item['sha256'],'embedding_sha256':sha256(output/'embedding.npz'),
            'projection':'fixed random projection, seed 95831, normalized on external gene bank only',
            'genes':len(genes),'source':label,'source_terms':'PRESAGE knowledge embeddings CC BY-NC 4.0'})
        del data
        print(label,'external gene embeddings',len(genes),flush=True)
    for name in names:
        base=ROOT/'data/architecture_resources'/('transfer-'+name if with_transfer else name)
        output=ROOT/'data/architecture_resources'/(('enriched-' if with_transfer else 'semantic-only-')+name);output.mkdir(exist_ok=True)
        if (output/'manifest.json').exists():continue
        resources=load_npz(base/'resources.npz');banks=[resources['bank']];masks=[resources['source_mask']]
        source_names=resources['source_names'].tolist();coverage={}
        for label in ['GenePT','ESM2']:
            source=ROOT/'data/architecture_resources'/('semantic-'+label)/'embedding.npz'
            data=load_npz(source);lookup={str(g):i for i,g in enumerate(data['genes'])}
            values=np.zeros((len(resources['targets']),16),np.float32);known=np.zeros(len(values),bool)
            for i,g in enumerate(resources['targets']):
                if str(g) in lookup:values[i]=data['embedding'][lookup[str(g)]];known[i]=True
            banks.append(values[:,None]);masks.append(known[:,None]);source_names.append(label);coverage[label]=int(known.sum())
        resources['bank']=np.concatenate(banks,axis=1);resources['source_mask']=np.concatenate(masks,axis=1)
        resources['source_names']=np.array(source_names)
        np.savez_compressed(output/'resources.npz',**resources)
        write_json(output/'manifest.json',{'base_sha256':sha256(base/'resources.npz'),
            'resources_sha256':sha256(output/'resources.npz'),'semantic_target_coverage':coverage,
            'benchmark_responses_used_for_semantic_features':False})

def dataset_resources(name):
    base=ROOT/'data/architecture_resources'/name
    output=ROOT/'data/architecture_resources'/('transfer-'+name);output.mkdir(exist_ok=True)
    if (output/'manifest.json').exists():return
    resources=load_npz(base/'resources.npz');pretrained=load_npz(ROOT/'data/architecture_resources/transfer/pretrained.npz')
    lookup={str(g):i for i,g in enumerate(pretrained['genes'])}
    basis=np.zeros((len(resources['genes']),pretrained['output_basis'].shape[1]),np.float32)
    for i,g in enumerate(resources['gene_symbols']):
        if str(g) in lookup:basis[i]=pretrained['output_basis'][lookup[str(g)]]
    resources.update(transfer_weight=pretrained['weight'],transfer_bias=pretrained['bias'],transfer_output_basis=basis)
    np.savez_compressed(output/'resources.npz',**resources)
    write_json(output/'manifest.json',{'base_sha256':sha256(base/'resources.npz'),
        'transfer_sha256':sha256(ROOT/'data/architecture_resources/transfer/pretrained.npz'),
        'resources_sha256':sha256(output/'resources.npz'),'mapped_output_genes':int(np.sum(np.any(basis!=0,axis=1))),
        'transfer_context':'independent K562 CRISPRi screen; separate current-context pair head'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--semantic-only',action='store_true');parser.add_argument('--datasets',nargs='+',default=['norman-log2','wessels-log2','wessels-expanded-log2'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        folder=extract()
        if not args.semantic_only:
            pretrain(folder)
            for name in args.datasets:dataset_resources(name)
        semantic_resources(folder,args.datasets,not args.semantic_only)


if __name__=='__main__':main()
