"""Prepare separately registered broader coverage and reference-only resources."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from scipy import sparse
import anndata
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import read_json,write_json,load_npz,sha256,array_hash
from opal.data.prepare import load_raw,prepare
from opal.profiles import library_normalize,effect
from opal.models.advanced import reference_embedding


def expanded():
    output=ROOT/'data/prepared/wessels-expanded-log2'
    if output.exists():
        if not (output/'profiles.npz').exists():
            raise ValueError('Incomplete expanded preparation; inspect before resuming')
        return
    config=read_json(ROOT/'configs/datasets/wessels-log2.json')
    config['output']=str(output.relative_to(ROOT))
    config['qc']['minimum_cells']=15
    config['qc_amendment']='Separate exploratory 15-cell track; specified from metadata coverage before fitting; full historical track unchanged.'
    write_json(ROOT/'configs/datasets/wessels-expanded-log2.json',config)
    write_json(ROOT/'configs/experiments/architecture-expansion.json',{
        'minimum_cells':15,'matching':'same exact stratum-support and construct rules as historical preparation',
        'seed':1729,'purpose':'broader-coverage exploratory architecture experiments',
        'historical_outcomes_known':True,'independent_confirmation':False,
        'source_qc_sha256':sha256(ROOT/'data/prepared/wessels-log2/qc_report.json'),
        'no_historical_evaluator_used_for_selection':True})
    prepare(config)


def protected_expanded():
    output=ROOT/'data/prepared/wessels-expanded-protected-log2'
    if (output/'profiles.npz').exists():return
    config=read_json(ROOT/'configs/datasets/wessels-expanded-log2.json')
    config['output']=str(output.relative_to(ROOT))
    old=load_npz(ROOT/'data/prepared/wessels-log2/profiles.npz')
    config['preserved_split']={'development':old['targets'][old['pairs']].tolist(),
        'protected':old['targets'][old['query_pairs']].tolist(),
        'source_profiles_sha256':sha256(ROOT/'data/prepared/wessels-log2/profiles.npz')}
    config['qc_amendment']='Exploratory 15-cell expansion preserving all original 40 development and 18 holdout pair roles; previously explored expanded responses are not independent confirmation.'
    write_json(ROOT/'configs/datasets/wessels-expanded-protected-log2.json',config)
    prepare(config)

def symbol_mapping():
    source=ROOT/'data/raw/NormanWeissman2019_filtered.h5ad'
    a=anndata.read_h5ad(source,backed='r')
    result=dict(zip(a.var['ensemble_id'].astype(str),a.var.index.astype(str)))
    a.file.close()
    return result


def joint_reference_draws(name,data,draws,seed,return_draws=False):
    base=name
    config=read_json(ROOT/f'configs/datasets/{base}.json')
    dataset=ROOT/'data/prepared'/name
    x,genes,labels,strata,cells,digest,source_n=load_raw(config)
    mapping=config['condition_map']; qc=config['qc']
    totals=np.asarray(x.sum(1)).ravel()
    valid=(totals>=qc['minimum_library'])&(totals<=qc['maximum_library'])
    valid &= np.array([mapping[l]['context']==config['context'] and not mapping[l].get('exclude',False) for l in labels])
    x,labels,strata=x[valid],labels[valid],strata[valid]
    prep=read_json(dataset/'preprocessing.json');keep=np.array(prep['gene_mask'],bool)
    if not np.array_equal(genes[keep],data['genes']):raise ValueError('Reference gene order mismatch')
    # Normalize libraries before selecting output genes.
    x=library_normalize(x,prep['normalizer'])[:,keep].tocsr()
    pair_groups=read_json(dataset/'qc_report.json')['pair_groups']
    development_groups=[pair_groups[int(i)] for i in data['pair_indices']]
    single_labels={t:next(l for l,v in mapping.items() if v['components']==[t] and np.any(labels==l)) for t in data['targets']}
    required=set(single_labels.values())|{l for group in development_groups for l in group}
    groups={(l,s):np.flatnonzero((labels==l)&(strata==s)) for l in sorted(required) for s in sorted(set(strata[labels==l]))}
    controls=np.array([len(mapping[l]['components'])==0 for l in labels])
    control_groups={s:np.flatnonzero(controls&(strata==s)) for s in sorted(set(strata))}
    rng=np.random.default_rng(seed)
    def profiles(means,control):
        return {l:np.mean([effect(means[l,s],control[s],config['scale']) for ll,s in groups if ll==l],axis=0) for l in required}
    point=profiles({k:np.asarray(x[idx].mean(0)).ravel() for k,idx in groups.items()},
                   {s:np.asarray(x[idx].mean(0)).ravel() for s,idx in control_groups.items()})
    point_single=np.stack([point[single_labels[t]] for t in data['targets']])
    point_double=np.stack([np.mean([point[l] for l in group],axis=0) for group in development_groups])
    if not np.allclose(point_single,data['singles'],atol=2e-6) or not np.allclose(point_double,data['doubles'],atol=2e-6):
        raise ValueError('Raw reconstruction differs from frozen development profiles')
    singles=[];doubles=[]
    def sample(idx):
        weights=sparse.csr_matrix(rng.multinomial(len(idx),np.full(len(idx),1/len(idx)),size=draws)/len(idx))
        return (weights@x[idx]).toarray()
    # A shared control draw and single draw are reused for all relevant pair residuals.
    controls={s:sample(idx) for s,idx in control_groups.items()}
    means={k:sample(idx) for k,idx in groups.items()}
    sampled=profiles(means,controls)
    singles=np.stack([sampled[single_labels[t]] for t in data['targets']],axis=1).astype(np.float32)
    doubles=np.stack([np.mean([sampled[l] for l in group],axis=0) for group in development_groups],axis=1).astype(np.float32)
    residual=doubles-singles[:,data['pairs'][:,0]]-singles[:,data['pairs'][:,1]]
    if return_draws:return singles,doubles,residual
    return singles, doubles.var(0,ddof=1), residual.var(0,ddof=1)


def resources(name,draws=16,seed=90317):
    folder=ROOT/'data/architecture_resources'/name
    if (folder/'manifest.json').exists():
        m=read_json(folder/'manifest.json')
        if sha256(folder/'resources.npz')!=m['resources_sha256']:raise ValueError('Resource cache changed')
        return
    folder.mkdir(exist_ok=True)
    dataset=ROOT/'data/prepared'/name;data=load_npz(dataset/'profiles.npz')
    mapping=symbol_mapping()
    symbols=np.array([mapping.get(str(g).split('.')[0],'') for g in data['genes']])
    gene_list=np.loadtxt(ROOT/'OneDrive_1_10-5-2026/scfoundation_genes.txt',dtype=str)
    go=sparse.load_npz(ROOT/'OneDrive_1_10-5-2026/go_mask_19264.npz').tocsr()
    if go.shape!=(len(gene_list),len(gene_list)):raise ValueError('GO schema mismatch')
    rng=np.random.default_rng(4147)
    projection=rng.normal(size=(len(gene_list),16))/4
    graph=np.asarray(go@projection)/np.sqrt(np.maximum(np.asarray(go.sum(1)),1))
    lookup={str(t):i for i,t in enumerate(gene_list)}
    graph=(graph-graph.mean(0))/np.maximum(graph.std(0),1e-6)
    def align(values,labels):
        output=np.zeros((len(labels),values.shape[1]),np.float32)
        known=np.zeros(len(labels),bool)
        for i,t in enumerate(labels):
            if str(t) in lookup: output[i]=values[lookup[str(t)]];known[i]=True
        return output,known
    target_graph,target_known=align(graph,data['targets']); gene_graph,gene_known=align(graph,symbols)
    annotations=[]; names=[]
    for line in (ROOT/'data/architecture_resources/hallmark.txt').read_text().splitlines():
        fields=line.split('\t');names.append(fields[0]); annotations.append(set(fields[2:]))
    membership=np.array([[float(s in pathway) for pathway in annotations] for s in symbols],np.float32)
    target_membership=np.array([[float(s in pathway) for pathway in annotations] for s in data['targets']],np.float32)
    measured,projection,mean,scale=reference_embedding(data['singles'],16)
    dimension=16
    def pad(features):
        return np.pad(features,((0,0),(0,dimension-features.shape[1])))[:,:dimension].astype(np.float32)
    pathway,_,_,_=reference_embedding(target_membership,16)
    bank=np.stack([pad(measured),target_graph,pad(pathway)],axis=1)
    mask=np.stack([np.ones(len(data['targets']),bool),target_known,target_membership.sum(1)>0],axis=1)
    print(name,'sampling shared references',flush=True)
    sampled,double_variance,residual_variance=joint_reference_draws(name,data,draws,seed)
    feature_draws=pad(((sampled-mean)@projection/scale).reshape(-1,measured.shape[1])).reshape(draws,len(data['targets']),16)
    arrays={'bank':bank,'source_mask':mask,'membership':membership,'gene_prior':gene_graph,
        'single_variance':sampled.var(0,ddof=1),'double_variance':double_variance,'residual_variance':residual_variance,
        'single_draw_features':feature_draws,'targets':data['targets'],'genes':data['genes'],'gene_symbols':symbols,
        'source_names':np.array(['measured_single','GO_graph','Hallmark']), 'module_names':np.array(names),
        'measurement_projection':projection.astype(np.float32),'measurement_mean':mean.astype(np.float32),
        'measurement_scale':scale.astype(np.float32)}
    np.savez_compressed(folder/'resources.npz',**arrays)
    write_json(folder/'manifest.json',{'dataset':name,'seed':seed,'draws':draws,
        'resources_sha256':sha256(folder/'resources.npz'),'profiles_sha256':sha256(dataset/'profiles.npz'),
        'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'data/architecture_resources/hallmark.txt',ROOT/'OneDrive_1_10-5-2026/go_mask_19264.npz',ROOT/'OneDrive_1_10-5-2026/scfoundation_genes.txt']},
        'matched_target_graph':int(target_known.sum()),'matched_output_graph':int(gene_known.sum()),
        'output_genes':len(symbols),'source_policy':'GO and Hallmark contain annotations only; expression features and bases use allowed singles',
        'bootstrap_policy':'controls/singles and DEVELOPMENT doubles only; no query double construction or evaluator read',
        'uncertainty_scope':'technical cell sampling with frozen QC, normalizer and gene panel; not biological variance',
        'query_response_access':False})
    print(name,'resources ready',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--expanded',action='store_true')
    parser.add_argument('--protected',action='store_true')
    parser.add_argument('--datasets',nargs='+',default=['norman-log2','wessels-log2'])
    parser.add_argument('--draws',type=int,default=16)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.expanded:expanded()
        if args.protected:protected_expanded()
        for name in args.datasets:resources(name,args.draws)


if __name__=='__main__':main()
