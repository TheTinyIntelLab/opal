"""Qualify local UPR factorial metadata, freeze source predictions, then score."""
import argparse
import csv
import gzip
import time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from scipy.io import mmread
from threadpoolctl import threadpool_limits
from opal.artifacts import read_json, write_json, sha256, array_hash, environment
from opal.profiles import validate_counts, library_normalize, mean_profile, effect
from opal.contextual.contracts import ReferenceBank, SCHEMA
from opal.state.contracts import validate_response_sources
from tools.run_stage1 import load_data, select_kernel, kernel_prediction
from tools.improve_residual_model import fold_sets

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/experiments/adamson-transfer-20261007.json'
FILE_PREFIX='GSM2406677_10X005_'
CONSTRUCTS={
    '3x_neg_ctrl_pMJ144-1':(), '3x_neg_ctrl_pMJ144-2':(),
    'ATF6_only_pMJ145':('ATF6',), 'PERK_only_pMJ146':('EIF2AK3',), 'IRE1_only_pMJ148':('ERN1',),
    'ATF6_PERK_pMJ150':('ATF6','EIF2AK3'), 'ATF6_IRE1_pMJ152':('ATF6','ERN1'),
    'PERK_IRE1_pMJ154':('EIF2AK3','ERN1'), 'ATF6_PERK_IRE1_pMJ158':('ATF6','EIF2AK3','ERN1')}
BACKGROUNDS={'1':'tunicamycin4ug-mL','2':'thapsigargin100nM','3':'DMSO'}
PRIMARY='https://oar.princeton.edu/bitstream/88435/pr16t0gw6t/1/A%20Multiplexed%20Single-Cell%20CRISPR%20Screening%20Platform%20Enables%20Systematic%20Dissection%20of%20the%20Unfolded%20Protein%20Response.pdf'


def canonical_construct(label):
    if label not in CONSTRUCTS:raise ValueError('Unknown construct; never infer components from substrings')
    return tuple(sorted(CONSTRUCTS[label]))


def joined_metadata(raw):
    with gzip.open(raw/(FILE_PREFIX+'barcodes.tsv.gz'),'rt') as f:barcodes=[line.strip() for line in f]
    if len(set(barcodes))!=len(barcodes):raise ValueError('Duplicate full barcode')
    obs=pd.read_csv(raw/(FILE_PREFIX+'cell_identities.csv.gz'),index_col='cell BC',dtype=str)
    if not obs.index.is_unique:raise ValueError('Duplicate guide metadata barcode')
    obs=obs.reindex(barcodes);obs['barcode']=barcodes
    obs['gemgroup']=[b.rsplit('-',1)[-1] for b in barcodes]
    if set(obs['gemgroup'])!=set(BACKGROUNDS):raise ValueError('Undeclared capture group')
    good=(obs['good coverage']=='True')&(obs['number of cells']=='1')&obs['guide identity'].isin(CONSTRUCTS)
    return obs,good.to_numpy()


def gene_alignment(source,query):
    if len(set(map(str,source)))!=len(source) or len(set(map(str,query)))!=len(query):raise ValueError('Duplicate gene identity')
    lookup={str(g):i for i,g in enumerate(query)}
    source_indices=np.array([i for i,g in enumerate(source) if str(g) in lookup],dtype=int)
    query_indices=np.array([lookup[str(source[i])] for i in source_indices],dtype=int)
    if not len(source_indices):raise ValueError('No exact Ensembl overlap')
    return source_indices,query_indices


def qualify(raw,source,output):
    hashes={}
    for suffix in ['barcodes.tsv.gz','cell_identities.csv.gz','genes.tsv.gz','matrix.mtx.txt.gz']:
        path=raw/(FILE_PREFIX+suffix);expected=read_json(path.with_name(path.name+'.download.json'))
        hashes[path.name]=sha256(path)
        if hashes[path.name]!=expected['sha256'] or path.stat().st_size!=expected['bytes']:raise ValueError('Original source hash/size changed')
    obs,good=joined_metadata(raw)
    genes=pd.read_csv(raw/(FILE_PREFIX+'genes.tsv.gz'),sep='\t',header=None,usecols=[0],dtype=str)[0].to_numpy()
    with np.load(source/'profiles.npz',allow_pickle=False) as f:
        sg=f['genes'];targets=f['targets'];pairs=f['pairs'];si,qi=gene_alignment(sg,genes)
        held_names=[list(p) for p in [('ATF6','EIF2AK3'),('ATF6','ERN1'),('EIF2AK3','ERN1')]]
        rows=[dict(biological_group='K562/CRISPRa/Norman2019',interventions=list(map(str,targets[p])),dose='declared genetic construct',response_kind='pair') for p in pairs]
        validate_response_sources('unseen_pair_observed',dict(fitting=rows,preprocessing=rows,pretraining=[],calibration=[]),held_pairs=held_names)
    counts=Counter((g,label) for g,label in zip(obs.loc[good,'gemgroup'],obs.loc[good,'guide identity']))
    if any(sum(n for (g,label),n in counts.items() if g==gem and canonical_construct(label)==components)<50 for gem in BACKGROUNDS for components in set(CONSTRUCTS.values())):raise ValueError('Factorial metadata coverage incomplete')
    write_json(output/'qualification.json',dict(status='qualified_original_factorial_external_feasibility',raw_hashes=hashes,
        source_profiles_sha256=sha256(source/'profiles.npz'),source_preprocessing_sha256=sha256(source/'preprocessing.json'),
        source_gene_indices=si.tolist(),query_gene_indices=qi.tolist(),aligned_genes=len(si),
        backgrounds=BACKGROUNDS,background_mapping_provenance=dict(url=PRIMARY,page=29,verified_before_expression=True),
        conditions=[dict(gemgroup=g,construct=l,components=list(canonical_construct(l)),good_singlet_cells=n) for (g,l),n in sorted(counts.items())],
        barcode_scope='Full original barcode including suffix; duplicates across capture groups cannot be merged',
        genuine_higher_order=True,unique_triple_identities=1,higher_order_scoring=False,
        license='GEO public local scientific reuse with original attribution; explicit redistribution license unspecified; no raw data/weights redistributed',
        source_url='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2406677',count_provenance='Original Cellranger1.1 UMI gene-barcode matrix per GEO SOFT',
        expression_access=False,prior_local_scores=False,public_exposure='Published source; not guaranteed hidden from public research/pretraining',
        independent_cultures=False,context_capture_confounding=True,protocol_sha256=sha256(CONFIG)))


def check_budget(start,seconds):
    if time.perf_counter()-start>seconds or psutil.Process().memory_info().rss>read_json(CONFIG)['budget']['rss_bytes']:raise RuntimeError('Registered wall/RSS budget exceeded')


def prepare(raw,source,qpath,output):
    start=time.perf_counter();q=read_json(qpath);config=read_json(CONFIG)
    if q['protocol_sha256']!=sha256(CONFIG):raise ValueError('Qualified protocol changed')
    for name,checksum in q['raw_hashes'].items():
        if sha256(raw/name)!=checksum:raise ValueError('Qualified raw source changed')
    obs,good=joined_metadata(raw)
    with gzip.open(raw/(FILE_PREFIX+'matrix.mtx.txt.gz'),'rb') as f:x=mmread(f).T.tocsr()
    validate_counts(x,q['count_provenance'])
    if x.shape[0]!=len(obs):raise ValueError('Count/barcode orientation mismatch')
    libraries=np.asarray(x.sum(axis=1)).ravel();names=obs['guide identity'].fillna('').to_numpy();groups=obs['gemgroup'].to_numpy()
    reference=good&np.array([label in CONSTRUCTS and len(CONSTRUCTS[label])<=1 for label in names])&(libraries>=500)
    upper=float(np.quantile(libraries[reference],.995));keep=good&(libraries>=500)&(libraries<=upper)
    norm=read_json(source/'preprocessing.json')['normalizer']
    if norm!=14048.:raise ValueError('Registered source normalizer changed')
    indices=np.array(q['query_gene_indices']);targets=np.array(['ATF6','EIF2AK3','ERN1']);contexts=np.array(list(BACKGROUNDS.values()))
    controls=[];singles=[];responses=[];pairs=[];ci=[];condition_counts=[]
    def mean(rows):
        if len(rows)<50:raise ValueError('Registered minimum cells not met; no population narrowing')
        # Freeze full-library normalization before selecting aligned genes.
        return mean_profile(library_normalize(x[rows],norm))[indices]
    for context,gem in enumerate(BACKGROUNDS):
        control_rows=np.flatnonzero(keep&(groups==gem)&np.isin(names,[k for k,v in CONSTRUCTS.items() if not v]))
        control=mean(control_rows);controls.append(control);single=[]
        condition_counts.append(dict(context=str(contexts[context]),kind='control',cells=len(control_rows)))
        for target in targets:
            label=next(k for k,v in CONSTRUCTS.items() if v==(target,));rows=np.flatnonzero(keep&(groups==gem)&(names==label))
            single.append(effect(mean(rows),control,'log2'));condition_counts.append(dict(context=str(contexts[context]),kind=str(target),cells=len(rows)))
        singles.append(single)
        for pair in [[0,1],[0,2],[1,2]]:
            components=tuple(sorted(targets[pair]));label=next(k for k in CONSTRUCTS if canonical_construct(k)==components)
            rows=np.flatnonzero(keep&(groups==gem)&(names==label));responses.append(effect(mean(rows),control,'log2'));pairs.append(pair);ci.append(context)
            condition_counts.append(dict(context=str(contexts[context]),kind=' + '.join(components),cells=len(rows)))
        check_budget(start,config['budget']['preparation_seconds'])
    controls=np.array(controls);singles=np.array(singles)
    with np.load(source/'profiles.npz',allow_pickle=False) as f:genes=f['genes'][np.array(q['source_gene_indices'])]
    records={str(c):dict(biological_state='K562/'+str(c),modality='CRISPRi',assay='10x_scRNAseq',study='Adamson2016',source='GSM2406677',batch=gem,replicate='capture library; independent culture unverified') for gem,c in zip(BACKGROUNDS,contexts)}
    metadata=dict(schema=SCHEMA,scale='log2',context_records=records,
        intervention_records={str(t):dict(canonical_target=str(t),reagent=next(k for k,v in CONSTRUCTS.items() if v==(t,)),dose='exact author construct',time='publisher epistasis endpoint',modality='CRISPRi') for t in targets},
        feature_names=[str(g) for g in genes],reference_access='controls_and_measured_singles_all_declared_contexts',
        context_features_source='controls_and_metadata',synthetic=False,
        source_sha256=q['raw_hashes'][FILE_PREFIX+'matrix.mtx.txt.gz'],gene_panel_uses_query_controls=False,
        endpoint='Source-gene-aligned observed-single external feasibility; no independently initiated culture claim')
    bank=ReferenceBank(dict(singles=singles,available=np.ones(singles.shape[:2],bool),controls=np.log2(1+controls),features=np.log2(1+controls),targets=targets,genes=genes,contexts=contexts),metadata)
    bank.save(output/'references')
    evaluator=output/'evaluator';evaluator.mkdir()
    np.savez_compressed(evaluator/'pairs.npz',responses=np.array(responses),pairs=np.array(pairs),contexts=np.array(ci))
    write_json(output/'preparation.json',dict(status='prepared_references_and_separate_evaluator',qualification_sha256=sha256(qpath),
        reference_hash=bank.fingerprint,genes=len(genes),source_gene_indices=q['source_gene_indices'],pairs=pairs,contexts=ci,
        evaluator_sha256=sha256(evaluator/'pairs.npz'),qc=dict(minimum_library=500,maximum_library=upper,minimum_cells=50),
        library_normalizer=norm,conditions=condition_counts,elapsed_seconds=time.perf_counter()-start,
        completion_rss_bytes=psutil.Process().memory_info().rss,peak_working_set_bytes=psutil.Process().memory_info().peak_wset,
        triple_policy='Original genuine triple construct retained in qualification inventory; no triple response scores generated'))


def freeze(source,prepared,qpath,output):
    start=time.perf_counter();config=read_json(CONFIG);q=read_json(qpath)
    if sha256(source/'profiles.npz')!=q['source_profiles_sha256'] or sha256(CONFIG)!=q['protocol_sha256']:raise ValueError('Source/protocol changed')
    data=load_data(source.parent.parent,'norman-log2');gi=np.array(q['source_gene_indices'])
    for k in ['a','b','y']:data[k]=data[k][:,gi]
    data['residual']=data['y']-data['a']-data['b'];data['residual_gram']=data['residual']@data['residual'].T/len(gi)
    train=np.arange(len(data['pairs']));inner=fold_sets(data['pairs'],'random',config['seeds']['split'],3)
    bank=ReferenceBank.load(prepared/'references');prep=read_json(prepared/'preparation.json');pair=np.array(prep['pairs']);ci=np.array(prep['contexts']);bank.validate_rows(pair,ci)
    query_a=bank.arrays['singles'][ci,pair[:,0]];query_b=bank.arrays['singles'][ci,pair[:,1]];query=np.arange(len(train),len(train)+len(pair))
    extended={**data,'a':np.concatenate([data['a'],query_a]),'b':np.concatenate([data['b'],query_b]),
        'pairs':np.concatenate([data['pairs'],len(data['targets'])+pair]),'residual':np.concatenate([data['residual'],np.zeros_like(query_a)])}
    selections={};searches={};pred=dict(additive=query_a+query_b)
    for family in config['menu']['families']:
        choice,search=select_kernel(data,train,inner,family,config['menu']);selections[family]=choice;searches[family]=search
        pred[family]=pred['additive']+kernel_prediction(extended,family,config['menu'],train,query,choice['config'])
        check_budget(start,config['budget']['fitting_seconds'])
    simple=min(['mean_residual','global_correction','single_sum'],key=lambda k:selections[k]['mse']);pair_family=min(['pair_product','mixed'],key=lambda k:selections[k]['mse'])
    pred['selected_simple']=pred[simple];pred['selected_pair']=pred[pair_family]
    np.savez_compressed(output/'predictions.npz',pairs=pair,contexts=ci,**pred)
    write_json(output/'freeze.json',dict(status='source_selected_predictions_frozen_before_evaluation',selected_simple_family=simple,selected_pair_family=pair_family,
        selections=selections,search=searches,protocol_sha256=sha256(CONFIG),reference_hash=bank.fingerprint,
        prepared_manifest_sha256=sha256(prepared/'preparation.json'),evaluator_sha256=prep['evaluator_sha256'],
        prediction_sha256=sha256(output/'predictions.npz'),elapsed_seconds=time.perf_counter()-start,completion_rss_bytes=psutil.Process().memory_info().rss,
        evaluator_opened=False,source_gene_indices=gi.tolist(),environment=environment()))


def evaluate(prepared,frozen,output):
    config=read_json(CONFIG);record=read_json(frozen/'freeze.json');prep=read_json(prepared/'preparation.json');bank=ReferenceBank.load(prepared/'references')
    if record['protocol_sha256']!=sha256(CONFIG) or record['reference_hash']!=bank.fingerprint or record['prepared_manifest_sha256']!=sha256(prepared/'preparation.json'):raise ValueError('Frozen input changed')
    if sha256(frozen/'predictions.npz')!=record['prediction_sha256'] or sha256(prepared/'evaluator/pairs.npz')!=record['evaluator_sha256']:raise ValueError('Frozen predictions/evaluator changed')
    if (frozen/'evaluation-access.json').exists():raise ValueError('This prediction freeze was already evaluated; reuse cannot be fresh')
    write_json(frozen/'evaluation-access.json',dict(status='outcomes_opened_for_registered_evaluation',evaluation_directory=str(output),protocol_sha256=sha256(CONFIG)))
    with np.load(prepared/'evaluator/pairs.npz',allow_pickle=False) as f:y=f['responses'];pair=f['pairs'];ci=f['contexts']
    with np.load(frozen/'predictions.npz',allow_pickle=False) as f:
        if not np.array_equal(pair,f['pairs']) or not np.array_equal(ci,f['contexts']):raise ValueError('Frozen prediction order differs')
        errors={k:np.mean((f[k]-y)**2,axis=1) for k in f.files if k not in ['pairs','contexts']}
    rng=np.random.default_rng(config['seeds']['bootstrap']);weights=[]
    for _ in range(2000):
        sc=np.bincount(rng.integers(3,size=3),minlength=3);tc=np.bincount(rng.integers(3,size=3),minlength=3)
        w=sc[ci]*tc[pair[:,0]]*tc[pair[:,1]]
        if w.sum():weights.append(w)
    comparisons={}
    for k,error in errors.items():
        d=errors['selected_simple']-error;lo,hi=np.quantile([np.average(d,weights=w) for w in weights],[.025,.975]);gain=float(1-error.mean()/errors['selected_simple'].mean())
        comparisons[k]=dict(mse=float(error.mean()),relative_gain=gain,paired_lower=float(lo),paired_upper=float(hi),diagnostic_materiality=bool(gain>=.05 and lo>0),
            by_background={str(bank.arrays['contexts'][c]):float(error[ci==c].mean()) for c in range(3)})
    with (output/'condition-errors.csv').open('w',newline='') as f:
        rows=[dict(background=str(bank.arrays['contexts'][c]),pair=' + '.join(bank.arrays['targets'][p]),**{k:float(v[i]) for k,v in errors.items()}) for i,(p,c) in enumerate(zip(pair,ci))]
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(output/'results.json',dict(status='completed_external_factorial_feasibility',models=comparisons,conditions=9,unique_pairs=3,backgrounds=3,
        aligned_genes=prep['genes'],frozen_selection=dict(simple=record['selected_simple_family'],pair=record['selected_pair_family']),
        bootstrap_nonempty_draws=len(weights),uncertainty=config['uncertainty'],independent_cultures=False,independent_joint_confirmation=False,
        freeze_sha256=sha256(frozen/'freeze.json'),prediction_sha256=record['prediction_sha256'],evaluator_sha256=record['evaluator_sha256'],
        caveat='Drug backgrounds alias capture libraries; only three pair identities; modality/study/chemical-background shifts confounded. First local score does not supply powered or sealed pretreatment-state confirmation.'))
    print(comparisons,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['qualify','prepare','freeze','evaluate']);p.add_argument('--raw',type=Path,default=ROOT/'data/raw/adamson-upr')
    p.add_argument('--source',type=Path,default=ROOT/'data/prepared/norman-log2');p.add_argument('--qualification',type=Path);p.add_argument('--prepared',type=Path);p.add_argument('--frozen',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'execution.json',dict(status='started',phase=args.phase,protocol_sha256=sha256(CONFIG)))
    try:
        with threadpool_limits(limits=1):
            if args.phase=='qualify':qualify(args.raw,args.source,args.output)
            elif args.phase=='prepare':prepare(args.raw,args.source,args.qualification,args.output)
            elif args.phase=='freeze':freeze(args.source,args.prepared,args.qualification,args.output)
            else:evaluate(args.prepared,args.frozen,args.output)
        write_json(args.output/'execution.json',dict(status='completed',phase=args.phase,protocol_sha256=sha256(CONFIG),code_sha256=sha256(__file__)))
    except Exception as error:
        write_json(args.output/'execution.json',dict(status='failed',phase=args.phase,error=repr(error)));raise


if __name__=='__main__':main()
