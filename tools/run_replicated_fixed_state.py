"""Metadata qualification, preparation, rehearsal, freeze and custody scoring.

Run --help for subcommands. Demo is synthetic engineering evidence only.
"""
import argparse
import json
from pathlib import Path
import sys
import itertools
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.replicated.contracts import qualify, split_pairs, write_json, sha
from opal.replicated.data import prepare, query_references, validate_preparation_access
from opal.replicated.workflow import freeze, evaluate, load_bank, restore_predictions
from opal.replicated.models import select, fit_candidate, predict_candidate, balanced_weights
from opal.replicated.evaluation import reliability, empirical_power, cube, decomposition


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def diagnostics(bank, menu, seed=1729):
    if np.any(np.isin(bank['partitions'],['test','guard'])):raise ValueError('Rehearsal received held outcomes')
    selection=select(bank,menu);val=bank['partitions']=='validation';train=bank['partitions']=='train'
    validation={k:v[val] if k in {'singles','responses','pair_ids','culture_ids','row_ids','partitions'} else v for k,v in bank.items()}
    w=balanced_weights(validation['pair_ids'],validation['culture_ids'])
    model=fit_candidate(selection['reference'],bank,train)
    pred=predict_candidate(selection['reference'],model,validation)
    controls={name:float(w@np.mean((p-validation['responses'])**2,axis=1)) for name,p in {
        'exact_response':validation['responses'],'additive':validation['singles'].sum(1),
        'zero_effect':np.zeros_like(validation['responses']),'shared_reference':pred}.items()}
    # Cross-fit globally grouped pairs; shared corrections cannot fit the query pair.
    pairs=np.unique(bank['pair_ids']);rng=np.random.default_rng(seed);rng.shuffle(pairs)
    crossfit=np.empty_like(bank['responses'])
    for held in np.array_split(pairs,min(5,len(pairs))):
        mask=~np.isin(bank['pair_ids'],held);query=~mask
        fitted=fit_candidate(selection['reference'],bank,mask)
        qb={k:v[query] if k in {'singles','responses','pair_ids','culture_ids','row_ids','partitions'} else v for k,v in bank.items()}
        crossfit[query]=predict_candidate(selection['reference'],fitted,qb)
    raw=bank['responses']-bank['singles'].sum(1);departure=bank['responses']-crossfit
    coverage=[];remaining=set(bank['pair_ids'][train]);ordered=[];degrees={}
    while remaining:
        pair=min(remaining,key=lambda p:(sum(degrees.get(c,0) for c in json.loads(p)),sha_text(str(p)+str(seed))))
        remaining.remove(pair);ordered.append(pair)
        for c in json.loads(pair):degrees[c]=degrees.get(c,0)+1
    for fraction in (.25,.5,.75,1.):
        retained=ordered[:max(1,int(len(ordered)*fraction))]
        subset=train&np.isin(bank['pair_ids'],retained)
        m=fit_candidate(selection['candidate'],bank,subset)
        p=predict_candidate(selection['candidate'],m,validation)
        degree={}
        for pair in retained:
            for c in json.loads(pair):degree[c]=degree.get(c,0)+1
        coverage.append({'fraction':fraction,'distinct_training_pairs':len(retained),'partner_degrees':degree,
                         'validation_mse':float(w@np.mean((p-validation['responses'])**2,axis=1)),
                         'validation_pair_ids':sorted(set(validation['pair_ids'])),'retuned':False})
    report={'selection':selection,'metric_controls':controls,'coverage_curve':coverage,
            'crossfit_shared_departure_norms':np.linalg.norm(departure,axis=1).tolist(),
            'raw_residual_norms':np.linalg.norm(raw,axis=1).tolist(),
            'access_log':{'development_only':bank['row_ids'].tolist()},'evaluator_access':False,
            'independent_reference_decomposition':decomposition(bank['singles'].sum(1),crossfit,bank['singles'].sum(1),bank['responses']),
            'reference_precision':'requires independently partitioned whole reference banks; not inferred from coverage',
            'biological_claim':'only qualified independent cultures support a biological interpretation'}
    try:report['reliability']=reliability(bank)
    except ValueError as exc:report['reliability']={'blocked_on':str(exc)}
    return report


def sha_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def demo(output, config):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20261008);targets=[f'T{i}' for i in range(8)]
    genes=np.asarray([f'G{i}' for i in range(12)])
    pairs=[json.dumps(list(p),separators=(',',':')) for p in itertools.combinations(targets,2)]
    split={p:('train' if i<16 else 'validation' if i<22 else 'test') for i,p in enumerate(pairs)}
    base=rng.normal(0,.3,(8,len(genes)));shared=rng.normal(0,.05,len(genes));metadata=[]
    def bank(cultures,query=False):
        arrays={k:[] for k in ('singles','responses','pair_ids','culture_ids','row_ids','partitions')}
        for culture in cultures:
            refs=base+rng.normal(0,.01,base.shape)
            for i,p in enumerate(pairs):
                if (split[p]=='test')!=query:continue
                a,b=[targets.index(c) for c in json.loads(p)]
                row_id=culture+'_'+str(i)
                arrays['singles'].append([refs[a],refs[b]])
                arrays['responses'].append(refs[a]+refs[b]+shared+.05*base[a]*base[b]+rng.normal(0,.01,len(genes)))
                arrays['pair_ids'].append(p);arrays['culture_ids'].append(culture);arrays['row_ids'].append(row_id)
                arrays['partitions'].append(split[p])
                metadata.append({'row_id':row_id,'study_id':'synthetic','source_id':'synthetic','culture_id':culture,
                    'culture_initiation':None,'culture_evidence':None,'replicate_type':'synthetic',
                    'state_id':'fixed','assay_id':'synthetic','processing_batch_id':'synthetic',
                    'control_stratum_id':culture,'condition_kind':'pair','components':json.loads(p),
                    'reagent_ids':json.loads(p),'doses':[1,1],'dose_units':['a.u.','a.u.'],
                    'treatment_time':'24h','treatment_order':'simultaneous','partition':split[p],
                    'count_provenance':'synthetic engineering suite'})
        result={k:np.asarray(v) for k,v in arrays.items()}
        result.update(genes=genes,scale=np.asarray('synthetic_effect'),units=np.asarray('a.u.'))
        return result
    dev=bank(['dev1','dev2','dev3','dev4']);q=bank(['query1','query2','query3','query4'],True)
    np.savez_compressed(output/'development.npz',**dev)
    np.savez_compressed(output/'query_references.npz',**{k:v for k,v in q.items() if k not in {'responses','partitions'}})
    np.savez_compressed(output/'custodian_synthetic_outcomes.npz',**q,reference_additive=q['singles'].sum(1))
    write_json(output/'metadata.json',metadata)
    cfg=read(config);cfg.update(data_role='synthetic_engineering',confirmation_cultures=['query1','query2','query3','query4'])
    write_json(output/'protocol.json',cfg)
    write_json(output/'diagnostics.json',diagnostics(dev,cfg['menu']))
    selection=select(dev,cfg['menu']);train=dev['partitions']=='train'
    m=fit_candidate(selection['candidate'],dev,train);r=fit_candidate(selection['reference'],dev,train)
    rp=predict_candidate(selection['reference'],r,dev);cp=predict_candidate(selection['candidate'],m,dev)
    rv,_,_=cube(np.mean((rp-dev['responses'])**2,axis=1),dev['culture_ids'],dev['pair_ids'])
    cv,_,_=cube(np.mean((cp-dev['responses'])**2,axis=1),dev['culture_ids'],dev['pair_ids'])
    write_json(output/'power_report.json',empirical_power(rv,cv,[4,8],draws=100))
    freeze(output/'development.npz',output/'query_references.npz',output/'metadata.json',output/'protocol.json',output/'freeze')
    restored=restore_predictions(output/'freeze',output/'query_references.npz')
    with np.load(output/'freeze/predictions.npz') as f:
        for k,v in restored.items():np.testing.assert_allclose(v,f[k],rtol=1e-10,atol=1e-12)
    evaluate(output/'freeze',output/'custodian_synthetic_outcomes.npz',output/'evaluation')
    write_json(output/'engineering_verification.json',{'restoration_pass':True,'synthetic_only':True,
                                                      'scientific_gates_passed':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    for name in ('qualify','split'):
        sub=s.add_parser(name);sub.add_argument('--metadata',type=Path,required=True);sub.add_argument('--output',type=Path,required=True)
        if name=='split':sub.add_argument('--counts',nargs=3,type=int,required=True);sub.add_argument('--seed',type=int,default=1729)
    for name in ('prepare','prepare-sealed','query-references'):
        sub=s.add_parser(name);sub.add_argument('--metadata',type=Path,required=True);sub.add_argument('--counts',type=Path,required=True)
        sub.add_argument('--contract',type=Path,required=True);sub.add_argument('--output',type=Path,required=True)
        if name=='query-references':sub.add_argument('--queries',type=Path,required=True)
    sub=s.add_parser('freeze')
    for arg in ('development','query','metadata','protocol','output'):sub.add_argument('--'+arg,type=Path,required=True)
    sub=s.add_parser('evaluate')
    for arg in ('frozen','outcomes','output'):sub.add_argument('--'+arg,type=Path,required=True)
    sub.add_argument('--evidence',type=Path)
    sub=s.add_parser('diagnose');sub.add_argument('--development',type=Path,required=True);sub.add_argument('--protocol',type=Path,required=True);sub.add_argument('--output',type=Path,required=True)
    sub=s.add_parser('power');sub.add_argument('--pilot',type=Path,required=True);sub.add_argument('--cultures',type=int,nargs='+',required=True);sub.add_argument('--draws',type=int,default=1000);sub.add_argument('--output',type=Path,required=True)
    sub=s.add_parser('procedure-power')
    for arg in ('pilot','protocol','designs','output'):sub.add_argument('--'+arg,type=Path,required=True)
    sub.add_argument('--draws',type=int,default=100);sub.add_argument('--seed',type=int,default=1729)
    sub.add_argument('--effects',type=float,nargs='+',default=[0,.02,.05,.075,.10])
    sub.add_argument('--variance-multipliers',type=float,nargs='+',default=[1,2.25,4])
    sub=s.add_parser('protocol-check');sub.add_argument('--protocol',type=Path,required=True)
    sub.add_argument('--metadata',type=Path);sub.add_argument('--output',type=Path,required=True)
    sub=s.add_parser('amendment');sub.add_argument('--original',type=Path,required=True)
    sub.add_argument('--replacement',type=Path);sub.add_argument('--reason',required=True)
    sub.add_argument('--outcomes-seen',choices=['yes','no'],required=True)
    sub.add_argument('--affected-inputs',nargs='+',required=True);sub.add_argument('--output',type=Path,required=True)
    sub=s.add_parser('quote');sub.add_argument('--design',type=Path,required=True)
    sub.add_argument('--items',type=Path,required=True);sub.add_argument('--output',type=Path,required=True)
    sub=s.add_parser('allocation')
    for arg in ('metadata','counts','contract','protocol','output'):sub.add_argument('--'+arg,type=Path,required=True)
    sub=s.add_parser('demo');sub.add_argument('--output',type=Path,required=True);sub.add_argument('--protocol',type=Path,default=ROOT/'configs/experiments/replicated_fixed_state_v1.json')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior artifacts; use a new output path')
    if a.command=='demo':demo(a.output,a.protocol)
    elif a.command=='qualify':write_json(a.output,qualify(read(a.metadata)))
    elif a.command=='split':write_json(a.output,split_pairs(read(a.metadata),a.counts,a.seed))
    elif a.command in {'prepare','prepare-sealed','query-references'}:
        if a.output.suffix!='.npz':raise ValueError('Prepared bank output must have .npz suffix')
        if a.output.with_suffix('.preparation.json').exists():raise FileExistsError('Preserve prior preparation manifest')
        rows=read(a.metadata);contract=read(a.contract)
        queries=read(a.queries) if a.command=='query-references' else None
        if a.command=='query-references':
            if any(r['condition_kind']=='pair' for r in rows):raise ValueError('Query reference bank contains pair outcomes')
        else:validate_preparation_access(rows,contract,include_outcomes=a.command=='prepare-sealed')
        with np.load(a.counts,allow_pickle=False) as f:counts=f['counts'];genes=f['genes'].tolist()
        if a.command in {'prepare','prepare-sealed'}:bank,q=prepare(rows,counts,genes,contract,include_outcomes=a.command=='prepare-sealed')
        else:
            bank=query_references(rows,counts,genes,queries,contract)
            q=qualify(rows+queries,contract.get('aliases'))
        a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**bank)
        write_json(a.output.with_suffix('.preparation.json'),{'counts_sha256':sha(a.counts),'metadata_sha256':sha(a.metadata),
            'contract_sha256':sha(a.contract),'prepared_sha256':sha(a.output),'genes':genes,
            'command':a.command,'custodian_only':a.command=='prepare-sealed','secure_custody_claimed':False,
            'query_identities_sha256':sha(a.queries) if queries is not None else None,
            'source_row_ids':[r['row_id'] for r in rows],'matched_reference_rows':q['matches'],
            'original_counts_retained_at':str(a.counts),'row_mapping':bank['row_ids'].tolist()})
    elif a.command=='freeze':freeze(a.development,a.query,a.metadata,a.protocol,a.output)
    elif a.command=='evaluate':evaluate(a.frozen,a.outcomes,a.output,a.evidence)
    elif a.command=='diagnose':write_json(a.output,diagnostics(load_bank(a.development,True),read(a.protocol)['menu']))
    elif a.command=='power':
        with np.load(a.pilot,allow_pickle=False) as f:report=empirical_power(f['reference_errors'],f['candidate_errors'],a.cultures,a.draws)
        write_json(a.output,report)
    elif a.command=='procedure-power':
        from opal.replicated.power import procedure_power
        cfg=read(a.protocol)
        report=procedure_power(load_bank(a.pilot,True),cfg['menu'],read(a.designs),a.draws,a.seed,
            a.effects,a.variance_multipliers,cfg['shuffle_seeds'],cfg['maximum_seconds'])
        report.update(pilot_sha256=sha(a.pilot),protocol_sha256=sha(a.protocol),designs_sha256=sha(a.designs),
            implementation_sha256=sha(ROOT/'src/replicated/power.py'))
        report['protocol_snapshot']=cfg
        report['designs_snapshot']=read(a.designs)
        report['implementation_dependencies_sha256']={name:sha(ROOT/'src/replicated'/name)
            for name in ('power.py','models.py','evaluation.py')}
        import scipy
        report['runtime']={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__}
        write_json(a.output,report)
    elif a.command=='protocol-check':
        from opal.replicated.protocol import validate_protocol
        write_json(a.output,validate_protocol(read(a.protocol),read(a.metadata) if a.metadata else None))
    elif a.command=='amendment':
        from opal.replicated.protocol import amendment
        amendment(a.original,a.output,a.reason,a.outcomes_seen=='yes',a.affected_inputs,a.replacement)
    elif a.command=='quote':
        from opal.replicated.planning import quote_design
        report=quote_design(read(a.design),read(a.items))
        report.update(design_sha256=sha(a.design),quote_items_sha256=sha(a.items))
        write_json(a.output,report)
    elif a.command=='allocation':
        from opal.replicated.diagnostics import reference_allocation
        rows=read(a.metadata);contract=read(a.contract)
        validate_preparation_access(rows,contract)
        with np.load(a.counts,allow_pickle=False) as f:x=f['counts'];genes=f['genes'].tolist()
        write_json(a.output,reference_allocation(rows,x,genes,contract,read(a.protocol)['menu']))


if __name__=='__main__':main()
