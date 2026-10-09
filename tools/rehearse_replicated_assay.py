"""Exercise the count-to-frozen-score CLI on synthetic aggregate reads only."""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.replicated.contracts import write_json,sha
from opal.replicated.workflow import load_bank,restore_predictions


def rehearse(output,protocol):
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve prior assay rehearsals')
    output.mkdir(parents=True)
    rng=np.random.default_rng(20261009)
    targets=[f'SYNTHETIC_T{i}' for i in range(8)]
    genes=np.array([f'SYNTHETIC_G{i}' for i in range(12)])
    pairs=list(itertools.combinations(targets,2))
    assignment={p:'train' if i<16 else 'validation' if i<22 else 'test' for i,p in enumerate(pairs)}
    effects={t:rng.normal(0,.12,len(genes)) for t in targets}
    baseline=rng.normal(7,.15,len(genes))
    all_rows=[];groups={k:([],[]) for k in ('development','query-reference','sealed')}
    def add(group,culture,components,partition,repeat=0):
        name=culture+'_'+('_'.join(components) if components else 'control')+'_'+str(repeat)
        row={'row_id':name,'study_id':'synthetic-assay-rehearsal','source_id':'synthetic-generator-v1',
            'culture_id':culture,'culture_initiation':None,'culture_evidence':None,
            'state_id':'synthetic-state','assay_id':'synthetic-aggregate-reads','processing_batch_id':culture,
            'control_stratum_id':culture,'condition_kind':{0:'control',1:'single',2:'pair'}[len(components)],
            'components':list(components),'reagent_ids':list(components),'doses':[1]*len(components),
            'dose_units':['synthetic-unit']*len(components),'treatment_time':'synthetic-fixed-time',
            'treatment_order':'simultaneous','replicate_type':'synthetic','count_provenance':'synthetic integer read generator',
            'partition':partition}
        log_mean=baseline+sum((effects[t] for t in components),start=np.zeros(len(genes)))
        if len(components)==2:log_mean=log_mean+.02*effects[components[0]]*effects[components[1]]
        count=rng.poisson(np.exp(log_mean+rng.normal(0,.005,len(genes))))
        groups[group][0].append(row);groups[group][1].append(count);all_rows.append(row)
    development=[f'synthetic-dev{i}' for i in range(4)]
    confirmation=[f'synthetic-query{i}' for i in range(4)]
    for culture in development+confirmation:
        group='development' if culture in development else 'query-reference'
        for repeat in range(2):
            add(group,culture,[],'reference',repeat)
            for target in targets:add(group,culture,[target],'reference',repeat)
        for pair in pairs:
            if (culture in confirmation)!=(assignment[pair]=='test'):continue
            add('sealed' if culture in confirmation else 'development',culture,pair,assignment[pair])
    for group,(rows,counts) in groups.items():
        write_json(output/(group+'-metadata.json'),rows)
        np.savez_compressed(output/(group+'-counts.npz'),counts=np.asarray(counts),genes=genes)
    queries=groups['sealed'][0]
    write_json(output/'query-identities.json',queries)
    write_json(output/'all-metadata.json',all_rows)
    contract={'confirmation_cultures':confirmation,'held_pair_ids':[json.dumps(list(p),separators=(',',':')) for p in pairs if assignment[p]=='test'],
              'normalization':'library_total','transform':'log2_after_reference_mean','library_target':10000,
              'pseudocount':1,'scale':'log2_aggregate','units':'log2','aliases':{}}
    write_json(output/'contract.json',contract)
    config=json.loads(Path(protocol).read_text());config.update(data_role='synthetic_engineering',confirmation_cultures=confirmation)
    write_json(output/'protocol.json',config)
    # The custodian alone combines its reference counts with sealed pair counts.
    sealed_rows=groups['query-reference'][0]+queries
    sealed_counts=groups['query-reference'][1]+groups['sealed'][1]
    write_json(output/'custodian-metadata.json',sealed_rows)
    np.savez_compressed(output/'custodian-counts.npz',counts=np.asarray(sealed_counts),genes=genes)
    records=[]
    def command(name,**kwargs):
        argv=[sys.executable,str(ROOT/'tools/run_replicated_fixed_state.py'),name]
        for key,value in kwargs.items():argv+=['--'+key.replace('_','-'),str(value)]
        completed=subprocess.run(argv,capture_output=True,text=True)
        records.append({'command':name,'returncode':completed.returncode})
        if completed.returncode:raise RuntimeError(completed.stdout+completed.stderr)
    command('prepare',metadata=output/'development-metadata.json',counts=output/'development-counts.npz',
            contract=output/'contract.json',output=output/'development.npz')
    command('query-references',metadata=output/'query-reference-metadata.json',counts=output/'query-reference-counts.npz',
            queries=output/'query-identities.json',contract=output/'contract.json',output=output/'query.npz')
    command('prepare-sealed',metadata=output/'custodian-metadata.json',counts=output/'custodian-counts.npz',
            contract=output/'contract.json',output=output/'sealed.npz')
    command('diagnose',development=output/'development.npz',protocol=output/'protocol.json',output=output/'diagnostics.json')
    command('freeze',development=output/'development.npz',query=output/'query.npz',metadata=output/'all-metadata.json',
            protocol=output/'protocol.json',output=output/'freeze')
    restored=restore_predictions(output/'freeze',output/'query.npz')
    with np.load(output/'freeze/predictions.npz',allow_pickle=False) as frozen:
        for name,prediction in restored.items():np.testing.assert_allclose(prediction,frozen[name],rtol=1e-10,atol=1e-12)
    command('evaluate',frozen=output/'freeze',outcomes=output/'sealed.npz',output=output/'evaluation')
    query=load_bank(output/'query.npz');sealed=load_bank(output/'sealed.npz',True)
    for key in ('singles','pair_ids','culture_ids','row_ids','genes','scale','units'):
        np.testing.assert_array_equal(query[key],sealed[key])
    access=json.loads((output/'freeze/access.json').read_text())
    queried_targets={target for row in queries for target in row['components']}
    reference_ids={r['row_id'] for r in groups['query-reference'][0]
                   if r['condition_kind']=='control' or set(r['components'])<=queried_targets}
    assert set(access['query_reference_rows'])==reference_ids
    result=json.loads((output/'evaluation/results.json').read_text())
    assert not result['scientific_confirmation_claimed'] and not result['primary_claim_pass']
    report={'schema':'opal-synthetic-assay-rehearsal-v1','commands':records,'synthetic_only':True,
            'original_integer_counts':True,'exact_reference_lineage_verified':True,'query_outcomes_absent':True,
            'query_and_custodian_reference_transforms_agree':True,'restoration_pass':True,
            'scientific_confirmation_claimed':False,'development_pairs':len(load_bank(output/'development.npz',True)['row_ids']),
            'confirmation_pairs':len(query['row_ids']),'query_reference_rows':len(reference_ids),
            'protocol_sha256':sha(output/'protocol.json'),'code_sha256':sha(__file__)}
    write_json(output/'verification.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/experiments/replicated_fixed_state_v1.json')
    a=p.parse_args();print(json.dumps(rehearse(a.output,a.protocol),indent=2))


if __name__=='__main__':main()
