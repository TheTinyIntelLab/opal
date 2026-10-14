"""Execute the separate, registered crossed-single feasibility study on cached counts."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from opal.artifacts import read_json,write_json,sha256,fingerprint
from opal.contextual.contracts import ReferenceBank
from opal.context_contrast.resources import Budget
from opal.context_contrast.artifacts import save_state,load_state,save_npz,load_npz
from opal.context_contrast.contracts import Unsupported
from opal.crossed_single import fit,predict,select,grouped_folds,FAMILIES

CONFIG=ROOT/'configs/experiments/opal_crossed_single_v1.json'


def execute(output,config,budget):
    # Pin inputs and code before accessing any newly scored expression output.
    raw=ROOT/'data/raw/state/SrivatsanTrapnell2020_sciplex3.h5ad'
    primary=ROOT/'data/raw/identity-reference-20261007/GSM4150378_sciPlex3_pData.txt.gz'
    if sha256(raw)!=config['source_sha256'] or sha256(primary)!=config['primary_metadata_sha256']:
        raise ValueError('Raw or primary assignment identity mismatch')
    provenance=read_json(ROOT/'results/identity-reference-20261007/provenance-v3/audit.json')
    if provenance['matched_cells']!=provenance['source_cells'] or any(provenance['condition_mismatches'].values()):
        raise ValueError('Primary condition provenance unqualified')
    budget.write_json(output/'registration.json',dict(config=config,config_sha256=sha256(CONFIG),
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/crossed_single.py',ROOT/'tools/prepare_state_source.py']},
        primary_provenance=provenance,owner_execution_authorized=True,
        evidence='exposed single-response development only',independent_confirmation=False))
    old=sys.argv
    try:
        sys.argv=['prepare_state_source.py','--output',str(output/'prepared'),'--maximum-interventions','96',
                  '--doses','1000','--genes','512','--wall-seconds','1800','--rss-bytes','8000000000']
        runpy.run_path(str(ROOT/'tools/prepare_state_source.py'),run_name='__main__')
    finally:sys.argv=old
    budget.check()
    bank=ReferenceBank.load(output/'prepared');a=bank.arrays
    if len(a['targets'])!=96 or len(a['genes'])!=512:
        raise Unsupported('Registered 96 compound/512 gene preparation unsupported')
    context=np.array([bank.metadata['context_records'][str(c)]['biological_state'] for c in a['contexts']])
    effects=[]
    for line in config['contexts']:
        rows=np.flatnonzero(context==line)
        if not a['available'][rows].any(0).all():raise Unsupported('Missing declared line/compound support')
        effects.append(np.stack([a['singles'][rows[a['available'][rows,i]],i].mean(0) for i in range(96)]))
    effects=np.array(effects);assignment=grouped_folds(96,config['split_seed'])
    budget.write_json(output/'qualification.json',dict(status='qualified_exposed_single_response_only',
        compounds=a['targets'].tolist(),genes=a['genes'].tolist(),source_sha256=config['source_sha256'],
        primary_assignments_verified=True,independent_culture_initiation_verified=False,
        endpoint=bank.metadata['normalization'],outer_assignment=assignment.tolist(),query_vehicles_permitted=True))
    attempts=[];records=[];invocation=0
    for target,line in enumerate(config['contexts']):
        others=[j for j in range(3) if j!=target]
        x=np.transpose(effects[others],(1,0,2));y=effects[target]
        for held in range(3):
            train=np.flatnonzero(assignment!=held);query=np.flatnonzero(assignment==held)
            folder=output/f'{line}-{held}';folder.mkdir()
            inputs=save_npz(folder/'fit.npz',budget,x=x[train],y=y[train],identities=train)
            queries=save_npz(folder/'query.npz',budget,x=x[query],identities=query)
            evaluator=save_npz(folder/'evaluator.npz',budget,y=y[query],identities=query)
            # Fit code receives no evaluator object or path.
            def record(stage,state,details):
                nonlocal invocation
                invocation+=1
                path=output/'states'/f'{line}-{held}-{stage}'
                if state is not None:save_state(path,state,budget)
                attempts.append(dict(target=line,outer=held,stage=stage,state_path=path.relative_to(output).as_posix() if state else None,**details))
                budget.check()
            train_data=load_npz(folder,inputs);query_data=load_npz(folder,queries)
            choices,locked,scores=select(train_data['x'],train_data['y'],config,record)
            states={};unsupported={}
            for family in FAMILIES:
                choice=choices[family]
                if choice is None:
                    states[family]=None;unsupported[family]='required nested fit unsupported'
                    record('final-'+family,None,dict(spec=dict(family=family),error=unsupported[family]))
                    continue
                try:
                    states[family]=fit(train_data['x'],train_data['y'],choice['spec'],config)
                    record('final-'+family,states[family],dict(spec=choice['spec'],error=None))
                except Unsupported as error:
                    states[family]=None;unsupported[family]=str(error)
                    record('final-'+family,None,dict(spec=choice['spec'],error=str(error)))
            if states[locked] is None:raise Unsupported('Locked reference final fit unsupported')
            fallback=predict(states[locked],query_data['x'])
            predictions={f:predict(states[f],query_data['x']) if states[f] is not None else fallback.copy() for f in FAMILIES}
            predictions['selector']=fallback.copy()
            # Restore all final inference assets and require exact predictions.
            for family,state in states.items():
                if state is not None:
                    restored=load_state(output/'states'/f'{line}-{held}-final-{family}')
                    if not np.array_equal(predict(restored,query_data['x']),predictions[family]):
                        raise ValueError('Exact saved-state replay failed')
            prediction=save_npz(folder/'predictions.npz',budget,**predictions)
            record_value=dict(target=line,outer=held,identities=query.tolist(),fit=inputs,query=queries,
                evaluator=evaluator,predictions=prediction,locked_reference=locked,unsupported=unsupported,
                selection=scores,choices=choices,replay_exact=True,path=folder.relative_to(output).as_posix())
            records.append(record_value)
            budget.write_json(folder/'selection.json',record_value)
            print('sealed',line,held,'fits',invocation,flush=True)
    if invocation!=config['logical_invocations']:raise ValueError('Complete fit inventory mismatch')
    budget.write_json(output/'attempts.json',attempts)
    seal=dict(status='sealed_before_evaluation',records=records,logical_invocations=invocation,
              config_sha256=sha256(CONFIG),attempts_sha256=sha256(output/'attempts.json'))
    budget.write_json(output/'prediction_seal.json',seal)
    seal_hash=sha256(output/'prediction_seal.json')
    errors={};fallback_counts={};unsupported_counts={}
    for line in config['contexts']:
        losses={f:np.empty(96) for f in (*FAMILIES,'selector')};fallback_counts[line]={f:0 for f in FAMILIES}
        for record in records:
            if record['target']!=line:continue
            folder=output/record['path'];p=load_npz(folder,record['predictions']);e=load_npz(folder,record['evaluator'])
            ids=e['identities']
            if ids.tolist()!=record['identities']:raise ValueError('Evaluator identity mismatch')
            for family in losses:losses[family][ids]=np.mean((p[family]-e['y'])**2,axis=1)
            for family in record['unsupported']:fallback_counts[line][family]+=len(ids)
        errors[line]=losses
        unsupported_counts[line]=[f for f,n in fallback_counts[line].items() if n]
    rng=np.random.default_rng(config['sensitivity_seed'])
    draws=rng.integers(0,96,size=(config['sensitivity_draws'],96))
    results={};passed=True
    for line in config['contexts']:
        losses=errors[line];candidate=losses['pca_concat'];reference=losses['selector']
        difference=reference-candidate;bounds=np.quantile(difference[draws].mean(1),[.025,.975])
        gain=1-float(candidate.mean()/reference.mean()) if reference.mean()>0 else None
        controls=['pca_mean','pca_difference','kernel','shuffled_concat']
        okay=bool(gain is not None and gain>=.05 and bounds[0]>0 and not unsupported_counts[line]
                  and all(candidate.mean()<losses[f].mean() for f in controls))
        passed &= okay
        results[line]=dict(population=96,mse={f:float(v.mean()) for f,v in losses.items()},
            candidate_relative_gain=gain,conditional_paired_loss_bounds=bounds.tolist(),
            spending_filter_passed=okay,fallback_counts=fallback_counts[line])
    if sha256(output/'prediction_seal.json')!=seal_hash:raise ValueError('Evaluator mutated prediction seal')
    budget.write_json(output/'metrics.json',dict(targets=results,primary=config['primary'],
        uncertainty=config['uncertainty'],coupled_compound_draws=config['sensitivity_draws']))
    budget.write_json(output/'decision.json',dict(status='completed_exposed_feasibility',
        spending_filter_passed=passed,complete_population=True,logical_invocations=invocation,
        independent_biological_confirmation=False,pair_composition_claim=False,frontier_claim=False,
        unseen_context_claim=False,required_control_fallback=unsupported_counts,
        next_action='separate reliability pilot required before any confirmation' if passed else 'close finite rank-16 probe; no rescue sweep'))
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);config=read_json(CONFIG)
    accepted=dict(config['budget'],approved=True,owner='project owner in this conversation',
        authorization_source='persistent owner execution authorization and do whatever looks best instruction',
        experiment_id=config['experiment_id'],operation='prepare_fit_evaluate',output=str(args.output.resolve()),
        protocol_hash=fingerprint(config),approved_at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    budget=Budget(args.output,accepted)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            result=execute(args.output,config,budget)
        budget.receipt('completed_exposed_feasibility')
        print(result,flush=True)
    except Exception as error:
        budget.write_failure_json(args.output/'failure.json',dict(status='failed_partial_attempt',reason=str(error),
            partial_assets_retained=True,scientific_success=False))
        budget.receipt('failed_partial_attempt')
        raise


if __name__=='__main__':main()
