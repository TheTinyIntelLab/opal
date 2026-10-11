"""Refresh compact evidence-backed local closeout without rewriting old records."""
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import read_json,sha256,write_json
from prepare_information import OUT,LOCAL,resource_check
from measure_information_process import retained


def run():
    delivery_authorized=(OUT/'verification/delivery_authorization.json').exists()
    summary=read_json(OUT/'development/summary.json')
    rows=[read_json(p) for p in (OUT/'development/folds').glob('*.json')]
    receipts=[read_json(p) for p in (OUT/'receipts').glob('*.json')]
    sources=read_json(OUT/'measurement/summaries.json')
    labels={'unique_pairs':summary['unique_pairs'],'pair_stratum_profiles':{'TNFA-positive':432,'TNFA-negative':208},
        'evaluated_profile_events':sum(r['held_pairs'] for r in rows),'outer_training_profile_events':sum(r['training_pairs'] for r in rows),
        'inner_training_profile_events_per_arm_config':sum(sum(v[0] for v in r['inner_train_validation_counts']) for r in rows),
        'inner_validation_profile_events_per_arm_config':sum(sum(v[1] for v in r['inner_train_validation_counts']) for r in rows),
        'candidate_slots':len(rows)*4*13,
        'nonzero_inner_head_fit_calls_including_failures':sum(48*sum(v is not None for v in r['inner_base_selections']) for r in rows),
        'nonzero_final_head_fit_calls':sum(a['chosen'] is not None for r in rows for a in r['arms'].values()),
        'distinct_control_single_cells':sum(s['control_cells']+sum(s['single_cells']) for c in sources for s in c['references']),
        'cell_level_pair_outcomes_for_information_models':0,'new_independent_cultures':0,
        'representation_labels':'controls only for input projection; current acquired pairs only for output basis',
        'selection_labels':'all nested stopping/selection/representation label roles included in allowed acquired banks',
        'interpretation':'Repeated access events are not independent measurements; one shuffle seed and one precision draw per shared reference.'}
    write_json(OUT/'development/label_accounting.json',labels)
    ranks=read_json(OUT/'diagnostics/ranks.json')
    energies=[i['retained_energy'] for r in ranks for i in r.get('interaction',[]) if i.get('retained_energy') is not None]
    write_json(OUT/'diagnostics/summary.json',{'folds':len(ranks),'full_row_rank_main_designs':sum(r.get('main_rank')==r['pairs'] for r in ranks),
        'retained_interaction_energy_range':[min(energies),max(energies)],'projection_changed':False,
        'verdict':'No near-complete annihilation for the selected references in these new folds; no inference about all historical folds or absent biology.',
        'error_attribution':'Per-fold Base error, correction energy and twice error/correction product; residual MSE is response MSE in shared coordinates.',
        'metric_controls':'Exact zero error, constant and fixed identity shuffle are evaluator-only; no metric switching.'})
    dependencies={
        'native_pertspectra':'Exact pinned StringDB.HQ.txt and Homo_sapiens.gene_info hashes/terms, ordered graph and author filter/normalization parity. Public construction recipe located and author50-cell metadata count65648 reproduced; cached graph absent. Two graph-source assets total188097304 bytes and exceed retention headroom.',
        'txpert_closure':'Approved storage and supported isolated Python>=3.12/CUDA2.6 closure with all310 direct dependencies, resolved transitive lock, public graph/weight provenance; torch wheel204120469 bytes exceeds headroom.',
        'presage_contract':'Author-supported PCA decoder, vector mask signature and condition-per-pair reducer. Two intervention rows cannot be averaged without that contract.',
        'native_task_matching':'Genuine native GEARS/CPA cell/graph/gene/covariate contract. SLIM signed profiles and NAIAD scalar-to-transcriptomic output need disclosed adaptations.',
        'external_qualification':'GSM7056151 sample-specific size/support, exact physical dose/control/single matching, initiation/plate lineage, derivative terms, complete prior-access attestation and independent custody.',
        'culture_pilot':'Qualified independently initiated matched cultures, shared-reference covariance, procedure simulations and statistical review; final count/useful margin/power/price remain unknown.',
        'remote_ci':'Separate push/dispatch authorization on exact reviewed revision; inspect Windows/Linux3.11/3.14 matrix cells. No commits or upload authorized here.',
        'redistribution':'Separate source/graph/embedding/derivative weight terms and publication/redistribution authorization.',
        'broader_metric_scope':'Technical-duplicate/cross-half DEG, pathway and affected-gene weighting were not in the executed finite freeze. Any future diagnostics require a separately declared design; no post-result endpoint expansion in this cycle.'}
    if delivery_authorized:
        dependencies['remote_ci']='Commit/push authorized by owner follow-up. Inspect every Windows/Linux Python3.11/3.14 on-push integrity cell on the delivered revision; no remote success inferred from local checks.'
    write_json(OUT/'report/dependencies.json',dependencies)
    write_json(OUT/'report/claims.json',{
        'information_gate':{'status':'complete_negative','evidence':'development/summary.json','verdict':'Administrative gate fails; TNF-positive loses to equally tuned mean/capacity controls. Retain Base.'},
        'tnf_negative_partner_information':{'status':'inconclusive','evidence':'development/folds/','verdict':'Nested support unavailable, all heads use Base; no claim that variance information is absent.'},
        'fresh_superiority':{'status':'blocked_external','evidence':'confirmation/source_qualification.json','verdict':'No qualified sealed source, biological useful margin or independent cultures.'},
        'native_competitiveness':{'status':'blocked_external','evidence':'comparators/','verdict':'Zero fully eligible contemporary native comparisons.'},
        'new_workload_measurements':{'status':'complete_verified','evidence':'profiling/ and comparators/*-profile.json','verdict':'Fresh training/replay measurements; folds/hardware/input tiers differ, no Pareto dominance proof.'},
        'historical_cpu_and_native_training_ram':{'status':'historically_unknown','verdict':'Not backfilled by new profiling.'},
        'frontier_interactions_cultures_transfer_uncertainty_savings':{'status':'blocked_external','verdict':'No supporting independent biological evidence.'},
        'comprehensive_metric_qualification':{'status':'conditional_not_started','evidence':'metrics/saved_controls.json','verdict':'Only frozen exact/constant/identity-shuffle controls executed; broader duplicate/DEG/pathway diagnostics were not prespecified or retrospectively added.'},
        'section15_branches':{'status':'conditional_not_started','verdict':'Activation gates unpassed; contracts refuse execution.'}})
    work={
        'P00':('complete_verified','preservation.json','none','Current inventory and historical evidence hashes preserved'),
        'P01':('complete_verified','development/execution_freeze.json','none','Finite protocol frozen; arithmetic correction documented before any fit'),
        'P02':('complete_verified','comparators/','native_pertspectra,txpert_closure,presage_contract','Bounded pinned-source/contract repairs complete; native eligibility remains blocked'),
        'P03':('blocked_external','measurement/ and confirmation/','external_qualification','Local cards complete; source/independent-unit qualification unavailable'),
        'P04':('complete_verified','diagnostics/','none','Rank, shared-anchor and frozen exact/constant/shuffle controls executed; broader secondary diagnostics unexecuted'),
        'P05':('complete_verified','src/information.py','none','Minimal symmetric summary/ridge heads and safe fallback/load contracts implemented'),
        'P06':('complete_negative','development/summary.json','none','All96 fold records complete; reference retained; TNF-negative partner support inconclusive'),
        'P07':('complete_verified','profiling/ and comparators/','native_task_matching','New Opal and three adaptation training/replay profiles; historical RAM still unknown'),
        'P08':('blocked_external','confirmation/','external_qualification','No confirmation opened or scored; dependency/custody handoff complete'),
        'P09':('blocked_external','docs/next_cycle/opal_information_custody_handoff.md','culture_pilot','Local pilot/power protocol delivered; actual independent cultures missing'),
        'P10':('complete_verified','verification/ and receipts/','remote_ci','Local tests/replay/installed package verified; remote matrix awaits authorized delivery run' if delivery_authorized else 'Local tests/replay/installed package verified; remote matrix pending authorization'),
        'P11':('complete_verified','docs/next_cycle/opal_information_report.md','redistribution','Local report/cards/README/manuscript and review package from saved evidence'),
        'P12':('complete_verified','resource_ledger.json and task_status.json','report/dependencies.json','Finite local closeout; external scientific program remains open'),
        'C01':('conditional_not_started','report/claims.json','section15 gates','No branch activated')}
    hashes={p:sha256(ROOT/p) for p in ['configs/experiments/opal_information_v1.json','src/information.py','tools/run_information.py','tools/prepare_information.py']}
    tasks=[{'id':i,'status':s,'owner_role':'local implementation agent','dependency':d,'code_config_hashes':hashes,
        'source_hashes':{c['source']:c['bank_sha256'] for c in sources},'label_access_roles':'exposed development only; no fresh held-expression access',
        'resource_allowance':{'cpu_core_hours':24,'retained_bytes':3000000000},'attempted_actions':v,'outputs':o,
        'verification_evidence':'receipts/, verification/, development/execution_freeze.json','engineering_delivered':i!='C01',
        'comparison_eligible':False,'scientific_verdict':v,'next_action':d if d!='none' else 'retain Base; no experiment expansion'} for i,(s,o,d,v) in work.items()]
    write_json(OUT/'task_status.json',{'tasks':tasks,'commits':'owner authorized small lowercase commits and push; git history is authoritative' if delivery_authorized else 'not authorized; none created','scientific_program_complete':False})
    sizes=resource_check();sizes.update(retained([ROOT/'build',ROOT/'src/opal_research.egg-info']))
    cpu=sum(r.get('cpu_seconds_observed',0) for r in receipts)
    ledger={'physical_roots':sizes,'artifact_bytes':sum(sizes.values()),'artifact_cap_bytes':3000000000,
        'cycle_bytes':sizes.get(str(LOCAL.resolve()),0)+sizes.get(str(OUT.resolve()),0),'maximum_new_bytes':120000000,
        'free_bytes':{v:shutil.disk_usage(v+':/').free for v in ['C','D','E']},'reserve_bytes':1073741824,'readonly_volumes':['D','E'],
        'cpu_core_hours_observed':cpu/3600,'cpu_seconds_observed_process_tree':cpu,'cpu_core_hours_cap':24,
        'receipt_count':len(receipts),'failed_or_incomplete_receipts':[r['id'] for r in receipts if r['status']!='complete'],
        'historical_cpu_complete':None,'historical_native_training_ram':None,'new_cpu_complete_certification':False,
        'unknown_costs':'Bootstrap shell/tool/web work and monitor overhead have no exact CPU counters; sampled last-PID totals can miss final increments and short-lived children. Earlier incomplete launch receipts remain visible. Observed CPU is a lower bound, never a complete total.',
        'double_counting':'Each PID cumulative CPU counted once; embedded stage CPU is not added again. Physical overlapping roots excluded.',
        'raw_existing_asset_bytes':3142132144,'raw_asset_classification':'Pre-existing cached input explicitly outside inherited artifact ledger; no new raw download/copy.',
        'gpu_seconds':None,'gpu_disclosure':'RTX4060 allocated/reserved memory peaks observed; wall time is not GPU kernel/device time.'}
    if ledger['artifact_bytes']>3000000000 or ledger['cycle_bytes']>120000000:raise RuntimeError('Storage cap exceeded')
    write_json(OUT/'resource_ledger.json',ledger)
    efficiency=[]
    for method in ['GEARS','CPA','PertSpectra']:
        t=read_json(OUT/f'comparators/{method}-training-profile.json');r=read_json(OUT/f'comparators/{method}-replay-profile.json')
        efficiency.append({'method':method,'fully_eligible':False,'classification':'protocol adaptation','mse':t['mse'],
            'train_wall_seconds':t.get('wall_seconds_including_fit_graph_predict',t.get('wall_seconds')),
            'train_peak_host_bytes':t['peak_stage_rss_sampled_bytes'],'checkpoint_bytes':t['checkpoint_bytes'],'reference_bytes':t['reference_bytes'],
            'replay_max_abs':r['replay_max_abs'],'device':t.get('gpu',{}).get('device','CPU'),'historical_training_ram_bytes':None})
    write_json(OUT/'report/efficiency.json',{'native_adaptations':efficiency,'Opal_training':read_json(OUT/'profiling/Opal-training.json'),
        'Opal_replay':read_json(OUT/'profiling/Opal-replay.json'),'fully_eligible_native_comparators':0,
        'equivalent_workload_hardware_pareto_proof':False,'limitations':'New Opal lexical fold differs from historical native fold; CPU/GPU/input tiers and stage/process memory scopes differ.'})
    print('Task/claim/dependency/label/resource/efficiency records refreshed',ledger['artifact_bytes'],cpu/3600)


if __name__=='__main__':run()
