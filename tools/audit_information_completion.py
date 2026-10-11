"""Consolidate saved evidence without new fitting, predictions or metric selection."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import read_json,write_json,sha256
from prepare_information import OUT,resource_check


def run():
    resource_check(1000000)
    freeze=read_json(OUT/'development/execution_freeze.json')
    assert sha256(ROOT/'configs/experiments/opal_information_v1.json')==freeze['configuration_sha256']
    assert sha256(ROOT/'docs/next_cycle/opal_information_protocol.md')==freeze['protocol_sha256']
    for path,digest in freeze['code'].items():assert sha256(ROOT/path)==digest,path
    folds=[read_json(p) for p in sorted((OUT/'development/folds').glob('*.json'))]
    assert len(folds)==96
    for row in folds:
        assert sha256(Path(row['checkpoint'])/'predictions.npz')==row['prediction_sha256']
        assert row['metric_controls']['exact']['mse']==0
        assert row['metric_controls']['exact']['retrieval_mean_rank']==1
    metrics={'scope':'Only previously frozen and scored metrics, consolidated from saved fold records; no new scoring.',
        'records':[{k:r[k] for k in ['id','source','split','precision','held_pairs','metric_controls']} for r in folds],
        'primary_endpoint_unchanged':True,'candidate_selection_reopened':False,
        'exact_control_zero_mse_folds':96,'exact_control_first_retrieval_rank_folds':96,
        'limits':'Exact/constant/identity-shuffle controls were prespecified and scored after prediction freeze. Technical-duplicate/cross-half DEG, pathway and affected-gene weighted analyses were not in the executed finite protocol. They were not run or retrospectively registered; no biological futility or comprehensive metric-qualification claim.'}
    write_json(OUT/'metrics/saved_controls.json',metrics)
    prepared=ROOT/'results/opal_model_series_v1/pertspectra/access-v4'
    lineage={'source':'exposed PertSpectra inhouse genetic screen; TNF contexts separate',
        'original_evidence':{str(p.relative_to(ROOT)):sha256(p) for p in prepared.glob('*.json')},
        'current_measurement_cards':{str(p.relative_to(ROOT)):sha256(p) for p in (OUT/'measurement').glob('*.json') if p.name!='lineage_links.json'},
        'assignment':'Gene-level deposited guide mixtures; guide/target length mismatch excluded by original preparation. No matched-guide causal equivalence claimed.',
        'references':'Control/single means remain original aggregate library10000 log2 arithmetic; control-only log-cell PCA and measured-single variance are extra inputs.',
        'independent_unit':'SampleIndex is a source/technical stratum, not established independent culture initiation.',
        'physical_dose_time_and_initiation':'No new lineage evidence; exact assignments remain unknown where prior deposited records do not establish them.',
        'rights':'Pinned upstream code license audits do not establish raw-data/graph/embedding/derivative weight redistribution rights. No redistribution authorized.',
        'new_source_or_pair_cell_access':False,'original_support_and_exclusions_preserved':True}
    write_json(OUT/'measurement/lineage_links.json',lineage)
    receipts=[]
    for path in sorted((OUT/'receipts').glob('*.json')):
        r=read_json(path)
        receipts.append({'id':r['id'],'receipt_sha256':sha256(path),'status':r['status'],
            'command':r['command'],'start_utc':r['start_utc'],'end_utc':r.get('end_utc'),
            'resource_values':'Original receipt is authoritative; missing values remain unknown.',
            'source_config_label_context':'development/execution_freeze.json and development/label_accounting.json; comparator/profiling cards define native workload roles.',
            'code_hash_scope':'Frozen scientific code hashes verified unchanged now; tooling snapshots not present in original receipts cannot be reconstructed retrospectively.'})
    write_json(OUT/'verification/receipt_context_index.json',{'receipts':receipts,
        'execution_freeze_sha256':sha256(OUT/'development/execution_freeze.json'),
        'label_ledger_sha256':sha256(OUT/'development/label_accounting.json'),
        'historical_receipts_modified':False,'complete_cpu_certification':False})
    tasks=read_json(OUT/'task_status.json')['tasks']
    assert not any(t['status']=='pending_local' for t in tasks)
    write_json(OUT/'verification/completion_audit.json',{
        'frozen_code_config_protocol_unchanged':True,'frozen_prediction_hashes_verified':96,
        'pending_local_task_ids':[],'blocking_tasks':[{k:t[k] for k in ['id','status','dependency']} for t in tasks if t['status']=='blocked_external'],
        'conditional_tasks':[t['id'] for t in tasks if t['status']=='conditional_not_started'],
        'saved_metric_control_consolidation':'metrics/saved_controls.json',
        'measurement_lineage_consolidation':'measurement/lineage_links.json',
        'receipt_context_consolidation':'verification/receipt_context_index.json',
        'requirements_not_satisfied_by_software':['independent cultures/empirical biological power','qualified sealed fresh confirmation','native comparator parity','complete historical/new CPU certification','remote CI/registration/publication/weight redistribution'],
        'broader_secondary_metric_scope':'Not executed in this frozen study; a future separately frozen design is required. No post-result metrics or fits added.',
        'scientific_program_complete':False,'bounded_local_scope_complete':True})
    print('96 frozen records and code/config/protocol hashes verified; no pending_local task; evidence indexes saved')


if __name__=='__main__':run()
