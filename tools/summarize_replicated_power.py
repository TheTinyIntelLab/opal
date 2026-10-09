"""Summarize immutable procedure sensitivity runs without pooling shared draws."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(paths):
    sources=[];rows=[];seen=set()
    for path in paths:
        path=Path(path);record=json.loads(path.read_text(encoding='utf-8'))
        if record.get('confirmation_design_qualified') is not False or not record.get('selection_repeated'):
            raise ValueError('Expected explicitly unqualified full-procedure sensitivity')
        sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'pilot_sha256':record['pilot_sha256'],'protocol_sha256':record['protocol_sha256'],
                        'elapsed_seconds':record['elapsed_seconds'],'seed':record['seed']})
        for row in record['reports']:
            key=(json.dumps(row['design'],sort_keys=True),row['effect'],row['variance_multiplier'])
            if key in seen:raise ValueError('Duplicate design/effect/variance scenario')
            seen.add(key);rows.append({**row,'source':str(path)})
    if not rows:raise ValueError('No simulation scenarios')
    null=[r for r in rows if r['effect']==0]
    detection=[r for r in rows if r['effect']==.05]
    return {'schema':'opal-procedure-sensitivity-summary-v1','sources':sources,'scenarios':rows,
            'simulated_studies':sum(r['draws'] for r in rows),
            'variance_multipliers':sorted({r['variance_multiplier'] for r in rows}),
            'maximum_null_positive_detection':max(r['positive_detection_power'] for r in null) if null else None,
            'minimum_null_interval_coverage':min(r['interval_coverage_given_feasible'] for r in null) if null else None,
            'maximum_5percent_positive_detection':max(r['positive_detection_power'] for r in detection) if detection else None,
            'maximum_5percent_full_gate':max(r['full_gate_power'] for r in detection) if detection else None,
            'infeasible_alternatives':sum(r['infeasible_draws'] for r in rows),
            'confirmation_design_qualified':False,'scientific_sample_size_recommended':None,
            'scope':'Synthetic conditional local-alternative sensitivity; no biological qualification',
            'draws_pooled':False,'scenarios_independent':False,
            'limitations':['Identical seeds reuse random blocks across variance scenarios; trial counts are execution counts, not independent pooled evidence',
                           'Point estimates retain scenario-specific Monte Carlo intervals',
                           'Protocol hashes can differ across preserved budget-only amendments; inspect the stored protocol snapshots',
                           'Infeasible alternatives remain failures rather than favorable redraws',
                           'Deletion diagnostics and sensitivity scenarios cannot replace a registered primary endpoint']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior sensitivity summaries')
    record=summarize(args.inputs);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('simulated_studies','variance_multipliers','maximum_null_positive_detection',
                                         'minimum_null_interval_coverage','maximum_5percent_positive_detection',
                                         'maximum_5percent_full_gate','confirmation_design_qualified')},indent=2))


if __name__=='__main__':main()
