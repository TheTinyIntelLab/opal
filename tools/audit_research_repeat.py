"""Check registered reruns against preserved predictions and scientific gates."""
import argparse
from pathlib import Path
import subprocess
import numpy as np
from opal.artifacts import read_json,write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];repeat=root/'results/recheck-20261007'
    tracks=[('stage1/run-v1','stage1-20261007/run-v1'),('stage1/main-v1','stage1-20261007/main-v1'),
        ('stage1/identity-control-v1','stage1-20261007/identity-control-v1'),
        ('dose/dose-v2','dose-design-20261007/dose-v2'),('identity/run-v1','identity-reference-20261007/run-v1'),
        ('joint/run-v1','joint-development-20261007/run-v1'),('panel/run-v1','panel-scale-20261007/run-v1')]
    arrays=[];metrics=[]
    for new_name,old_name in tracks:
        new=repeat/new_name;old=root/'results'/old_name
        if not new.is_dir():raise ValueError('Rerun missing: '+new_name)
        for saved in sorted(old.rglob('*.npz')):
            current=new/saved.relative_to(old)
            if not current.exists():raise ValueError('Rerun array missing: '+str(current))
            with np.load(saved,allow_pickle=False) as a,np.load(current,allow_pickle=False) as b:
                if set(a.files)!=set(b.files):raise ValueError('Array schema changed')
                drift=0.
                for key in a.files:
                    if a[key].shape!=b[key].shape:raise ValueError('Array shape changed')
                    if a[key].dtype.kind in 'fc':
                        delta=float(np.max(np.abs(a[key]-b[key]),initial=0));drift=max(drift,delta)
                        if not np.isfinite(b[key]).all():raise ValueError('Nonfinite rerun predictions')
                    elif not np.array_equal(a[key],b[key]):raise ValueError('Identity or availability changed')
            arrays.append(dict(original=str(saved.relative_to(root)),rerun=str(current.relative_to(root)),max_absolute_drift=drift,
                original_sha256=sha256(saved),rerun_sha256=sha256(current)))
        for saved in sorted(old.rglob('results.json')):
            current=new/saved.relative_to(old);a=read_json(saved);b=read_json(current)
            field=next((k for k in ['models','comparisons'] if k in a),None)
            if field:
                original_models=a[field] if isinstance(a[field],dict) else {str(i):v for i,v in enumerate(a[field])}
                current_models=b[field] if isinstance(b[field],dict) else {str(i):v for i,v in enumerate(b[field])}
                changed=[]
                for key,value in original_models.items():
                    for name in ['mse','relative_gain','development_gate','gate','paired_lower','paired_upper','diagnostic_95_interval']:
                        if name in value:
                            if name in ['development_gate','gate']:
                                if value[name]!=current_models[key][name]:raise ValueError('Scientific gate changed: '+str(current)+'/'+key)
                            elif not np.allclose(value[name],current_models[key][name],rtol=1e-10,atol=1e-12):
                                changed.append(dict(model=key,metric=name,original=value[name],rerun=current_models[key][name]))
                metrics.append(dict(original=str(saved.relative_to(root)),rerun=str(current.relative_to(root)),matching_gate_models=len(original_models),numerical_changes=changed))
    upr=root/'results/adamson-transfer-20261007';record=read_json(upr/'evaluation-v1/results.json')
    with np.load(root/'data/prepared/adamson-transfer-20261007-v1/evaluator/pairs.npz',allow_pickle=False) as f:y=f['responses']
    with np.load(upr/'freeze-v1/predictions.npz',allow_pickle=False) as f:
        for name,r in record['models'].items():
            mse=float(np.mean((f[name]-y)**2));assert np.isclose(mse,r['mse'],rtol=1e-12,atol=0)
            assert r['diagnostic_materiality']==(r['relative_gain']>=.05 and r['paired_lower']>0)
    assert not record['models']['selected_pair']['diagnostic_materiality']
    assert record['models']['additive']['mse']<record['models']['selected_simple']['mse']
    raw=subprocess.check_output(['git','status','--porcelain','-z'],cwd=root,stderr=subprocess.DEVNULL)
    initial=(root/'tmp/pre-main-transfer-status.bin').read_bytes()
    deletions=lambda s:sorted(x for x in s.split(b'\0') if x.startswith(b' D '))
    assert deletions(raw)==deletions(initial)
    write_json(args.output/'audit.json',dict(status='passed_identity_accounting_and_unchanged_gates',arrays=arrays,metrics=metrics,
        current_code_sha256=sha256(__file__),protected_deletions=len(deletions(raw)),historical_baseline='84de7e36eae20531e815be9de5a4cb2fa9b94ce9',
        numerical_environments={k:read_json(root/path)['environment'] for k,path in [('original','results/stage1-20261007/qualification/qualification.json'),('rerun','results/recheck-20261007/stage1/qualification/qualification.json')]},
        upr_independent_score_accounting=True,upr_gate=False,upr_additivity_stronger=True,
        interpretation='Reruns preserve identities and scientific gates, with every numerical drift retained. Retraining is not promised to be bitwise identical. Reruns are exposed development, not additional independent samples. Diagnostic bounds hold reference means and selection fixed; biological confirmation remains unevaluable.'))
    print('Verified',len(arrays),'array archives and',len(metrics),'metric records; UPR accounting and protected deletions pass.')


if __name__=='__main__':main()
