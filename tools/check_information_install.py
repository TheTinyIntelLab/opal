"""Verify an extracted review wheel outside the checkout, offline."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def run(label,python,package_root='package'):
    local=Path('C:/Users/USER/Desktop/opal-information-20261011')
    wheel=local/package_root/'opal_research-0.1.3.dev0-py3-none-any.whl'
    destination=local/('installed'+label)
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(wheel) as archive:
        members=archive.namelist()
        if 'opal/information.py' not in members:raise ValueError('Information module missing')
        if any(n.endswith(('.npz','.h5ad','.npy','.pt')) or Path(n).name.lower()=='plan.md' for n in members):raise ValueError('Private package member')
        archive.extractall(destination/'package')
    with tarfile.open(local/package_root/'opal_research-0.1.3.dev0.tar.gz') as archive:
        if any(n.name.endswith(('.npz','.h5ad','.npy','.pt')) or Path(n.name).name.lower()=='plan.md' for n in archive):raise ValueError('Private sdist member')
    env=dict(os.environ,PYTHONPATH=str(destination/'package'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    # Installed training/scoring on the actual preserved TNF-positive example.
    script=ROOT/'tools/check_installed_series.py'
    example=Path('C:/Users/USER/Desktop/opal-model-series-20261010-v2/installed-example')
    result=subprocess.run([str(Path(python).resolve()),str(script),'--output',str(destination/'series-verification'),
        '--example',str(example),'--require-installed','--config',str(ROOT/'configs/experiments/opal_model_series_v2.json')],
        cwd=destination,env=env,capture_output=True,text=True,timeout=1800)
    (destination/'series.stdout.txt').write_text(result.stdout);(destination/'series.stderr.txt').write_text(result.stderr)
    if result.returncode:raise RuntimeError(result.stderr+result.stdout)
    # Exercise actual new head fitting, prediction, save/load and known-result parity.
    installed_script=destination/'information_example.py'
    installed_script.write_text('''
import json, pathlib, sys
import numpy as np
import opal
from opal.artifacts import read_json, write_json
from opal.information import fit_head, save_head, load_head, predict_head
from opal.series_bilinear import predict_reference, weights
root=pathlib.Path('C:/Users/USER/Desktop/code/thetinyintellab/opal')
out=root/'results/opal_information_v1'
row=read_json(out/'development/folds/TNFA-positive-global_pair-0-3-original.json')
source=read_json(out/'measurement/TNFA-positive.json')
with np.load(source['bank'],allow_pickle=False) as z: bank={k:z[k] for k in ['singles','responses','pair_ids','culture_ids']}
with np.load(source['summary'],allow_pickle=False) as z: summary={k:z[k] for k in z.files}
train=(bank['culture_ids']=='3')&np.isin(bank['pair_ids'],row['train_pair_ids'])
held=(bank['culture_ids']=='3')&np.isin(bank['pair_ids'],row['held_pair_ids'])
dev={k:v[train] for k,v in bank.items()}
q={k:v[held] for k,v in bank.items() if k!='responses'}
base=opal.load_model(pathlib.Path(row['checkpoint'])/'base')
contract=read_json(pathlib.Path(row['checkpoint'])/'freeze.json')['contract']
residual=dev['responses']-predict_reference(base.config,base.fitted['reference'],dev)
head=fit_head(summary['means_projected'][train],summary['variance'][train],residual,weights(dev),row['arms']['variance']['chosen'],'variance')
save_head(head,'head-example',contract)
pred=predict_reference(base.config,base.fitted['reference'],q)+predict_head(head,summary['means_projected'][held],summary['variance'][held],512)
replay=predict_reference(base.config,base.fitted['reference'],q)+predict_head(load_head('head-example',contract),summary['means_projected'][held],summary['variance'][held],512)
with np.load(pathlib.Path(row['checkpoint'])/'predictions.npz',allow_pickle=False) as z: expected=z['variance']
difference=float(np.max(np.abs(expected-pred)))
assert difference<1e-10
assert np.allclose(pred,replay,atol=1e-10,rtol=0)
assert not pathlib.Path(opal.__file__).is_relative_to(root)
write_json('information-verification.json',{'python':sys.version,'package_path':opal.__file__,'real_data':True,'training_prediction_safe_replay':'passed','source_replay_max_abs':difference,'outside_checkout':True,'dependencies':'cached existing environment, no install/download','evaluation_role':'installed verification only; not new scientific confirmation'})
print(difference)
''')
    result=subprocess.run([str(Path(python).resolve()),str(installed_script)],cwd=destination,env=env,capture_output=True,text=True,timeout=1800)
    (destination/'information.stdout.txt').write_text(result.stdout);(destination/'information.stderr.txt').write_text(result.stderr)
    if result.returncode:raise RuntimeError(result.stderr+result.stdout)
    import json
    evidence=ROOT/'results/opal_information_v1/verification';evidence.mkdir(exist_ok=True)
    (evidence/('installed'+label+'.json')).write_text(json.dumps({'information':json.loads((destination/'information-verification.json').read_text()),
        'series':json.loads((destination/'series-verification/verification.json').read_text()),'package_arrays_weights_excluded':True},indent=2))
    print(label,'actual installed series and information head fit/predict/score/replay passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--python',required=True)
    p.add_argument('--package-root',default='package')
    a=p.parse_args();run(a.label,a.python,a.package_root)
