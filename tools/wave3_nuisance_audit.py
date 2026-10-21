"""Read-only arithmetic audit; no fitting or selection changes."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import load_npz, read_json, write_json, sha256
from opal.scalar_screen import block_assignments
from tools.wave3_nuisance import Guard,CONFIG,predict,registered,FAMILIES
from threadpoolctl import threadpool_limits

def audit(out):
    if (out/'arithmetic_audit.json').exists(): raise FileExistsError('Preserve audit')
    reg=registered(out); result=read_json(out/'results.json'); selection=read_json(out/'selection.json')
    guard=Guard(out,CONFIG,consumed=read_json(out/'replay.json')['resources'])
    q=read_json(Path(reg['q1b'])/'registration.json'); bank=load_npz(q['paths'][2])
    prepared=load_npz(Path(reg['q1b'])/'prepared.npz'); features=load_npz(out/'features.npz')['features']
    # Independently available Q1b values verify the new raw-stream aggregation.
    assert np.allclose(features[:,:,0],prepared['means'][...,4].mean(1),rtol=0,atol=1e-12)
    assert np.allclose(features[:,:,1],prepared['means'][...,5].mean(1),rtol=0,atol=1e-12)
    assert np.allclose(features[:,:,3],prepared['means'][...,3].mean(1),rtol=0,atol=1e-12)
    records=[]
    for c,context in enumerate(('K562','Jurkat')):
        task='horlbeck_'+context; predictions=load_npz(out/(task+'_predictions.npz'))
        y=bank['responses'][c,predictions['test_indices']]
        for name,stats in result['tasks'][task]['metrics'].items():
            assert abs(np.mean((predictions[name]-y)**2)-stats['mse'])<1e-15
        splits=block_assignments(dict(genes=bank['genes'],pairs=bank['pairs'],accession='GSE116198'))
        for rec in [r for r in selection['records'] if r['task']==task]:
            guard.check(); fold=rec['fold']; val=np.flatnonzero(splits[fold]==1); dest=out/task/f'fold-{fold}'
            for family in FAMILIES:
                model=load_npz(dest/(family+'.npz')); f=features[c] if family=='nuisance_nodes' else features[c,:,:5]
                prediction=predict(model,f,bank['pairs'],val,len(bank['genes']))
                expected=rec['menus'][family]['validation_mse'][rec['menus'][family]['chosen']]
                assert abs(np.mean((prediction-bank['responses'][c,val])**2)-expected)<1e-15
            records.append(dict(task=task,fold=fold,winner=rec['winner']))
        pairs=bank['pairs'][predictions['test_indices']]; active,inverse=np.unique(pairs,return_inverse=True); edges=inverse.reshape(-1,2)
        delta=(predictions['aligned']-y)**2-(predictions['selected']-y)**2
        rng=np.random.default_rng(20261022); boot=[]
        for _ in range(2000):
            counts=np.bincount(rng.integers(len(active),size=len(active)),minlength=len(active)); weight=counts[edges].prod(1)
            if weight.sum(): boot.append(float(np.dot(weight,delta)/weight.sum()))
        assert np.array_equal(np.quantile(boot,[.025,.975]),result['tasks'][task]['comparisons']['selected']['uncertainty']['gene_conditional_95'])
    guard.check(); write_json(out/'arithmetic_audit.json',dict(passed=True,checks=['Q1b sum/abs/min covariate reconstruction','all test MSEs','all selected validation penalties/families','both primary conditional intervals'],
        records=records,script_sha256=sha256(__file__),results_sha256=sha256(out/'results.json'),resources_including_audit=guard.receipt(),
        limitation='read-only audit by same workstream; not independent biological evidence or optimizer replay'))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    with threadpool_limits(1): audit(a.output)
