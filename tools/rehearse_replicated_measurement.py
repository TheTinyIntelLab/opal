"""Bounded technical rehearsal using development banks only, never evaluators."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.replicated.contracts import write_json,sha
from opal.replicated.models import ProfileKernel
from tools.run_replicated_fixed_state import diagnostics


def cosine_summary(a,b):
    norm=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    defined=norm>1e-12
    values=np.sum(a[defined]*b[defined],axis=1)/norm[defined]
    pooled=float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>1e-12 else None
    return {'defined_pairs':int(defined.sum()),'undefined_pairs':int((~defined).sum()),
            'mean_pair_cosine':float(values.mean()) if len(values) else None,'pooled_cosine':pooled,
            'norms_a':np.linalg.norm(a,axis=1).tolist(),'norms_b':np.linalg.norm(b,axis=1).tolist()}


def rehearse(directory,menu,seed):
    directory=Path(directory)
    with np.load(directory/'profiles.npz',allow_pickle=False) as f:
        # Do not read any historical evaluator or held-pair response arrays.
        singles=f['singles'];doubles=f['doubles'];pairs=f['pairs'];genes=f['genes'];targets=f['targets']
    with np.load(directory/'profile_uncertainty.npz',allow_pickle=False) as f:
        halves=[(f[f'singles_{h}'],f[f'doubles_{h}']) for h in (0,1)]
    identities=[];component_profiles=[]
    for pair in pairs:
        ordered=sorted(pair,key=lambda i:str(targets[i]))
        identities.append(json.dumps([str(targets[i]) for i in ordered],separators=(',',':')))
        component_profiles.append(singles[ordered])
    pair_ids=np.asarray(identities);order=np.random.default_rng(seed).permutation(len(pairs))
    partitions=np.full(len(pairs),'train',dtype='U10');partitions[order[:max(1,len(pairs)//5)]]='validation'
    bank={'singles':np.asarray(component_profiles),'responses':doubles,'pair_ids':pair_ids,
          'culture_ids':np.asarray(['exposed_technical_population']*len(pairs)),
          'row_ids':np.asarray([f'development_{i}' for i in range(len(pairs))]),'partitions':partitions,
          'genes':genes,'scale':np.asarray('historical_log2_mean_effect'),'units':np.asarray('log2')}
    report=diagnostics(bank,menu,seed)
    cfg=report['selection']['reference'];shared={'kind':'global','alpha':cfg.get('alpha',1),'bandwidth':1}
    # Full source-only shared correction, cross-fitted by global pair identity.
    half_residual=[];half_departures=[]
    for s,d in halves:
        x=np.asarray([s[sorted(p,key=lambda i:str(targets[i]))] for p in pairs]);pred=np.empty_like(d)
        for held in np.array_split(order,5):
            train=np.setdiff1d(np.arange(len(pairs)),held)
            model=ProfileKernel(shared).fit(x[train],d[train]);pred[held]=model.predict(x[held])
        half_residual.append(d-x.sum(1));half_departures.append(d-pred)
    residual=doubles-bank['singles'].sum(1)
    report['cosine_audit']={'raw_halves':cosine_summary(*half_residual),
        'after_crossfit_shared':cosine_summary(*half_departures),
        'zero_effect_vs_observed_residual':cosine_summary(-bank['singles'].sum(1),residual),
        'full_residual_norms':np.linalg.norm(residual,axis=1).tolist(),
        'definitions':{'half_residual':'half double effect minus half single A minus half single B',
                       'zero_effect_correction':'negative observed additive prediction'},
        'shared_preprocessing':'full-data reference QC, normalization target and gene universe are fixed',
        'control_matching':'half-specific stratum controls follow saved preparation contract',
        'control_reuse':'whole single matrices retain shared control dependence within halves',
        'biological_reliability_established':False}
    train=partitions=='train';val=~train;precision=[]
    # Preserves each half's entire single matrix across all dependent pairs.
    # Mixes existing measurements; it does not claim actual read-depth subsampling.
    for part in ('singles','doubles'):
        for fraction in (0.,.25,.5,1.):
            errors=[]
            for s,d in halves:
                sx=(1-fraction)*singles+fraction*s if part=='singles' else singles
                dy=(1-fraction)*doubles+fraction*d if part=='doubles' else doubles
                x=np.asarray([sx[sorted(p,key=lambda i:str(targets[i]))] for p in pairs])
                model=ProfileKernel(shared).fit(x[train],dy[train])
                errors.append(float(np.mean((model.predict(x[val])-doubles[val])**2)))
            precision.append({'varied':part,'half_measurement_mixture':fraction,'validation_mse_by_half':errors})
    report['reference_precision_sensitivity']={'rows':precision,'fixed_validation_pair_ids':pair_ids[val].tolist(),
        'scope':'technical half/full mixture sensitivity, not an empirical depth allocation curve',
        'control_precision_separate':'blocked: saved effects do not identify control counts separately',
        'shared_draws':'complete half single matrix reused across all pairs',
        'full_measured_target_reused':True}
    report['inputs']={name:sha(directory/name) for name in ('profiles.npz','profile_uncertainty.npz','preprocessing.json','source_manifest.json')}
    report['exposure']='previously exposed development only; no fresh confirmation'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=ROOT/'data/prepared')
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/experiments/replicated_fixed_state_v1.json');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();menu=json.loads(a.protocol.read_text())['menu']
    with threadpool_limits(limits=1):
        for name in ('norman-log2','wessels-log2'):
            write_json(a.output/(name+'.json'),rehearse(a.data_root/name,menu,1729))
            if time.monotonic()-start>300:raise RuntimeError('Bounded rehearsal exceeded 300 seconds')
    write_json(a.output/'summary.json',{'status':'complete_technical_rehearsal','seconds':time.monotonic()-start,
        'maximum_seconds':300,'evaluator_access':False,'new_architectures':False,'biological_reliability_established':False,
        'control_only_allocation':'requires development raw/reference sampling under qualified assay contract'})


if __name__=='__main__':main()
