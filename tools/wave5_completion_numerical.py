"""Root execution of A's frozen marginally grounded nine-cell no-fit matrix."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.wave5_completion_common import ROOT, OUT, digest, gates, write


def array_hash(value):
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def discrepancy(value, reference):
    import numpy as np
    delta=np.abs(value-reference)
    active=delta!=0
    relative=np.divide(delta,np.abs(reference),out=np.zeros_like(delta),where=reference!=0)
    return dict(max_abs=float(delta.max(initial=0)),max_relative_nonzero_reference=float(relative.max(initial=0)),zero_reference_nonzero_errors=int(np.count_nonzero(active&(reference==0))),affected_rows=np.flatnonzero(active if active.ndim==1 else active.any(axis=1)).tolist(),bitwise_equal=bool(np.array_equal(value,reference)),atol_1e12_rtol0=bool(np.all(delta<=1e-12)))


def trace(s,pairs,ascending):
    import numpy as np
    from opal.models.fitness import design
    ordered=np.sort(s['singles'][pairs],axis=1)
    coordinates=(ordered-s['input_center'])/s['input_scale']
    distance=np.square(coordinates[:,None,:]-s['centers'][None,:,:]).sum(2)
    kernel=np.exp(-distance/(2*float(s['bandwidth'])**2))
    if ascending:
        white=np.zeros((len(pairs),int(s['n_centers'])))
        for j in range(kernel.shape[1]):white=white+kernel[:,j:j+1]*s['whitener'][j]
    else:white=kernel@s['whitener']
    features=np.concatenate((design(s['singles'],pairs,bool(s['nodes'])),white),axis=1)
    products=(features-s['center'])/s['scale']*s['coefficient']
    if ascending:
        projection=np.zeros(len(pairs))
        for column in products.T:projection=projection+column
    else:projection=(features-s['center'])/s['scale']@s['coefficient']
    prediction=(s['singles'][pairs].sum(1)+float(s['offset']))+projection
    return dict(kernel=kernel,whitening=white,features=features,products=products,projection=projection,prediction=prediction)


def exact_rounded_sum(products):
    return float(sum((Fraction(float(v)) for v in products),Fraction()))


def run(output):
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_limits,threadpool_info
    from opal.flexible_prediction import FixedOrderFlexiblePredictor,FIELDS
    from opal.flexible_singles import FlexibleSingles
    freeze=gates();started=time.perf_counter();cpu=time.process_time()
    def forbidden(*args,**kwargs):raise RuntimeError('No fitting in numerical completion')
    FlexibleSingles.fit=FlexibleSingles.prepare=FlexibleSingles.solve=forbidden
    checkpoint=Path('E:/opal-wave2-20261021/q2b-v1/horlbeck_K562/fold-0/nodes-1.npz')
    original=FixedOrderFlexiblePredictor.from_legacy_checkpoint(checkpoint,'c48569c57c367542ebaad6400fe1c9da4383042a85019891902bc09391b65de3')
    baseline={k:v.copy() for k,v in original._state.items() if k in FIELDS};baseline['format_version']=np.array(1)
    support=np.flatnonzero(baseline['support']);rng=np.random.default_rng(510523)
    bootstrap=rng.choice(baseline['singles'][support],len(support),replace=True)
    pairs=[]
    while len(pairs)<257:
        row=rng.choice(support,2,replace=True)
        if row[0]!=row[1]:pairs.append(row)
    pairs=np.array(pairs);permutation=np.random.default_rng(41).permutation(len(pairs))
    def summary(a):return dict(shape=list(a.shape),quantiles=np.quantile(a,[0,.01,.25,.5,.75,.99,1]).tolist(),sha256=array_hash(a))
    cells=[];tuples=None
    with threadpool_limits(1),tempfile.TemporaryDirectory(prefix='opal-completion-grid-',dir=ROOT/'tmp') as temporary:
        path=Path(temporary)/'state.npz'
        for width in (.5,1.,2.):
            for scale in (2.**-10,1.,2.**10):
                gates();cell_start=time.perf_counter()
                s={k:v.copy() for k,v in baseline.items()}
                median=np.median(baseline['singles'][support]);s['singles'][support]=median+(bootstrap-median)*width
                k=int(s['n_centers']);s['scale'][-k:]*=scale
                np.savez(path,**s);model=FixedOrderFlexiblePredictor.from_legacy_checkpoint(path,digest(path))
                fixed=trace(s,pairs,True);blas=trace(s,pairs,False);actual=model.predict(pairs)
                controls={}
                for size in (1,17,63,257):controls['partition-'+str(size)]=discrepancy(np.concatenate([model.predict(pairs[j:j+size]) for j in range(0,len(pairs),size)]),actual)
                for name,query,restore in [('reverse',pairs[::-1],lambda x:x[::-1]),('permutation',pairs[permutation],lambda x:x[np.argsort(permutation)]),('orientation',pairs[:,::-1],lambda x:x),('C',np.ascontiguousarray(pairs),lambda x:x),('F',np.asfortranarray(pairs),lambda x:x)]:controls[name]=discrepancy(restore(model.predict(query)),actual)
                strided=np.empty((len(pairs),4),dtype=pairs.dtype);strided[:,::2]=pairs
                controls['strided']=discrepancy(model.predict(strided[:,::2]),actual)
                repeated=np.tile(pairs[:1],(257,1));controls['repeated']=discrepancy(model.predict(repeated),np.full(257,actual[0]))
                stages={name:discrepancy(fixed[name],blas[name]) for name in ('kernel','whitening','features','projection','prediction')}
                oracle=[]
                for row in (0,1,2,3,63,127,255,256):
                    for column in (0,k//2,k-1):
                        exact=exact_rounded_sum(fixed['kernel'][row]*s['whitener'][:,column])
                        oracle.append(dict(stage='whitening',row=row,column=column,reference=exact,fixed_error=float(fixed['whitening'][row,column]-exact),blas_error=float(blas['whitening'][row,column]-exact)))
                    exact=exact_rounded_sum(fixed['products'][row]);oracle.append(dict(stage='projection',row=row,reference=exact,fixed_error=float(fixed['projection'][row]-exact)))
                weighted=(fixed['whitening']-blas['whitening'])*(s['coefficient'][-k:]/s['scale'][-k:])
                signed=weighted.sum(1);delta=fixed['prediction']-blas['prediction'];triangle=np.abs(weighted).sum(1)
                row=dict(width=width,scale_multiplier=scale,state_sha256={name:array_hash(value) for name,value in s.items()},distribution=dict(singles=summary(s['singles'][support]),kernel=summary(fixed['kernel']),whitening=summary(fixed['whitening']),nonlinear_scale=summary(s['scale'][-k:]),weighted_coefficient=summary(s['coefficient'][-k:]/s['scale'][-k:])),production_vs_ascending=discrepancy(actual,fixed['prediction']),controls=controls,stages=stages,first_divergent_operation=next((name for name,v in stages.items() if not v['bitwise_equal']),None),oracle_scope='correctly rounded exact sum of already-rounded float64 products; excludes product and upstream kernel error',reduction_oracles=oracle,oracle_max_fixed_abs=max(abs(o['fixed_error']) for o in oracle),oracle_max_blas_whitening_abs=max(abs(o['blas_error']) for o in oracle if 'blas_error' in o),signed_weighted_max_abs=float(np.max(abs(signed))),triangle_max=float(triangle.max()),minimum_bound_slack=float(np.min(triangle-np.abs(signed))),reconstruction_residual_max_abs=float(np.max(abs(delta-signed))),prediction_sha256=array_hash(actual),wall_seconds=time.perf_counter()-cell_start)
                if width==scale==1:
                    tiled=np.tile(pairs,(8,1))[:2051];full=model.predict(tiled)
                    row['boundary_anchor']={str(size):discrepancy(np.concatenate([model.predict(tiled[j:j+size]) for j in range(0,len(tiled),size)]),full) for size in (2047,2048,2049,2051)}
                    indices=sorted(set(np.linspace(0,len(s['scale'])-1,126,dtype=int).tolist()+[int(np.argmin(s['scale'])),int(np.argmax(abs(s['coefficient']/s['scale'])))]))
                    tuples=dict(executor='root',freeze_sha256=freeze,checkpoint_sha256=digest(checkpoint),tool_sha256=digest(__file__),query_sha256=array_hash(pairs),state_sha256=row['state_sha256'],selection='union126 evenly spaced,min scale,maxabs beta/scale;central row0',tuples=[dict(column_index=i,feature=float(fixed['features'][0,i]),center=float(s['center'][i]),scale=float(s['scale'][i]),coefficient=float(s['coefficient'][i])) for i in indices])
                cells.append(row)
                if time.perf_counter()-started>360 or time.process_time()-cpu>360:raise RuntimeError('Numerical time cap')
                if not np.isfinite(actual).all():raise ValueError('Unexpected nonfinite prediction')
    assert len(cells)==9 and tuples is not None
    assert all(c['production_vs_ascending']['bitwise_equal'] for c in cells)
    gates();write(OUT/'A-sensitivity-tuples.json',tuples)
    result=dict(executor='root after A provider failure',freeze_sha256=freeze,checkpoint_sha256=digest(checkpoint),synthetic_only=True,fits=0,scores=0,outcome_arrays_loaded=False,seed=510523,original_singles=summary(baseline['singles'][support]),query_sha256=array_hash(pairs),support_sha256=array_hash(baseline['support']),distribution_limit='marginal bootstrap plus altered widths/scales; not actual joint assay distribution',cells=cells,all_self_controls_bitwise=all(all(c['bitwise_equal'] for c in row['controls'].values()) for row in cells),threadpools=threadpool_info(),wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,os_lifetime_peak_bytes=max(psutil.Process().memory_info().rss,getattr(psutil.Process().memory_info(),'peak_wset',0)))
    write(output,result)
    print(json.dumps({k:result[k] for k in ('executor','all_self_controls_bitwise','wall_seconds','cpu_seconds','os_lifetime_peak_bytes')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    run(args.output)
