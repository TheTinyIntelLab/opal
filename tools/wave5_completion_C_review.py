"""Root execution of frozen C constructions, explicitly not independent review."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.wave5_completion_common import ROOT,OUT,digest,gates,write


def run():
    import numpy as np
    from opal.flexible_prediction import FixedOrderFlexiblePredictor
    from opal.evaluation_contracts import metrics_v2
    from opal.flexible_singles import FlexibleSingles
    from tools.wave5_completion_numerical import trace,discrepancy
    freeze=gates()
    def forbidden(*args,**kwargs):raise RuntimeError('No fitting in C-designed review')
    FlexibleSingles.fit=FlexibleSingles.prepare=FlexibleSingles.solve=forbidden
    values=np.array([2.**54,1.,-2.**54]);epsilon=2.**-26
    gram=np.array([[1.,1.],[1.,1.+epsilon],[1.,1.-epsilon],[1.,1.]])
    exact_gram=sum(Fraction(float(v))**2 for v in gram[:,1])
    constructions=dict(cancellation=dict(exact=float(sum(map(Fraction,values))),ascending=float(np.add.accumulate(values)[-1])),gram=dict(exact_increment=str(exact_gram-4),rounded=float((gram.T@gram)[1,1])),amplification=(2.**-40/2.**-20)/(2.**-40),disjoint_tolerance_intervals=bool(Fraction(0)+Fraction(1e-12)<Fraction(3e-12)-Fraction(1e-12)))
    receipt=json.loads((OUT/'A-sensitivity-tuples.json').read_text(encoding='utf8'))
    if receipt['freeze_sha256']!=freeze or len(receipt['tuples'])>128:raise ValueError('Tuple custody mismatch')
    sensitivity=[]
    for t in receipt['tuples']:
        feature,center,scale,beta=(t[k] for k in ('feature','center','scale','coefficient'))
        before=(feature-center)/scale*beta
        before_exact=(Fraction(feature)-Fraction(center))/Fraction(scale)*Fraction(beta)
        for nominal in (0.,2.**-52*max(1.,abs(feature)),-2.**-52*max(1.,abs(feature)),2.**-40*max(1.,abs(feature)),-2.**-40*max(1.,abs(feature))):
            changed=feature+nominal;realized=changed-feature
            after=(changed-center)/scale*beta
            after_exact=(Fraction(changed)-Fraction(center))/Fraction(scale)*Fraction(beta)
            exact_change=after_exact-before_exact;actual_change=after-before
            sensitivity.append(dict(column=t['column_index'],nominal=nominal,realized=realized,exact_change=float(exact_change),actual_change=actual_change,change_error=float(Fraction(actual_change)-exact_change),baseline_rounding_error=float(Fraction(before)-before_exact),triangle_bound=abs(realized*beta/scale),zero_realization=realized==0,atol_1e12_rtol0=abs(float(Fraction(actual_change)-exact_change))<=1e-12))
    query=np.tile(np.array([[0,1],[1,2],[2,0]]),(11,1))[:32]
    state=dict(format_version=1,nodes=True,alpha=1.,n_centers=2,seed=1,singles=np.array([-.5,.125,.25]),support=np.ones(3,bool),input_center=np.zeros(2),input_scale=np.ones(2),center_rows=np.array([0,1]),centers=np.array([[-1.,0.],[0.,1.]]),whitener=np.array([[1.,2.],[-1.,.5]]),bandwidth=1.,center=np.zeros(8),scale=np.ones(8),offset=.125,coefficient=np.arange(8,dtype=float)/8)
    with tempfile.TemporaryDirectory(prefix='opal-completion-C-',dir=ROOT/'tmp') as temporary:
        path=Path(temporary)/'legacy.npz';np.savez(path,**state)
        model=FixedOrderFlexiblePredictor.from_legacy_checkpoint(path,digest(path));expected=model.predict(query)
        control={str(size):discrepancy(np.concatenate([model.predict(query[j:j+size]) for j in range(0,len(query),size)]),expected) for size in (1,3,7,32)}
        permutation=np.random.default_rng(510023).permutation(32)
        control['permutation']=discrepancy(model.predict(query[permutation])[np.argsort(permutation)],expected)
        control['reverse']=discrepancy(model.predict(query[::-1])[::-1],expected)
        control['orientation']=discrepancy(model.predict(query[:,::-1]),expected)
        control['F']=discrepancy(model.predict(np.asfortranarray(query)),expected)
        strided=np.zeros((32,4),int);strided[:,::2]=query;control['strided']=discrepancy(model.predict(strided[:,::2]),expected)
        constructed=trace(state,query,True)
        reduction_errors=[]
        for row in range(32):
            exact=sum((Fraction(float(v)) for v in constructed['products'][row]),Fraction())
            reduction_errors.append(float(Fraction(float(constructed['projection'][row]))-exact))
        saved=Path(temporary)/'fixed.npz';model.save(saved)
        replay=bool(np.array_equal(FixedOrderFlexiblePredictor.load(saved,digest(saved)).predict(query),expected))
        refusals={}
        for name,field,value in [('scale','scale',np.zeros(8)),('nonfinite','singles',np.array([np.nan,.125,.25])),('dtype','singles',np.ones(3,np.float32)),('shape','whitener',np.ones((1,2)))]:
            data={**state,field:value};np.savez(path,**data)
            try:FixedOrderFlexiblePredictor.from_legacy_checkpoint(path,digest(path));refusals[name]=False
            except ValueError:refusals[name]=True
        np.savez(path,**{**state,'support':np.array([True,True,False])});unsupported=FixedOrderFlexiblePredictor.from_legacy_checkpoint(path,digest(path))
        try:unsupported.predict(np.array([[0,2]]));refusals['unsupported']=False
        except ValueError:refusals['unsupported']=True
        try:FixedOrderFlexiblePredictor.load(saved,'0'*64);refusals['hash']=False
        except ValueError:refusals['hash']=True
        with np.load(saved) as z:data={k:z[k].copy() for k in z.files}
        for name,value in [('bytes',np.array(b'opal-flexible-fixed-order-f64-v1')),('numeric',np.array(1))]:
            np.savez(path,**{**data,'prediction_contract':value})
            try:FixedOrderFlexiblePredictor.load(path);refusals['contract-'+name]=False
            except ValueError:refusals['contract-'+name]=True
    scalar=dict(unsigned_mse=metrics_v2(np.array([255,0],np.uint8),np.array([0,255],np.uint8))['mse'],signed_mse=metrics_v2(np.array([2**62,-2**62],np.int64),np.array([-2**62,2**62],np.int64))['mse'])
    assert scalar['unsigned_mse']==65025 and scalar['signed_mse']==float(2**126)
    assert all(r['bitwise_equal'] for r in control.values()) and all(refusals.values()) and replay
    assert all(r['actual_change']==0 for r in sensitivity if r['nominal']==0)
    gates()
    return dict(executor='root fallback; C independent execution unavailable',independent_validation=False,freeze_sha256=freeze,tuple_sha256=digest(OUT/'A-sensitivity-tuples.json'),synthetic_only=True,fits=0,scores=0,constructions=constructions,sensitivity=sensitivity,sensitivity_rows=len(sensitivity),max_change_error=max(abs(r['change_error']) for r in sensitivity),sensitivity_atol_failures=sum(not r['atol_1e12_rtol0'] for r in sensitivity),api_controls=control,exact_replay=replay,refusals=refusals,max_projection_reduction_error=max(map(abs,reduction_errors)),scalar=scalar,historical_results_retained=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();result=run();write(args.output,result)
    print(json.dumps({k:result[k] for k in ('executor','sensitivity_rows','max_change_error','sensitivity_atol_failures')}))
