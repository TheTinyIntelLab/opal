"""Prospective hand-oracle audit of v2 metrics/checkpoints, synthetic only."""
import argparse
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import tempfile
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.wave5_completion_common import gates, write


def audit():
    import numpy as np
    from opal.evaluation_contracts import metrics_v2, strict_load_completion_v2
    from opal.block_completion import ScalarCompletion
    freeze = gates()
    records = []

    def metric(name, y, p):
        y, p = np.asarray(y), np.asarray(p)
        record = dict(name=name, target_dtype=str(y.dtype), prediction_dtype=str(p.dtype))
        if y.ndim == 1 and y.shape == p.shape and len(y) and y.dtype.kind in 'iuf' and p.dtype.kind in 'iuf' and np.isfinite(y).all() and np.isfinite(p).all():
            errors = [Fraction(float(b))-Fraction(float(a)) for a,b in zip(y,p)]
            oracle = sum(e*e for e in errors)/len(errors)
            record.update(rational_mse=str(oracle), nonzero_error=any(errors))
            try:
                record['rounded_rational_mse'] = float(oracle)
            except OverflowError:
                record['rounded_rational_mse'] = 'overflow'
        try:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter('always')
                record['result'] = metrics_v2(y,p)
                record['warnings'] = [str(w.message) for w in seen]
            record['accepted'] = True
            record['silent_nonzero_loss_to_zero'] = bool(record.get('nonzero_error') and record['result']['mse'] == 0)
        except ValueError as error:
            record.update(accepted=False, refusal=str(error))
        records.append(record)

    for dtype in (np.uint8,np.int64,np.float32,np.float64):
        metric('dtype-'+np.dtype(dtype).name,np.array([0,1,2,3],dtype),np.array([1,2,3,4],dtype))
    for exponent in (-600,-500,0,500,600):
        scale = np.ldexp(1.,exponent)
        metric('scale-'+str(exponent),np.array([0.,1.,-1.,2.])*scale,np.array([1.,2.,0.,3.])*scale)
    for shift in (0.,float(2**20)):
        metric('shift-'+str(shift),np.array([0.,1.,2.,3.])+shift,np.array([1.,2.,3.,4.])+shift)
    for name,y,p in [('singleton',[1.],[2.]),('constant',[1.,1.],[2.,2.]),('empty',[],[]),('misaligned',[1.,2.],[1.]),('boolean',[True],[False]),('complex',[1j],[2j]),('nan',[np.nan],[0.]),('infinity',[0.],[np.inf])]:
        metric(name,y,p)

    checkpoints=[]
    with tempfile.TemporaryDirectory(prefix='opal-completion-evaluator-',dir=Path(__file__).resolve().parents[1]/'tmp') as temporary:
        path=Path(temporary)/'fixture.npz'
        baseline=dict(rank=2,penalty=.01,steps=25,seed=1,format_version=2,normalized_upper=np.arange(6,dtype=np.float32)/10,support=np.ones(3,bool),scale=1.,best_step=25,parameter_count=12)
        cases=[('valid',{}),('partial-support',{'support':np.array([True,True,False])})]
        cases += [('scale-'+str(e),{'scale':np.ldexp(1.,e)}) for e in (-1074,0,1023)]
        cases += [(name,{field:value}) for name,field,value in [('boolean-rank','rank',True),('array-rank','rank',np.array([2])),('bad-step','best_step',26),('bad-count','parameter_count',11),('bad-support','support',np.ones(3,int)),('bad-upper','normalized_upper',np.zeros(6,np.float64)),('nonfinite','normalized_upper',np.full(6,np.nan,np.float32)),('overflow','normalized_upper',np.full(6,np.finfo(np.float32).max,np.float32))]]
        cases += [('missing',{}),('extra',{}),('duplicate',{}),('truncated',{})]
        for name,mutation in cases:
            data={**baseline,**mutation}
            if name=='overflow':data['scale']=np.finfo(float).max
            if name=='missing':del data['rank']
            if name=='extra':data['unexpected']=1
            np.savez(path,**data)
            if name=='duplicate':
                buffer=io.BytesIO();np.save(buffer,np.array(2))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with zipfile.ZipFile(path,'a') as archive:archive.writestr('rank.npy',buffer.getvalue())
            if name=='truncated':path.write_bytes(path.read_bytes()[:25])
            r=dict(name=name)
            try:
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter('always')
                    loaded=strict_load_completion_v2(path)
                r.update(accepted=True,exact_historical_replay=bool(np.array_equal(loaded.matrix,ScalarCompletion.load(path).matrix)),warnings=[str(w.message) for w in seen],nonzero_normalized_entries=int(np.count_nonzero(loaded.normalized)),nonzero_reconstructed_entries=int(np.count_nonzero(loaded.matrix)))
                r['silent_operator_underflow']=bool(np.any((loaded.normalized!=0)&(loaded.matrix==0)))
                if name=='partial-support':
                    try:loaded.predict(np.array([[0,2]]));r['unsupported_query_refused']=False
                    except ValueError:r['unsupported_query_refused']=True
            except (ValueError,zipfile.BadZipFile,EOFError) as error:
                r.update(accepted=False,refusal=type(error).__name__+': '+str(error))
            checkpoints.append(r)
    gates()
    return dict(freeze_sha256=freeze,executor='root',synthetic_only=True,biological_fits=0,biological_scores=0,metrics=records,checkpoints=checkpoints,metric_underflow_cases=[r['name'] for r in records if r.get('silent_nonzero_loss_to_zero')],operator_underflow_cases=[r['name'] for r in checkpoints if r.get('silent_operator_underflow')])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=audit();write(args.output,result)
    print(json.dumps({k:result[k] for k in ('executor','metric_underflow_cases','operator_underflow_cases')}))
