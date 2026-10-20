import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from opal.information import (fit_head, load_head, predict_head, projected_variance,
    reference_projection, require_branch, save_head, split_masks, stratified_shuffle)
from opal.series import Unavailable


def example():
    rng = np.random.default_rng(81)
    means = np.zeros((80,2,1))
    variance = rng.uniform(.2,3,(80,2,1))
    response = variance.sum(1) @ np.array([[1.,-2.,.5]])
    return means,variance,response


def test_informative_equal_mean_variance_and_null():
    means,variance,response=example()
    cfg={'input_rank':1,'output_rank':1,'alpha':.0001}
    with threadpool_limits(limits=1):
        head=fit_head(means[:60],variance[:60],response[:60],np.repeat(1/60,60),cfg,'variance')
        prediction=predict_head(head,means[60:],variance[60:],3)
    assert np.mean((prediction-response[60:])**2)<1e-5
    assert np.mean((response[60:]-response[:60].mean(0))**2)>.1
    np.testing.assert_allclose(prediction,predict_head(head,means[60:,::-1],variance[60:,::-1],3),atol=1e-12)
    with pytest.raises(Unavailable,match='Output rank'):
        fit_head(means[:60],variance[:60],np.zeros((60,3)),np.repeat(1/60,60),cfg,'variance')
    assert np.array_equal(predict_head(None,means,variance,3),np.zeros((80,3)))


def test_control_only_projection_and_missing_summary():
    rng=np.random.default_rng(2)
    controls=rng.normal(size=(30,4))
    center,basis=reference_projection(controls,2)
    assert np.allclose(projected_variance(controls,basis),np.var((controls-center)@basis,axis=0,ddof=1))
    with pytest.raises(Unavailable): reference_projection(np.ones((20,4)),2)
    means,var,y=example()
    head=fit_head(means,var,y,np.repeat(1/80,80),{'input_rank':1,'output_rank':1,'alpha':.1},'variance')
    var[3]=np.nan
    assert np.array_equal(predict_head(head,means,var,3)[3],np.zeros(3))


@pytest.mark.parametrize('kind',['global_pair','partner'])
def test_global_exclusion_and_complete_population(kind):
    pairs=np.asarray([json.dumps([a,b]) for a in ['a','b','c','d','e'] for b in ['f','g','h']]*2)
    count=np.zeros(len(pairs),int)
    for fold in range(3):
        train,test=split_masks(pairs,kind,fold)
        assert not set(pairs[train])&set(pairs[test])
        if kind=='partner':
            # At least one held component of every query never appears in training.
            train_targets={t for p in pairs[train] for t in json.loads(p)}
            assert all(any(t not in train_targets for t in json.loads(p)) for p in pairs[test])
        count+=test
    assert np.all(count==1)


def test_held_mutation_does_not_fit_transform():
    means,var,y=example()
    train=np.arange(60)
    first=fit_head(means[train],var[train],y[train],np.repeat(1/60,60),{'input_rank':1,'output_rank':1,'alpha':.1},'variance')
    y[60:]=1e9
    second=fit_head(means[train],var[train],y[train],np.repeat(1/60,60),{'input_rank':1,'output_rank':1,'alpha':.1},'variance')
    assert np.array_equal(first['basis'],second['basis'])
    assert np.array_equal(first['coefficients'],second['coefficients'])


def test_strata_shuffle_and_singleton_refusal():
    values=np.arange(6,dtype=float)[:,None]
    shuffled,records=stratified_shuffle(values,[10,12,25,30,51,100],9)
    for group in [(0,1),(2,3),(4,5)]:
        assert set(shuffled[list(group),0])==set(values[list(group),0])
        assert not np.any(shuffled[list(group)]==values[list(group)])
    assert np.array_equal(shuffled,stratified_shuffle(values,[10,12,25,30,51,100],9)[0])
    assert np.isnan(stratified_shuffle(values[:1],[10],9)[0]).all()


def test_safe_replay_contract_and_tampering(tmp_path):
    means,var,y=example()
    head=fit_head(means,var,y,np.repeat(1/80,80),{'input_rank':1,'output_rank':1,'alpha':.1},'variance')
    contract={'source':'fixed','units':'log2','genes':['a','b','c'],'reference':'hash'}
    save_head(head,tmp_path/'head',contract)
    loaded=load_head(tmp_path/'head',contract)
    assert np.array_equal(predict_head(head,means,var,3),predict_head(loaded,means,var,3))
    with pytest.raises(ValueError,match='mismatch'): load_head(tmp_path/'head',dict(contract,units='count'))
    with pytest.raises(ValueError,match='mismatch'): load_head(tmp_path/'head',contract,'deployment')
    with (tmp_path/'head/parameters.npz').open('ab') as f: f.write(b'tamper')
    with pytest.raises(ValueError,match='hash'): load_head(tmp_path/'head',contract)


@pytest.mark.parametrize('branch',['distribution','context','uncertainty','acquisition','scaling','confirmation'])
def test_unpassed_gates_refuse(branch):
    with pytest.raises(Unavailable,match='requires'): require_branch(branch,{})


def test_process_failure_and_child_accounting(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    import measure_information_process as monitor
    monkeypatch.setattr(monitor,'OUT',tmp_path)
    # A sampled monitor explicitly may miss short-lived children. Keep this
    # fixture alive until its memory sample is observed, rather than depending
    # on a CPU loop taking longer than scheduling/receipt I/O on the runner.
    acknowledgement=tmp_path/'child-observed'
    original_memory_info=monitor.psutil.Process.memory_info
    def acknowledge_child(process):
        memory=original_memory_info(process)
        if (tmp_path/f'{process.pid}.ready').exists():
            acknowledgement.touch()
        return memory
    monkeypatch.setattr(monitor.psutil.Process,'memory_info',acknowledge_child)
    child=(f'import os,time; from pathlib import Path; sum(i*i for i in range(2000000)); '
           f'Path({str(tmp_path)!r},str(os.getpid())+".ready").touch(); '
           f'ack=Path({str(acknowledgement)!r}); deadline=time.monotonic()+10\n'
           'while not ack.exists() and time.monotonic()<deadline: time.sleep(.005)\n'
           'assert ack.exists(), "monitor did not observe child"')
    code=f'import subprocess,sys; subprocess.run([sys.executable,"-c",{child!r}]); sys.exit(3)'
    assert monitor.run([sys.executable,'-c',code],'failure-fixture',30)==3
    record=json.loads((tmp_path/'receipts/failure-fixture.json').read_text())
    assert record['status']=='failed' and len(record['pids'])>=2
    assert record['cpu_seconds_observed']>0 and record['peak_tree_rss_bytes']>0
    with pytest.raises(FileExistsError): monitor.run([sys.executable,'-c','pass'],'failure-fixture')
