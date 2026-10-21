"""Prerequisites for wave-three nuisance fits; synthetic outcomes only."""
import numpy as np
import pytest
from tools import wave3_nuisance as w

def test_construct_formula_and_double_outcome_absence():
    x=np.array([-2.,-1.]); y=np.array([-1.,-3.])
    f=w.construct_features(x,y,x+.2,y-.4)
    assert np.array_equal(f[:,3],[-2.,-3.])
    assert np.array_equal(f[:,0],x+y)
    assert np.array_equal(f[:,2],x*y)
    assert np.array_equal(f[:,5],x-y)
    assert np.allclose(f[:,9],.2)
    assert np.allclose(f[:,10],-.08)

def fixture():
    rng=np.random.default_rng(5); pairs=np.array([(a,b) for a in range(10) for b in range(a+1,10)])
    x=rng.normal(size=(len(pairs),5)); y=x@np.arange(5)+rng.normal(size=len(pairs))*.1
    return x,pairs,y,np.arange(30),np.arange(30,38),np.arange(38,45)

@pytest.mark.parametrize('nodes',[False,True])
def test_ridge_matches_dense_and_held_test_labels_do_not_enter(nodes):
    x,p,y,tr,va,te=fixture(); model,menu=w.ridge_grid(x,p,y,tr,va,nodes,10,lambda:None)
    z=w.design(x,p,tr,nodes,10); center=z.mean(0); scale=np.maximum(z.std(0),1e-8); z=(z-center)/scale
    beta=np.linalg.solve(z.T@z+menu['alpha']*np.eye(z.shape[1]),z.T@(y[tr]-y[tr].mean()))
    assert np.allclose(beta,model['coefficient'],atol=1e-10)
    changed=y.copy(); changed[te]=1e10
    other,other_menu=w.ridge_grid(x,p,changed,tr,va,nodes,10,lambda:None)
    assert menu==other_menu
    assert all(np.array_equal(model[k],other[k]) for k in model)
    assert np.array_equal(w.predict(model,x,p,te,10),w.predict(other,x,p,te,10))

def test_scaler_uses_training_covariates_only():
    x,p,y,tr,va,te=fixture(); model,_=w.ridge_grid(x,p,y,tr,va,False,10,lambda:None)
    x[te]=1e10; other,_=w.ridge_grid(x,p,y,tr,va,False,10,lambda:None)
    assert np.array_equal(model['center'],other['center'])
    assert np.array_equal(model['coefficient'],other['coefficient'])

def test_guard_callback_and_serialization(tmp_path):
    x,p,y,tr,va,te=fixture()
    def stop(): raise RuntimeError('stop')
    with pytest.raises(RuntimeError,match='stop'): w.ridge_grid(x,p,y,tr,va,True,10,stop)
    model,_=w.ridge_grid(x,p,y,tr,va,True,10,lambda:None)
    np.savez_compressed(tmp_path/'m.npz',**model)
    assert np.array_equal(w.predict(model,x,p,te,10),w.predict(w.load_npz(tmp_path/'m.npz'),x,p,te,10))

def test_frozen_signature_and_no_overwrite(tmp_path):
    assert all(len(v)==64 for v in w.signature().values())
    with pytest.raises(FileExistsError): w.freeze(argparse_namespace(tmp_path))
    f=tmp_path/'input'; f.write_text('a'); frozen={str(f):w.sha256(f)}
    w.check_files(frozen); f.write_text('b')
    with pytest.raises(ValueError,match='drift'): w.check_files(frozen)

def argparse_namespace(path):
    from argparse import Namespace
    return Namespace(output=path)

def test_existing_block_controls_grouped_and_train_only():
    x,p,y,tr,va,te=fixture()
    screen=dict(genes=np.array([f'G{i:02}' for i in range(10)]),pairs=p,accession='GSE116198',singles=np.arange(10.),counts=np.full((len(p),2),8))
    split=w.block_assignments(screen)[0]
    for control in w.residual_permutations(screen,split):
        assert np.array_equal(np.sort(control['permutation']),np.flatnonzero(split==0))
        assert not np.isin(control['permutation'],np.flatnonzero(split==2)).any()

def test_resource_configuration():
    assert w.CONFIG['rss_bytes']==1024**3
    assert w.CONFIG['wall_seconds']==1800
    assert w.CONFIG['artifact_bytes']==100*1024**2
    assert len(w.CONFIG['control_seeds'])==3
