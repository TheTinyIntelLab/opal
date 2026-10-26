"""Ground-truth and contract checks for the prospective synthetic tournament."""
import hashlib
import numpy as np
import pytest
import torch

from opal.frontier_tournament import (make_world, budget_rows, pair_kernel,
    fit_models, Predictor, select_records, residual_permutation)


@pytest.fixture(scope='module')
def fitted():
    torch.set_num_threads(1)
    d = make_world('additive',77101,0.,genes=12)
    t,v,_ = budget_rows(d,1.)
    # Small fixture partition may have no two-partner support; this seed does.
    records = fit_models(d['singles'],d['features'],d['pairs'][t],d['observed'][0,t],
                         d['pairs'][v],d['observed'][0,v],71,controls=False,
                         completion_steps=5,neural_steps=5)
    return d,records


@pytest.mark.parametrize('noise',[0.,.1])
@pytest.mark.parametrize('mask',['complete','independent','input_quality'])
def test_causal_alias_and_redraw(noise,mask):
    a = make_world('lowrank',77102,noise,mask)
    b = make_world('technical_persistent',77102,noise,mask)
    c = make_world('technical_redrawn',77102,noise,mask)
    for key in a: assert np.array_equal(a[key],b[key]),key
    assert np.array_equal(a['observed'][0],c['observed'][0])
    assert not np.array_equal(a['observed'][1],c['observed'][1])


def test_budget_missingness_and_grouping():
    d = make_world('additive',77103,.1)
    t,v,cost = budget_rows(d,.25)
    assert (len(t),len(v),cost['charged_pair_union']) == (221,88,309)
    d['available'][t[0]] = False
    tt,vv,newcost = budget_rows(d,.25)
    assert np.array_equal(tt,t[1:]) and np.array_equal(vv,v)
    assert newcost['charged_pair_union'] == 309
    roles = np.concatenate([d[k] for k in ('train','validation','test')])
    assert np.array_equal(np.sort(roles),np.arange(1770))
    assert np.array_equal(d['constructs'].mean(2),d['observed'])


def test_kernel_symmetric_psd_identity_signal():
    s = np.zeros(5)
    p = np.array([[0,1],[0,2],[3,4]])
    k = pair_kernel(s,p,p)
    assert np.allclose(k,k.T,atol=1e-15,rtol=0)
    assert np.linalg.eigvalsh(k).min() >= -1e-14
    assert k[0,1] > k[0,2]  # identical singles, distinct shared-gene identity
    assert np.array_equal(k,pair_kernel(s,p[:,::-1],p))


def test_exact_additive_oracle_and_safe_replay(fitted,tmp_path):
    d,records = fitted
    _,simple,_ = select_records(records)
    assert simple['config'] == 'sum' and simple['validation_mse'] == 0
    query = d['pairs'][d['test']]
    for i,record in enumerate(records):
        model = record['model']
        query_supported = query[model.supported(query)]
        path = tmp_path/f'{i}.npz'
        model.save(path)
        loaded = Predictor.load(path,hashlib.sha256(path.read_bytes()).hexdigest())
        assert np.array_equal(model.predict(query_supported),loaded.predict(query_supported))
        assert np.allclose(model.predict(query_supported),model.predict(query_supported[:,::-1]),atol=1e-14,rtol=0)
        with pytest.raises(FileExistsError): model.save(path)
        with pytest.raises(ValueError,match='identity'): Predictor.load(path,'0'*64)


def test_unavailable_inputs_and_overlap_refuse(fitted):
    d,_ = fitted
    t,v,_ = budget_rows(d,1.)
    kwargs = dict(seed=1,controls=False,completion_steps=1,neural_steps=1)
    args = [d['singles'],d['features'],d['pairs'][t],d['observed'][0,t],d['pairs'][v],d['observed'][0,v]]
    bad = list(args); bad[1] = np.column_stack((d['singles'],d['singles']))
    with pytest.raises(ValueError,match='identities only'): fit_models(*bad,**kwargs)
    bad = list(args); bad[4] = d['pairs'][t[:2],::-1]; bad[5] = d['observed'][0,t[:2]]
    with pytest.raises(ValueError,match='leakage'): fit_models(*bad,**kwargs)


def test_unsupported_query_and_poison_exclusion(fitted):
    d,records = fitted
    model = records[0]['model']
    state = dict(model.state); state['support'] = state['support'].copy(); state['support'][0] = False
    with pytest.raises(ValueError,match='TRAIN partners'):
        Predictor(state).predict(np.array([[0,1]]))
    t,v,_ = budget_rows(d,1.)
    poisoned = dict(d,observed=d['observed'].copy())
    poisoned['observed'][1] = np.nan
    poisoned['observed'][0,d['test']] = np.nan
    other = fit_models(d['singles'],d['features'],d['pairs'][t],poisoned['observed'][0,t],
                       d['pairs'][v],poisoned['observed'][0,v],71,controls=False,
                       completion_steps=5,neural_steps=5)
    for first,second in zip(records,other,strict=True):
        assert first['validation_mse'] == second['validation_mse']
        for key in first['model'].state:
            assert np.array_equal(first['model'].state[key],second['model'].state[key])


def test_shuffle_preserves_graph_and_residual_multiset():
    d = make_world('sparse',77106,.1)
    t,_,_ = budget_rows(d,.25)
    permutation,fraction = residual_permutation(d['pairs'][t],d['singles'],71)
    assert fraction >= .8
    assert np.array_equal(np.sort(permutation),np.arange(len(t)))
