import io,pickle
import numpy as np
import pytest
from opal.models.experts import FeatureRidgeResidual,MeasurementGP,retrieve_residual
from opal.inference import export_members,Predictor
from test_advanced_architecture import fixture


def test_prior_ridge_matches_runner_and_safe_export(tmp_path):
    from tools.study_architecture import prior_ridge
    singles,pairs,doubles,r=fixture()
    train=np.arange(6);query=np.array([6,7])
    model=FeatureRidgeResidual(singles,r['bank'],10).fit(pairs[train],doubles[train])
    np.testing.assert_allclose(model.predict(pairs[query]),prior_ridge(singles,r,pairs,doubles,train,query,10))
    np.testing.assert_allclose(model.predict(pairs[query]),model.predict(pairs[query,::-1]))
    data={'singles':singles,'targets':np.array(list('abcdefg')),'genes':np.array([f'g{i}' for i in range(32)]),
        'control':np.zeros(32),'pairs':pairs[:6],'query_pairs':pairs[6:]}
    export_members([{'type':'feature_ridge','model':model,'config':{'alpha':10},'seed':0}],data,tmp_path,{'context':'fixture'})
    np.testing.assert_allclose(Predictor(tmp_path).predict('a','g')['effect'],model.predict(pairs[6:])[0])


def test_gp_variance_symmetric_finite_and_query_response_free():
    singles,pairs,doubles,r=fixture()
    model=MeasurementGP(singles,r['bank']).fit(pairs[:6],doubles[:6],np.full((6,32),.1))
    prediction,std=model.predict(pairs[6:],True)
    swapped,swapped_std=model.predict(pairs[6:,::-1],True)
    np.testing.assert_allclose(prediction,swapped);np.testing.assert_allclose(std,swapped_std)
    assert np.isfinite(std).all() and np.all(std>=0)
    doubles[6:]+=999
    np.testing.assert_allclose(prediction,model.predict(pairs[6:]))


def test_retrieval_excludes_identity_and_uses_training_responses_only():
    singles,pairs,doubles,r=fixture();train=np.arange(6);query=np.array([6,7])
    prediction=retrieve_residual(singles,r['bank'],pairs,doubles,train,query)
    doubles[query]+=999
    np.testing.assert_allclose(prediction,retrieve_residual(singles,r['bank'],pairs,doubles,train,query))
    with pytest.raises(ValueError,match='overlaps'):
        retrieve_residual(singles,r['bank'],pairs,doubles,train,np.array([0]))


def test_data_only_unpickler_rejects_executable_globals():
    from tools.prepare_transfer import DataOnlyUnpickler
    class Unsafe:
        def __reduce__(self):return eval,('1+1',)
    with pytest.raises(pickle.UnpicklingError,match='Unsupported'):
        DataOnlyUnpickler(io.BytesIO(pickle.dumps(Unsafe()))).load()
    array=np.arange(10,dtype=np.float32)
    loaded=DataOnlyUnpickler(io.BytesIO(pickle.dumps({'gene':array}))).load()
    np.testing.assert_array_equal(loaded['gene'],array)


def test_independent_transfer_encoder_and_decoder_roundtrip():
    from opal.models.advanced import AdvancedResidual
    singles,pairs,doubles,r=fixture()
    r.update(transfer_weight=np.ones((8,8),np.float32)*.03,transfer_bias=np.zeros(8,np.float32),
        transfer_output_basis=np.eye(32,4,dtype=np.float32))
    config={'epochs':3,'width':8,'rank':3,'priors':True,'transfer':True}
    model=AdvancedResidual(singles,r,config).fit(pairs[:6],doubles[:6],seed=73)
    assert model.counts[2]==4
    restored=AdvancedResidual.from_state(model.state(),config)
    np.testing.assert_allclose(model.predict(pairs[6:]),restored.predict(pairs[6:]),atol=1e-6)


def test_prior_kernel_psd_symmetry_missing_sources_and_export(tmp_path):
    from opal.models.prior_kernel import PriorKernelResidual
    singles,pairs,doubles,r=fixture()
    config={'sources':[0,1,2],'alpha':1,'bandwidth':1,'prior_weight':.1}
    r['source_mask'][0,1]=False
    model=PriorKernelResidual(singles,r['bank'],r['source_mask'],config).fit(pairs[:6],doubles[:6])
    gram=model.kernel(pairs,pairs)
    assert np.linalg.eigvalsh(gram).min()>-1e-8
    np.testing.assert_allclose(model.predict(pairs[6:]),model.predict(pairs[6:,::-1]))
    data={'singles':singles,'targets':np.array(list('abcdefg')),'genes':np.array([f'g{i}' for i in range(32)]),
        'control':np.zeros(32),'pairs':pairs[:6],'query_pairs':pairs[6:]}
    export_members([{'type':'prior_kernel','model':model,'config':config,'seed':0}],data,tmp_path,{'context':'fixture'})
    np.testing.assert_allclose(Predictor(tmp_path).predict('a','g')['effect'],model.predict(pairs[6:])[0])


def test_preserved_split_keeps_old_identity_roles_and_duplicates():
    from opal.splits import preserved_pair_split
    pairs=np.array([(str(i),str(i+1)) for i in range(15)]+[('1','0')])
    split=preserved_pair_split(pairs,[('0','1'),('1','2')],[('2','3')],folds=3)
    assert 0 in split['development'] and 1 in split['development'] and 15 in split['development']
    assert 2 in split['test']
    assert not set(split['development'])&set(split['test'])
    with pytest.raises(ValueError,match='overlap'):
        preserved_pair_split(pairs,[('0','1')],[('1','0')],folds=3)


def test_linear_weighted_decoder_checkpoint():
    from opal.models.advanced import AdvancedResidual
    singles,pairs,doubles,r=fixture()
    config={'epochs':5,'rank':3,'width':8,'priors':True,'linear':True,'structured':True,'signal_loss':1}
    model=AdvancedResidual(singles,r,config).fit(pairs[:6],doubles[:6],np.full((6,32),.1),seed=81)
    restored=AdvancedResidual.from_state(model.state(),config)
    np.testing.assert_allclose(model.predict(pairs[6:]),restored.predict(pairs[6:]),atol=1e-6)
    np.testing.assert_allclose(model.attention_weights().sum(1),1)


def test_linear_decoder_scales_only_training_features():
    import torch
    from opal.models.advanced import AdvancedResidual
    singles,pairs,doubles,r=fixture();r['bank'][:,1:]*=1e5
    model=AdvancedResidual(singles,r,{'epochs':10,'rank':3,'width':8,'priors':True,'linear':True}).fit(pairs[:6],doubles[:6],seed=81)
    features=model.network.features(torch.tensor(model.bank),torch.tensor(model.mask),torch.tensor(model.reliability),torch.tensor(pairs[:6]))
    np.testing.assert_allclose(model.network.feature_mean.numpy(),features.mean(0).numpy(),rtol=1e-6,atol=1e-6)
    assert model.network.feature_scale.max()>1e6
    assert np.abs(model.predict(pairs[6:])).max()<100


def test_bounded_linear_head_roundtrip_and_extrapolation():
    from opal.models.advanced import AdvancedResidual
    singles,pairs,doubles,r=fixture()
    config={'rank':3,'width':8,'priors':True,'linear':True,'linear_clip':8}
    model=AdvancedResidual(singles,r,config).fit(pairs[:6],doubles[:6],seed=81)
    model.bank[6]*=1e6
    prediction=model.predict(pairs[6:])
    restored=AdvancedResidual.from_state(model.state(),config)
    np.testing.assert_allclose(prediction,restored.predict(pairs[6:]),atol=1e-6)
    assert np.abs(prediction).max()<100
