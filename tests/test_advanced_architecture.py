import numpy as np
import pytest
import torch
from opal.models.advanced import AdvancedResidual, SourceFusion, signal_weights, check_pairs
from opal.inference import export_members, Predictor


def fixture(seed=17):
    rng=np.random.default_rng(seed)
    singles=rng.normal(size=(7,32))
    pairs=np.array([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(0,6),(1,5)])
    doubles=singles[pairs[:,0]]+singles[pairs[:,1]]+.2*(singles[pairs[:,0]]*singles[pairs[:,1]])
    bank=rng.normal(size=(7,3,8)).astype(np.float32)
    resources={'bank':bank,'source_mask':np.ones((7,3),bool),'single_variance':np.full((7,32),.1),
        'membership':np.tile(np.eye(4),(8,1)),'gene_prior':rng.normal(size=(32,5)),
        'single_draw_features':np.stack([bank[:,0]+rng.normal(0,.1,(7,8)) for _ in range(4)]).astype(np.float32)}
    return singles,pairs,doubles,resources


@pytest.mark.parametrize('config',[{'noise':True},{'priors':True},{'structured':True},{'signal_loss':1},
                                   {'priors':True,'noise':True,'structured':True,'signal_loss':1}])
def test_symmetry_and_safe_roundtrip(config,tmp_path):
    singles,pairs,doubles,resources=fixture()
    config={**config,'epochs':6,'rank':3,'width':8}
    model=AdvancedResidual(singles,resources,config).fit(pairs[:6],doubles[:6],np.full((6,32),.02),seed=41)
    prediction=model.predict(pairs[6:])
    np.testing.assert_allclose(prediction,model.predict(pairs[6:,::-1]),atol=1e-6)
    np.testing.assert_allclose(model.predict(pairs[6:],0),model.base.predict(pairs[6:]),atol=1e-10)
    restored=AdvancedResidual.from_state(model.state(),config)
    np.testing.assert_allclose(prediction,restored.predict(pairs[6:]),atol=1e-6)
    data={'singles':singles,'targets':np.array(list('abcdefg')),'genes':np.array([f'g{i}' for i in range(32)]),
          'control':np.zeros(32),'pairs':pairs[:6],'query_pairs':pairs[6:]}
    member={'type':'advanced','model':model,'config':config,'seed':41}
    export_members([member],data,tmp_path,{'context':'fixture','scale':'log2'})
    predictor=Predictor(tmp_path)
    output=predictor.predict('a','g')
    np.testing.assert_allclose(output['effect'],prediction[0],atol=1e-6)
    np.testing.assert_allclose(output['additive']+output['residual'],output['effect'],atol=1e-10)
    with pytest.raises(ValueError,match='Missing observed single'):
        predictor.predict('absent','g')
    with (tmp_path/'member_0.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):Predictor(tmp_path)


def test_heldout_responses_do_not_enter_fit():
    singles,pairs,doubles,resources=fixture()
    config={'epochs':4,'rank':3,'width':8,'priors':True,'structured':True}
    first=AdvancedResidual(singles,resources,config).fit(pairs[:6],doubles[:6],seed=3).predict(pairs[6:])
    doubles[6:]+=10000
    second=AdvancedResidual(singles,resources,config).fit(pairs[:6],doubles[:6],seed=3).predict(pairs[6:])
    np.testing.assert_array_equal(first,second)


def test_signal_weights_bound_and_reliability():
    y=np.array([[.1,2.,10.]])
    reliable=signal_weights(y,np.zeros_like(y),1)
    noisy=signal_weights(y,np.full_like(y,1e9),1)
    np.testing.assert_allclose(reliable.mean(1),1)
    assert reliable.max()/reliable.min()<=5
    np.testing.assert_allclose(noisy,1,atol=1e-5)
    np.testing.assert_array_equal(signal_weights(y,np.zeros_like(y),0),np.ones_like(y))


def test_exact_quadratic_matches_gene_space_objective():
    rng=np.random.default_rng(8)
    basis=rng.normal(size=(13,5));weights=rng.uniform(.2,2,(6,13));residual=rng.normal(size=(6,13));z=rng.normal(size=(6,5))
    q=np.einsum('gl,ng,gk->nlk',basis,weights,basis)/13
    cross=(weights*residual)@basis/13
    compressed=(np.einsum('nl,nlk,nk->n',z,q,z)-2*(z*cross).sum(1)).mean()
    direct=np.mean(weights*(z@basis.T-residual)**2)-np.mean(weights*residual**2)
    np.testing.assert_allclose(compressed,direct,atol=1e-12)


def test_missing_sources_receive_no_attention():
    torch.manual_seed(4)
    fusion=SourceFusion(3,5,7)
    bank=torch.randn(4,3,5);mask=torch.tensor([[True,False,True]]*4)
    first,weights=fusion(bank,mask,torch.ones(4))
    bank[:,1]+=1000
    second,_=fusion(bank,mask,torch.ones(4))
    torch.testing.assert_close(first,second)
    torch.testing.assert_close(weights[:,1],torch.zeros(4))


@pytest.mark.parametrize('pairs',[np.array([[0,0]]),np.array([[0,7]]),np.array([[0.,1.]]),np.array([0,1])])
def test_invalid_query_pairs_fail_closed(pairs):
    with pytest.raises(ValueError):check_pairs(pairs,7)
