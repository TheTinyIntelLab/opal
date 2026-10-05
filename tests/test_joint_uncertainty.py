import numpy as np
import pytest
from opal.models.covariance_kernel import noise_covariance,covariance_smoother,CovarianceKernel
from opal.models.noise_kernel import measurement_smoother
from opal.inference import Predictor,export_members
from test_advanced_architecture import fixture
from tools.improve_residual_model import candidates


def test_diagonal_equivalence_and_shared_covariance_psd():
    singles,pairs,doubles,r=fixture();config,kernel=candidates(singles,pairs)[1]
    covariance=np.diag(np.arange(8)+1.)+.5*np.ones((8,8))
    train=np.arange(6);query=np.array([6,7]);block=covariance[np.ix_(train,train)]
    actual=covariance_smoother(kernel,train,query,config['alpha'],noise_covariance(block,.5,0))
    np.testing.assert_allclose(actual,measurement_smoother(kernel,train,query,config['alpha'],np.diag(block),.5),atol=1e-9)
    correlated=noise_covariance(block,1,1)
    assert np.linalg.eigvalsh(correlated).min()>0
    np.testing.assert_allclose(covariance_smoother(kernel,train,query,1,correlated).sum(1),1,atol=1e-9)
    covariance[query,query]+=1e9
    np.testing.assert_allclose(noise_covariance(covariance[np.ix_(train,train)],.5,0),noise_covariance(block,.5,0))


def test_covariance_checkpoint_and_pair_symmetry(tmp_path):
    singles,pairs,doubles,r=fixture();config={**candidates(singles,pairs)[1][0],'noise_strength':1,'correlation':.5}
    covariance=np.eye(6)+.2*np.ones((6,6))
    model=CovarianceKernel(singles,config).fit(pairs[:6],doubles[:6],covariance)
    np.testing.assert_allclose(model.predict(pairs[6:]),model.predict(pairs[6:,::-1]))
    data={'singles':singles,'targets':np.array(list('abcdefg')),'genes':np.array([f'g{i}' for i in range(32)]),'control':np.zeros(32),'pairs':pairs[:6],'query_pairs':pairs[6:]}
    export_members([{'type':'covariance_kernel','model':model,'config':config,'seed':0}],data,tmp_path,{'context':'fixture'})
    np.testing.assert_allclose(Predictor(tmp_path).predict('a','g')['effect'],model.predict(pairs[6:])[0])
    state=model.state();state['training_noise']=np.ones((2,2))
    with pytest.raises(ValueError,match='schema'):CovarianceKernel(singles,config).restore(state)


@pytest.mark.parametrize('matrix',[np.array([[1,2],[2,1]]),np.array([[1,1],[0,1]]),np.full((2,2),np.nan)])
def test_invalid_covariance_rejected(matrix):
    with pytest.raises(ValueError):noise_covariance(matrix)


def test_finite_sample_rank_and_unattainable_radius():
    from opal.reliability import error_radius
    assert error_radius(np.arange(9.),.9)['radius']==8
    assert error_radius(np.arange(7.),.9)['radius'] is None
    assert error_radius(np.arange(7.),.8)['radius']==6
    with pytest.raises(ValueError):error_radius([-1,1])


def test_calibration_labels_do_not_select_or_fit_the_risk_model():
    from tools.study_retuned_curves import risk_diagnostic
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(67)
    singles=rng.normal(size=(12,20))
    pairs=np.array([(i,j) for i in range(12) for j in range(i+1,12)])
    doubles=singles[pairs[:,0]]+singles[pairs[:,1]]+.1*rng.normal(size=(len(pairs),20))
    data={'singles':singles,'pairs':pairs,'doubles':doubles}
    covariance=np.eye(len(pairs));base=candidates(singles,pairs)
    train=np.arange(50);val=np.arange(50,len(pairs))
    with threadpool_limits(limits=1):
        first=risk_diagnostic(data,covariance,base,train,val,'random',42)
        assert first['status']=='complete'
        doubles[first['calibration']]+=100
        second=risk_diagnostic(data,covariance,base,train,val,'random',42)
    assert first['selected_config']==second['selected_config']
    np.testing.assert_array_equal(first['outer_rmse'],second['outer_rmse'])
    assert second['radii'][0]['radius']>first['radii'][0]['radius']


def test_singular_measurement_covariance_can_roundtrip_with_positive_kernel():
    singles,pairs,doubles,_=fixture()
    config={'kind':'pair_product','bandwidth':1,'alpha':1,'noise_strength':1,'correlation':1}
    model=CovarianceKernel(singles,config).fit(pairs[:6],doubles[:6],np.ones((6,6)))
    restored=CovarianceKernel(singles,config).restore(model.state())
    np.testing.assert_allclose(restored.predict(pairs[6:]),model.predict(pairs[6:]))
    with pytest.raises(ValueError,match='semidefinite'):noise_covariance(np.array([[0.,1e-9],[1e-9,0.]]))
