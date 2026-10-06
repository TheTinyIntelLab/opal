import numpy as np
from opal.models.kernel import single_kernel, symmetric_kernel, global_kernel, smoother, prediction_error
from opal.models.kernel import KernelResidual
from opal.training import fit
from opal.inference import export_members, Predictor
import pytest


def test_symmetric_pair_kernel_is_psd_and_order_invariant():
    rng = np.random.default_rng(77)
    singles = rng.normal(size=(6, 12))
    pairs = np.array([(0, 1), (2, 4), (1, 5), (0, 4)])
    single = single_kernel(singles)
    for interaction in [False, True]:
        kernel = symmetric_kernel(single, pairs, pairs, interaction)
        swapped = symmetric_kernel(single, pairs[:, ::-1], pairs, interaction)
        np.testing.assert_allclose(kernel, swapped, atol=1e-12)
        assert np.linalg.eigvalsh(kernel).min() > -1e-12


def test_gram_error_matches_gene_space_and_intercept():
    rng = np.random.default_rng(4)
    singles = rng.normal(size=(6, 12))
    pairs = np.array([(0, 1), (2, 4), (1, 5), (0, 4), (2, 5)])
    kernel = global_kernel(singles, pairs)
    train, query = np.array([0, 1, 2]), np.array([3, 4])
    weights = smoother(kernel, train, query, 0.1)
    np.testing.assert_allclose(weights.sum(1), 1)
    residual = rng.normal(size=(5, 12))
    direct = np.mean((weights @ residual[train] - residual[query]) ** 2)
    efficient = prediction_error(weights, train, query, residual @ residual.T / 12)
    np.testing.assert_allclose(direct, efficient)
    shifted = residual + rng.normal(size=12)
    shifted_mse = np.mean((weights @ shifted[train] - shifted[query]) ** 2)
    np.testing.assert_allclose(direct, shifted_mse)


@pytest.mark.parametrize('kind', ['shared_offset', 'global', 'single_sum', 'pair_product', 'global_plus_pair', 'global_plus_pair_10'])
def test_kernel_fit_matches_smoother_and_safe_export(tmp_path, kind):
    rng = np.random.default_rng(12)
    singles = rng.normal(size=(6, 13))
    pairs = np.array([(0, 1), (2, 4), (1, 5), (0, 4), (2, 5)])
    doubles = rng.normal(size=(5, 13))
    config = {'family': 'kernel', 'kind': kind, 'alpha': 0.1, 'bandwidth': 1}
    model = KernelResidual(singles, config).fit(pairs[:3], doubles[:3])
    residual = doubles - singles[pairs[:, 0]] - singles[pairs[:, 1]]
    weights = smoother(model.kernel(pairs, pairs), np.arange(3), np.arange(3, 5), .1)
    expected = singles[pairs[3:, 0]] + singles[pairs[3:, 1]] + weights @ residual[:3]
    np.testing.assert_allclose(model.predict(pairs[3:]), expected, atol=1e-11)
    np.testing.assert_allclose(model.predict(pairs[3:, ::-1]), expected, atol=1e-11)
    targets = np.array(list('abcdef'))
    data = {'singles': singles, 'targets': targets, 'genes': np.array([f'g{i}' for i in range(13)]),
            'control': np.zeros(13), 'pairs': pairs[:3], 'query_pairs': pairs[3:]}
    member = fit(config, singles, pairs[:3], doubles[:3])
    export_members([member], data, tmp_path, {'context': 'assay', 'scale': 'log2'})
    predictor = Predictor(tmp_path)
    np.testing.assert_allclose(predictor.predict('a', 'e')['effect'], expected[0], atol=1e-11)
    with pytest.raises(ValueError, match='Unsupported context'):
        predictor.predict('a', 'e', context='other-assay')
    with (tmp_path / 'member_0.npz').open('ab') as handle:
        handle.write(b'tampering')
    with pytest.raises(ValueError, match='checksum'):
        Predictor(tmp_path)


import numpy as np
import pytest
from opal.models.noise_kernel import NoiseAwareKernel,measurement_smoother,noise_diagonal
from opal.models.kernel import KernelResidual,smoother
from opal.inference import export_members,Predictor
from test_advanced_architecture import fixture


def test_zero_noise_strength_matches_centered_kernel():
    singles,pairs,doubles,r=fixture()
    config={'alpha':1,'kind':'linear','noise_strength':0}
    # Reuse a registered kernel configuration rather than assuming a feature implementation.
    from tools.improve_residual_model import candidates
    config,kernel=candidates(singles,pairs)[1]
    train=np.arange(6);query=np.array([6,7])
    actual=measurement_smoother(kernel,train,query,config['alpha'],np.arange(6)+1,0)
    np.testing.assert_allclose(actual,smoother(kernel,train,query,config['alpha']),atol=1e-9)
    np.testing.assert_allclose(actual.sum(1),1,atol=1e-9)


def test_noise_kernel_export_symmetry_and_training_only(tmp_path):
    singles,pairs,doubles,r=fixture()
    from tools.improve_residual_model import candidates
    config={**candidates(singles,pairs)[1][0],'noise_strength':1}
    model=NoiseAwareKernel(singles,config).fit(pairs[:6],doubles[:6],np.arange(6)+1)
    expected=model.predict(pairs[6:])
    doubles[6:]+=999
    np.testing.assert_allclose(expected,model.predict(pairs[6:,::-1]))
    data={'singles':singles,'targets':np.array(list('abcdefg')),'genes':np.array([f'g{i}' for i in range(32)]),
          'control':np.zeros(32),'pairs':pairs[:6],'query_pairs':pairs[6:]}
    export_members([{'type':'noise_kernel','model':model,'config':config,'seed':0}],data,tmp_path,{'context':'fixture'})
    np.testing.assert_allclose(Predictor(tmp_path).predict('a','g')['effect'],expected[0])
    state=model.state();state['training_noise']=np.ones(2)
    with pytest.raises(ValueError,match='schema'):NoiseAwareKernel(singles,config).restore(state)


@pytest.mark.parametrize('variance,strength',[([-1,1],1),([np.nan,1],1),([1,2],-1),([1,2],np.nan)])
def test_invalid_measurement_noise_rejected(variance,strength):
    with pytest.raises(ValueError):noise_diagonal(variance,strength)


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
