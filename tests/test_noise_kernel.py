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
