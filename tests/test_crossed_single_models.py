import numpy as np
import pytest
from opal.crossed_single import fit,predict,grouped_folds,menu
from opal.context_contrast.artifacts import save_state,load_state
from opal.context_contrast.contracts import Unsupported

CONFIG=dict(penalties=[.1,1,10],representation_rank=16,rank_tolerance=1e-10,
            condition_limit=1e8,alignment_seed=20261015)


def test_all_families_finite_and_exact_portable_replay(tmp_path):
    rng=np.random.default_rng(24);x=rng.normal(size=(48,2,24));y=.6*x.mean(1)+.3
    q=rng.normal(size=(3,2,24))
    for spec in menu(CONFIG):
        state=fit(x,y,spec,CONFIG)
        path=tmp_path/('state'+str(len(list(tmp_path.glob('state*')))))
        save_state(path,state)
        expected=predict(state,q)
        np.testing.assert_array_equal(predict(load_state(path),q),expected)
        assert expected.shape==(3,24) and np.isfinite(expected).all()


def test_query_outcomes_never_enter_fitted_state():
    rng=np.random.default_rng(7);x=rng.normal(size=(60,2,24));y=rng.normal(size=(60,24))
    state=fit(x[:40],y[:40],dict(family='pca_concat',alpha=1),CONFIG)
    before=predict(state,x[40:]);y[40:]+=1000
    after=predict(fit(x[:40],y[:40],dict(family='pca_concat',alpha=1),CONFIG),x[40:])
    np.testing.assert_array_equal(before,after)


def test_fixed_rank_refusal_and_complete_global_identity_folds():
    with pytest.raises(Unsupported,match='rank-16'):
        fit(np.ones((40,2,24)),np.ones((40,24)),dict(family='pca_mean',alpha=1),CONFIG)
    a=grouped_folds(96,20261014)
    assert np.bincount(a).tolist()==[32,32,32]
    np.testing.assert_array_equal(a,grouped_folds(96,20261014))
    assert len(menu(CONFIG))==22
