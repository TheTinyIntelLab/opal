"""Reporting panels and held-target labels must obey source isolation."""
import numpy as np
import pytest
from tools.run_panel_scale import panels, partner_contract
from opal.splits import partner_splits


def test_reporting_panels_ignore_query_labels_and_resolve_ties():
    rng=np.random.default_rng(22)
    data={k:rng.normal(size=(10,20)) for k in ['y','a','b']}
    before=panels(data,np.arange(7))
    data['y'][7:]=np.nan
    after=panels(data,np.arange(7))
    for key in before:np.testing.assert_array_equal(before[key],after[key])
    tied={k:np.zeros((3,20)) for k in ['y','a','b']}
    np.testing.assert_array_equal(panels(tied,[0,1])['strong'],[0,1])
    assert len(before['strong'])==len(before['residual_proxy'])==2


def test_partner_categories_use_declared_targets_and_preserve_overlap():
    pairs=np.array([[i,j] for i in range(7) for j in range(i+1,7)])
    splits=partner_splits(pairs.astype(str),20261007,3,minimum_development=3,minimum_test=1)
    folds=[dict(fold=i,train=f['development'],query=f['test']) for i,f in enumerate(splits) if f['feasible']]
    contract=partner_contract(pairs,folds,20261007)
    assert contract['occurrences']>contract['unique_conditions']==len(pairs)
    for i,fold in enumerate(contract['folds']):
        assert contract['intersections'][i][i]==len(fold['query'])
        assert set(fold['held_count'])=={1,2}
    folds[0]['train']=folds[0]['train']+[folds[0]['query'][0]]
    with pytest.raises(ValueError):partner_contract(pairs,folds,20261007)
