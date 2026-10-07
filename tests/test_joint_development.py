"""Consequential joint holdout invariants, including multiple dose rows."""
import numpy as np
from tools.run_joint_development import joint_folds
from opal.state.contracts import validate_isolation


def test_joint_folds_exclude_all_doses_and_cover_each_query_once():
    pairs = np.array([['a', 'b'], ['a', 'c'], ['b', 'd'], ['c', 'd'], ['a', 'e'], ['b', 'e']])
    canonical = np.tile(np.repeat(pairs, 2, axis=0), (5, 1))
    groups = np.repeat(['s1', 's2', 's3', 's4', 's5'], 12)
    folds = joint_folds(canonical, groups)
    assert sorted(i for f in folds for i in f['query']) == list(range(60))
    for fold in folds:
        train, query = np.array(fold['train']), np.array(fold['query'])
        validate_isolation(train, query, groups, pairs=canonical, held_pair=True)
        assert not set(map(tuple, canonical[train])) & set(map(tuple, canonical[query]))
        assert set(train) | set(query) | set(fold['guard']) == set(range(60))
        for inner in fold['inner']:
            validate_isolation(np.array(inner['train']), np.array(inner['query']), groups[train], pairs=canonical[train], held_pair=True)


def test_sparse_joint_metadata_fails_without_outcome_redraw():
    import pytest
    pairs = np.array([['a', 'b'], ['a', 'c'], ['a', 'b'], ['a', 'c']])
    with pytest.raises(ValueError):
        joint_folds(pairs, np.array(['s1', 's1', 's2', 's2']))
