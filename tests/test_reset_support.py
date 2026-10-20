import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('reset_support', Path(__file__).parents[1]/'tools/reset_support.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_degree_groups_use_train_only_and_partition_metadata():
    pairs = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]])
    split = np.array([0, 0, 0, 0, 2, 2])
    groups, metadata = module.support_groups(pairs, split, np.array(['A', 'B', 'C', 'D']), {'A', 'D'})
    # Train degrees A=3,B=2,C=2,D=1, not degrees in the complete query graph.
    assert metadata['cutpoints'] == [2., 2.]
    assert metadata['active_gene_tertile_counts'] == [1, 0, 3]
    assert sum(int(groups[f'novel_gene_endpoints_{k}'].sum()) for k in range(3)) == 2
    assert groups['novel_gene_endpoints_1'].tolist() == [True, True]
    assert groups['endpoint_train_degree_tertiles_0_2'].tolist() == [True, True]
    assert groups['endpoint_train_degree_tertiles_1_1'].sum() == 0


def test_support_requires_unique_canonical_supported_pairs():
    genes = np.array(['A', 'B', 'C'])
    with pytest.raises(ValueError, match='canonical'):
        module.support_groups(np.array([[1, 0], [0, 2]]), [0, 2], genes, set())
    with pytest.raises(ValueError, match='canonical'):
        module.support_groups(np.array([[0, 1], [0, 1]]), [0, 2], genes, set())
    with pytest.raises(ValueError, match='support'):
        module.support_groups(np.array([[0, 1], [0, 2]]), [0, 2], genes, set())


def test_empty_subgroup_is_reported_without_nan_or_inference():
    score = module.subgroup_score(np.array([['A', 'B']]), np.array([1.]), np.array([2.]),
                                 np.array([3.]), np.array([False]))
    assert score['pairs'] == score['genes'] == 0
    assert score['completion_mse'] is None
    assert score['relative_gain'] is None
    assert not score['qualifies_descriptive_interval']


def test_baseline_metrics_and_conditional_bootstrap_are_paired():
    names = np.array([['A', 'B'], ['A', 'C'], ['B', 'C']])
    target = np.array([0., 1., 2.])
    score = module.subgroup_score(names, target, target + 1, target + 2, np.ones(3, dtype=bool))
    assert score['completion_mse'] == 1.
    assert score['main_effect_mse'] == 4.
    assert score['relative_gain'] == .75
    first = module.paired_gene_interval(names, np.full(3, -3.), draws=50)
    assert first == module.paired_gene_interval(names, np.full(3, -3.), draws=50)
    assert first['conditional_gene_loss_difference_95'] == [-3., -3.]
    assert 0 < first['nonempty_draws'] <= 50
