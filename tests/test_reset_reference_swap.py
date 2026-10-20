import importlib.util
from pathlib import Path

import numpy as np
import pytest

from opal.artifacts import write_json

spec = importlib.util.spec_from_file_location('reset_reference_swap', Path(__file__).resolve().parents[1]/'tools/reset_reference_swap.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    names = np.array([['A', 'B'], ['A', 'C'], ['B', 'C']])
    base = np.array([[0., 1.], [1., 2.], [2., 3.]])
    correction = np.array([.1, -.2, .3])
    return dict(names=names, minimum=base, additive=base*2,
        original_main_effect=np.ones(3), original_completion=np.ones(3)+correction,
        target=base+correction[:, None], strict_novel_total=3, missing_single_pairs=0)


def test_frozen_correction_subtraction_and_mean_target_are_exact():
    p = fixture()
    result = m.analyze({k: p for k in m.COHORTS})
    primary = result['cohorts']['environment1']['primary']['minimum']
    assert primary['reference_plus_old_correction']['mse'] < 1e-30
    assert primary['relative_mse_gain'] == pytest.approx(1.)
    assert primary['reference']['mse'] == pytest.approx(np.mean([.01, .04, .09]))
    assert result['correction_weight'] == 1
    assert primary['old_correction_local_residual_centered_pearson'] == pytest.approx(1.)


def test_cross_infection_uses_opposite_target_not_same_infection():
    p = fixture()
    result = m.analyze({k: p for k in m.COHORTS})
    cross = result['cohorts']['environment1']['cross_infection']['minimum']
    assert cross['infection1_singles_to_infection2']['reference_plus_old_correction']['mse'] == pytest.approx(1.)
    assert cross['infection2_singles_to_infection1']['reference_plus_old_correction']['mse'] == pytest.approx(1.)


def test_zero_correction_is_identical_to_reference_and_constant_correlation_none():
    p = fixture()
    result = m.compare(p['names'], np.arange(3.), np.zeros(3), np.arange(3.)+1)
    assert result['reference'] == result['reference_plus_old_correction']
    assert result['paired_difference']['gene_conditional_95'] == [0., 0.]
    assert result['old_correction_local_residual_centered_pearson'] is None


def test_alignment_finite_values_and_gene_dependence_prerequisites():
    p = fixture()
    with pytest.raises(ValueError, match='Aligned'):
        m.compare(p['names'], np.zeros(3), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match='Finite'):
        m.compare(p['names'], np.zeros(3), np.full(3, np.nan), np.zeros(3))
    names = p['names'].copy()
    names[1] = names[0]
    with pytest.raises(ValueError, match='canonical'):
        m.compare(names, np.zeros(3), np.ones(3), np.zeros(3))


def test_aggregate_replay_is_deterministic_without_fitting():
    p = fixture()
    a = m.analyze({k: p for k in m.COHORTS})
    assert a == m.analyze({k: p for k in m.COHORTS})
    assert 'no fitting' in a['label']
    assert not hasattr(m, 'GrowthReference')
    assert not hasattr(m, 'FitnessCompletion')


def test_failed_prerequisite_blocks_run_and_freeze_refuses_existing_output(tmp_path):
    write_json(tmp_path/'protocol.json', dict(files={}, code=m.signature()))
    write_json(tmp_path/'prefit_tests.json', dict(code=m.signature(), returncode=1))
    with pytest.raises(ValueError, match='passing'):
        m.execute(tmp_path)
    with pytest.raises(FileExistsError):
        m.freeze(tmp_path, Path('missing'))
