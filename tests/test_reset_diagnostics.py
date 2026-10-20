import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('reset_diagnostics', Path(__file__).parents[1]/'tools/reset_diagnostics.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    names = np.array([['A', 'B'], ['A', 'C'], ['B', 'C']])
    d = dict(pair_names=names, single_genes=np.array(['A', 'B', 'C']),
             singles=np.array([[-1., -2.], [-3., -4.], [-5., -6.]]),
             gamma=np.array([[-3., -4.], [-5., -6.], [-5., -6.]]), counts=np.array([[2, 2], [4, 4], [8, 8]]))
    f = dict(pair_names=names.copy(), rows=np.arange(3), strata=np.array([m.NOVEL]*3))
    f.update({k: np.zeros(3) for k in ('completion', 'main_effect', 'minimum', 'additive', 'zero')})
    return d, f


def test_panel_singles_counts_and_no_missing_imputation():
    d, f = fixture()
    p = m.panel(d, f)
    np.testing.assert_array_equal(p['minimum'], d['gamma'])
    assert p['strict_novel_total'] == 3
    d['single_genes'] = d['single_genes'][:2]
    d['singles'] = d['singles'][:2]
    p = m.panel(d, f)
    assert p['missing_single_pairs'] == 2
    assert p['names'].tolist() == [['A', 'B']]


def test_wrong_rows_and_duplicates_rejected():
    d, f = fixture()
    f['rows'] = np.array([1, 0, 2])
    with pytest.raises(ValueError, match='misaligned'):
        m.panel(d, f)
    f['rows'] = np.array([0, 0, 2])
    with pytest.raises(ValueError, match='row assignment'):
        m.panel(d, f)
    with pytest.raises(ValueError, match='canonical'):
        m.names_valid(np.array([['A', 'B'], ['B', 'A']]))


def test_nonfinite_and_baseline_correctness():
    d, f = fixture()
    d['gamma'][0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        m.panel(d, f)
    assert m.metrics(np.array([2., 3.]), np.array([1., 2.]))['mse'] == 1
    with pytest.raises(ValueError, match='Aligned'):
        m.metrics(np.zeros(2), np.zeros(3))
    assert m.correlation(np.ones(3), np.arange(3)) is None


def test_cross_infection_uses_opposite_target():
    s = np.array([[0., 10.], [1., 11.]])
    result = m.cross_infection(s, s.copy())
    assert result['infection1_singles_to_infection2']['mse'] == 100
    assert result['infection2_singles_to_infection1']['mse'] == 100
    assert result['residual_pearson'] is None


def test_conditional_constant_advantage_deterministic():
    d, _ = fixture()
    a = m.conditional_interval(d['pair_names'], -np.ones(3), draws=100)
    assert a['gene_conditional_95'] == [-1., -1.]
    assert a == m.conditional_interval(d['pair_names'], -np.ones(3), draws=100)


def test_integrity_drift_rejected(tmp_path):
    p = tmp_path/'file'
    p.write_bytes(b'original')
    hashes = {str(p): m.sha256(p)}
    m.verify(hashes)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Frozen input differs'):
        m.verify(hashes)


def test_synthetic_end_to_end_diagnostic_and_panel_thresholds():
    d, f = fixture()
    p = m.panel(d, f)
    result = m.analyze({c: p for c in m.COHORTS})
    score = result['cohorts']['environment1']
    assert [score['coverage'][str(t)]['pairs'] for t in (2, 4, 8)] == [3, 2, 1]
    assert score['coverage']['2']['methods']['local_minimum']['mse'] == 0
    assert len(result['matched_strict_novel_transfers']) == 6
    assert result == m.analyze({c: p for c in m.COHORTS})
