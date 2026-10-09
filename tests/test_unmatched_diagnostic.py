"""Unmatched-arm references and complete query-pair exclusion."""
import numpy as np
import pytest

from opal.replicated.unmatched_diagnostic import lncap_array_design, unmatched_replicate_diagnostic


def fixture():
    rows, values = [], []
    for components, profiles in [((), [0, 0]), (('a',), [1, 1]), (('b',), [2, 2]),
                                 (('c',), [3, 3]), (('a', 'b'), [4, 6]),
                                 (('a', 'c'), [5, 7]), (('b', 'c'), [6, 8])]:
        for value in profiles:
            accession = 'GSM' + str(len(rows))
            rows.append({'accession': accession, 'components': components})
            values.append([value, value])
    return rows, [r['accession'] for r in rows], ['p1', 'p2'], np.asarray(values, float)


def test_arm_means_and_query_pair_exclusion_have_hand_computed_losses():
    rows, columns, probes, values = fixture()
    report = unmatched_replicate_diagnostic(rows, columns, probes, values)
    assert report['equal_pair_mean_mse']['zero'] == pytest.approx(113 / 3)
    assert report['equal_pair_mean_mse']['additive'] == 5
    assert report['equal_pair_mean_mse']['same_pair_other_samples'] == 4
    assert report['equal_pair_mean_mse']['other_pair_mean_offset'] == 1
    values[8] += 9
    changed = unmatched_replicate_diagnostic(rows, columns, probes, values)
    pair = changed['pair_diagnostics'][0]
    assert pair['sample_loss_values']['same_pair_other_samples'][0] == 49
    assert pair['sample_loss_values']['other_pair_mean_offset'] == [64, 1]
    assert not changed['independent_culture_blocks_inferred']
    assert not changed['same_pair_transfer_eligible_for_original_held_pair_protocol']


def test_order_invariance_and_missing_or_nonfinite_references_reject():
    rows, columns, probes, values = fixture()
    assert unmatched_replicate_diagnostic(rows, columns, probes, values) == unmatched_replicate_diagnostic(
        rows[::-1], columns[::-1], probes, values[::-1])
    with pytest.raises(ValueError, match='both singles'):
        unmatched_replicate_diagnostic(rows[:2] + rows[4:], columns[:2] + columns[4:],
                                       probes, np.concatenate([values[:2], values[4:]]))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match='Finite'):
        unmatched_replicate_diagnostic(rows, columns, probes, values)


def test_lncap_metadata_requires_all_42_unambiguous_profiles():
    treatments = ['vehicle control', 'bicalutamide', 'SAHA', '17-AAG',
                  '17-AAG+bicalutamide', '17-AAG+SAHA', 'bicalutamide+SAHA']
    text = ''.join(f'^SAMPLE = GSM{i * 6 + repeat}\n!Sample_title = LNCaP_{arm}_biological replicate {repeat}\n'
                   '!Sample_characteristics_ch1 = cell line: LNCaP\n'
                   for i, arm in enumerate(treatments) for repeat in range(1, 7))
    assert len(lncap_array_design(text)) == 42
    with pytest.raises(ValueError, match='Duplicate treatment'):
        lncap_array_design(text.replace('LNCaP_SAHA_biological replicate 6', 'LNCaP_SAHA_biological replicate 5'))
    with pytest.raises(ValueError, match='Unexpected'):
        lncap_array_design(text.replace('cell line: LNCaP', 'cell line: other', 1))
