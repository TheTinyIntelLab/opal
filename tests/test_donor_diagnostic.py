"""Complete donor matching and honest same-pair transfer boundaries."""
import numpy as np
import pytest
import gzip

from opal.replicated.donor_diagnostic import (TREATMENTS, matched_donors,
                                            single_pair_donor_diagnostic, parse_brief_samples, read_series_matrix)


def fixture():
    rows, values = [], []
    for donor in range(3):
        for arm, effect in zip(('E', 'V', 'EL', 'VL'), (0, 1, 2, 4)):
            accession = f'GSM{donor}{arm}'
            rows.append({'accession': accession, 'title': f'AA_{donor}_{arm}',
                         'characteristics': {'treated with': TREATMENTS[arm], 'cell type': 'Primary monocytes',
                                             'subject status': 'healthy individual', 'ancestry': 'African-American'}})
            values.append([-3 + donor + effect, 2 + effect])
    return rows, [r['accession'] for r in rows], ['p1', 'p2'], np.asarray(values, float)


def test_processed_scale_and_complete_donor_exclusion():
    rows, columns, probes, values = fixture()
    report = single_pair_donor_diagnostic(rows, columns, probes, values)
    assert report['equal_donor_mean_mse'] == {'zero': 16., 'additive': 1., 'same_pair_other_donors': 0.}
    assert not report['same_pair_transfer_eligible_for_original_held_pair_protocol']
    values[3] += 9  # Query donor's observed pair cannot alter its training correction.
    changed = single_pair_donor_diagnostic(rows, columns, probes, values)
    assert changed['donor_loss_values']['same_pair_other_donors'][0] == 81.
    assert changed['donor_loss_values']['same_pair_other_donors'][1] == 20.25


def test_reference_mismatch_missing_arms_and_nonfinite_reject():
    rows, columns, probes, values = fixture()
    with pytest.raises(ValueError, match='Complete'):
        single_pair_donor_diagnostic(rows[:-1], columns[:-1], probes, values[:-1])
    rows[1]['characteristics']['treated with'] = 'lipopolysaccharide'
    with pytest.raises(ValueError, match='disagree'):
        matched_donors(rows, columns)
    rows, columns, probes, values = fixture()
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match='Finite'):
        single_pair_donor_diagnostic(rows, columns, probes, values)


def test_metadata_order_and_matrix_columns_do_not_change_donor_weight():
    rows, columns, probes, values = fixture()
    original = single_pair_donor_diagnostic(rows, columns, probes, values)
    order = np.arange(len(columns))[::-1]
    assert original == single_pair_donor_diagnostic(rows[::-1], [columns[i] for i in order], probes, values[order])
    with pytest.raises(ValueError, match='expression tables'):
        parse_brief_samples('!sample_table_begin\nID_REF\tVALUE')


def test_series_matrix_requires_terminator_and_preserves_probe_ids(tmp_path):
    path = tmp_path / 'matrix.txt.gz'
    text = '!series_matrix_table_begin\n"ID_REF"\t"GSM1"\t"GSM2"\n"p1"\t-2\t3\n'
    path.write_bytes(gzip.compress(text.encode()))
    with pytest.raises(ValueError, match='terminator'):
        read_series_matrix(path)
    path.write_bytes(gzip.compress((text + '!series_matrix_table_end\n').encode()))
    columns, probes, values = read_series_matrix(path)
    assert columns == ['GSM1', 'GSM2'] and probes == ['p1']
    np.testing.assert_array_equal(values, [[-2], [3]])
