"""Processed scale and same-stratum reference enforcement."""
import itertools

import numpy as np
import pytest

from opal.replicated.bulk_diagnostic import processed_bulk_diagnostic


def fixture():
    rows, columns, values = [], [], []
    doses = {'T': 20, 'M': 10, 'W': 5}
    names = {'T': 'Tamoxifen', 'M': 'Mefloquine', 'W': 'Withaferin'}
    for hour, repeat, kind in itertools.product((0, 3, 6, 9, 12, 24), 'ABC',
                                               ('DMSO', 'T', 'M', 'W', 'TM', 'TW', 'MW', 'TMW')):
        name = f'{kind}_{hour}{repeat}'
        columns.append(name)
        row = {'title': 'LNCaP ' + name, 'accession': name, 'cell line': 'LNCaP cells',
               'hours of treatment': str(hour), 'replicate': repeat,
               'treatment_protocol': 'in 0.3% DMSO for the viability time courses',
               'treatment': '0.3% DMSO' if kind == 'DMSO' else ' and '.join(names[c] for c in kind)}
        if kind != 'DMSO':
            row.update({'solvent': 'DMSO', 'treatment concentration (um)': ' and '.join(str(doses[c]) for c in kind)})
        rows.append(row)
        values.append(np.asarray([-2., 3.]) + (0 if kind == 'DMSO' else len(kind)))
    return rows, columns, np.asarray(values)


def test_processed_negative_values_and_time_strata_are_preserved():
    rows, columns, values = fixture()
    report = processed_bulk_diagnostic(rows, columns, values, ['g1', 'g2'])
    assert len(report['strata']) == 6
    assert report['analyzed_samples'] == 126
    assert not report['biological_measurement_gate_pass']
    assert not report['confidence_intervals_computed']
    for stratum in report['strata']:
        assert stratum['equal_repeat_mean_mse']['additive_mse'] == pytest.approx(0)


def test_dose_vehicle_and_reference_mismatches_reject():
    rows, columns, values = fixture()
    rows[1]['treatment concentration (um)'] = '21'
    with pytest.raises(ValueError, match='dose'):
        processed_bulk_diagnostic(rows, columns, values, ['g1', 'g2'])
    rows[1]['treatment concentration (um)'] = '20'
    rows[0]['treatment'] = '0.4% DMSO'
    with pytest.raises(ValueError, match='vehicle'):
        processed_bulk_diagnostic(rows, columns, values, ['g1', 'g2'])
    rows[0]['treatment'] = '0.3% DMSO'
    with pytest.raises(ValueError, match='match source'):
        processed_bulk_diagnostic(rows, columns[:-1], values[:-1], ['g1', 'g2'])


def test_column_reordering_and_no_biological_pooling():
    rows, columns, values = fixture()
    first = processed_bulk_diagnostic(rows, columns, values, ['g1', 'g2'])
    order = np.arange(len(columns))[::-1]
    second = processed_bulk_diagnostic(rows, [columns[i] for i in order], values[order], ['g1', 'g2'])
    assert [s['equal_repeat_mean_mse'] for s in first['strata']] == [s['equal_repeat_mean_mse'] for s in second['strata']]
    assert not second['independent_culture_initiation_verified']
    assert not second['primary_claim_pass']
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match='Finite log-CPM'):
        processed_bulk_diagnostic(rows, columns, values, ['g1', 'g2'])
