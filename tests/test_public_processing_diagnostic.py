"""Source joins and non-advancement checks for exploratory public readouts."""
import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from opal.replicated.public_diagnostic import processing_diagnostic


def fixture():
    rows, counts = [], []
    for run in ('r1', 'r2', 'r3'):
        for a, b, value in [('DMSO', 'DMSO', [10, 10]), ('A', 'DMSO', [20, 10]),
                            ('DMSO', 'B', [10, 20]), ('A', 'B', [30, 10]),
                            ('B', 'A', [10, 30])]:
            rows.append({'': f'{run}_{a}_{b}', 'Run': run, 'Drug1': a, 'Drug2': b})
            counts.append(value)
    return rows, np.asarray(counts)


def test_technical_variants_collapse_and_never_advance():
    rows, counts = fixture()
    report = processing_diagnostic(rows, counts, ['g1', 'g2'])
    assert report['complete_nominal_pair_count'] == 1
    assert not report['primary_claim_pass']
    assert not report['biological_measurement_gate_pass']
    assert not report['confidence_intervals_computed']
    assert len(report['reference_matches']['r1']['["A","B"]']['pair_rows']) == 2
    control = np.log2(1 + np.asarray([5000, 5000]))
    observed = (np.log2(1 + np.asarray([[7500, 2500], [2500, 7500]])) - control).mean(0)
    assert report['per_run_losses'][0]['zero_effect_mse'] == pytest.approx(np.mean(observed ** 2))
    assert report['per_run_losses'][0]['forbidden_repeated_pair_other_run_mse'] == pytest.approx(0)


def test_unusable_reference_cannot_silently_enter_panel():
    rows, counts = fixture()
    counts[0] = 0
    with pytest.raises(ValueError, match='No complete nominal'):
        processing_diagnostic(rows, counts, ['g1', 'g2'])


def test_corrected_values_and_duplicate_identities_rejected():
    rows, counts = fixture()
    with pytest.raises(ValueError, match='integer aggregate'):
        processing_diagnostic(rows, counts + .5, ['g1', 'g2'])
    with pytest.raises(ValueError, match='genes'):
        processing_diagnostic(rows, counts, ['g', 'g'])
    rows[0][''] = rows[1]['']
    with pytest.raises(ValueError, match='source treatment'):
        processing_diagnostic(rows, counts, ['g1', 'g2'])


def test_matrix_join_uses_source_ids_and_rejects_missing_columns(tmp_path):
    spec = importlib.util.spec_from_file_location('public_combi_runner', Path(__file__).parents[1] / 'tools/diagnose_public_combi.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'matrix.csv.gz'
    with gzip.open(path, 'wt') as f:
        f.write(',b,a\ng1,2,1\ng2,4,3\n')
    counts, genes = module.load_matrix(path, ['a', 'b'])
    np.testing.assert_array_equal(counts, [[1, 3], [2, 4]])
    assert genes == ['g1', 'g2']
    with pytest.raises(ValueError, match='match metadata'):
        module.load_matrix(path, ['a', 'c'])
    with pytest.raises(ValueError, match='byte budget'):
        module.load_matrix(path, ['a', 'b'], maximum_uncompressed_bytes=1)
