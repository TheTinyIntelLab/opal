import pytest
from tools.prepare_emeraldbay import condition


def test_dose_and_unit_are_part_of_treatment_identity():
    a = condition("[('A', 1, 'uM'), ('B', 2, 'uM')]")
    assert a == condition("[('B', 2, 'uM'), ('A', 1.0, 'uM')]")
    assert a != condition("[('A', 0.1, 'uM'), ('B', 2, 'uM')]")
    singles = {condition("[('A', 1, 'uM')]")[0], condition("[('B', 0.2, 'uM')]")[0]}
    assert not all(s in singles for s in a)


@pytest.mark.parametrize('value', ["[('A', -1, 'uM')]", "[('A', 1, 'nM')]", "[('A', 1, 'uM'), ('A', 2, 'uM')]", "[]", "__import__('os').getcwd()"])
def test_invalid_source_treatments_are_rejected(value):
    with pytest.raises((ValueError, SyntaxError)):
        condition(value)


import pytest
from tools.prepare_dspin import treatment


def test_exact_dose_matching_and_control_activation():
    assert treatment('CONTROL_CD3_2') == ()
    assert treatment('Dasatinib_0p1_Halcinonide_0_CD3_1') == (('Dasatinib', .1),)
    assert treatment('Dasatinib_1000_Halcinonide_1000_CD3_2') == (('Dasatinib', 1000.), ('Halcinonide', 1000.))
    with pytest.raises(ValueError, match='Unmapped'):
        treatment('CONTROL_1')
    with pytest.raises(ValueError, match='Unmapped'):
        treatment('ambiguous_treatment')


import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from opal.artifacts import write_json, sha256
from opal.contextual.contracts import ReferenceBank
from tools.prepare_state_source import main


def test_count_preparation_matches_one_shared_baseline_and_balanced_plates(tmp_path, monkeypatch):
    records, counts = [], []
    for line in ['A549', 'K562', 'MCF7']:
        for plate in ['P1', 'P2']:
            for drug in ['control', 'A', 'B', 'C', 'D']:
                # Unequal treatment cell counts must not silently weight plates.
                cells = 2 if plate == 'P1' and drug == 'A' else 1
                for i in range(cells):
                    records.append({'cell_line': line, 'plate': plate, 'well': drug, 'perturbation': drug,
                                    'dose_value': 0 if drug == 'control' else 1000, 'dose_unit': 'nM', 'time': 24})
                    counts.append([10 if plate == 'P1' else 20, 5 if drug == 'control' else 15, 2, 1])
    raw = tmp_path / 'raw'; raw.mkdir()
    path = raw / 'SrivatsanTrapnell2020_sciplex3.h5ad'
    data = ad.AnnData(sparse.csr_matrix(np.asarray(counts)), obs=pd.DataFrame(records, index=[f'cell{i}' for i in range(len(records))]),
                     var=pd.DataFrame({'ensembl_id': ['ENSG1', 'ENSG2', 'ENSG3', 'ENSG4']}, index=['G1', 'G2', 'G3', 'G4']))
    data.write_h5ad(path)
    write_json(raw / 'verified.json', {'source_sha256': sha256(path), 'processing_sha256': 'synthetic_test_fixture'})
    output = tmp_path / 'prepared'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--raw', str(raw), '--output', str(output), '--genes', '4'])
    main()
    bank = ReferenceBank.load(output)
    def transform(values):
        values = np.asarray(values, float)
        return np.log2(1 + 10000 * values / values.sum())
    expected_control = transform([10, 5, 2, 1])
    expected_treated = transform([10, 15, 2, 1])
    np.testing.assert_allclose(bank.arrays['controls'][0], expected_control)
    np.testing.assert_allclose(bank.arrays['singles'][0, 0] + bank.arrays['controls'][0], expected_treated)
    assert bank.metadata['gene_panel_uses_query_controls']
    assert bank.metadata['context_records']['A549@P1']['selection_group'] == 'A549'
    assert len(bank.arrays['contexts']) == 6
