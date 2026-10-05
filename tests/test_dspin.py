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
