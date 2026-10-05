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
