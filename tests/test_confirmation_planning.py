import pytest
from opal.confirmation_planning import paired_power, minimum_blocks, sensitivity


def test_sample_size_is_minimal_under_explicit_planning_model():
    n = minimum_blocks(.5, .9)
    assert paired_power(n, .5, .0125) >= .9
    assert paired_power(n-1, .5, .0125) < .9
    assert minimum_blocks(.3, .9) > n
    assert minimum_blocks(.5, .8) < n
    assert minimum_blocks(.1, .9, maximum=3) is None
    with pytest.raises(ValueError):
        minimum_blocks(0)
    with pytest.raises(ValueError):
        paired_power(1, .5, .0125)


def test_sensitivity_never_becomes_empirical_power_or_chosen_design():
    r = sensitivity()
    assert len(r['rows']) == 10
    assert r['sample_size_recommended'] is False and r['empirical_power_reviewed'] is False
    assert r['family_one_sided_alpha_upper'] == 2*r['alpha_per_context']
    for row in r['rows']:
        assert row['two_context_assay_outcomes'] == 60*row['independent_blocks_per_context']
        assert row['independently_initiated_cultures'] == 6*row['independent_blocks_per_context']
