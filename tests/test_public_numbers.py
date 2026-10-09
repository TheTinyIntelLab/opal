"""Numerical grounding rejects changed sentences and changed compact sources."""
import pytest
from tools.audit_public_numbers import literal_value,check
from opal.replicated.contracts import sha


def test_scientific_notation_and_percent_rounding_are_explicit():
    assert literal_value(r'2.22709\times10^{-4}')[0]==pytest.approx(.000222709)
    value,tolerance,percent=literal_value(r'-12.79\%')
    assert value==-12.79 and percent and tolerance==pytest.approx(.00500001)


def test_audit_fails_when_public_text_or_source_changes(tmp_path):
    sentence=tmp_path/'README.md';sentence.write_text('MSE is 0.0074.\n')
    source=tmp_path/'results.json';source.write_text('{"mse":0.00740358}')
    record={'literals':1,'claims':[{'surface':'README.md','line':1,'context':'MSE is 0.0074.',
        'value':.0074,'rounding_tolerance':.00005,'matching_sources':[{
            'source_path':'results.json','source_sha256':sha(source),'displayed_value':.00740358}]}]}
    assert check(record,tmp_path)['checked_literals']==1
    source.write_text('{"mse":0.009}')
    with pytest.raises(ValueError,match='source'):check(record,tmp_path)
    sentence.write_text('MSE is 0.009.\n')
    with pytest.raises(ValueError,match='sentence'):check(record,tmp_path)
