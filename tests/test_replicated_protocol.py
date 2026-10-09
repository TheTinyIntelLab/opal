"""Protocols and amendments must not manufacture timing or scientific support."""
import copy
import json
from pathlib import Path
import pytest
from opal.replicated.protocol import validate_protocol,statistical_evidence,amendment
from opal.replicated.contracts import sha,write_json


def config():
    return json.loads((Path(__file__).parents[1]/'configs/experiments/replicated_fixed_state_v1.json').read_text())


def test_protocol_rejects_postulated_reference_and_unbounded_selection():
    cfg=config();assert validate_protocol(cfg)['configurations']==16
    cfg['menu'][0]['kind']='global_plus_pair'
    with pytest.raises(ValueError,match='non-product reference'):validate_protocol(cfg)
    cfg=config();cfg['menu'][1]['id']=cfg['menu'][0]['id']
    with pytest.raises(ValueError,match='unique'):validate_protocol(cfg)
    cfg=config();cfg['maximum_seconds']=float('inf')
    with pytest.raises(ValueError,match='protocol field'):validate_protocol(cfg)


def test_scientific_protocol_requires_backed_empirical_design(tmp_path):
    cfg=config();cfg.update(data_role='prospective_confirmation',confirmation_cultures=['a','b','c'])
    with pytest.raises(ValueError,match='reviewed empirical power'):validate_protocol(cfg)
    evidence=tmp_path/'power.json'
    write_json(evidence,{'confirmation_design_qualified':False,'independent_cultures_verified':False})
    cfg['statistical_design']={'pilot_report':'power.json','interval_coverage_report':'power.json',
        'pilot_report_sha256':sha(evidence),'interval_coverage_report_sha256':sha(evidence),
        'review_source':'primary reviewer record','confirmation_cultures':3,
        'stopping_rule':'fixed_sample_size_no_outcome_peeking','estimand':'fixed_pair_panel_new_cultures'}
    assert validate_protocol(cfg)['valid']
    with pytest.raises(ValueError,match='does not qualify'):statistical_evidence(cfg,tmp_path)
    write_json(evidence,{'confirmation_design_qualified':True,'independent_cultures_verified':True,
                         'qualification_status':'accepted_for_fixed_panel'})
    with pytest.raises(ValueError,match='Changed reviewed'):statistical_evidence(cfg,tmp_path)


def test_exposed_amendment_preserves_original_and_requires_fresh_confirmation(tmp_path):
    original=tmp_path/'original.json';original.write_text('{"endpoint":"mse"}')
    digest=sha(original)
    record=amendment(original,tmp_path/'amendment.json','Endpoint changed after scoring',True,['primary endpoint'])
    assert sha(original)==digest and record['original_preserved']
    assert record['fresh_confirmation_required_for_changed_claim']
    assert not record['public_timestamp_claimed']
    with pytest.raises(FileExistsError):amendment(original,tmp_path/'amendment.json','again',False,['endpoint'])
