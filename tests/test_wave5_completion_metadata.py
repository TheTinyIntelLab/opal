"""Policy examples are invented metadata, never an actual source schema."""
import copy

import pytest

from tools.wave5_completion_metadata import fixture,validate_synthetic_v2,variants


@pytest.mark.parametrize('source',['spidr','shen2017'])
def test_declared_metadata_never_qualifies_source_or_endpoint(source):
    result=validate_synthetic_v2(fixture(source))
    assert result['declared_endpoint_metadata_ready']
    assert not result['declared_independent_initiations']
    assert not result['numeric_endpoint_qualified']
    assert not result['source_truth_verified']
    assert result['observed_eligibility'] is None
    assert not result['response_access_authorized']


@pytest.mark.parametrize('case',['renamed-exact-duplicate','contradictory-alias','second-control-right','second-control-left','second-variant-missing','second-variant-conflict','duplicate-semantic-time','unused-guide','unused-single','whitespace-id','aggregate-513'])
def test_registered_ambiguous_relations_refuse(case):
    payload=dict(variants())[case]
    with pytest.raises(ValueError):validate_synthetic_v2(payload)


@pytest.mark.parametrize('case',['baseline','basis-only-alias','second-variant-valid','record-permutation'])
def test_registered_consistent_relations_pass_without_source_claim(case):
    result=validate_synthetic_v2(dict(variants())[case])
    assert result['design_consistent'] and not result['source_truth_verified']


@pytest.mark.parametrize('case',['opaque-time','wrong-origin'])
def test_unknown_semantic_times_do_not_establish_endpoint(case):
    assert not validate_synthetic_v2(dict(variants())[case])['declared_endpoint_metadata_ready']


def test_distinct_documented_initiations_are_a_declaration_only():
    distinct=validate_synthetic_v2(dict(variants())['distinct-initiations'])
    unknown=validate_synthetic_v2(dict(variants())['unknown-initiation'])
    assert distinct['declared_independent_initiations']
    assert not unknown['declared_independent_initiations']
    assert not distinct['source_truth_verified']
