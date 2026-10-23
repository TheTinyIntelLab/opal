import copy

import pytest

from tools.continuation_colo1_container_v1 import EXPECTED, URL, project_identity


def fixture():
    return dict(entries=[{**EXPECTED, 'links': {'container': URL}}])


@pytest.mark.parametrize('field', list(EXPECTED))
def test_changed_identity_field_cannot_authorize_a_container_request(field):
    value = fixture()
    value['entries'][0][field] = 'wrong'
    with pytest.raises(ValueError):
        project_identity(value)


def test_boolean_size_cannot_impersonate_an_integer():
    value = fixture()
    value['entries'][0]['size'] = True
    with pytest.raises(ValueError, match='size'):
        project_identity(value)


def test_duplicate_entry_key_cannot_shadow_identity():
    value = fixture()
    value['entries'].append(copy.deepcopy(value['entries'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        project_identity(value)


@pytest.mark.parametrize('url', [URL+'?token=synthetic', 'https://example.com/container'])
def test_foreign_or_signed_endpoint_substitution_refused(url):
    value = fixture()
    value['entries'][0]['links']['container'] = url
    with pytest.raises(ValueError, match='substitution'):
        project_identity(value)


def test_actual_list_identity_projects_only_the_registered_endpoint():
    value = fixture()
    value['entries'][0]['metadata'] = dict(arbitrary='discard')
    result = project_identity(value)
    assert result['technical_fields'] == EXPECTED
    assert result['container_url'] == URL
    assert 'discard' not in str(result)
