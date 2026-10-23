import pytest

from tools.continuation_paralog_github_history_v1 import decode, validate_refs, validate_tree


def test_catalog_duplicate_keys_refused_before_schema_interpretation():
    with pytest.raises(ValueError, match='Duplicate catalog key'):
        decode(b'{"sha":"untrusted","sha":"overwritten"}')


@pytest.mark.parametrize('value', [None, 'not-a-git-id', 'a' * 39, 3])
def test_invalid_catalog_identity_cannot_create_fetch_target(value):
    with pytest.raises(ValueError, match='Invalid Git identity'):
        validate_refs([{'sha': value}], 'commits')


@pytest.mark.parametrize('change', [
    {'truncated': True}, {'tree': {}},
    {'tree': [{'path': 'METADATA/map.tsv', 'type': 'blob', 'sha': 'b' * 40, 'size': -1}]},
    {'tree': [{'path': '../METADATA/map.tsv', 'type': 'blob', 'sha': 'b' * 40, 'size': 1}]},
])
def test_incomplete_or_malformed_tree_cannot_establish_filename_coverage(change):
    tree = dict(sha='a' * 40, truncated=False, tree=[])
    tree.update(change)
    with pytest.raises(ValueError):
        validate_tree(tree)


def test_duplicate_tree_paths_refused_even_with_distinct_blob_identities():
    entries = [dict(path='METADATA/map.tsv', type='blob', sha=h * 40, size=1) for h in ('b', 'c')]
    with pytest.raises(ValueError, match='duplicate tree path'):
        validate_tree(dict(sha='a' * 40, truncated=False, tree=entries))


def test_catalog_keeps_only_filename_identity_and_size_without_content():
    tree = dict(sha='a' * 40, truncated=False, tree=[dict(path='METADATA/map.tsv', type='blob', sha='b' * 40, size=4)])
    result = validate_tree(tree)
    assert result['entry_count'] == 1
    assert result['entries'] == tree['tree']
    refs = validate_refs([dict(name='tag', commit={'sha': 'c' * 40}, arbitrary_body='discard')], 'tags')
    assert refs == [dict(name='tag', sha='c' * 40)]
