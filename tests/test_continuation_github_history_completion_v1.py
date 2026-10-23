import pytest

from tools.continuation_paralog_github_history_completion_v1 import FAILED, EMPTY, descriptor


def test_empty_tree_requires_exact_git_object_identity_and_matching_commit():
    assert EMPTY == '4b825dc642cb6eb9a060e54bf8d69288fbee4904'
    assert descriptor(dict(sha=FAILED, tree={'sha': EMPTY}, parents=[]))['empty_tree_by_git_identity']
    assert not descriptor(dict(sha=FAILED, tree={'sha': 'b' * 40}, parents=[]))['empty_tree_by_git_identity']


@pytest.mark.parametrize('change', [
    {'sha': 'a' * 40}, {'tree': {'sha': 'bad'}},
    {'parents': {}}, {'parents': [{'sha': None}]}, {'parents': ['bad']},
])
def test_malformed_descriptor_cannot_explain_refused_tree(change):
    value = dict(sha=FAILED, tree={'sha': EMPTY}, parents=[])
    value.update(change)
    with pytest.raises(ValueError):
        descriptor(value)
