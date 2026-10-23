"""Code catalog/identity and metadata-only projection refusal fixtures."""
import hashlib
import zlib

import pytest

from tools.continuation_colo1_provenance_v1 import metadata_view,select_blob,verify_code


def test_tree_requires_exact_unique_regular_blob():
    entry=dict(path='a.R',mode='100644',type='blob',size=3,sha='a'*40)
    assert select_blob(dict(truncated=False,tree=[entry]),'a.R',3)['bytes']==3
    for tree,path,size in [(dict(truncated=True,tree=[entry]),'a.R',3),
                           (dict(truncated=False,tree=[entry,entry]),'a.R',3),
                           (dict(truncated=False,tree=[entry]),'b.R',3),
                           (dict(truncated=False,tree=[entry]),'a.R',4),
                           (dict(truncated=False,tree=[dict(entry,mode='120000')]),'a.R',3)]:
        with pytest.raises(ValueError):
            select_blob(tree,path,size)


def test_whole_code_identity_before_decoding():
    raw=b'abc'
    entry=dict(bytes=3,git_blob_sha1=hashlib.sha1(b'blob 3\0abc').hexdigest())
    crc=zlib.crc32(raw)&0xffffffff
    assert verify_code(raw,entry,crc)['bytes']==3
    for value,expected_crc in [(b'abd',crc),(b'ab',crc),(raw,0)]:
        with pytest.raises(ValueError,match='before decoding'):
            verify_code(value,entry,expected_crc)


def test_projection_has_no_neighboring_lines():
    rows,view=metadata_view('unrelated <- secret_output\nIDs_colo1 <- readRDS(path)\nplot(response)')
    assert rows==[(2,'IDs_colo1 <- readRDS(path)')]
    assert 'secret_output' not in view and 'response' not in view


@pytest.mark.parametrize('line',['# [1] Gene1 1.23','1 Gene1 1.23','> Gene1',
                                 'x <- data.frame(Gene1="a", fitness=c(1.23))'])
def test_output_like_selected_lines_refuse(line):
    with pytest.raises(ValueError,match='Output-like'):
        metadata_view(line)


def test_projection_limits_are_enforced_before_display():
    with pytest.raises(ValueError,match='line cap'):
        metadata_view('\n'.join(['Gene1 <- x']*201))
    with pytest.raises(ValueError,match='byte cap'):
        metadata_view('Gene1 <- "'+'x'*25000+'"')
