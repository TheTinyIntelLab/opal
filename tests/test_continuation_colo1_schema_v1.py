"""Synthetic schema/corruption challenges, never source data or parser downloads."""
import bz2
import gzip
import hashlib
import lzma
from types import SimpleNamespace as NS

import pytest

from tools.continuation_colo1_schema_v1 import (
    decompress, frame_schema, graph_kinds, parse_verified, schema_gate, summarize,
)


def node(kind, value=None, attributes=None, tag=None, referenced_object=None):
    return NS(info=NS(type=NS(name=kind)), value=value, attributes=attributes,
              tag=tag, referenced_object=referenced_object)


def chars(*values):
    return node('STR', [node('CHAR', s.encode()) for s in values])


def frame(names, columns):
    end = node('NILVALUE')
    cls = node('LIST', (chars('data.frame'), end), tag=node('SYM', node('CHAR', b'class')))
    attrs = node('LIST', (chars(*names), cls), tag=node('SYM', node('CHAR', b'names')))
    return node('VEC', columns, attributes=attrs)


def test_bad_hash_cannot_call_parser():
    def forbidden(*args, **kwargs):
        pytest.fail('Parser called before byte verification')
    with pytest.raises(ValueError, match='before parsing'):
        parse_verified(b'X\nsynthetic', '0'*64, forbidden)


@pytest.mark.parametrize('compress', [gzip.compress, bz2.compress, lzma.compress])
def test_bounded_decompression_and_nested_refusal(compress):
    raw = b'X\n' + b'0'*100
    assert decompress(compress(raw), 102) == raw
    with pytest.raises(ValueError, match='cap'):
        decompress(compress(raw), 101)
    with pytest.raises(ValueError, match='single-layer'):
        decompress(compress(compress(raw)))


def test_verified_parser_is_nonexpanding_rds():
    raw = b'X\nsynthetic'
    calls = []
    def parser(data, **kwargs):
        calls.append((data, kwargs))
        return 'parsed'
    assert parse_verified(raw, hashlib.sha256(raw).hexdigest(), parser) == ('parsed', len(raw))
    assert calls == [(raw, {'expand_altrep':False, 'extension':'.rds'})]


@pytest.mark.parametrize('name,issue', [('unregistered_alias','unknown_field'),
                                       ('fitness','outcome_like_field')])
def test_unknown_and_outcome_fields_stop(name, issue):
    schema, _ = frame_schema(frame([name], [chars('withheld')]))
    assert not schema['gate']['passed']
    assert issue in schema['gate']['issues']
    assert 'withheld' not in str(schema)


def test_duplicate_fields_and_unequal_lengths_stop():
    schema, _ = frame_schema(frame(['ID','ID'], [chars('a'),chars('a','b')]))
    assert set(schema['gate']['issues']) == {'duplicate_field_names','inconsistent_column_lengths'}


@pytest.mark.parametrize('kind', ['CLO','ENV','EXTPTR','ALTREP','BCODE'])
def test_executable_and_unsupported_nodes_refuse(kind):
    with pytest.raises(ValueError, match='Unsupported serialized R kind'):
        graph_kinds(node(kind))


def test_duplicate_construct_id_is_not_eligible():
    result = summarize({'ID':['a','a','b']})
    assert result['duplicate_construct_id_rows'] == 1
    assert not result['construct_id_gate_passed']


def test_guide_identity_conflict_and_symmetric_pair_counts():
    table = {'ID':['x','y'], 'sgRNA1_WGE_ID':['g','g'],
             'sgRNA1_WGE_Sequence':['AAA','CCC'], 'sgRNA1_Approved_Symbol':['A','B'],
             'sgRNA2_WGE_ID':['h','h'], 'sgRNA2_WGE_Sequence':['TTT','TTT'],
             'sgRNA2_Approved_Symbol':['B','A']}
    result = summarize(table)
    assert result['ordered_arm_consistency']['sgRNA1_']['conflicting_guide_ids'] == 1
    assert result['canonical_gene_pairs'] == 1
    assert result['canonical_guide_pairs'] == 1
    assert result['construct_id_gate_passed']
    assert result['missing_by_field']['ID'] == 0


def test_empty_missing_and_row_cap_are_explicit():
    result = summarize({'ID':['',None]})
    assert result['missing_by_field']['ID'] == 2
    assert not result['construct_id_gate_passed']
    assert 'annotation_row_cap' in schema_gate(['ID'], [150001], ['STR'], ['data.frame'])['issues']
    assert 'unsupported_root_class' in schema_gate(['ID'], [1], ['STR'], ['tbl_df','data.frame'])['issues']
