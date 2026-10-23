"""Synthetic R matrix ordering, dimensions and pre-conversion field gates."""
from types import SimpleNamespace as NS

import pytest

from tools.continuation_colo1_matrix_v4 import decode_columns, inspect_matrix, matrix_gate


def node(kind, value=None, attributes=None, tag=None, referenced_object=None):
    return NS(info=NS(type=NS(name=kind)), value=value, attributes=attributes,
              tag=tag, referenced_object=referenced_object)


def chars(*values):
    return node('STR', [node('CHAR', v.encode()) for v in values])


def matrix():
    end = node('NILVALUE')
    labels = node('VEC', [end, chars('ID','Notes')])
    rest = node('LIST', (labels,end), tag=node('SYM',node('CHAR',b'dimnames')))
    attrs = node('LIST', (node('INT',[3,2]),rest), tag=node('SYM',node('CHAR',b'dim')))
    return node('STR', chars('a','b','c','note-a','note-b','note-c').value, attributes=attrs)


def test_column_major_order_is_preserved():
    root = matrix()
    schema = inspect_matrix(root)
    assert schema['gate']['passed']
    assert decode_columns(root,schema) == {'ID':['a','b','c'],
                                          'Notes':['note-a','note-b','note-c']}
    assert 'note-a' not in str(schema)


@pytest.mark.parametrize('shape', [[],[2],[2,3,4],[0,2],[-1,2],[True,2],[1.5,2]])
def test_invalid_dimensions_refuse(shape):
    assert matrix_gate(shape,['ID','Notes'],6)['issues'] == ['invalid_dimensions']


@pytest.mark.parametrize('names,issue', [([], 'missing_or_inconsistent_column_names'),
    (['ID','ID'],'duplicate_field_names'), (['ID','unknown_alias'],'unknown_field'),
    (['ID','fitness'],'outcome_like_field')])
def test_bad_fields_refuse_even_if_pass_flag_is_forged(names,issue):
    gate = matrix_gate([3,2],names,6)
    assert issue in gate['issues']
    with pytest.raises(ValueError,match='before conversion'):
        decode_columns(matrix(),dict(dimensions=[3,2],column_names=names,gate={'passed':True}))


def test_shape_and_storage_limits_are_independent():
    assert 'matrix_size_cap' in matrix_gate([150001,1],['ID'],150001)['issues']
    assert 'matrix_size_cap' in matrix_gate([1,65],['ID']*65,65)['issues']
    assert matrix_gate([3,2],['ID','Notes'],5)['issues'] == ['inconsistent_storage_length']


def test_malformed_dimension_labels_refuse():
    root = matrix()
    root.attributes.value[1].value[0].value = [node('NILVALUE')]
    with pytest.raises(ValueError,match='dimension-label'):
        inspect_matrix(root)
