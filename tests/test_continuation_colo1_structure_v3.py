"""Partial-schema diagnostics must never turn structural support into eligibility."""
from types import SimpleNamespace as NS

from tools.continuation_colo1_structure_v3 import inspect_structure, structural_counts


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


def test_unknown_field_is_reported_without_values():
    result = inspect_structure(frame(['unknown'], [chars('never-print-this')]))
    assert result['schema_gate']['issues'] == ['unknown_field']
    assert 'never-print-this' not in str(result)
    assert not result['field_values_converted']
    assert not result['field_values_aggregated']


def test_opaque_kind_preserves_partial_frame_schema():
    root = frame(['ID'], [chars('hidden')])
    root.attributes.value[1].value = (chars('data.frame'), node('NILVALUE'))
    root.attributes.value[1].value[0].attributes = node('ALTREP')
    result = inspect_structure(root)
    assert result['column_names'] == ['ID']
    assert result['unsupported_kinds'] == ['ALTREP']
    assert result['schema_gate']['passed']
    assert not result['field_values_aggregated']


def test_executable_object_is_counted_without_evaluation_or_value_traversal():
    class Unreadable:
        def __iter__(self):
            raise AssertionError('Opaque value traversed')
    assert structural_counts(node('ENV', Unreadable())) == {'ENV':1}


def test_bad_class_keeps_structural_counts():
    root = frame(['ID'], [chars('hidden')])
    root.attributes.value[1].value = (node('INT', [1]), node('NILVALUE'))
    result = inspect_structure(root)
    assert result['schema_inspection_error_type'] == 'ValueError'
    assert result['completed_structural_operations'] == ['root_kind','root_attributes']
    assert result['serialized_node_kinds']['INT'] == 1


def test_reference_cycles_refuse_without_object_repr():
    root = node('REF')
    root.referenced_object = root
    result = inspect_structure(root)
    assert result['schema_inspection_error_type'] == 'ValueError'
    assert result['serialized_node_kinds'] == {'REF':1}
