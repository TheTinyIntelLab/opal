"""Native labels never silently acquire biological or WGE identities."""
from types import SimpleNamespace as NS

import pytest

from tools.continuation_colo1_native_v5 import NAMES,decode_native,native_gate,summarize_native


def table():
    return dict(zip(NAMES,[['a','b','c'],['n']*3,['m']*3,
        ['g1','g2','g3'],['g2','g1','g3'],['A','B','C'],['B','A','C']]))


def test_canonical_pairs_reversal_and_graph_components():
    result=summarize_native(table())
    assert result['native_target_label_pairs']['ordered']['distinct_keys']==3
    assert result['native_target_label_pairs']['canonical']['distinct_keys']==2
    assert result['native_target_label_pairs']['reversed_nonidentical_ordered_keys']==2
    assert result['native_target_label_graph']['component_sizes']==[2,1]
    assert result['native_target_label_graph']['nonself_unique_edges']==1
    assert result['native_target_label_pairs']['identical_native_label_rows']==1
    assert not result['biological_label_identity_established']
    assert not result['actual_response_eligibility_established']


def test_duplicate_ids_and_missingness_are_distinct():
    data=table();data['ID']=['a','a',None];data['Note']=['',' ','NA']
    result=summarize_native(data)
    assert result['duplicate_construct_id_rows']==1
    assert result['field_missingness']['ID']['r_na']==1
    assert result['field_missingness']['Note']==dict(r_na=0,empty_string=1,whitespace_only=1,literal_uppercase_NA=1)


def test_cross_arm_and_within_arm_conflicts_are_separate():
    data=table();data['sgRNA1_ID']=['g1','g1','g3']
    data['Gene2']=['X','Y','Z']
    result=summarize_native(data)
    assert result['guide_to_target_label_consistency_by_arm']['1']['guide_labels_with_multiple_target_labels']==1
    assert result['cross_arm_guide_labels_with_multiple_target_labels']==2


def test_exact_gate_refuses_unknown_fields_order_and_dimensions():
    assert native_gate([78920,7],NAMES,552440)
    for shape,names,total in [([78921,7],NAMES,552447),([78920,7],NAMES[::-1],552440),
        ([78920,7],list(NAMES[:-1])+['WGE_ID'],552440),([78920,7],NAMES,552439)]:
        assert not native_gate(shape,names,total)


def node(kind,value=None,attributes=None,tag=None,referenced_object=None):
    return NS(info=NS(type=NS(name=kind)),value=value,attributes=attributes,
              tag=tag,referenced_object=referenced_object)


def chars(values):
    return node('STR',[node('CHAR',v.encode()) for v in values])


def test_column_major_decode_is_gated_before_values():
    end=node('NILVALUE');labels=node('VEC',[end,chars(NAMES)])
    rest=node('LIST',(labels,end),tag=node('SYM',node('CHAR',b'dimnames')))
    attrs=node('LIST',(node('INT',[3,7]),rest),tag=node('SYM',node('CHAR',b'dim')))
    data=table();root=node('STR',chars([v for n in NAMES for v in data[n]]).value,attributes=attrs)
    with pytest.raises(ValueError,match='before decoding'):
        decode_native(root)
    decoded,structure=decode_native(root,expected_shape=(3,7))
    assert decoded==data and structure['storage_order']=='R column-major'


def test_summary_refuses_inconsistent_lengths():
    data=table();data['ID']=['a']
    with pytest.raises(ValueError,match='structure mismatch'):
        summarize_native(data)
