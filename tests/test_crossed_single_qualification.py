import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("crossed_source",Path(__file__).parents[1]/"tools/qualify_crossed_single_source.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_literal_assignments_reject_code_mixtures_and_nonphysical_doses():
    assert module.assignments("[('drug',1,'uM')]")["dose"] == 1
    for value in ("__import__('os').getcwd()", "[('a',1,'uM'),('b',1,'uM')]", "[('a',-1,'uM')]", "[('a',True,'uM')]", "[('a',1,'')]", [('a',10**500,'uM')]):
        assert module.assignments(value) is None


def test_catalog_labels_and_repeated_plates_never_certify_cultures_or_edges():
    rows=[dict(sample='ctrl',plate='1',drug='DMSO_TF'),
          dict(sample='a',plate='1',drug='a',drugname_drugconc="[('a',1,'uM')]"),
          dict(sample='b',plate='2',drug='a',drugname_drugconc="[('a',1,'uM')]")]
    result=module.summarize(rows,['c1','c2','c3'])
    assert not result['qualified'] and result['catalog_context_count']==3
    assert result['measured_crossed_edge_count'] is None
    assert result['independently_initiated_culture_count'] is None
    assert 'missing_plate_vehicle' in result['reasons']
    assert len(result['repeated_assignment_plate_groups'])==1
    assert not result['expression_accessed'] and not result['pair_composition_claim']


def test_duplicate_samples_and_invalid_assignments_remain_in_decision():
    result=module.summarize([dict(sample='a',plate='1',drug='x',drugname_drugconc='bad')]*2,[])
    assert 'duplicate_sample_ids' in result['reasons']
    assert result['invalid_sample_ids']==['a','a']
    assert module.summarize([dict(sample=None,plate=None,drug='x',drugname_drugconc="[('x',1,'uM')]")],[])['literal_single_sample_rows']==0
