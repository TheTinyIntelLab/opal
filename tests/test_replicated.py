"""Failure-oriented checks for metadata, isolation, symmetry and frozen scoring."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from opal.replicated.contracts import identity,qualify,split_pairs,validate_access,write_json,sha
from opal.replicated.data import prepare,query_references
from opal.replicated.models import ComponentAdditive,ProfileKernel,balanced_weights,shuffled_profiles
from opal.replicated.evaluation import decomposition,shared_reference_draws,paired_interval,empirical_power,cube,score
from opal.replicated.workflow import freeze,evaluate,restore_predictions,load_bank
from tools.run_replicated_fixed_state import demo


def row(name,components,culture='culture1',kind=None,partition='train',doses=None):
    return {'row_id':name,'study_id':'study','source_id':'source','culture_id':culture,
            'culture_initiation':'2026-10-01 independent seed','culture_evidence':{
                'kind':'independent_initiation','culture_id':culture,'record_id':'initiation-'+culture,'source':'lab notebook accession'},
            'state_id':'K562','assay_id':'bulk','processing_batch_id':'batch1','control_stratum_id':culture,
            'condition_kind':kind or {0:'control',1:'single',2:'pair'}[len(components)],
            'components':components,'reagent_ids':components,'doses':doses or [1]*len(components),
            'dose_units':['uM']*len(components),'treatment_time':'12h','treatment_order':'simultaneous',
            'replicate_type':'independent_culture','count_provenance':'aggregate aligned reads','partition':partition}


def contract():
    return {'confirmation_cultures':['query1'],'normalization':'library_total','transform':'log2_after_reference_mean',
            'library_target':10000,'pseudocount':1,'scale':'log2_aggregate','units':'log2','aliases':{}}


def test_dose_assignment_and_aliased_reverse_grouping():
    a=row('p',['A','B'],doses=[1,2]);b=row('q',['B','gA'],doses=[2,1]);b['reagent_ids']=['B','A']
    aliases={'gA':'A'}
    assert identity(a)==identity(b,aliases)
    assert identity(a,dose_specific=True)==identity(b,aliases,True)
    b['doses']=[1,2]
    assert identity(a,dose_specific=True)!=identity(b,aliases,True)
    # Identity grouping remains global even across different doses and states.
    b['partition']='test';b['state_id']='other'
    with pytest.raises(ValueError,match='cross partitions'):validate_access([a,b],{},[],[],aliases)


def test_alias_chains_cannot_evade_global_leakage_or_profile_order():
    aliases={'x':'Z','Z':'A'}
    a=row('p',['A','B']);b=row('held',['B','x'],partition='test')
    assert identity(a)==identity(b,aliases)
    with pytest.raises(ValueError,match='cross partitions'):
        validate_access([a,b],{},[],[],aliases)
    with pytest.raises(ValueError,match='Cyclic'):
        identity(a,{'A':'Z','Z':'A'})
    rows=[row('c',[],partition='reference'),row('a',['A'],partition='reference'),
          row('b',['B'],partition='reference'),row('p',['x','B'])]
    rows[-1]['reagent_ids']=['A','B']
    cfg=contract();cfg['aliases']=aliases
    bank,_=prepare(rows,np.array([[100,100],[180,20],[20,180],[90,110]]),['g1','g2'],cfg)
    assert bank['pair_ids'][0]=='["A","B"]' and bank['singles'][0,0,0]>0


@pytest.mark.parametrize('stage',['normalization','PCA','gene_selection','pretraining','calibration','early_stopping','fitting'])
def test_held_pair_in_any_response_consuming_stage_fails(stage):
    a=row('held',['A','B'],partition='test')
    with pytest.raises(ValueError,match='Forbidden pair'):validate_access([a],{stage:['held']},[identity(a)],[])


def test_technical_run_never_qualifies_independent_culture():
    rows=[row('c',[],partition='reference'),row('a',['A'],partition='reference'),row('b',['B'],partition='reference'),row('p',['A','B'])]
    assert qualify(rows)['independent_cultures_verified']
    rows[0]['culture_initiation']=None;rows[0]['replicate_type']='technical_run'
    q=qualify(rows)
    assert not q['independent_cultures_verified'] and q['role']=='replicated_diagnostic_only'
    rows[2]['doses']=[2]
    assert 'p' in qualify(rows)['rejected']
    rows[0]['culture_initiation']='2026-10-01';rows[0]['replicate_type']='independent_culture'
    rows[0]['culture_evidence']='plate1'
    assert not qualify(rows)['independent_cultures_verified']


@pytest.mark.parametrize('guard', [True, False])
def test_adapter_rejects_prohibited_metadata_before_reading_counts(guard):
    class UnreadableCounts:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Prohibited counts were accessed')
    rows=[row('p',['alias','B'],partition='guard' if guard else 'train')]
    cfg=contract();cfg['aliases']={'alias':'A'};cfg['held_pair_ids']=['["A","B"]']
    with pytest.raises(ValueError,match='sealed|Forbidden pair'):
        prepare(rows,UnreadableCounts(),['g'],cfg)


@pytest.mark.parametrize('command',['prepare','query-references','allocation'])
def test_preparation_cli_rejects_forbidden_rows_before_opening_count_file(tmp_path,monkeypatch,command):
    import sys
    from tools.run_replicated_fixed_state import main
    write_json(tmp_path/'metadata.json',[row('held',['A','B'],partition='test')])
    write_json(tmp_path/'contract.json',contract())
    args=['runner',command,'--metadata',str(tmp_path/'metadata.json'),'--contract',str(tmp_path/'contract.json'),
          '--counts',str(tmp_path/'does-not-exist.npz'),'--output',str(tmp_path/'prepared.npz')]
    if command=='query-references':args+=['--queries',str(tmp_path/'metadata.json')]
    if command=='allocation':args+=['--protocol',str(tmp_path/'contract.json')]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(ValueError,match='sealed|pair outcomes'):main()
    assert not (tmp_path/'prepared.npz').exists()


def test_preparation_cli_preserves_orphaned_manifest(tmp_path,monkeypatch):
    import sys
    from tools.run_replicated_fixed_state import main
    manifest=tmp_path/'prepared.preparation.json';manifest.write_text('prior preparation record')
    monkeypatch.setattr(sys,'argv',['runner','prepare','--metadata','missing','--contract','missing',
                                  '--counts','missing','--output',str(tmp_path/'prepared.npz')])
    with pytest.raises(FileExistsError,match='preparation manifest'):main()
    assert manifest.read_text()=='prior preparation record'


def test_query_bank_cannot_hide_responses_in_an_unexpected_array(tmp_path):
    bank={'singles':np.ones((1,2,3)),'pair_ids':np.array(['["A","B"]']),'culture_ids':np.array(['c']),
          'row_ids':np.array(['q']),'genes':np.array(['g1','g2','g3']),'scale':np.asarray('log2'),'units':np.asarray('log2'),
          'innocent_name':np.zeros((1,3))}
    np.savez_compressed(tmp_path/'bad.npz',**bank)
    with pytest.raises(ValueError,match='Unexpected bank fields'):load_bank(tmp_path/'bad.npz')


def test_metadata_split_deterministic_and_global():
    rows=[row(f'p{i}',list(pair)) for i,pair in enumerate([('A','B'),('A','C'),('A','D'),('B','C'),('B','D'),('C','D')])]
    rows.append(row('variant',['B','A'],culture='other',doses=[2,3]))
    a=split_pairs(rows,[3,1,2],11);b=split_pairs(list(reversed(rows)),[3,1,2],11)
    assert a['assignment']==b['assignment'];assert a['rows']['p0']==a['rows']['variant']
    with pytest.raises(ValueError):split_pairs(rows,[4,1,2],11)


def test_adapter_matches_references_and_preserves_canonical_profile_order():
    rows=[row('c',[],partition='reference'),row('a',['A'],partition='reference'),row('b',['B'],partition='reference'),row('p',['B','A'])]
    x=np.array([[100,100],[180,20],[20,180],[90,110]])
    bank,q=prepare(rows,x,['g1','g2'],contract())
    query=copy.deepcopy(rows[-1]);query['partition']='test'
    refs=query_references(rows[:3],x[:3],['g1','g2'],[query],contract())
    np.testing.assert_allclose(bank['singles'],refs['singles'])
    assert bank['singles'][0,0,0]>0
    with pytest.raises(ValueError,match='sealed'):prepare(rows[:3]+[query],x,['g1','g2'],contract())
    with pytest.raises(ValueError,match='pair outcomes'):query_references(rows,x,['g1','g2'],[query],contract())
    bad=copy.deepcopy(rows);bad[2]['reagent_ids']=['different reagent']
    assert 'p' in qualify(bad)['rejected']


@pytest.mark.parametrize('key,value',[('library_target',float('inf')),('pseudocount',float('nan')),('pseudocount',0)])
def test_query_adapter_rejects_invalid_normalization_constants(key,value):
    refs=[row('c',[],partition='reference'),row('a',['A'],partition='reference'),row('b',['B'],partition='reference')]
    cfg=contract();cfg[key]=value
    with pytest.raises(ValueError,match='Normalization constants'):
        query_references(refs,np.ones((3,2)),np.array(['g1','g2']),[row('q',['A','B'],partition='test')],cfg)


def test_component_incidence_is_symmetric_and_unseen_component_correction_is_zero():
    p=np.asarray(['["A","B"]','["A","C"]','["B","C"]'])
    model=ComponentAdditive(.1).fit(p,np.array([[1.,2],[3.,4],[2.,1]]))
    np.testing.assert_allclose(model.predict(['["A","B"]']),model.predict(['["B","A"]']))
    np.testing.assert_allclose(model.design(['["unseen","A"]']),model.design(['["A"]']))
    np.testing.assert_allclose(model.coefficients.sum(0),0,atol=1e-12)


def test_profile_kernel_is_symmetric_and_query_single_is_an_actual_input():
    rng=np.random.default_rng(7);x=rng.normal(size=(9,2,4));y=x.sum(1)+rng.normal(0,.1,(9,4))
    model=ProfileKernel({'kind':'global_plus_pair','alpha':1}).fit(x,y)
    np.testing.assert_allclose(model.predict(x),model.predict(x[:,::-1]),atol=1e-10)
    altered=x.copy();altered[0,0]+=1
    assert not np.allclose(model.predict(x)[0],model.predict(altered)[0])


def test_shuffling_profiles_breaks_alignment_and_reuses_complete_references():
    pairs=np.asarray(['["A","B"]','["A","C"]','["B","C"]'])
    bank={'pair_ids':pairs,'culture_ids':np.asarray(['c']*3),'singles':np.array([[[1.,0],[2.,0]],[[1.,0],[3.,0]],[[2.,0],[3.,0]]])}
    shuffled=shuffled_profiles(bank,3)
    assert not np.array_equal(shuffled['singles'],bank['singles'])
    np.testing.assert_array_equal(shuffled['singles'][0,0],shuffled['singles'][1,0])


def test_equal_culture_weighting_and_shared_single_draws():
    values=np.array([1.,2.,5.,6.]);pairs=np.array(['AB','AC','AB','AC']);cultures=np.array(['c1','c1','c2','c2'])
    original=balanced_weights(pairs,cultures)@values
    index=[0,0,1,2,3]
    assert balanced_weights(pairs[index],cultures[index])@values[index]==pytest.approx(original)
    p=np.array([[0,1],[0,2],[1,2]])
    additive,ctrl,single=shared_reference_draws(np.zeros(2),np.ones((3,2)),p,np.random.default_rng(4),.1,.2)
    np.testing.assert_allclose(additive[0]+additive[1]-additive[2],2*(single[0]-ctrl))


def test_decomposition_and_same_anchor_are_exact_with_reference_error_cancellation():
    input=np.array([[1.,2.]]);pred=np.array([[.9,1.8]]);truth=np.array([[.8,1.7]]);dagger=np.array([[.7,1.5]])
    result=decomposition(input,pred,dagger,truth)
    assert result['cross_term']<0
    assert result['total']==pytest.approx(result['main']+result['residual']+result['cross_term'])
    assert result['same_anchor_residual_mse']==pytest.approx(result['total'])


def test_independent_reference_accounting_preserves_equal_unit_weights():
    x=np.array([[1.,2.],[3.,4.],[2.,3.],[4.,5.]])
    pred=x*.9;truth=x*.8;independent=x*.7
    pairs=np.array(['AB','AC','AB','AC']);cultures=np.array(['c1','c1','c2','c2'])
    original=decomposition(x,pred,independent,truth,balanced_weights(pairs,cultures))
    idx=[0,0,0,1,2,3]
    repeated=decomposition(x[idx],pred[idx],independent[idx],truth[idx],balanced_weights(pairs[idx],cultures[idx]))
    for key in original:assert repeated[key]==pytest.approx(original[key])


@pytest.fixture
def frozen_demo(tmp_path):
    output=tmp_path/'demo';demo(output,Path(__file__).parents[1]/'configs/experiments/replicated_fixed_state_v1.json')
    # Demo was scored once. A second freeze allows evaluator failure checks.
    freeze(output/'development.npz',output/'query_references.npz',output/'metadata.json',output/'protocol.json',output/'untouched')
    return output


def test_outcomes_do_not_affect_frozen_predictions_and_restoration(frozen_demo):
    output=frozen_demo
    original=(output/'freeze/predictions.npz').read_bytes()
    with np.load(output/'custodian_synthetic_outcomes.npz') as f:bank={k:f[k] for k in f.files}
    bank['responses']+=100
    np.savez_compressed(output/'changed_outcomes.npz',**bank)
    report=evaluate(output/'untouched',output/'changed_outcomes.npz',output/'changed_report')
    assert not report['primary_claim_pass'] and not report['scientific_confirmation_claimed']
    assert original==(output/'freeze/predictions.npz').read_bytes()
    restored=restore_predictions(output/'freeze',output/'query_references.npz')
    with np.load(output/'freeze/predictions.npz') as f:
        for role,pred in restored.items():np.testing.assert_allclose(pred,f[role])
    with pytest.raises(FileExistsError):evaluate(output/'untouched',output/'changed_outcomes.npz',output/'another_report')


@pytest.mark.parametrize('field',['pair_ids','culture_ids','singles'])
def test_restoration_rejects_changed_query_identities_and_references(frozen_demo,field):
    output=frozen_demo
    with np.load(output/'query_references.npz') as f:bank={k:f[k] for k in f.files}
    if field=='singles':bank[field][0,0,0]+=1
    elif field=='pair_ids':bank[field]=np.roll(bank[field],1)
    else:bank[field][0]='other'
    np.savez_compressed(output/'changed_query.npz',**bank)
    with pytest.raises(ValueError,match='query identities|query references'):
        restore_predictions(output/'freeze',output/'changed_query.npz')


def test_frozen_gene_diagnostics_ignore_duplicate_technical_rows(frozen_demo):
    output=frozen_demo
    with np.load(output/'development.npz') as f:bank={k:f[k] for k in f.files}
    index=np.r_[np.arange(len(bank['row_ids'])),np.repeat(0,20)]
    for key in ('singles','responses','row_ids','pair_ids','culture_ids','partitions'):
        bank[key]=bank[key][index]
    bank['row_ids']=np.asarray(bank['row_ids'].tolist()[:-20]+[f'duplicate-{i}' for i in range(20)])
    metadata=json.loads((output/'metadata.json').read_text())
    original=next(r for r in metadata if r['row_id']==str(bank['row_ids'][0]))
    metadata.extend(dict(original,row_id=f'duplicate-{i}') for i in range(20))
    np.savez_compressed(output/'duplicated.npz',**bank)
    write_json(output/'duplicated_metadata.json',metadata)
    freeze(output/'duplicated.npz',output/'query_references.npz',output/'duplicated_metadata.json',output/'protocol.json',output/'duplicated_freeze')
    with np.load(output/'freeze/predictions.npz') as original,np.load(output/'duplicated_freeze/predictions.npz') as repeated:
        for key in ('secondary_gene_panel','secondary_gene_weights'):
            np.testing.assert_allclose(original[key],repeated[key])
    access=json.loads((output/'duplicated_freeze/access.json').read_text())['stages']
    lookup=dict(zip(bank['row_ids'],bank['partitions']))
    assert all(lookup[i]=='train' for i in access['fitting'])
    assert all(lookup[i]=='validation' for i in access['selection'])


@pytest.mark.parametrize('field',['genes','units','scale','row_ids','pair_ids'])
def test_evaluator_rejects_schema_changes(frozen_demo,field):
    output=frozen_demo
    with np.load(output/'custodian_synthetic_outcomes.npz') as f:bank={k:f[k] for k in f.files}
    if bank[field].ndim:bank[field]=bank[field][::-1]
    else:bank[field]=np.asarray('changed')
    np.savez_compressed(output/'bad.npz',**bank)
    with pytest.raises(ValueError,match='schema mismatch'):evaluate(output/'untouched',output/'bad.npz',output/'bad_report')


def test_evaluator_rejects_changed_prediction_hash(frozen_demo):
    p=frozen_demo/'untouched/predictions.npz'
    with p.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):evaluate(frozen_demo/'untouched',frozen_demo/'custodian_synthetic_outcomes.npz',frozen_demo/'bad_report')


def test_concurrent_score_cannot_publish_a_second_report(frozen_demo,monkeypatch):
    import opal.replicated.workflow as workflow
    original=workflow.score
    receipt=frozen_demo/'untouched/evaluation_receipt.json'
    def competing_score(*args,**kwargs):
        result=original(*args,**kwargs)
        receipt.write_text('{"status":"complete","competing_evaluator":true}')
        return result
    monkeypatch.setattr(workflow,'score',competing_score)
    with pytest.raises(FileExistsError):
        evaluate(frozen_demo/'untouched',frozen_demo/'custodian_synthetic_outcomes.npz',frozen_demo/'second_report')
    assert not (frozen_demo/'second_report').exists()
    assert json.loads(receipt.read_text())['competing_evaluator']


def test_freeze_rejects_measured_memory_budget_overrun(frozen_demo,monkeypatch):
    import opal.replicated.resources as resources
    monkeypatch.setattr(resources,'memory_evidence',lambda:{'peak_memory_bytes':3*1024**3,
        'memory_scope':'test process peak','measurement_method':'injected budget overrun'})
    output=frozen_demo
    with pytest.raises(RuntimeError,match='memory budget'):
        freeze(output/'development.npz',output/'query_references.npz',output/'metadata.json',output/'protocol.json',output/'oversized')
    assert not (output/'oversized').exists()


def test_processing_role_does_not_bypass_exact_reference_support(frozen_demo):
    output=frozen_demo;cfg=json.loads((output/'protocol.json').read_text())
    cfg['data_role']='processing_diagnostic'
    write_json(output/'processing_protocol.json',cfg)
    with pytest.raises(ValueError,match='exact matched reference metadata'):
        freeze(output/'development.npz',output/'query_references.npz',output/'metadata.json',output/'processing_protocol.json',output/'unmatched_processing')


def test_technical_metadata_cannot_be_promoted_by_changing_protocol_role(frozen_demo):
    output=frozen_demo;cfg=json.loads((output/'protocol.json').read_text())
    cfg.update(data_role='prospective_confirmation',status='registered',registration_url='https://example.org/registration')
    write_json(output/'scientific_protocol.json',cfg)
    with pytest.raises(ValueError,match='independent-initiation'):
        freeze(output/'development.npz',output/'query_references.npz',output/'metadata.json',output/'scientific_protocol.json',output/'bad_freeze')


@pytest.mark.parametrize('registered,url,verified',[
    ('2000-01-01T00:00:00+00:00','https://example.org/registration',True),
    ('2099-01-01T00:00:00+00:00','https://example.org/registration',False),
    ('2000-01-01T00:00:00+00:00','https://example.org/other',False)])
def test_registration_review_requires_matching_record_and_prior_timestamp(frozen_demo,registered,url,verified):
    output=frozen_demo;cfg=json.loads((output/'protocol.json').read_text())
    cfg['registration_url']='https://example.org/registration'
    write_json(output/'registration_protocol.json',cfg)
    frozen=output/'registration_freeze'
    manifest=freeze(output/'development.npz',output/'query_references.npz',output/'metadata.json',output/'registration_protocol.json',frozen)
    assert not manifest['public_registration_established']
    write_json(output/'review.json',{'freeze_sha256':sha(frozen/'freeze_manifest.json'),
        'registration':{'verified':True,'registered_utc':registered,'url':url,'verification_source':'custodian record review'}})
    report=evaluate(frozen,output/'custodian_synthetic_outcomes.npz',output/'registration_report',output/'review.json')
    assert report['public_registration_review_verified']==verified
    assert not report['scientific_confirmation_claimed']


def test_incomplete_support_and_pseudoreplication_are_rejected():
    with pytest.raises(ValueError,match='Incomplete'):cube(np.array([1.,2.,3.]),['a','a','b'],['AB','AC','AB'])
    with pytest.raises(ValueError,match='three'):paired_interval(np.array([1.,2.]),np.array([.9,1.9]))
    with pytest.raises(ValueError):empirical_power(np.ones((2,10)),np.ones((2,10)),[10])


def test_full_gate_does_not_advance_failed_primary_or_synthetic_measurement():
    truth=np.zeros((8,2));ref=np.ones((8,2));cand=.99*ref
    pred={'reference':ref,'candidate':cand,'additive':1.2*ref,'shuffle_1':1.1*ref}
    cultures=np.repeat(['c1','c2','c3','c4'],2);pairs=np.tile(['AB','AC'],4)
    report=score(pred,truth,cultures,pairs,pred['additive'],measurement_pass=True,attribution_pass=True)
    assert not report['primary_claim_pass']
    report=score(pred,truth,cultures,pairs,pred['additive'])
    assert report['decision']=='measurement_unqualified'
