"""Invented metadata policy experiments; no source data or access grant."""
import argparse
import copy
import json
from pathlib import Path
import runpy
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import wave5_source_contract as old
from tools.wave5_completion_common import ROOT,gates,write

CONTRACT='opal-synthetic-metadata-consistency-v2'


def fixture(source='spidr'):
    data=runpy.run_path(str(ROOT/'tests/test_wave5_source_contract.py'))['fixture'](source)
    return dict(design=data,construct_assignments=[dict(construct='ab',left='a',right='b',gene_a='A',gene_b='B',split='test')],time_semantics=[dict(label=t,elapsed=e,origin='transduction') for t,e in [('d3',0.),('d14',11.),('d28',25.)]],initiation_evidence=[dict(id='shared',basis='invented explicit lineage',status='documented')])


def validate_synthetic_v2(payload):
    """Check a declared synthetic common-control policy, never source truth."""
    import math
    if set(payload)!={'design','construct_assignments','time_semantics','initiation_evidence'}:raise ValueError('Unexpected prospective policy fields')
    data=copy.deepcopy(payload['design'])
    total=sum(len(data[k]) for k in ('guides','samples','constructs','matches','anchors','assignments'))+sum(len(payload[k]) for k in ('construct_assignments','time_semantics','initiation_evidence'))
    if total>512:raise ValueError('Aggregate fixture row cap')
    for table in ('guides','samples','constructs'):
        for row in data[table]:
            if not isinstance(row['id'],str) or not row['id'].strip() or row['id']!=row['id'].strip():raise ValueError('Meaningful exact identifiers required')
    physical={}
    for guide in data['guides']:
        key=tuple(guide[k] for k in ('gene','sequence','arm','role'))
        annotations=tuple((guide[k]['state'],guide[k]['value']) for k in ('transcript','mismatch'))
        if key in physical and physical[key]!=annotations:raise ValueError('Contradictory physical guide annotations')
        physical[key]=annotations
        for k in ('transcript','mismatch'):guide[k]['basis']='prospective identity ignores documentary basis text'
    result=old.validate_metadata(data)
    guides={g['id']:g for g in data['guides']};constructs={c['id']:c for c in data['constructs']}
    anchors={a['sample']:constructs[a['construct']] for a in data['anchors']}
    def identity(g):return tuple(g[k] for k in ('gene','sequence','arm','role'))+tuple((g[k]['state'],g[k]['value']) for k in ('transcript','mismatch'))
    for match in data['matches']:
        for side,other in [('left','right'),('right','left')]:
            single=constructs[match[side+'_single']['construct']]
            if identity(guides[single[other]])!=identity(guides[anchors[match['sample']][other]]):raise ValueError('Prospective common-control anchor mismatch')
    doubles={c['id']:c for c in data['constructs'] if all(guides[c[side]]['role']=='target' for side in ('left','right'))}
    seen={};pair_splits={}
    for assignment in payload['construct_assignments']:
        if set(assignment)!={'construct','left','right','gene_a','gene_b','split'}:raise ValueError('Concrete assignment schema')
        c=doubles.get(assignment['construct'])
        if c is None or c['id'] in seen or (assignment['left'],assignment['right'])!=(c['left'],c['right']):raise ValueError('Concrete ordered assignment mismatch')
        pair=tuple(sorted((guides[c['left']]['gene'],guides[c['right']]['gene'])))
        if tuple(sorted((assignment['gene_a'],assignment['gene_b'])))!=pair or assignment['split'] not in {'train','validation','test'}:raise ValueError('Concrete canonical assignment mismatch')
        if pair in pair_splits and pair_splits[pair]!=assignment['split']:raise ValueError('Concrete variants cross splits')
        declared={a['split'] for a in data['assignments'] if tuple(sorted((a['gene_a'],a['gene_b'])))==pair}
        if declared!={assignment['split']}:raise ValueError('Concrete and pair declarations disagree')
        pair_splits[pair]=assignment['split'];seen[c['id']]=True
    if set(seen)!=set(doubles):raise ValueError('Missing concrete double assignment')
    used_constructs=set(doubles)|{a['construct'] for a in data['anchors']}|{m[k]['construct'] for m in data['matches'] for k in ('left_single','right_single')}
    if used_constructs!=set(constructs):raise ValueError('Unlinked construct')
    if {c[k] for c in constructs.values() for k in ('left','right')}!=set(guides):raise ValueError('Unlinked guide')
    times={};elapsed=set()
    for row in payload['time_semantics']:
        if set(row)!={'label','elapsed','origin'} or not isinstance(row['label'],str) or not row['label'].strip() or row['label'] in times:raise ValueError('Semantic time schema')
        e=row['elapsed']
        if isinstance(e,bool) or not isinstance(e,(int,float)) or not math.isfinite(e) or e<0:raise ValueError('Finite elapsed time required')
        if (row['origin'],e) in elapsed:raise ValueError('Duplicate semantic time')
        elapsed.add((row['origin'],e));times[row['label']]=row
    endpoint_ready=True
    for sample in data['samples']:
        semantic=times.get(sample['time'])
        if semantic is None or semantic['origin']!=sample['time_origin']:endpoint_ready=False
    if set(times)!={s['time'] for s in data['samples']}:endpoint_ready=False
    evidence={}
    for row in payload['initiation_evidence']:
        if set(row)!={'id','basis','status'} or row['id'] in evidence:raise ValueError('Initiation evidence schema')
        evidence[row['id']]=row
    ids={s['initiation'] for s in data['samples']}
    known=all(i.lower() not in {'unknown','unresolved','na','none'} and i in evidence and evidence[i]['status']=='documented' and bool(str(evidence[i]['basis']).strip()) for i in ids)
    result.update(contract=CONTRACT,aggregate_rows=total,synthetic_only=True,declared_endpoint_metadata_ready=endpoint_ready,declared_independent_initiations=bool(known and len(ids)>=2),numeric_endpoint_qualified=False,source_truth_verified=False)
    return result


def variants():
    cases=[]
    def add(name,mutate):
        p=fixture();mutate(p);cases.append((name,p))
    cases.append(('baseline',fixture()))
    def alias(p,contradict=False,duplicate=False):
        d=p['design'];g=copy.deepcopy(d['guides'][0]);g['id']='alias';g['transcript']['basis']='second independent document'
        if contradict:g['transcript']['value']='contradictory_tx'
        d['guides'].append(g)
        if duplicate:d['constructs'].append(dict(id='duplicate_single',left='alias',right='n2'))
        else:d['constructs'][1]['left']='alias'
    add('basis-only-alias',lambda p:alias(p))
    add('renamed-exact-duplicate',lambda p:alias(p,duplicate=True))
    add('contradictory-alias',lambda p:alias(p,contradict=True))
    def control(p,side):
        d=p['design'];g=copy.deepcopy(d['guides'][3 if side=='right' else 2]);g['id']='other_control';g['sequence']='ACAC';d['guides'].append(g)
        d['constructs'][1 if side=='right' else 2][side]='other_control'
    add('second-control-right',lambda p:control(p,'right'))
    add('second-control-left',lambda p:control(p,'left'))
    def second(p,mode):
        d=p['design'];g=copy.deepcopy(d['guides'][0]);g.update(id='a2',sequence='ACAC');d['guides'].append(g)
        d['constructs'] += [dict(id='a2b',left='a2',right='b'),dict(id='a2n',left='a2',right='n2')]
        for match in list(d['matches']):
            m=copy.deepcopy(match);m['double']='a2b';m['left_single']['construct']='a2n';d['matches'].append(m)
        if mode!='missing':p['construct_assignments'].append(dict(construct='a2b',left='a2',right='b',gene_a='A',gene_b='B',split='train' if mode=='conflict' else 'test'))
    for mode in ('valid','missing','conflict'):add('second-variant-'+mode,lambda p,mode=mode:second(p,mode))
    add('opaque-time',lambda p:p['design']['samples'][0].update(time='opaque'))
    add('duplicate-semantic-time',lambda p:p['time_semantics'][1].update(elapsed=0.))
    add('wrong-origin',lambda p:p['time_semantics'][0].update(origin='unrelated_origin'))
    def initiation(p,mode):
        for sample in p['design']['samples']:sample['initiation']='unknown' if mode=='unknown' else sample['culture']
        if mode!='unknown':p['initiation_evidence']=[dict(id=c,basis='invented documented distinct initiation',status='documented') for c in ('c1','c2')]
    add('unknown-initiation',lambda p:initiation(p,'unknown'))
    add('distinct-initiations',lambda p:initiation(p,'distinct'))
    add('unused-guide',lambda p:p['design']['guides'].append({**copy.deepcopy(p['design']['guides'][0]),'id':'unused','sequence':'ACAC'}))
    add('unused-single',lambda p:p['design']['constructs'].append(dict(id='unused',left='a',right='n2')))
    add('whitespace-id',lambda p:p['design']['guides'][0].update(id=' '))
    add('aggregate-513',lambda p:p['construct_assignments'].extend([copy.deepcopy(p['construct_assignments'][0]) for _ in range(513-sum(len(p['design'][k]) for k in ('guides','samples','constructs','matches','anchors','assignments'))-sum(len(p[k]) for k in ('construct_assignments','time_semantics','initiation_evidence')))]))
    reversed_fixture=fixture()
    for value in reversed_fixture['design'].values():
        if isinstance(value,list) and value and isinstance(value[0],dict):value.reverse()
    cases.append(('record-permutation',reversed_fixture))
    return cases


def diagnose():
    freeze=gates();results=[]
    for name,payload in variants():
        row=dict(case=name)
        for label,operation in [('historical',lambda:old.validate_metadata(payload['design'])),('prospective',lambda:validate_synthetic_v2(payload))]:
            try:row[label]=dict(accepted=True,result=operation())
            except ValueError as error:row[label]=dict(accepted=False,refusal=str(error))
        results.append(row)
    gates()
    return dict(executor='root after B/D provider failure',freeze_sha256=freeze,synthetic_only=True,network_requests=0,source_qualified={'spidr':False,'shen2017':False},observed_eligibility=None,response_access_authorized=False,response_header_access_authorized=False,cases=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=diagnose();write(args.output,result);print(json.dumps(dict(cases=len(result['cases']),source_qualified=result['source_qualified'])))
