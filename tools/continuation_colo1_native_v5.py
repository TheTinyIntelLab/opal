"""Descriptive native-label metadata audit; no WGE/gene/response qualification."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.continuation_colo1_schema_v2 import (
    attributes,deref,digest,graph_kinds,kind,parse_verified,sha,strings,text,
)

NAMES=('ID','Note','MyNote','sgRNA1_ID','sgRNA2_ID','Gene1','Gene2')


def native_gate(shape,names,total,expected_shape=(78920,7)):
    return (tuple(shape)==tuple(expected_shape) and list(names)==list(NAMES) and
            total==shape[0]*shape[1] and shape[0]<=150000)


def decode_native(root,expected_shape=(78920,7)):
    root=deref(root)
    if kind(root)!='STR':
        raise ValueError('Native character matrix required')
    attrs=attributes(root)
    dims=deref(attrs['dim'])
    if kind(dims)!='INT' or len(dims.value)!=2:
        raise ValueError('Native integer matrix dimensions required')
    shape=[int(v) for v in dims.value]
    labels=deref(attrs['dimnames'])
    if kind(labels)!='VEC' or len(labels.value)!=2:
        raise ValueError('Native matrix labels required')
    names=strings(labels.value[1])
    if not native_gate(shape,names,len(root.value),expected_shape):
        raise ValueError('Native exact shape/name/storage gate refused before decoding')
    counts=graph_kinds(root)
    rows=shape[0]
    table={}
    for i,name in enumerate(names):
        table[name]=[None if deref(v).value is None else text(v)
                     for v in root.value[i*rows:(i+1)*rows]]
    return table,dict(root_kind='STR',dimensions=shape,column_names=names,
                      storage_order='R column-major',serialized_node_kinds=counts)


def complete(value):
    # No label normalization or inferred string sentinels.
    return value is not None and value!=''


def multiplicities(counter):
    values=list(counter.values())
    return dict(distinct_keys=len(counter),rows=sum(values),
                minimum=min(values,default=0),maximum=max(values,default=0),
                multiplicity_histogram={str(k):v for k,v in sorted(Counter(values).items())})


def pair_summary(first,second):
    pairs=[(a,b) for a,b in zip(first,second) if complete(a) and complete(b)]
    ordered=Counter(pairs)
    canonical=Counter(tuple(sorted(p)) for p in pairs)
    keys=set(ordered)
    return dict(complete_rows=len(pairs),incomplete_rows=len(first)-len(pairs),
                ordered=multiplicities(ordered),canonical=multiplicities(canonical),
                identical_native_label_rows=sum(a==b for a,b in pairs),
                reversed_nonidentical_ordered_keys=sum(a!=b and (b,a) in keys for a,b in keys),
                canonical_label_count_sha256=digest(sorted((list(k),v) for k,v in canonical.items())))


def graph_summary(first,second):
    graph=defaultdict(set)
    for a,b in zip(first,second):
        if not complete(a) or not complete(b):
            continue
        graph[a];graph[b]
        if a!=b:
            graph[a].add(b);graph[b].add(a)
    remaining=set(graph)
    sizes=[]
    while remaining:
        start=next(iter(remaining))
        remaining.remove(start)
        pending=[start]
        size=0
        while pending:
            current=pending.pop();size+=1
            new=graph[current]&remaining
            remaining.difference_update(new)
            pending.extend(new)
        sizes.append(size)
    degrees=[len(v) for v in graph.values()]
    return dict(native_target_label_nodes=len(graph),components=len(sizes),
                component_sizes=sorted(sizes,reverse=True),
                nonself_unique_edges=sum(degrees)//2,
                distinct_other_neighbor_degree_histogram={str(k):v for k,v in sorted(Counter(degrees).items())},
                degree_convention='Distinct other native labels; self-label rows do not add a neighbor.')


def summarize_native(table):
    if tuple(table)!=NAMES or len({len(v) for v in table.values()})!=1:
        raise ValueError('Native converted table structure mismatch')
    rows=len(table['ID'])
    stats={}
    for name,column in table.items():
        stats[name]=dict(r_na=sum(v is None for v in column),
                         empty_string=sum(v=='' for v in column),
                         whitespace_only=sum(isinstance(v,str) and v!='' and v.strip()=='' for v in column),
                         literal_uppercase_NA=sum(v=='NA' for v in column))
    ids=[v for v in table['ID'] if complete(v)]
    all_rows=Counter(zip(*(table[n] for n in NAMES)))
    mapping=defaultdict(set)
    arms={}
    for arm in (1,2):
        local=defaultdict(set)
        for guide,target in zip(table[f'sgRNA{arm}_ID'],table[f'Gene{arm}']):
            if complete(guide) and complete(target):
                local[guide].add(target);mapping[guide].add(target)
        arms[str(arm)]=dict(distinct_complete_guide_labels=len(local),
                            guide_labels_with_multiple_target_labels=sum(len(v)>1 for v in local.values()))
    categories={}
    for name in ('Note','MyNote'):
        count=Counter(table[name])
        categories[name]=dict(distinct_categories=len(count),
                              frequencies=multiplicities(count),
                              category_frequency_sha256=digest(sorted(count.items(),key=lambda p:(p[0] is not None,p[0] or ''))))
    return dict(annotation_rows=rows,field_missingness=stats,
                complete_construct_id_rows=len(ids),distinct_complete_construct_ids=len(set(ids)),
                duplicate_construct_id_rows=len(ids)-len(set(ids)),
                exact_duplicate_annotation_rows=rows-len(all_rows),
                guide_to_target_label_consistency_by_arm=arms,
                cross_arm_distinct_complete_guide_labels=len(mapping),
                cross_arm_guide_labels_with_multiple_target_labels=sum(len(v)>1 for v in mapping.values()),
                native_target_label_pairs=pair_summary(table['Gene1'],table['Gene2']),
                native_guide_label_pairs=pair_summary(table['sgRNA1_ID'],table['sgRNA2_ID']),
                native_target_label_graph=graph_summary(table['Gene1'],table['Gene2']),
                categorical_annotations=categories,ordered_metadata_sha256=digest(table),
                completeness_rule='Only R NA and exact empty strings are incomplete. Whitespace and literal NA are counted separately, never normalized.',
                biological_label_identity_established=False,guide_sequence_identity_established=False,
                actual_response_eligibility_established=False)


def main():
    out=ROOT/'results/research-continuation'
    output=out/'colo1-native-receipt-v5.json'
    if output.exists():
        raise FileExistsError(output)
    freeze_path=out/'colo1-native-freeze-v5.json'
    binding_path=out/'colo1-native-code-binding-v5.json'
    freeze=json.loads(freeze_path.read_bytes());binding=json.loads(binding_path.read_bytes())
    if sha(freeze_path)!=binding['freeze_sha256']:
        raise ValueError('Native freeze drift')
    parent=json.loads((out/'colo1-schema-freeze-v1.json').read_bytes())
    manifest=json.loads((out/'colo1-parser-install-manifest-v1.json').read_bytes())
    for path,expected in {**freeze['repo_sha256'],**parent['repo_sha256'],**parent['local_input_sha256'],
                         **binding['code_sha256'],**manifest['installed_sha256']}.items():
        if sha(ROOT/path)!=expected:
            raise ValueError('Native audit prerequisite drift: '+path)
    if shutil.disk_usage(ROOT).free<2**30:
        raise RuntimeError('Unchanged disk reserve')
    sys.path.insert(0,manifest['target'])
    import rdata
    source='tmp/colo1-annotation-acquisition-v2/IDs_colo1.rds'
    result=dict(schema='opal-colo1-native-receipt-v5',freeze_sha256=sha(freeze_path),
                code_binding_sha256=sha(binding_path),annotation_sha256=sha(ROOT/source),
                source_gets=0,response_objects_opened=0,fits=0,scores=0,independent_review=False)
    stage='hash_decompress_parse'
    try:
        parsed,size=parse_verified((ROOT/source).read_bytes(),parent['local_input_sha256'][source],rdata.parser.parse_data)
        result['decompressed_bytes']=size
        stage='native_structure_decode'
        table,structure=decode_native(parsed.object)
        result['observed_schema']=structure
        stage='descriptive_native_label_counts'
        result['native_metadata_summary']=summarize_native(table)
        result['status']='native_metadata_described'
    except Exception as error:
        result.update(status='native_audit_refused',error_type=type(error).__name__,refusal_stage=stage)
    result['source_qualified_for_response_ingestion']=False
    with output.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],source_qualified=False)))


if __name__=='__main__':
    main()
