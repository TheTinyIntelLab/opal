"""No-request, no-data-deserialization verification of native audit custody/counts."""
import argparse
import json
from pathlib import Path

from continuation_colo1_provenance_v1 import metadata_view,verify_code
from continuation_colo1_schema_v2 import sha

ROOT=Path(__file__).resolve().parents[1]


def reconcile_multiplicities(summary):
    histogram={int(k):v for k,v in summary['multiplicity_histogram'].items()}
    if sum(histogram.values())!=summary['distinct_keys']:
        raise ValueError('Multiplicity key total mismatch')
    if sum(k*v for k,v in histogram.items())!=summary['rows']:
        raise ValueError('Multiplicity row total mismatch')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    out=ROOT/'results/research-continuation'
    artifacts=[];checked={}
    for prefix in ('colo1-provenance-','colo1-native-'):
        artifacts.extend(sorted(out.glob(prefix+'*.json')))
    for path in artifacts:
        obj=json.loads(path.read_bytes())
        for key in ('repo_sha256','local_input_sha256','code_sha256'):
            for target,expected in obj.get(key,{}).items():
                if sha(ROOT/target)!=expected:
                    raise ValueError('Native custody drift: '+target)
                checked[target]=expected
        if 'returncode' in obj:
            if not obj['passed'] or obj['returncode'] or obj['stopped'] or obj['bound_drift']:
                raise ValueError('Unexpected native command failure')
            for target,expected in obj['bound_sha256'].items():
                if sha(target)!=expected:
                    raise ValueError('Command binding drift')
            if sha(obj['log'])!=obj['log_sha256']:
                raise ValueError('Command log drift')
    parent=json.loads((out/'colo1-schema-freeze-v1.json').read_bytes())
    manifest=json.loads((out/'colo1-parser-install-manifest-v1.json').read_bytes())
    for target,expected in {**parent['repo_sha256'],**parent['local_input_sha256'],
                            **manifest['installed_sha256']}.items():
        if sha(ROOT/target)!=expected:
            raise ValueError('Parent/parser custody drift: '+target)
        checked[target]=expected
    provenance=json.loads((out/'colo1-provenance-receipt-v1.json').read_bytes())
    freeze=json.loads((out/'colo1-provenance-freeze-v1.json').read_bytes())
    source=ROOT/'tmp/colo1-annotation-provenance-v1/Rscript_final.R'
    if verify_code(source.read_bytes(),provenance['selected_blob'],freeze['expected_crc32'])!=provenance['verified_code']:
        raise ValueError('Pinned source identity drift')
    selected,view=metadata_view(source.read_text(encoding='utf8'))
    if [n for n,_ in selected]!=provenance['selected_line_numbers']:
        raise ValueError('Metadata source projection drift')
    import hashlib
    if hashlib.sha256(view.encode()).hexdigest()!=provenance['metadata_view_sha256']:
        raise ValueError('Metadata view hash drift')
    receipt=json.loads((out/'colo1-native-receipt-v5.json').read_bytes())
    if receipt['status']!='native_metadata_described' or receipt['source_qualified_for_response_ingestion']:
        raise ValueError('Native descriptive/source-status drift')
    summary=receipt['native_metadata_summary'];rows=summary['annotation_rows']
    if rows!=78920 or summary['distinct_complete_construct_ids']!=rows or summary['duplicate_construct_id_rows']:
        raise ValueError('Native construct counts drift')
    for name in ('native_target_label_pairs','native_guide_label_pairs'):
        pairs=summary[name]
        if pairs['complete_rows']+pairs['incomplete_rows']!=rows:
            raise ValueError('Native pair completeness mismatch')
        for mode in ('ordered','canonical'):
            reconcile_multiplicities(pairs[mode])
            if pairs[mode]['rows']!=pairs['complete_rows']:
                raise ValueError('Native pair multiplicity mismatch')
    graph=summary['native_target_label_graph']
    degree={int(k):v for k,v in graph['distinct_other_neighbor_degree_histogram'].items()}
    if (sum(graph['component_sizes'])!=graph['native_target_label_nodes'] or
        len(graph['component_sizes'])!=graph['components'] or
        sum(degree.values())!=graph['native_target_label_nodes'] or
        sum(k*v for k,v in degree.items())!=2*graph['nonself_unique_edges']):
        raise ValueError('Graph handshake/component reconciliation failed')
    for category in summary['categorical_annotations'].values():
        reconcile_multiplicities(category['frequencies'])
        if category['frequencies']['rows']!=rows:
            raise ValueError('Category count reconciliation failed')
    if any(summary[k] for k in ('biological_label_identity_established','guide_sequence_identity_established',
                                'actual_response_eligibility_established')):
        raise ValueError('Unsupported biological identity claim')
    result=dict(schema='opal-colo1-native-custody-v5',checked_files=len(checked),
                source_code_identity_verified=True,metadata_source_projection_verified=True,
                native_count_and_graph_invariants_verified=True,new_requests=0,
                source_data_deserializations=0,source_qualified=False,fits=0,scores=0)
    with args.output.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
