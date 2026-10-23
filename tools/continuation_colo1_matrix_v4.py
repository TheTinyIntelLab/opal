"""Versioned character-matrix adapter with the original exact field restrictions."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.continuation_colo1_schema_v2 import (
    ALLOWED, OUTCOME, attributes, deref, graph_kinds, kind, parse_verified,
    sha, strings, summarize, text,
)


def matrix_gate(shape, names, total):
    issues = []
    if (len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
                              for n in shape)):
        return dict(passed=False, issues=['invalid_dimensions'])
    rows, columns = shape
    if rows > 150000 or columns > 64:
        issues.append('matrix_size_cap')
    if rows*columns != total:
        issues.append('inconsistent_storage_length')
    if len(names) != columns:
        issues.append('missing_or_inconsistent_column_names')
    if len(names) != len(set(names)):
        issues.append('duplicate_field_names')
    if any(OUTCOME.search(n) for n in names):
        issues.append('outcome_like_field')
    if any(n not in ALLOWED for n in names):
        issues.append('unknown_field')
    return dict(passed=not issues, issues=issues,
                unknown_fields=[n for n in names if n not in ALLOWED])


def inspect_matrix(root):
    root = deref(root)
    if kind(root) != 'STR':
        raise ValueError('Character matrix root required')
    attrs = attributes(root)
    dims = deref(attrs['dim'])
    if kind(dims) != 'INT' or len(dims.value) != 2:
        raise ValueError('Two integer dimensions required')
    shape = [int(v) for v in dims.value]
    dimnames = deref(attrs['dimnames'])
    if kind(dimnames) != 'VEC' or len(dimnames.value) != 2:
        raise ValueError('Two dimension-label components required')
    column_labels = deref(dimnames.value[1])
    names = [] if kind(column_labels) in ('NIL','NILVALUE') else strings(column_labels)
    gate = matrix_gate(shape, names, len(root.value))
    schema = dict(root_kind='STR', dimensions=shape, column_names=names,
                  storage_length=len(root.value), storage_order='R column-major',
                  serialized_node_kinds=graph_kinds(root), gate=gate)
    return schema


def decode_columns(root, schema):
    # Independently rerun the gate so a forged passed flag cannot enable decoding.
    root = deref(root)
    gate = matrix_gate(schema['dimensions'], schema['column_names'], len(root.value))
    if not gate['passed']:
        raise ValueError('Matrix field gate refused before conversion')
    rows = schema['dimensions'][0]
    output = {}
    for index, name in enumerate(schema['column_names']):
        col = root.value[index*rows:(index+1)*rows]
        output[name] = [None if deref(v).value is None else text(v) for v in col]
    return output


def main():
    out = ROOT/'results/research-continuation'
    output = out/'colo1-matrix-receipt-v4.json'
    if output.exists():
        raise FileExistsError(output)
    binding_path = out/'colo1-matrix-code-binding-v4.json'
    binding = json.loads(binding_path.read_bytes())
    freeze_path = out/'colo1-matrix-freeze-v4.json'
    if sha(freeze_path) != binding['freeze_sha256']:
        raise ValueError('Matrix freeze drift')
    freeze = json.loads(freeze_path.read_bytes())
    parent = json.loads((out/'colo1-schema-freeze-v1.json').read_bytes())
    manifest = json.loads((out/'colo1-parser-install-manifest-v1.json').read_bytes())
    for path, expected in {**freeze['repo_sha256'], **parent['repo_sha256'],
                           **parent['local_input_sha256'], **binding['code_sha256'],
                           **manifest['installed_sha256']}.items():
        if sha(ROOT/path) != expected:
            raise ValueError('Matrix prerequisite drift: '+path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    sys.path.insert(0, manifest['target'])
    import rdata
    source = 'tmp/colo1-annotation-acquisition-v2/IDs_colo1.rds'
    result = dict(schema='opal-colo1-matrix-receipt-v4',
                  freeze_sha256=sha(freeze_path), code_binding_sha256=sha(binding_path),
                  annotation_sha256=sha(ROOT/source), source_gets=0,
                  response_objects_opened=0, fits=0, scores=0, independent_review=False)
    stage = 'hash_decompress_parse'
    try:
        parsed, size = parse_verified((ROOT/source).read_bytes(), parent['local_input_sha256'][source],
                                      rdata.parser.parse_data)
        result['decompressed_bytes'] = size
        stage = 'matrix_structure'
        schema = inspect_matrix(parsed.object)
        result['observed_schema'] = schema
        if schema['gate']['passed']:
            stage = 'permitted_metadata_aggregation'
            result['metadata_summary'] = summarize(decode_columns(parsed.object, schema))
            result['status'] = 'metadata_matrix_audited'
        else:
            result['status'] = 'matrix_gate_refused_aggregation'
    except Exception as error:
        result.update(status='audit_refused', error_type=type(error).__name__, refusal_stage=stage)
    result['source_qualified_for_response_ingestion'] = False
    with output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], receipt=str(output))))


if __name__ == '__main__':
    main()
