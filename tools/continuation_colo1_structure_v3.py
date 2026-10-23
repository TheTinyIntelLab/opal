"""Structural-only successor: preserve partial schema, never aggregate values."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.continuation_colo1_schema_v2 import (
    SAFE_KINDS, attributes, deref, kind, parse_verified, schema_gate, sha, strings,
)


def structural_counts(root):
    seen, counts, pending = set(), Counter(), [root]
    while pending:
        obj = pending.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if len(seen) > 2_000_000:
            raise ValueError('node_cap')
        name = kind(obj)
        counts[name] += 1
        pending.extend((obj.attributes, obj.tag, obj.referenced_object))
        if name in ('STR','VEC','LIST'):
            pending.extend(obj.value)
        elif name == 'SYM':
            pending.append(obj.value)
        # No traversal, evaluation or expansion of opaque/executable values.
    return dict(sorted(counts.items()))


def inspect_structure(root):
    result = {'field_values_converted':False, 'field_values_aggregated':False}
    operations = []
    try:
        root = deref(root)
        result['root_kind'] = kind(root)
        operations.append('root_kind')
        attrs = attributes(root)
        result['root_attribute_names'] = sorted(attrs)
        operations.append('root_attributes')
        result['root_classes'] = strings(attrs['class']) if 'class' in attrs else []
        operations.append('root_classes')
        result['column_names'] = strings(attrs['names']) if 'names' in attrs else []
        operations.append('column_names')
        if kind(root) == 'VEC':
            columns = [deref(c) for c in root.value]
            result['column_kinds'] = [kind(c) for c in columns]
            result['column_lengths'] = [len(c.value) if kind(c) in ('STR','INT','REAL','LGL')
                                        else -1 for c in columns]
            result['schema_gate'] = schema_gate(result['column_names'], result['column_lengths'],
                                                result['column_kinds'], result['root_classes'])
            operations.append('column_structure')
    except Exception as error:
        result['schema_inspection_error_type'] = type(error).__name__
    try:
        counts = structural_counts(root)
        result['serialized_node_kinds'] = counts
        result['unsupported_kinds'] = sorted(set(counts)-SAFE_KINDS)
    except Exception as error:
        result['node_inspection_error_type'] = type(error).__name__
    result['completed_structural_operations'] = operations
    return result


def main():
    out = ROOT/'results/research-continuation'
    output = out/'colo1-structure-receipt-v3.json'
    if output.exists():
        raise FileExistsError(output)
    binding_path = out/'colo1-structure-code-binding-v3.json'
    binding = json.loads(binding_path.read_bytes())
    freeze_path = out/'colo1-structure-freeze-v3.json'
    if sha(freeze_path) != binding['freeze_sha256']:
        raise ValueError('Structural freeze drift')
    freeze = json.loads(freeze_path.read_bytes())
    parent = json.loads((out/'colo1-schema-freeze-v1.json').read_bytes())
    manifest = json.loads((out/'colo1-parser-install-manifest-v1.json').read_bytes())
    for path, expected in {**freeze['repo_sha256'], **parent['repo_sha256'],
                           **parent['local_input_sha256'], **binding['code_sha256'],
                           **manifest['installed_sha256']}.items():
        if sha(ROOT/path) != expected:
            raise ValueError('Structural prerequisite drift: '+path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    sys.path.insert(0, manifest['target'])
    import rdata
    source = 'tmp/colo1-annotation-acquisition-v2/IDs_colo1.rds'
    parsed, size = parse_verified((ROOT/source).read_bytes(), parent['local_input_sha256'][source],
                                  rdata.parser.parse_data)
    result = dict(schema='opal-colo1-structure-receipt-v3',
                  freeze_sha256=sha(freeze_path), code_binding_sha256=sha(binding_path),
                  annotation_sha256=sha(ROOT/source), decompressed_bytes=size,
                  structure=inspect_structure(parsed.object), source_gets=0,
                  response_objects_opened=0, fits=0, scores=0,
                  source_qualified_for_response_ingestion=False, independent_review=False)
    with output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(receipt=str(output), metadata_values_aggregated=False)))


if __name__ == '__main__':
    main()
