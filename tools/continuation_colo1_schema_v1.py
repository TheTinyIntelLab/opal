"""Nonexecuting audit of one preclassified, hash-pinned annotation object."""
import bz2
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import lzma
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-continuation'
FIELD_SUFFIXES = ('WGE_ID', 'WGE_Sequence', 'Library', 'Approved_Symbol',
                  'Off_Target', 'Chr', 'Start', 'End', 'Strand', 'Confidence')
ALLOWED = frozenset(('ID', 'Notes', 'Notes1', 'Notes2', 'Scaffold', 'Linker',
                     'tRNA', 'Oligo_Sequence')) | frozenset(
    arm + suffix for arm in ('sgRNA1_', 'sgRNA2_') for suffix in FIELD_SUFFIXES)
OUTCOME = re.compile(r'lfc|fitness|count|score|pvalue|fdr|effect|response|hit|'
                     r'essentiality|expression|activity|qc|viability|phenotype', re.I)
SAFE_KINDS = frozenset(('NIL', 'NILVALUE', 'SYM', 'LIST', 'CHAR', 'LGL',
                        'INT', 'REAL', 'STR', 'VEC', 'REF'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True,
                     separators=(',', ':'), allow_nan=False).encode('utf8')
    return hashlib.sha256(raw).hexdigest()


def decompress(raw, cap=64 * 1024**2):
    """Bound expansion before the third-party parser sees any input."""
    constructor = (gzip.GzipFile if raw.startswith(b'\x1f\x8b') else
                   bz2.BZ2File if raw.startswith(b'BZh') else
                   lzma.LZMAFile if raw.startswith(b'\xfd7zXZ\0') else None)
    if constructor:
        with constructor(io.BytesIO(raw), 'rb') as stream:
            result = stream.read(cap + 1)
    else:
        result = raw
    if len(result) > cap:
        raise ValueError('Decompressed byte cap')
    if not result.startswith((b'X\n', b'B\n', b'A\n')):
        raise ValueError('Only a single-layer RDS serialization is supported')
    return result


def parse_verified(raw, expected, parser):
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Annotation byte identity mismatch before parsing')
    unpacked = decompress(raw)
    parsed = parser(unpacked, expand_altrep=False, extension='.rds')
    return parsed, len(unpacked)


def kind(obj):
    return obj.info.type.name


def deref(obj):
    seen = set()
    while obj is not None and kind(obj) == 'REF':
        if id(obj) in seen:
            raise ValueError('Cyclic R reference')
        seen.add(id(obj))
        obj = obj.referenced_object
    if obj is None:
        raise ValueError('Missing R object/reference')
    return obj


def text(obj):
    obj = deref(obj)
    if kind(obj) == 'SYM':
        return text(obj.value)
    if kind(obj) != 'CHAR' or not isinstance(obj.value, bytes):
        raise ValueError('Expected nonmissing metadata text')
    # ASCII names need no guessed locale. Actual field values may be UTF8.
    return obj.value.decode('utf8', errors='strict')


def attributes(obj):
    current = obj.attributes
    output = {}
    seen = set()
    while current is not None:
        current = deref(current)
        if kind(current) in ('NIL', 'NILVALUE'):
            break
        if id(current) in seen or kind(current) != 'LIST':
            raise ValueError('Unsupported/cyclic R attribute list')
        seen.add(id(current))
        name = text(current.tag)
        if name in output:
            raise ValueError('Duplicate R attribute')
        output[name] = current.value[0]
        current = current.value[1]
    return output


def strings(obj):
    obj = deref(obj)
    if kind(obj) != 'STR':
        raise ValueError('Expected R character vector')
    return [text(v) for v in obj.value]


def graph_kinds(root):
    """Walk structural nodes without converting or printing column values."""
    seen, counts, pending = set(), Counter(), [root]
    while pending:
        obj = pending.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if len(seen) > 2_000_000:
            raise ValueError('R node cap')
        name = kind(obj)
        counts[name] += 1
        if name not in SAFE_KINDS:
            raise ValueError('Unsupported serialized R kind: ' + name)
        pending.extend((obj.attributes, obj.tag, obj.referenced_object))
        if name in ('STR', 'VEC', 'LIST'):
            pending.extend(obj.value)
        elif name == 'SYM':
            pending.append(obj.value)
    return dict(sorted(counts.items()))


def schema_gate(names, lengths, kinds, classes):
    issues = []
    if classes != ['data.frame']:
        issues.append('unsupported_root_class')
    if len(names) != len(set(names)):
        issues.append('duplicate_field_names')
    if not names or len(names) != len(lengths) or len(names) != len(kinds):
        issues.append('column_structure')
    if len(set(lengths)) != 1:
        issues.append('inconsistent_column_lengths')
    if any(n < 0 or n > 150000 for n in lengths):
        issues.append('annotation_row_cap')
    if any(OUTCOME.search(n) for n in names):
        issues.append('outcome_like_field')
    if any(n not in ALLOWED for n in names):
        issues.append('unknown_field')
    if any(k not in ('STR', 'INT', 'REAL', 'LGL') for k in kinds):
        issues.append('unsupported_column_type')
    return dict(passed=not issues, issues=issues,
                unknown_fields=[n for n in names if n not in ALLOWED])


def frame_schema(root):
    root = deref(root)
    if kind(root) != 'VEC':
        raise ValueError('Root must be an R generic vector/data frame')
    attrs = attributes(root)
    classes = strings(attrs['class']) if 'class' in attrs else []
    names = strings(attrs['names']) if 'names' in attrs else []
    columns = [deref(c) for c in root.value]
    kinds = [kind(c) for c in columns]
    lengths = [len(c.value) if kind(c) in ('STR', 'INT', 'REAL', 'LGL') else -1
               for c in columns]
    gate = schema_gate(names, lengths, kinds, classes)
    result = dict(root_kind=kind(root), root_classes=classes, column_names=names,
                  column_kinds=kinds, column_lengths=lengths, gate=gate)
    # No value conversion or aggregation before the full schema gate.
    result['serialized_node_kinds'] = graph_kinds(root)
    return result, columns


def values(column):
    column = deref(column)
    if kind(column) == 'STR':
        return [None if deref(v).value is None else text(v) for v in column.value]
    # Only primitive columns without class-specific conversions are supported.
    if attributes(column):
        raise ValueError('Attributed numeric/factor columns require a new contract')
    import numpy as np
    output = []
    for value in column.value:
        if np.ma.is_masked(value) or not np.isfinite(value):
            output.append(None)
        else:
            output.append(value.item() if hasattr(value, 'item') else value)
    return output


def summarize(table):
    sizes = {len(v) for v in table.values()}
    if len(sizes) != 1:
        raise ValueError('Inconsistent converted column lengths')
    rows = next(iter(sizes))
    result = dict(annotation_rows=rows,
                  missing_by_field={n:sum(v is None or v == '' for v in col)
                                    for n, col in table.items()},
                  ordered_metadata_sha256=digest(table))
    if 'ID' in table:
        ids = [v for v in table['ID'] if v is not None and v != '']
        duplicate_rows = len(ids) - len(set(ids))
        result.update(duplicate_construct_id_rows=duplicate_rows,
                      construct_id_gate_passed=duplicate_rows == 0 and len(ids) == rows)
    else:
        result.update(construct_id_gate_passed=False, construct_id_reason='missing_ID')
    checks = {}
    for arm in ('sgRNA1_', 'sgRNA2_'):
        fields = [arm+s for s in ('WGE_ID', 'WGE_Sequence', 'Approved_Symbol')]
        if all(f in table for f in fields):
            mapping = defaultdict(set)
            complete = 0
            for guide, sequence, target in zip(*(table[f] for f in fields)):
                if any(v is None or v == '' for v in (guide, sequence, target)):
                    continue
                complete += 1
                mapping[guide].add((sequence, target))
            checks[arm] = dict(complete_rows=complete, distinct_guide_ids=len(mapping),
                               conflicting_guide_ids=sum(len(v) > 1 for v in mapping.values()))
        else:
            checks[arm] = dict(checkable=False, missing_fields=[f for f in fields if f not in table])
    result['ordered_arm_consistency'] = checks
    for label, suffix in (('gene', 'Approved_Symbol'), ('guide', 'WGE_ID')):
        fields = [a+suffix for a in ('sgRNA1_', 'sgRNA2_')]
        if all(f in table for f in fields):
            pairs = [tuple(sorted((str(a), str(b)))) for a, b in zip(*(table[f] for f in fields))
                     if a is not None and b is not None and a != '' and b != '']
            result['canonical_'+label+'_pairs'] = len(set(pairs))
    result['constituent_single_control_status'] = (
        'Ordered guide fields alone cannot establish measured constituent singles, '
        'control roles, sample/time/culture joins or initiation independence.')
    return result


def main():
    output = OUT/'colo1-schema-receipt-v1.json'
    if output.exists():
        raise FileExistsError(output)
    binding_path = OUT/'colo1-schema-code-binding-v1.json'
    binding = json.loads(binding_path.read_bytes())
    freeze_path = OUT/'colo1-schema-freeze-v1.json'
    if sha(freeze_path) != binding['freeze_sha256']:
        raise ValueError('Schema freeze drift')
    freeze = json.loads(freeze_path.read_bytes())
    manifest_path = OUT/'colo1-parser-install-manifest-v1.json'
    if sha(manifest_path) != binding['parser_manifest_sha256']:
        raise ValueError('Parser manifest drift')
    manifest = json.loads(manifest_path.read_bytes())
    for path, expected in {**freeze['repo_sha256'], **freeze['local_input_sha256'],
                           **binding['code_sha256'], **manifest['installed_sha256']}.items():
        if sha(ROOT/path) != expected:
            raise ValueError('Schema audit input/code drift: '+path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    sys.path.insert(0, manifest['target'])
    import rdata
    source = ROOT/'tmp/colo1-annotation-acquisition-v2/IDs_colo1.rds'
    expected = freeze['local_input_sha256'][source.relative_to(ROOT).as_posix()]
    result = dict(schema='opal-colo1-annotation-schema-receipt-v1',
                  freeze_sha256=sha(freeze_path), code_binding_sha256=sha(binding_path),
                  annotation_sha256=expected, parser_packages=manifest['packages'],
                  response_objects_opened=0, fits=0, scores=0, independent_review=False)
    try:
        parsed, size = parse_verified(source.read_bytes(), expected, rdata.parser.parse_data)
        result['decompressed_bytes'] = size
        schema, columns = frame_schema(parsed.object)
        result['observed_schema'] = schema
        if schema['gate']['passed']:
            result['metadata_summary'] = summarize(dict(zip(schema['column_names'], map(values, columns))))
            result['status'] = 'metadata_schema_audited'
        else:
            result['status'] = 'schema_gate_refused_aggregation'
    except Exception as error:
        result['status'] = 'audit_refused'
        # Controlled error types/messages contain no object repr or metadata rows.
        result['error_type'] = type(error).__name__
    result['source_qualified_for_response_ingestion'] = False
    with output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], receipt=str(output))))


if __name__ == '__main__':
    main()
