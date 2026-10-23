"""Read-only local custody check. Makes no requests and deserializes no source data."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    out = ROOT/'results/research-continuation'
    prefixes = ('colo1-validator-', 'colo1-container-', 'colo1-annotation-acquisition-',
                'colo1-parser-', 'colo1-schema-', 'colo1-structure-', 'colo1-matrix-')
    files = [p for p in sorted(out.glob('*.json')) if p.name.startswith(prefixes)]
    checked = {}
    commands = []
    for path in files:
        item = json.loads(path.read_bytes())
        for key in ('repo_sha256','local_input_sha256','external_sha256','code_sha256','installed_sha256'):
            for target, expected in item.get(key,{}).items():
                actual = sha(ROOT/target)
                if actual != expected:
                    raise ValueError('Custody drift: '+target)
                checked[target] = actual
        if 'returncode' in item:
            for target, expected in item['bound_sha256'].items():
                if sha(target) != expected:
                    raise ValueError('Command binding drift: '+target)
            if sha(item['log']) != item['log_sha256']:
                raise ValueError('Command log drift')
            if item['bound_drift'] or item['stopped']:
                raise ValueError('Command drift or cap interruption')
            expected_failed = path.name == 'colo1-schema-tests-v1.json'
            if item['passed'] == expected_failed or item['returncode'] != (1 if expected_failed else 0):
                raise ValueError('Unexpected command result: '+path.name)
            commands.append(path.name)
    known_hashes = {sha(p) for p in files} | set(checked.values())
    for path in files:
        item = json.loads(path.read_bytes())
        for key in ('freeze_sha256','parser_manifest_sha256','pins_sha256',
                    'setup_code_sha256','code_binding_sha256'):
            if key in item and item[key] not in known_hashes:
                raise ValueError('Unresolved artifact hash: '+path.name+'/'+key)
    acquisition = json.loads((out/'colo1-annotation-acquisition-receipt-v2.json').read_bytes())
    raw = (ROOT/acquisition['quarantine_path']).read_bytes()
    import zlib
    identities = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                      git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
                      crc32=zlib.crc32(raw)&0xffffffff)
    for key, value in identities.items():
        if acquisition[key] != value:
            raise ValueError('Acquired annotation identity drift: '+key)
    final = json.loads((out/'colo1-matrix-receipt-v4.json').read_bytes())
    schema = final['observed_schema']
    if (schema['dimensions'] != [78920,7] or schema['storage_length'] != 552440 or
        schema['column_names'] != ['ID','Note','MyNote','sgRNA1_ID','sgRNA2_ID','Gene1','Gene2'] or
        schema['gate']['passed'] or schema['gate']['issues'] != ['unknown_field'] or
        final['status'] != 'matrix_gate_refused_aggregation' or 'metadata_summary' in final or
        final['source_qualified_for_response_ingestion'] or final['fits'] or final['scores']):
        raise ValueError('Final schema/refusal drift')
    manifest = json.loads((out/'colo1-parser-install-manifest-v1.json').read_bytes())
    for name, expected in manifest['reused_host_dependencies'].items():
        if version(name) != expected:
            raise ValueError('Reused dependency version drift: '+name)
    result = dict(schema='opal-colo1-annotation-custody-v1', checked_files=len(checked),
                  verified_command_receipts=commands, annotation_identities=identities,
                  retained_expected_fixture_failure='colo1-schema-tests-v1.json',
                  actual_matrix_gate_passed=False, metadata_field_values_aggregated=False,
                  source_qualified=False, new_requests=0, source_deserializations=0, fits=0, scores=0)
    with args.output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
