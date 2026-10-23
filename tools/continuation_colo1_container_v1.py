"""Validate the observed list-shaped file identity, then inspect one JSON catalog."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import continuation_colo1_validator_v2 as diagnostic

OUT = ROOT / 'results/research-continuation'
URL = diagnostic.CONTENT.removesuffix('/content') + '/container'
EXPECTED = dict(key=diagnostic.KEY, file_id='ef21b1fb-f63b-4b48-b3a9-c833ed5ce093',
                version_id='9de8c0fe-5d01-49ab-b7d3-6cd575712816',
                bucket_id='e1720a6c-9860-4fb2-b550-d38daffa1838',
                size=89422914, checksum='md5:174cd1597901fac463a092dbbbf4c71d',
                status='completed', mimetype='application/zip')


def project_identity(value):
    if not isinstance(value, dict) or not isinstance(value.get('entries'), list):
        raise ValueError('Actual list-shaped technical catalog required')
    entries, seen = {}, set()
    for entry in value['entries']:
        if not isinstance(entry, dict) or not isinstance(entry.get('key'), str):
            raise ValueError('Malformed technical file entry')
        key = entry['key']
        if key in seen:
            raise ValueError('Duplicate technical file key')
        seen.add(key)
        entries[key] = entry
    entry = entries.get(diagnostic.KEY)
    if entry is None:
        raise ValueError('Expected file identity missing')
    for key, expected in EXPECTED.items():
        if type(entry.get(key)) is not type(expected) or entry[key] != expected:
            raise ValueError('File identity mismatch: ' + key)
    if not isinstance(entry.get('links'), dict) or entry['links'].get('container') != URL:
        raise ValueError('Container endpoint substitution refused')
    return dict(technical_fields={key: entry[key] for key in EXPECTED}, container_url=URL,
                created=entry.get('created'), updated=entry.get('updated'))


def project_container(value):
    if not isinstance(value, dict):
        return dict(actual_root_type=type(value).__name__, recognized_entry_catalog=False)
    result = dict(actual_root_type='object', top_level_fields=sorted(value), entries=[])
    entries = value.get('entries')
    result['entries_type'] = type(entries).__name__
    if isinstance(entries, list):
        items = [(None, row) for row in entries]
    elif isinstance(entries, dict):
        items = list(entries.items())
    else:
        result['recognized_entry_catalog'] = False
        return result
    result['recognized_entry_catalog'] = True
    for key, entry in items:
        row = dict(catalog_key=key) if key is not None else {}
        if not isinstance(entry, dict):
            row['actual_entry_type'] = type(entry).__name__
        else:
            row['technical_field_names'] = sorted(entry)
            for field in ('key', 'path', 'name', 'size', 'compressed_size', 'crc', 'crc32', 'type', 'is_dir', 'mimetype'):
                if field in entry and isinstance(entry[field], (str, int, float, bool, type(None))):
                    row[field] = entry[field]
            if isinstance(entry.get('links'), dict):
                row['redacted_links'] = {k: diagnostic.safe_link(v) for k, v in entry['links'].items()}
        result['entries'].append(row)
    return result


def main():
    output = OUT / 'colo1-container-receipt-v1.json'
    if output.exists():
        raise FileExistsError(output)
    freeze_path = OUT / 'colo1-container-freeze-v1.json'
    binding_path = OUT / 'colo1-container-code-binding-v1.json'
    freeze, binding = json.loads(freeze_path.read_bytes()), json.loads(binding_path.read_bytes())
    if diagnostic.sha(freeze_path) != binding['freeze_sha256']:
        raise ValueError('Container freeze drift')
    checks = {**freeze['repo_sha256'], **freeze['local_input_sha256'], **binding['code_sha256']}
    for path, expected in checks.items():
        if diagnostic.sha(ROOT / path) != expected:
            raise ValueError('Container input drift: ' + path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    raw = (ROOT/'tmp/colo1-validator-v2/file-catalog.json').read_bytes()
    identity = project_identity(json.loads(raw, object_pairs_hook=diagnostic.unique_object))
    quarantine = ROOT / 'tmp/colo1-container-v1'
    quarantine.mkdir(exist_ok=False)
    result = dict(schema='opal-colo1-container-catalog-v1', freeze_sha256=diagnostic.sha(freeze_path),
                  code_binding_sha256=diagnostic.sha(binding_path), file_identity=identity,
                  url=URL, technical_catalog_body_bytes_read=0, archive_body_bytes_read=0,
                  member_bodies_opened=0, response_tables_or_headers_opened=False,
                  source_qualified=False, fits=0, scores=0)
    started = time.perf_counter()
    response = None
    try:
        request = urllib.request.Request(URL, headers={'User-Agent': 'Opal-server-container-catalog/1.0',
                                                       'Accept': 'application/json', 'Accept-Encoding': 'identity'})
        try:
            response = urllib.request.build_opener(diagnostic.NoRedirect()).open(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        result.update(status=response.status, mime=response.headers.get_content_type())
        if response.status != 200 or response.headers.get_content_type() != 'application/json' or response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise ValueError('Container technical JSON envelope refused')
        raw = response.read(2 * 1024**2 + 1)
        result['technical_catalog_body_bytes_read'] = len(raw)
        if len(raw) > 2 * 1024**2:
            raise ValueError('Container catalog byte cap')
        value = json.loads(raw, object_pairs_hook=diagnostic.unique_object)
        result.update(raw_sha256=hashlib.sha256(raw).hexdigest(), selected_catalog=project_container(value))
        with (quarantine/'container.json').open('xb') as stream:
            stream.write(raw)
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if response is not None:
            response.close()
    result['elapsed_seconds'] = time.perf_counter() - started
    for path, expected in checks.items():
        if diagnostic.sha(ROOT / path) != expected:
            raise ValueError('Post-container drift: ' + path)
    with output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'mime', 'technical_catalog_body_bytes_read', 'archive_body_bytes_read', 'error') if k in result}))


if __name__ == '__main__':
    main()
