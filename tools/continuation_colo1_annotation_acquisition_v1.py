"""Acquire one documented row-annotation member as opaque, hash-checked bytes."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import continuation_colo1_container_v1 as container

OUT = ROOT / 'results/research-continuation'
MEMBER = 'ibarrioh-DualGuide_COLO1-cfb5eb9/Analysis/ENCORE/input/IDs_colo1.rds'
URL = container.URL + '/' + MEMBER


def select_member(catalog):
    if catalog.get('truncated') is not False or type(catalog.get('total')) is not int:
        raise ValueError('Complete technical member catalog required')
    entries = catalog.get('entries')
    if not isinstance(entries, list) or len(entries) != catalog['total']:
        raise ValueError('Member catalog count mismatch')
    matches = [e for e in entries if isinstance(e, dict) and e.get('key') == MEMBER]
    if len(matches) != 1:
        raise ValueError('Unique exact annotation member required')
    entry = matches[0]
    if type(entry.get('size')) is not int or entry['size'] != 277295:
        raise ValueError('Annotation member size mismatch')
    if type(entry.get('crc')) is not int or not 0 <= entry['crc'] < 2**32:
        raise ValueError('Member CRC identity required')
    if not isinstance(entry.get('links'), dict) or entry['links'].get('content') != URL:
        raise ValueError('Annotation endpoint substitution refused')
    return dict(key=MEMBER, size=entry['size'], crc=entry['crc'], content_url=URL)


def consume(response, expected_size):
    if response.status != 200:
        raise ValueError('Annotation HTTP status refused')
    if response.headers.get_content_type() not in {'application/octet-stream', 'application/x-r-rds', 'application/rds', 'binary/octet-stream'}:
        raise ValueError('Annotation binary MIME refused')
    if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise ValueError('Annotation transport encoding refused')
    if response.headers.get('Content-Length') != str(expected_size):
        raise ValueError('Annotation exact Content-Length required')
    return response.read(expected_size+1)


def identities(raw):
    prefix = b'blob ' + str(len(raw)).encode() + b'\0'
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                git_blob_sha1=hashlib.sha1(prefix+raw).hexdigest(), crc32=zlib.crc32(raw))


def main():
    output = OUT / 'colo1-annotation-acquisition-receipt-v1.json'
    if output.exists():
        raise FileExistsError(output)
    freeze_path = OUT / 'colo1-annotation-acquisition-freeze-v1.json'
    binding_path = OUT / 'colo1-annotation-acquisition-code-binding-v1.json'
    diagnostic = container.diagnostic
    freeze, binding = json.loads(freeze_path.read_bytes()), json.loads(binding_path.read_bytes())
    if diagnostic.sha(freeze_path) != binding['freeze_sha256']:
        raise ValueError('Annotation acquisition freeze drift')
    checks = {**freeze['repo_sha256'], **freeze['local_input_sha256'], **binding['code_sha256']}
    for path, expected in checks.items():
        if diagnostic.sha(ROOT / path) != expected:
            raise ValueError('Annotation input drift: ' + path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    catalog = json.loads((ROOT/'tmp/colo1-container-v1/container.json').read_bytes(), object_pairs_hook=diagnostic.unique_object)
    member = select_member(catalog)
    quarantine = ROOT / 'tmp/colo1-annotation-acquisition-v1'
    quarantine.mkdir(exist_ok=False)
    result = dict(schema='opal-colo1-annotation-acquisition-v1', freeze_sha256=diagnostic.sha(freeze_path),
                  code_binding_sha256=diagnostic.sha(binding_path), member=member, member_bytes_read=0,
                  metadata_identity_verified=False, deserialization_performed=False,
                  response_tables_or_headers_opened=False, source_qualified_for_response_ingestion=False,
                  whole_archive_downloaded=False, fits=0, scores=0)
    started = time.perf_counter()
    response = None
    try:
        request = urllib.request.Request(URL, headers={'User-Agent': 'Opal-opaque-annotation-identity/1.0', 'Accept-Encoding': 'identity'})
        try:
            response = urllib.request.build_opener(diagnostic.NoRedirect()).open(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        result.update(status=response.status, mime=response.headers.get_content_type(),
                      content_length=response.headers.get('Content-Length'))
        raw = consume(response, freeze['expected_bytes'])
        result['member_bytes_read'] = len(raw)
        result.update(identities(raw))
        valid = len(raw) == freeze['expected_bytes'] and result['git_blob_sha1'] == freeze['expected_git_blob_sha1'] and result['crc32'] == member['crc']
        name = 'IDs_colo1.rds' if valid else 'unverified.bin'
        with (quarantine/name).open('xb') as stream:
            stream.write(raw)
        result['quarantine_path'] = str(quarantine/name)
        if not valid:
            raise ValueError('Annotation byte identity mismatch')
        result['metadata_identity_verified'] = True
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if response is not None:
            response.close()
    result['elapsed_seconds'] = time.perf_counter() - started
    for path, expected in checks.items():
        if diagnostic.sha(ROOT / path) != expected:
            raise ValueError('Post-annotation drift: ' + path)
    with output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'member_bytes_read', 'metadata_identity_verified', 'error') if k in result}))


if __name__ == '__main__':
    main()
