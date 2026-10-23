"""Bounded metadata stream plus unchanged whole-object cryptographic identity."""
import io
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import continuation_colo1_annotation_acquisition_v1 as previous

OUT = ROOT / 'results/research-continuation'


def consume(response, expected_size, audit, sink):
    if response.status != 200:
        raise ValueError('Annotation HTTP status refused')
    if response.headers.get_content_type() not in {'application/octet-stream', 'application/x-r-rds', 'application/rds', 'binary/octet-stream'}:
        raise ValueError('Annotation binary MIME refused')
    if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise ValueError('Annotation transport encoding refused')
    length = response.headers.get('Content-Length')
    if length is not None and length != str(expected_size):
        raise ValueError('Contradictory Content-Length refused')
    remaining = expected_size + 1
    while remaining:
        block = response.read(min(65536, remaining))
        if not block:
            break
        sink.write(block)
        audit['member_bytes_read'] += len(block)
        remaining -= len(block)
        if remaining < 0:
            raise ValueError('Reader exceeded explicit byte request')


def verified(values, size, blob, crc):
    return values['bytes'] == size and values['git_blob_sha1'] == blob and values['crc32'] == crc


def main():
    diagnostic = previous.container.diagnostic
    output = OUT / 'colo1-annotation-acquisition-receipt-v2.json'
    if output.exists():
        raise FileExistsError(output)
    freeze_path = OUT / 'colo1-annotation-acquisition-freeze-v2.json'
    binding_path = OUT / 'colo1-annotation-acquisition-code-binding-v2.json'
    freeze, binding = json.loads(freeze_path.read_bytes()), json.loads(binding_path.read_bytes())
    if diagnostic.sha(freeze_path) != binding['freeze_sha256']:
        raise ValueError('Annotation v2 freeze drift')
    checks = {**freeze['repo_sha256'], **freeze['local_input_sha256'], **binding['code_sha256']}
    for path, expected in checks.items():
        if diagnostic.sha(ROOT / path) != expected:
            raise ValueError('Annotation v2 input drift: ' + path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    catalog = json.loads((ROOT/'tmp/colo1-container-v1/container.json').read_bytes(), object_pairs_hook=diagnostic.unique_object)
    member = previous.select_member(catalog)
    if member['size'] != freeze['expected_bytes'] or member['crc'] != freeze['expected_crc32']:
        raise ValueError('Annotation v2 member identity drift')
    quarantine = ROOT / 'tmp/colo1-annotation-acquisition-v2'
    quarantine.mkdir(exist_ok=False)
    result = dict(schema='opal-colo1-annotation-acquisition-v2', freeze_sha256=diagnostic.sha(freeze_path),
                  code_binding_sha256=diagnostic.sha(binding_path), member=member, member_bytes_read=0,
                  metadata_identity_verified=False, deserialization_performed=False,
                  response_tables_or_headers_opened=False, source_qualified_for_response_ingestion=False,
                  whole_archive_downloaded=False, fits=0, scores=0)
    begun = time.perf_counter()
    response = None
    sink = io.BytesIO()
    try:
        request = urllib.request.Request(previous.URL, headers={'User-Agent': 'Opal-bounded-annotation-identity/2.0', 'Accept-Encoding': 'identity'})
        try:
            response = urllib.request.build_opener(diagnostic.NoRedirect()).open(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        result.update(status=response.status, mime=response.headers.get_content_type(), content_length=response.headers.get('Content-Length'))
        consume(response, freeze['expected_bytes'], result, sink)
        result.update(previous.identities(sink.getvalue()))
        if not verified(result, freeze['expected_bytes'], freeze['expected_git_blob_sha1'], freeze['expected_crc32']):
            raise ValueError('Annotation v2 byte identity mismatch')
        result['metadata_identity_verified'] = True
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if response is not None:
            response.close()
    if sink.tell():
        name = 'IDs_colo1.rds' if result['metadata_identity_verified'] else 'unverified.bin'
        with (quarantine/name).open('xb') as stream:
            stream.write(sink.getvalue())
        result['quarantine_path'] = str(quarantine/name)
        result.update(previous.identities(sink.getvalue()))
    result['elapsed_seconds'] = time.perf_counter() - begun
    for path, expected in checks.items():
        if diagnostic.sha(ROOT / path) != expected:
            raise ValueError('Post-annotation v2 drift: ' + path)
    with output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'member_bytes_read', 'metadata_identity_verified', 'error') if k in result}))


if __name__ == '__main__':
    main()
