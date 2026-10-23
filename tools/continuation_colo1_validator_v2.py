"""Inspect technical catalog identity and zero-body HTTP validator metadata."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-continuation'
KEY = 'ibarrioh/DualGuide_COLO1-Release2.zip'
CATALOG = 'https://zenodo.org/api/records/17191951/files'
CONTENT = CATALOG + '/' + KEY + '/content'
FIELDS = ('ETag', 'Last-Modified', 'Accept-Ranges', 'Content-Type', 'Content-Length',
          'Content-Range', 'Content-Encoding', 'Content-MD5', 'Content-Digest',
          'Repr-Digest', 'Digest', 'X-Checksum-MD5', 'OC-Checksum', 'Cache-Control', 'Vary')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate catalog key')
        result[key] = value
    return result


def etag_class(value):
    if value is None:
        return 'absent'
    if re.fullmatch(r'"[\x21\x23-\x7e]{1,128}"', value):
        return 'accepted-by-v1-parser'
    if re.fullmatch(r'W/"[\x21\x23-\x7e]{1,128}"', value):
        return 'weak'
    return 'malformed-or-unsupported-by-v1-parser'


def envelope(response):
    fields = {}
    for name in FIELDS:
        value = response.headers.get(name)
        if value is not None:
            fields[name] = dict(value=value[:256], truncated=len(value) > 256)
    return dict(status=int(response.status), transport_fields=fields,
                etag_class=etag_class(response.headers.get('ETag')), body_bytes_read=0)


def safe_link(value):
    if not isinstance(value, str):
        return None
    parsed = urllib.parse.urlsplit(value)
    return dict(scheme=parsed.scheme, host=parsed.hostname, path=parsed.path,
                query_redacted=bool(parsed.query), fragment_redacted=bool(parsed.fragment))


def select_catalog(value):
    if not isinstance(value, dict):
        raise ValueError('Expected technical file catalog object')
    entries = value.get('entries')
    result = dict(top_level_fields=sorted(value), entries_is_mapping=isinstance(entries, dict))
    if not isinstance(entries, dict):
        return result
    result['catalog_keys'] = sorted(entries)
    entry = entries.get(KEY)
    if not isinstance(entry, dict):
        result['expected_key_found'] = False
        return result
    result['expected_key_found'] = True
    allowed = ('key', 'id', 'file_id', 'size', 'checksum', 'status', 'storage_class', 'created', 'updated')
    result['technical_file_fields'] = {k: entry[k] for k in allowed if k in entry and isinstance(entry[k], (str, int, float, bool, type(None)))}
    links = entry.get('links')
    if isinstance(links, dict):
        result['redacted_links'] = {k: safe_link(v) for k, v in links.items()}
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def fetch(kind):
    url = CONTENT if kind == 'range-headers' else CATALOG
    result = dict(kind=kind, url=url, observed_utc=datetime.now(timezone.utc).isoformat(), body_bytes_read=0)
    started = time.perf_counter()
    response = None
    try:
        headers = {'User-Agent': 'Opal-metadata-validator-diagnostic/2.0', 'Accept-Encoding': 'identity'}
        if kind == 'range-headers':
            headers['Range'] = 'bytes=89422892-89422913'
        else:
            headers['Accept'] = 'application/json'
        try:
            response = urllib.request.build_opener(NoRedirect()).open(urllib.request.Request(url, headers=headers), timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        result.update(envelope(response))
        if kind == 'file-catalog':
            if response.status != 200 or response.headers.get_content_type() != 'application/json':
                raise ValueError('Technical JSON catalog envelope refused')
            raw = response.read(2 * 1024**2 + 1)
            result['body_bytes_read'] = len(raw)
            if len(raw) > 2 * 1024**2:
                raise ValueError('Technical catalog byte ceiling')
            result.update(raw_sha256=hashlib.sha256(raw).hexdigest(),
                          selected_catalog=select_catalog(json.loads(raw, object_pairs_hook=unique_object)))
            path = ROOT / 'tmp/colo1-validator-v2/file-catalog.json'
            with path.open('xb') as stream:
                stream.write(raw)
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if response is not None:
            response.close()
    result['elapsed_seconds'] = time.perf_counter() - started
    return result


def main():
    output = OUT / 'colo1-validator-receipt-v2.json'
    if output.exists():
        raise FileExistsError(output)
    freeze_path = OUT / 'colo1-validator-freeze-v2.json'
    binding_path = OUT / 'colo1-validator-code-binding-v2.json'
    freeze, binding = json.loads(freeze_path.read_bytes()), json.loads(binding_path.read_bytes())
    if sha(freeze_path) != binding['freeze_sha256']:
        raise ValueError('Validator freeze drift')
    checks = {**freeze['repo_sha256'], **freeze['external_sha256'], **binding['code_sha256']}
    for path, expected in checks.items():
        if sha(ROOT / path) != expected:
            raise ValueError('Validator input drift: ' + path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    (ROOT / 'tmp/colo1-validator-v2').mkdir(exist_ok=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        requests = list(pool.map(fetch, ('file-catalog', 'range-headers')))
    for path, expected in checks.items():
        if sha(ROOT / path) != expected:
            raise ValueError('Post-validator drift: ' + path)
    result = dict(schema='opal-colo1-validator-diagnostic-v2', freeze_sha256=sha(freeze_path),
                  code_binding_sha256=sha(binding_path), requests=requests,
                  archive_body_bytes_read=requests[1]['body_bytes_read'], current_observation_not_v1_reconstruction=True,
                  source_qualified=False, response_tables_or_headers_opened=False, fits=0, scores=0)
    with output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps([dict(kind=r['kind'], status=r.get('status'), etag_class=r.get('etag_class'), error=r.get('error')) for r in requests]))


if __name__ == '__main__':
    main()
