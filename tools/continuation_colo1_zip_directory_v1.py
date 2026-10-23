"""Read only two exact structural ZIP ranges; never open member payloads."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-continuation'
QUARANTINE = ROOT / 'tmp/colo1-zip-directory-v1'
MIMES = {'application/zip', 'application/octet-stream', 'application/x-zip-compressed', 'binary/octet-stream'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def consume_range(response, start, end, total, audit, expected_etag=None):
    """Validate the complete envelope before the first body read."""
    count = end - start + 1
    headers = response.headers
    etag = headers.get('ETag', '')
    if response.status != 206:
        raise ValueError('Exact partial-response status required')
    if headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
        raise ValueError('Exact Content-Range required')
    if headers.get('Content-Length') != str(count):
        raise ValueError('Exact Content-Length required')
    if headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise ValueError('Encoded transport refused')
    if headers.get('Content-Type', '').split(';')[0].strip().lower() not in MIMES:
        raise ValueError('ZIP range MIME refused')
    if re.fullmatch(r'"[\x21\x23-\x7e]{1,128}"', etag) is None:
        raise ValueError('Strong nonempty ETag required')
    if expected_etag is not None and etag != expected_etag:
        raise ValueError('ETag changed across ranges')
    chunks = []
    remaining = count
    while remaining:
        block = response.read(min(65536, remaining))
        audit['body_bytes_read'] += len(block)
        if not block or len(block) > remaining:
            raise ValueError('Truncated or oversized range body')
        chunks.append(block)
        remaining -= len(block)
    return b''.join(chunks), etag


def parse_eocd(raw, total):
    if len(raw) != 22:
        raise ValueError('EOCD length')
    signature, disk, directory_disk, disk_count, count, length, offset, comment = struct.unpack('<4s4H2IH', raw)
    if signature != b'PK\x05\x06' or disk != 0 or directory_disk != 0 or comment != 0:
        raise ValueError('Unsupported EOCD signature/disk/comment')
    if disk_count != count or not 1 <= count <= 2048:
        raise ValueError('EOCD entry count')
    if not 1 <= length <= 512 * 1024 or offset + length != total - 22:
        raise ValueError('EOCD directory bounds')
    return dict(entry_count=count, directory_bytes=length, directory_start=offset,
                directory_end=offset + length - 1)


def parse_directory(raw, descriptor):
    if len(raw) != descriptor['directory_bytes']:
        raise ValueError('Directory length mismatch')
    rows, seen = [], set()
    position = 0
    while position < len(raw):
        if len(raw) - position < 46:
            raise ValueError('Truncated central record')
        fields = struct.unpack_from('<4s6H3I5H2I', raw, position)
        signature, made, needed, flags, method, _, _, crc, compressed, expanded, namesize, extrasize, commentsize, disk, _, external, offset = fields
        if signature != b'PK\x01\x02' or needed > 20 or disk != 0 or flags & 1 or method not in {0, 8}:
            raise ValueError('Unsupported central record')
        if max(compressed, expanded, offset) == 0xffffffff or offset + 30 + compressed > descriptor['directory_start']:
            raise ValueError('Member structural bounds/ZIP64 refusal')
        end = position + 46 + namesize + extrasize + commentsize
        if end > len(raw) or namesize == 0:
            raise ValueError('Truncated central filename/metadata')
        name = raw[position + 46:position + 46 + namesize].decode('utf8' if flags & 0x800 else 'cp437')
        path = PurePosixPath(name)
        if name in seen or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or '\0' in name or str(path) != name.rstrip('/'):
            raise ValueError('Unsafe or duplicate member path')
        mode = external >> 16
        kind = stat.S_IFMT(mode)
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError('Link or special ZIP member')
        if method == 0 and compressed != expanded:
            raise ValueError('Stored entry size mismatch')
        seen.add(name)
        rows.append(dict(name=name, kind='directory' if name.endswith('/') else 'regular',
                         compression=method, compressed_bytes=compressed, uncompressed_bytes=expanded,
                         crc32=f'{crc:08x}', local_header_offset=offset))
        position = end
    if len(rows) != descriptor['entry_count']:
        raise ValueError('Central-directory count mismatch')
    return rows


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def fetch_range(url, start, end, total, label, etag=None):
    row = dict(label=label, url=url, range=[start, end], body_bytes_read=0,
               observed_utc=datetime.now(timezone.utc).isoformat())
    begun = time.perf_counter()
    response = None
    raw = None
    accepted_etag = None
    try:
        headers = {'User-Agent': 'Opal-structural-ZIP-metadata/1.0', 'Accept-Encoding': 'identity',
                   'Range': f'bytes={start}-{end}'}
        if etag is not None:
            headers['If-Match'] = etag
        try:
            response = urllib.request.build_opener(NoRedirect()).open(urllib.request.Request(url, headers=headers), timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        row.update(status=response.status,
                   content_type=response.headers.get('Content-Type'),
                   content_range=response.headers.get('Content-Range'),
                   content_length=response.headers.get('Content-Length'),
                   content_encoding=response.headers.get('Content-Encoding'))
        raw, accepted_etag = consume_range(response, start, end, total, row, etag)
        row.update(sha256=hashlib.sha256(raw).hexdigest(), etag_sha256=hashlib.sha256(accepted_etag.encode()).hexdigest())
        path = QUARANTINE / (label + '.bin')
        with path.open('xb') as stream:
            stream.write(raw)
        row['quarantine_path'] = str(path)
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        row['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if response is not None:
            response.close()
    row['elapsed_seconds'] = time.perf_counter() - begun
    return row, raw, accepted_etag


def main():
    output = OUT / 'colo1-zip-directory-receipt-v1.json'
    if output.exists():
        raise FileExistsError(output)
    freeze_path = OUT / 'colo1-zip-directory-freeze-v1.json'
    binding_path = OUT / 'colo1-zip-directory-code-binding-v1.json'
    freeze, binding = json.loads(freeze_path.read_bytes()), json.loads(binding_path.read_bytes())
    if sha(freeze_path) != binding['freeze_sha256']:
        raise ValueError('ZIP metadata freeze drift')
    checks = {**freeze['sha256'], **binding['code_sha256']}
    for path, expected in checks.items():
        if sha(ROOT / path) != expected:
            raise ValueError('ZIP metadata input drift: ' + path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    QUARANTINE.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    first, raw, etag = fetch_range(freeze['url'], *freeze['initial_range'], freeze['archive_bytes'], 'eocd')
    requests, entries, descriptor, stopped = [first], [], None, first.get('error')
    if stopped is None:
        try:
            descriptor = parse_eocd(raw, freeze['archive_bytes'])
            second, directory, _ = fetch_range(freeze['url'], descriptor['directory_start'], descriptor['directory_end'], freeze['archive_bytes'], 'central-directory', etag)
            requests.append(second)
            if 'error' in second:
                stopped = second['error']
            else:
                entries = parse_directory(directory, descriptor)
        except (ValueError, UnicodeError, struct.error) as error:
            stopped = type(error).__name__ + ': ' + str(error)
    candidates = [e for e in entries if e['name'].endswith('/Analysis/ENCORE/input/IDs_colo1.rds') or e['name'] == 'Analysis/ENCORE/input/IDs_colo1.rds']
    nominal_candidate = len(candidates) == 1 and candidates[0]['kind'] == 'regular' and candidates[0]['uncompressed_bytes'] == 277295
    for path, expected in checks.items():
        if sha(ROOT / path) != expected:
            raise ValueError('Post-ZIP metadata drift: ' + path)
    result = dict(schema='opal-colo1-zip-directory-v1', freeze_sha256=sha(freeze_path),
                  code_binding_sha256=sha(binding_path), requests=requests, descriptor=descriptor,
                  entries=entries, stopped=stopped, row_annotation_candidates=candidates,
                  documented_size_candidate_present=nominal_candidate,
                  wall_seconds=time.perf_counter() - begun,
                  structural_body_bytes_read=sum(r['body_bytes_read'] for r in requests),
                  archive_catalog_md5_verified=False, whole_archive_downloaded=False,
                  member_bodies_opened=0, response_tables_or_headers_opened=False,
                  source_qualified=False, fits=0, scores=0)
    with output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(requests=len(requests), entries=len(entries), stopped=stopped,
                          structural_body_bytes_read=result['structural_body_bytes_read'],
                          documented_size_candidate_present=nominal_candidate)))


if __name__ == '__main__':
    main()
