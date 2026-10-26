"""Central-directory names for one Figshare ZIP. Never read a member payload."""
import hashlib
import json
import pathlib
import shutil
import stat
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import PurePosixPath

ROOT = pathlib.Path(__file__).resolve().parents[1]
REG = ROOT / 'docs/frontier_execution/Grok_source_registration_v1.md'
OUT = ROOT / 'results/frontier-execution-20261026/Grok_source_requests_v1.json'
LOCAL = ROOT / 'tmp/colo1_frontier_grok_v1'
URL = 'https://ndownloader.figshare.com/files/45433417'
ARCHIVE_BYTES = 1153775580
MAX_ENTRIES = 4096
MAX_DIRECTORY_BYTES = 262144
MAX_NAME_BYTES = 240
MIMES = {'application/zip', 'application/octet-stream', 'application/x-zip-compressed', 'binary/octet-stream'}
REDIRECTS = {301, 302, 303, 307, 308}
DOCUMENTED_BASENAMES = {
    'colo1_plm.rds': 'documented_lfc_sealed',
    'colo1_plm_scale.rds': 'documented_scaled_lfc_sealed',
    'colo1_c91.rds': 'documented_lfc_sealed',
    'colo1_plm_GEN.rds': 'documented_gene_lfc_sealed',
    'IDs_colo1.rds': 'documented_row_annotation_sealed',
    'COLO1_RUNMERGED_EXACT_ANNOTATED.txt': 'documented_raw_counts_sealed',
    'lib-COLO-1.stats': 'documented_stats_sealed',
    'All.stats': 'documented_stats_sealed',
    'paralog_scores_ryan.csv': 'documented_external_scores_sealed',
    'SCORE_median_lfc.rds': 'documented_external_lfc_sealed',
}


def status_of(response):
    value = getattr(response, 'status', None)
    return response.code if value is None else value


def validate_envelope(status, headers, start, end, total):
    """Refuse every non-exact partial response before a body read."""
    count = end - start + 1
    if count <= 0:
        raise ValueError('empty range')
    if status != 206:
        raise ValueError(f'status {status}; partial content required')
    if headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
        raise ValueError('Content-Range mismatch')
    if headers.get('Content-Length') != str(count):
        raise ValueError('Content-Length mismatch')
    encoding = headers.get('Content-Encoding')
    if encoding is not None and encoding.lower() != 'identity':
        raise ValueError('Content-Encoding refused')
    mime = (headers.get('Content-Type') or '').split(';', 1)[0].strip().lower()
    if mime not in MIMES:
        raise ValueError('MIME refused')
    return count


def read_exact(response, count, record):
    chunks = []
    remaining = count
    while remaining:
        block = response.read(remaining)
        record['body_bytes_read'] += len(block)
        if not block or len(block) > remaining:
            raise ValueError('range body length mismatch')
        chunks.append(block)
        remaining -= len(block)
    return b''.join(chunks)


def header_snapshot(response, record):
    headers = response.headers
    record.update(
        status=status_of(response),
        content_type=headers.get('Content-Type'),
        content_range=headers.get('Content-Range'),
        content_length=headers.get('Content-Length'),
        content_encoding=headers.get('Content-Encoding'),
    )
    etag = headers.get('ETag')
    if etag:
        record['etag_sha256'] = hashlib.sha256(etag.encode()).hexdigest()
    return etag


def fetch_range(transport, url, start, end, total, previous_etag=None):
    headers = {
        'User-Agent': 'Opal-source-qualification/1.0',
        'Accept-Encoding': 'identity',
        'Range': f'bytes={start}-{end}',
    }
    record = {'url': url, 'range': [start, end], 'body_bytes_read': 0, 'redirects': 0}
    current = url
    response = None
    try:
        for hop in (0, 1):
            response = transport(current, headers)
            etag = header_snapshot(response, record)
            status = record['status']
            if status in REDIRECTS:
                if hop == 1:
                    raise ValueError('second redirect refused')
                location = response.headers.get('Location')
                if not location:
                    raise ValueError('redirect without Location')
                parts = urllib.parse.urlsplit(location)
                if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password:
                    raise ValueError('redirect location refused')
                record['redirects'] = 1
                record['redirect_host'] = parts.hostname
                record['redirect_path'] = parts.path
                record['location_sha256'] = hashlib.sha256(location.encode()).hexdigest()
                response.close()
                response = None
                current = location
                continue
            if previous_etag and etag and previous_etag != etag:
                raise ValueError('ETag changed across ranges')
            count = validate_envelope(status, response.headers, start, end, total)
            raw = read_exact(response, count, record)
            record['sha256'] = hashlib.sha256(raw).hexdigest()
            return record, raw, etag
        raise ValueError('redirect loop')
    except Exception as exc:
        record['error'] = type(exc).__name__ + ': ' + str(exc)
        record.pop('sha256', None)
        return record, None, None
    finally:
        if response is not None:
            response.close()


def parse_eocd(raw, total):
    if len(raw) != 22:
        raise ValueError('EOCD length')
    signature, disk, directory_disk, disk_count, count, length, offset, comment = struct.unpack('<4s4H2IH', raw)
    if signature != b'PK\x05\x06':
        raise ValueError('EOCD signature')
    if disk != 0 or directory_disk != 0:
        raise ValueError('EOCD multidisk')
    if comment != 0:
        raise ValueError('EOCD comment')
    if count == 0xFFFF or length == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        raise ValueError('ZIP64 sentinel')
    if disk_count != count:
        raise ValueError('EOCD count mismatch')
    if not 1 <= count <= MAX_ENTRIES:
        raise ValueError(f'EOCD entry count {count} outside 1..{MAX_ENTRIES}')
    if not 1 <= length <= MAX_DIRECTORY_BYTES:
        raise ValueError(f'EOCD directory bytes {length} outside 1..{MAX_DIRECTORY_BYTES}')
    if offset + length != total - 22:
        raise ValueError('EOCD directory is not contiguous with EOCD')
    return {
        'entry_count': count,
        'directory_bytes': length,
        'directory_start': offset,
        'directory_end': offset + length - 1,
    }


def classify_entry(name, kind):
    if kind == 'directory':
        return 'directory_entry_sealed'
    base = PurePosixPath(name).name
    return DOCUMENTED_BASENAMES.get(base, 'unclassified_filename_sealed')


def parse_central_directory(raw, descriptor):
    if len(raw) != descriptor['directory_bytes']:
        raise ValueError('central-directory length mismatch')
    rows = []
    seen = set()
    position = 0
    while position < len(raw):
        if len(raw) - position < 46:
            raise ValueError('truncated central record')
        fields = struct.unpack_from('<4s6H3I5H2I', raw, position)
        (signature, made, needed, flags, method, _time, _date, crc, compressed, expanded,
         namesize, extrasize, commentsize, disk, _internal, external, offset) = fields
        del made
        if signature != b'PK\x01\x02':
            raise ValueError('central signature')
        if needed > 20 or disk != 0 or flags & 1 or method not in {0, 8}:
            raise ValueError('unsupported central record')
        if max(compressed, expanded, offset) == 0xFFFFFFFF:
            raise ValueError('ZIP64 central sentinel')
        if namesize == 0 or namesize > MAX_NAME_BYTES:
            raise ValueError('central filename length')
        end = position + 46 + namesize + extrasize + commentsize
        if end > len(raw):
            raise ValueError('truncated central filename')
        if offset > descriptor['directory_start'] or compressed > descriptor['directory_start']:
            raise ValueError('member bounds overlap central directory')
        if offset + 30 + compressed > descriptor['directory_start']:
            raise ValueError('member bounds overlap central directory')
        name = raw[position + 46:position + 46 + namesize].decode('utf-8' if flags & 0x800 else 'cp437')
        path = PurePosixPath(name)
        if (name in seen or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
                or '\0' in name or str(path) != name.rstrip('/')):
            raise ValueError('unsafe or duplicate member path')
        mode = stat.S_IFMT(external >> 16)
        if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError('link or special ZIP member')
        if method == 0 and compressed != expanded:
            raise ValueError('stored entry size mismatch')
        kind = 'directory' if name.endswith('/') else 'regular'
        seen.add(name)
        rows.append({
            'name': name,
            'kind': kind,
            'role': classify_entry(name, kind),
            'compression': method,
            'compressed_bytes': compressed,
            'uncompressed_bytes': expanded,
            'crc32': f'{crc:08x}',
            'local_header_offset': offset,
            'body_access': 'not_authorized',
        })
        position = end
    if len(rows) != descriptor['entry_count']:
        raise ValueError('central-directory count mismatch')
    return rows


def catalog_from_transport(transport, total, url=URL):
    """Return names from EOCD plus the central directory, or an exact blocker."""
    started = time.monotonic()
    first, eocd_raw, etag = fetch_range(transport, url, total - 22, total - 1, total)
    requests = [first]
    result = {
        'requests': requests,
        'descriptor': None,
        'entries': [],
        'stopped': first.get('error'),
        'eocd_sha256': None,
        'accepted_eocd': None,
        'accepted_directory': None,
        'response_data_opened': False,
        'member_bodies_opened': 0,
        'whole_archive_downloaded': False,
        'archive_md5_verified': False,
    }
    if eocd_raw is None:
        result['wall_seconds'] = time.monotonic() - started
        return result
    result['eocd_sha256'] = hashlib.sha256(eocd_raw).hexdigest()
    try:
        descriptor = parse_eocd(eocd_raw, total)
    except (ValueError, struct.error) as exc:
        result['stopped'] = type(exc).__name__ + ': ' + str(exc)
        result['wall_seconds'] = time.monotonic() - started
        return result
    result['descriptor'] = descriptor
    result['accepted_eocd'] = eocd_raw
    if time.monotonic() - started > 90:
        result['stopped'] = 'ValueError: overall timeout'
        result['wall_seconds'] = time.monotonic() - started
        return result
    second, directory_raw, _second_etag = fetch_range(
        transport, url, descriptor['directory_start'], descriptor['directory_end'], total, etag)
    requests.append(second)
    if directory_raw is None:
        result['stopped'] = second.get('error')
        result['wall_seconds'] = time.monotonic() - started
        return result
    result['accepted_directory'] = directory_raw
    try:
        result['entries'] = parse_central_directory(directory_raw, descriptor)
    except (ValueError, UnicodeError, struct.error) as exc:
        result['stopped'] = type(exc).__name__ + ': ' + str(exc)
        result['entries'] = []
    result['wall_seconds'] = time.monotonic() - started
    return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def urllib_transport(url, headers):
    request = urllib.request.Request(url, headers=headers, method='GET')
    try:
        return urllib.request.build_opener(_NoRedirect()).open(request, timeout=30)
    except urllib.error.HTTPError as error:
        return error


def main():
    if OUT.exists():
        raise RuntimeError('immutable receipt exists')
    if shutil.disk_usage(ROOT).free < 1024**3 + MAX_DIRECTORY_BYTES + 22:
        raise RuntimeError('disk reserve')
    LOCAL.mkdir(parents=True, exist_ok=True)
    result = catalog_from_transport(urllib_transport, ARCHIVE_BYTES, URL)
    for label, key in (('eocd', 'accepted_eocd'), ('central-directory', 'accepted_directory')):
        blob = result.pop(key)
        if blob is not None:
            path = LOCAL / f'{label}.bin'
            path.write_bytes(blob)
            result[f'{label}_quarantine'] = path.name
    result.update(
        registration_sha256=hashlib.sha256(REG.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        archive_bytes=ARCHIVE_BYTES,
        catalog_md5_unverified='45c84b5ec11dc17bf5967968c7097ea0',
        container_license_from_prior_catalog='CC BY 4.0',
        attribution='Cursor Grok 4.7 continuation; not an A completion',
    )
    OUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'stopped': result['stopped'],
        'entries': len(result['entries']),
        'requests': len(result['requests']),
        'body_bytes_read': sum(item['body_bytes_read'] for item in result['requests']),
        'member_bodies_opened': result['member_bodies_opened'],
    }))


if __name__ == '__main__':
    main()
