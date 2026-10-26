import io
import json
import struct
import zipfile

import pytest

from tools.colo1_frontier_grok_zipdir_v1 import (
    ARCHIVE_BYTES,
    DOCUMENTED_BASENAMES,
    REG,
    catalog_from_transport,
    fetch_range,
    parse_central_directory,
    parse_eocd,
    validate_envelope,
)


class FakeResponse:
    def __init__(self, status, headers, body=b'', trap=False):
        self.status = status
        self.headers = headers
        self._body = io.BytesIO(body)
        self.trap = trap
        self.closed = False

    def read(self, count):
        if self.trap:
            raise AssertionError('unapproved body read')
        return self._body.read(count)

    def close(self):
        self.closed = True


def ranged(blob, content_type='application/zip'):
    def transport(url, headers):
        del url
        start, end = (int(part) for part in headers['Range'].split('=', 1)[1].split('-', 1))
        body = blob[start:end + 1]
        return FakeResponse(206, {
            'Content-Type': content_type,
            'Content-Range': f'bytes {start}-{end}/{len(blob)}',
            'Content-Length': str(len(body)),
        }, body)
    return transport


def archive(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
        for name, payload in members:
            handle.writestr(name, payload)
    return stream.getvalue()


def test_registration_binds_the_exact_range_and_sealed_names():
    text = REG.read_text(encoding='utf-8')
    assert 'https://ndownloader.figshare.com/files/45433417' in text
    assert str(ARCHIVE_BYTES) in text
    assert '1153775558-1153775579' in text
    assert '4096' in text and '262144' in text
    for name in DOCUMENTED_BASENAMES:
        assert name in text


@pytest.mark.parametrize('status,headers', [
    (200, {'Content-Type': 'application/zip', 'Content-Range': 'bytes 0-21/100', 'Content-Length': '22'}),
    (206, {'Content-Type': 'text/html', 'Content-Range': 'bytes 0-21/100', 'Content-Length': '22'}),
    (206, {'Content-Type': 'application/zip', 'Content-Range': 'bytes 0-21/99', 'Content-Length': '22'}),
    (206, {'Content-Type': 'application/zip', 'Content-Range': 'bytes 0-21/100', 'Content-Length': '100'}),
    (206, {'Content-Type': 'application/zip', 'Content-Range': 'bytes 0-21/100', 'Content-Length': '22', 'Content-Encoding': 'gzip'}),
])
def test_bad_envelope_is_rejected_before_validation_returns(status, headers):
    with pytest.raises(ValueError):
        validate_envelope(status, headers, 0, 21, 100)


def test_rejected_status_does_not_read_a_body():
    def transport(url, headers):
        del url, headers
        return FakeResponse(200, {
            'Content-Type': 'application/octet-stream',
            'Content-Length': '1153775580',
        }, trap=True)
    record, raw, _etag = fetch_range(transport, 'https://example.test/file', 0, 21, 100)
    assert raw is None and record['body_bytes_read'] == 0
    assert 'status 200' in record['error']


def test_redirect_reads_only_the_followed_partial_body_and_hides_the_query():
    blob = b'A' * 22
    calls = []

    def transport(url, headers):
        calls.append((url, headers['Range']))
        if len(calls) == 1:
            return FakeResponse(302, {'Location': 'https://cdn.example/files/45433417?X-Amz-Signature=secret'}, trap=True)
        return FakeResponse(206, {
            'Content-Type': 'application/zip',
            'Content-Range': 'bytes 0-21/100',
            'Content-Length': '22',
            'ETag': '"same"',
        }, blob)

    record, raw, etag = fetch_range(transport, 'https://ndownloader.figshare.com/files/45433417', 0, 21, 100)
    assert raw == blob and etag == '"same"'
    assert record['redirect_host'] == 'cdn.example'
    assert record['redirect_path'] == '/files/45433417'
    assert 'secret' not in json.dumps(record)
    assert calls[1][0].endswith('secret')


def test_second_redirect_and_insecure_location_read_nothing():
    def redirect(_url, _headers):
        return FakeResponse(302, {'Location': 'https://cdn.example/again'}, trap=True)

    record, raw, _etag = fetch_range(redirect, 'https://example.test/file', 0, 21, 100)
    assert raw is None and record['body_bytes_read'] == 0
    assert 'second redirect' in record['error']

    def insecure(_url, _headers):
        return FakeResponse(302, {'Location': 'http://cdn.example/file'}, trap=True)

    record, raw, _etag = fetch_range(insecure, 'https://example.test/file', 0, 21, 100)
    assert raw is None and 'redirect location refused' in record['error']


def test_catalog_returns_names_and_roles_without_member_bytes():
    marker = b'PAYLOAD_MARKER_9f3a'
    blob = archive((
        ('notes/IDs_colo1.rds', marker),
        ('notes/other.txt', marker + b'-other'),
        ('notes/', b''),
    ))
    result = catalog_from_transport(ranged(blob), len(blob))
    visible = json.dumps(result['entries']) + result['accepted_eocd'].hex() + result['accepted_directory'].hex()
    assert result['stopped'] is None
    assert marker.hex() not in visible and b'PAYLOAD_MARKER_9f3a' not in result['accepted_directory']
    roles = {row['name']: row['role'] for row in result['entries']}
    assert roles['notes/IDs_colo1.rds'] == 'documented_row_annotation_sealed'
    assert roles['notes/other.txt'] == 'unclassified_filename_sealed'
    assert roles['notes/'] == 'directory_entry_sealed'
    assert all(row['body_access'] == 'not_authorized' for row in result['entries'])
    assert result['member_bodies_opened'] == 0


def test_changed_etag_blocks_the_directory_body():
    blob = archive((('a.txt', b'abc'),))
    calls = {'n': 0}

    def transport(url, headers):
        del url
        calls['n'] += 1
        start, end = (int(part) for part in headers['Range'].split('=', 1)[1].split('-', 1))
        body = blob[start:end + 1]
        if calls['n'] == 1:
            return FakeResponse(206, {
                'Content-Type': 'application/zip',
                'Content-Range': f'bytes {start}-{end}/{len(blob)}',
                'Content-Length': str(len(body)),
                'ETag': '"one"',
            }, body)
        return FakeResponse(206, {
            'Content-Type': 'application/zip',
            'Content-Range': f'bytes {start}-{end}/{len(blob)}',
            'Content-Length': str(len(body)),
            'ETag': '"two"',
        }, trap=True)

    result = catalog_from_transport(transport, len(blob))
    assert result['entries'] == []
    assert 'ETag changed' in result['stopped']
    assert result['requests'][1]['body_bytes_read'] == 0


@pytest.mark.parametrize('offset,value,message', [
    (20, 1, 'EOCD comment'),
    (4, 1, 'EOCD multidisk'),
    (16, 0xFFFFFFFF, 'ZIP64 sentinel'),
])
def test_unsafe_eocd_cannot_authorize_a_wider_read(offset, value, message):
    blob = archive((('a.txt', b'abc'),))
    eocd = bytearray(blob[-22:])
    fmt = '<H' if offset != 16 else '<I'
    struct.pack_into(fmt, eocd, offset, value)
    with pytest.raises(ValueError, match=message):
        parse_eocd(bytes(eocd), len(blob))


def test_entry_and_directory_caps_stop_before_a_second_interpretation():
    blob = archive((('a.txt', b'abc'),))
    eocd = bytearray(blob[-22:])
    struct.pack_into('<HH', eocd, 8, 4097, 4097)
    with pytest.raises(ValueError, match='entry count'):
        parse_eocd(bytes(eocd), len(blob))
    eocd = bytearray(blob[-22:])
    struct.pack_into('<I', eocd, 12, 262145)
    with pytest.raises(ValueError, match='directory bytes'):
        parse_eocd(bytes(eocd), len(blob))


@pytest.mark.parametrize('mutation', ['encrypted', 'symlink', 'overlap', 'stored', 'truncated', 'duplicate'])
def test_malformed_directory_projects_no_partial_catalog(mutation):
    blob = archive((('safe.txt', b'abc'), ('safe.txt', b'def')) if mutation == 'duplicate' else (('safe.txt', b'abc'),))
    descriptor = parse_eocd(blob[-22:], len(blob))
    directory = bytearray(blob[descriptor['directory_start']:descriptor['directory_end'] + 1])
    if mutation == 'encrypted':
        struct.pack_into('<H', directory, 8, 1)
    elif mutation == 'symlink':
        struct.pack_into('<I', directory, 38, 0o120777 << 16)
    elif mutation == 'overlap':
        struct.pack_into('<I', directory, 20, descriptor['directory_start'])
    elif mutation == 'stored':
        struct.pack_into('<H', directory, 10, 0)
    elif mutation == 'truncated':
        directory.pop()
        descriptor = dict(descriptor, directory_bytes=descriptor['directory_bytes'] - 1)
    with pytest.raises(ValueError):
        parse_central_directory(bytes(directory), descriptor)
