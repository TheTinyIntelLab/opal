import io
import struct
import zipfile

import pytest

from tools.continuation_colo1_zip_directory_v1 import consume_range, parse_eocd, parse_directory


class TrapResponse:
    status = 206

    def __init__(self):
        self.headers = {'Content-Type': 'application/zip', 'Content-Range': 'bytes 78-99/100',
                        'Content-Length': '22', 'ETag': '"stable"'}

    def read(self, count):
        raise AssertionError('Unapproved body read')


@pytest.mark.parametrize('status,changes,etag', [
    (200, {}, None), (403, {}, None), (302, {}, None),
    (206, {'Content-Range': None}, None), (206, {'Content-Range': 'bytes 0-21/100'}, None),
    (206, {'Content-Length': '100'}, None), (206, {'ETag': 'W/"weak"'}, None),
    (206, {'ETag': ''}, None), (206, {}, '"changed"'),
    (206, {'Content-Encoding': 'gzip'}, None), (206, {'Content-Type': 'text/html'}, None),
])
def test_refused_envelopes_cannot_read_any_archive_bytes(status, changes, etag):
    response = TrapResponse()
    response.status = status
    response.headers.update(changes)
    audit = {'body_bytes_read': 0}
    with pytest.raises(ValueError):
        consume_range(response, 78, 99, 100, audit, etag)
    assert audit['body_bytes_read'] == 0


def synthetic_archive(names=('bundle/annotation.txt',)):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(name, b'synthetic annotation')
    return stream.getvalue()


def test_structural_parser_agrees_with_standard_library_without_reading_member():
    raw = synthetic_archive()
    descriptor = parse_eocd(raw[-22:], len(raw))
    rows = parse_directory(raw[descriptor['directory_start']:-22], descriptor)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.infolist()[0]
    assert rows == [dict(name=info.filename, kind='regular', compression=info.compress_type,
                         compressed_bytes=info.compress_size, uncompressed_bytes=info.file_size,
                         crc32=f'{info.CRC:08x}', local_header_offset=info.header_offset)]


@pytest.mark.parametrize('offset,value', [(4, 1), (6, 1), (8, 0), (20, 1)])
def test_unsupported_eocd_cannot_authorize_a_directory_range(offset, value):
    raw = synthetic_archive()
    eocd = bytearray(raw[-22:])
    struct.pack_into('<H', eocd, offset, value)
    with pytest.raises(ValueError):
        parse_eocd(eocd, len(raw))


@pytest.mark.parametrize('names', [('../escape',), ('same', 'same')])
def test_unsafe_or_ambiguous_names_refused(names):
    raw = synthetic_archive(names)
    descriptor = parse_eocd(raw[-22:], len(raw))
    with pytest.raises(ValueError):
        parse_directory(raw[descriptor['directory_start']:-22], descriptor)


@pytest.mark.parametrize('mutation', ['encrypted', 'symlink', 'truncated', 'count'])
def test_malformed_directory_cannot_establish_a_candidate(mutation):
    raw = synthetic_archive()
    descriptor = parse_eocd(raw[-22:], len(raw))
    directory = bytearray(raw[descriptor['directory_start']:-22])
    if mutation == 'encrypted':
        struct.pack_into('<H', directory, 8, 1)
    elif mutation == 'symlink':
        struct.pack_into('<I', directory, 38, 0o120777 << 16)
    elif mutation == 'truncated':
        directory.pop()
        descriptor['directory_bytes'] -= 1
    else:
        descriptor['entry_count'] += 1
    with pytest.raises(ValueError):
        parse_directory(directory, descriptor)


def test_valid_range_reads_exactly_authorized_bytes_without_probe():
    response = TrapResponse()
    stream = io.BytesIO(b'a' * 22 + b'forbidden payload')
    response.read = stream.read
    audit = {'body_bytes_read': 0}
    raw, etag = consume_range(response, 78, 99, 100, audit)
    assert raw == b'a' * 22 and etag == '"stable"'
    assert audit['body_bytes_read'] == stream.tell() == 22
