from email.message import Message
import json

import pytest

from tools.continuation_colo1_validator_v2 import KEY, envelope, select_catalog, unique_object


class BodyTrap:
    def __init__(self, status, etag):
        self.status = status
        self.headers = Message()
        self.headers['Set-Cookie'] = 'synthetic-private-cookie'
        if etag is not None:
            self.headers['ETag'] = etag

    def read(self, *args):
        raise AssertionError('Archive body forbidden')

    readinto = readline = read


@pytest.mark.parametrize('status,etag,classification', [
    (206, None, 'absent'), (206, 'W/"weak"', 'weak'),
    (206, '"strong"', 'accepted-by-v1-parser'),
    (206, 'bare', 'malformed-or-unsupported-by-v1-parser'),
    (200, None, 'absent'), (302, None, 'absent'), (403, None, 'absent'),
])
def test_diagnostic_records_all_statuses_and_validator_classes_without_body(status, etag, classification):
    result = envelope(BodyTrap(status, etag))
    assert result['status'] == status and result['etag_class'] == classification
    assert result['body_bytes_read'] == 0
    assert 'synthetic-private-cookie' not in str(result)


def test_catalog_omits_arbitrary_metadata_and_redacts_signed_link_queries():
    value = dict(entries={KEY: dict(size=123, checksum='md5:synthetic', metadata={'unexpected': 'discard'},
                                   links={'content': 'https://example.com/file?signature=synthetic-secret#fragment'})})
    result = select_catalog(value)
    assert result['technical_file_fields'] == dict(size=123, checksum='md5:synthetic')
    assert 'discard' not in str(result) and 'synthetic-secret' not in str(result)
    assert result['redacted_links']['content']['query_redacted']


def test_duplicate_technical_identity_keys_are_refused():
    with pytest.raises(ValueError, match='Duplicate catalog key'):
        json.loads('{"checksum":"first","checksum":"second"}', object_pairs_hook=unique_object)
