from email.message import Message

import pytest

from tools.continuation_paralog_transport_v2 import envelope


class BodyTrap:
    def __init__(self, status, mime):
        self.status = status
        self.headers = Message()
        if mime is not None:
            self.headers['Content-Type'] = mime
        self.headers['Set-Cookie'] = 'synthetic-private-cookie'

    def read(self, *args):
        raise AssertionError('Body access forbidden')

    readinto = readline = read


@pytest.mark.parametrize('status,mime,admitted', [
    (200, 'text/html; charset=utf-8', True),
    (202, 'text/html', False),
    (403, 'application/json', False),
    (302, 'text/html', False),
])
def test_all_transport_cases_record_fields_without_reading_any_body(status, mime, admitted):
    response = BodyTrap(status, mime)
    response.headers['Location'] = 'https://example.com/file?token=synthetic'
    result = envelope(response)
    assert result['status'] == status
    assert result['mime'] == mime.split(';')[0]
    assert result['matches_v1_transport_gate'] == admitted
    assert result['body_bytes_read'] == result['redirects_followed'] == 0
    assert result['location'] == dict(host='example.com', path='/file', query_redacted=True)
    assert 'synthetic-private-cookie' not in str(result)
    assert 'token=synthetic' not in str(result)
