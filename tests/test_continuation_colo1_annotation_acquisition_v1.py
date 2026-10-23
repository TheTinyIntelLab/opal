from email.message import Message

import pytest

from tools.continuation_colo1_annotation_acquisition_v1 import MEMBER, URL, consume, select_member


class BodyTrap:
    status = 200

    def __init__(self):
        self.headers = Message()
        self.headers['Content-Type'] = 'application/octet-stream'
        self.headers['Content-Length'] = '277295'

    def read(self, *args):
        raise AssertionError('Unapproved member body read')


@pytest.mark.parametrize('status,change', [
    (302, {}), (403, {}), (200, {'Content-Type': 'text/html'}),
    (200, {'Content-Length': '123'}), (200, {'Content-Encoding': 'gzip'}),
])
def test_unapproved_member_envelope_cannot_read_body(status, change):
    response = BodyTrap()
    response.status = status
    for key, value in change.items():
        if key in response.headers:
            response.headers.replace_header(key, value)
        else:
            response.headers[key] = value
    with pytest.raises(ValueError):
        consume(response, 277295)


@pytest.mark.parametrize('change', ['truncated', 'duplicate', 'size', 'endpoint'])
def test_inconsistent_or_substituted_member_catalog_refused(change):
    entry = dict(key=MEMBER, size=277295, crc=123, links={'content': URL})
    value = dict(truncated=False, total=1, entries=[entry])
    if change == 'truncated':
        value['truncated'] = True
    elif change == 'duplicate':
        value['entries'].append(entry)
        value['total'] = 2
    elif change == 'size':
        entry['size'] = True
    else:
        entry['links']['content'] = 'https://example.com/other'
    with pytest.raises(ValueError):
        select_member(value)
