from email.message import Message
import io

import pytest

from tools.continuation_colo1_annotation_acquisition_v1 import identities
from tools.continuation_colo1_annotation_acquisition_v2 import consume, verified

ABC = 'f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f'


def test_missing_length_is_bounded_and_whole_object_identity_remains_required():
    class Response:
        status = 200
        headers = Message()
        headers['Content-Type'] = 'application/octet-stream'
        read = io.BytesIO(b'abc').read
    sink, audit = io.BytesIO(), dict(member_bytes_read=0)
    consume(Response(), 3, audit, sink)
    assert audit['member_bytes_read'] == 3
    assert verified(identities(sink.getvalue()), 3, ABC, 891568578)


def test_oversized_stream_cannot_read_past_registered_bound():
    class Response:
        status = 200
        headers = Message()
        headers['Content-Type'] = 'application/octet-stream'
    source = io.BytesIO(b'abcXYZ')
    response = Response()
    response.read = source.read
    sink, audit = io.BytesIO(), dict(member_bytes_read=0)
    consume(response, 3, audit, sink)
    assert audit['member_bytes_read'] == source.tell() == 4
    assert not verified(identities(sink.getvalue()), 3, ABC, 891568578)


@pytest.mark.parametrize('raw,blob,crc', [
    (b'abd', ABC, 891568578), (b'ab', ABC, 891568578),
    (b'abc', '0'*40, 891568578), (b'abc', ABC, 0),
])
def test_corruption_or_wrong_identity_cannot_pass(raw, blob, crc):
    assert not verified(identities(raw), 3, blob, crc)


def test_contradictory_length_still_closes_before_reading():
    class Response:
        status = 200
        headers = Message()
        headers['Content-Type'] = 'application/octet-stream'
        headers['Content-Length'] = '4'
        def read(self, count):
            raise AssertionError('Body forbidden')
    with pytest.raises(ValueError, match='Contradictory'):
        consume(Response(), 3, dict(member_bytes_read=0), io.BytesIO())
