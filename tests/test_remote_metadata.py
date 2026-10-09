"""Bounded remote HDF5 reads preserve object identity and reject full downloads."""
import importlib.util
import io
from pathlib import Path
import re

import pytest

spec = importlib.util.spec_from_file_location('remote_metadata',
    Path(__file__).parents[1] / 'tools/audit_remote_h5ad_metadata.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response(io.BytesIO):
    def __init__(self, data, start, end, total, status=206, etag='"stable"'):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': f'bytes {start}-{end}/{total}', 'ETag': etag}


def test_ranges_cache_and_handle_cross_block_seek_and_end_of_object(tmp_path):
    data = bytes(range(256)) * 513
    calls = []

    def opener(request, timeout):
        start, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', request.get_header('Range')).groups())
        assert request.get_header('If-match') == '"stable"'
        calls.append((start, end))
        return Response(data[start:end + 1], start, end, len(data))

    reader = module.RangeReader('https://example.test/object', len(data), '"stable"', tmp_path, opener)
    reader.seek(65530)
    assert reader.read(20) == data[65530:65550]
    reader.seek(65530)
    assert reader.read(20) == data[65530:65550]
    assert len(calls) == 2
    reader.seek(-10, 2)
    assert reader.read(100) == data[-10:]
    assert reader.read(1) == b''
    assert reader.used == len(data)
    with pytest.raises(ValueError, match='Seek'):
        reader.seek(-1)


@pytest.mark.parametrize('status,etag', [(200, '"stable"'), (206, '"changed"')])
def test_ignored_ranges_and_changed_identity_reject_before_reading_body(tmp_path, status, etag):
    def opener(request, timeout):
        return Response(b'x' * 100, 0, 99, 100, status, etag)

    reader = module.RangeReader('https://example.test/object', 100, '"stable"', tmp_path, opener)
    with pytest.raises(ValueError, match='ignored range or changed'):
        reader.read(1)
    assert reader.used == 0
    assert not reader.ledger


def test_budget_prevents_unbounded_or_second_block_download(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(request)
        return Response(b'x' * 65536, 0, 65535, 131072)

    reader = module.RangeReader('https://example.test/object', 131072, '"stable"', tmp_path, opener, budget=65536)
    with pytest.raises(ValueError, match='Read exceeds'):
        reader.read()
    assert not calls
    assert reader.read(1) == b'x'
    reader.seek(65536)
    with pytest.raises(ValueError, match='ranges exceed'):
        reader.read(1)
    assert len(calls) == 1


def test_sample_contexts_and_higher_order_assignments_are_not_pooled():
    strata = module.summarize_strata(['1'] * 6 + ['2'] * 2, ['TNFA+'] * 6 + ['TNFA-'] * 2,
        ['A', 'B', 'A|B', 'A|A', 'A|B|C', 'A|INTERGENIC1', 'A', 'A|B'])
    assert len(strata) == 2
    assert strata[0]['literal_distinct_two_target_count_excluding_intergenic'] == 1
    assert strata[0]['literal_pairs_with_both_single_labels'] == 1
    assert strata[1]['literal_pairs_with_both_single_labels'] == 0
    assert not strata[0]['independent_culture_identity_verified']
    with pytest.raises(ValueError, match='lengths'):
        module.summarize_strata(['1'], [], ['A'])
