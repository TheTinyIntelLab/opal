import gzip
import numpy as np
import pytest
from opal.gxe_reconstruction import coordinates, reconstruct_metrics
from opal.gxe_aggregate_validation import validate_aggregates
from opal.gxe_measurement import matrix_blocks


def test_integer_stream_agrees_with_frozen_reader_and_refuses_bad_entries(tmp_path):
    path = tmp_path/'counts.gz'
    with gzip.open(path, 'wt') as f:
        f.write('1\t1\t2\n2\t3\t4\n')
    a = list(coordinates(path, (2, 3), lambda: None, chunk_lines=1))
    old = matrix_blocks(path, lambda: None, expected_shape=(2, 3), chunk_lines=1)
    next(old)
    b = list(old)
    for i in range(3):
        np.testing.assert_array_equal(np.concatenate([r[i] for r in a]), np.concatenate([r[i] for r in b]))
    for text in ('1 1 1.5\n', '1 1 -2\n', '3 1 2\n', '1 1 2 extra\n'):
        with gzip.open(path, 'wt') as f:
            f.write(text)
        with pytest.raises(ValueError):
            list(coordinates(path, (2, 3), lambda: None))


def test_complete_supported_and_missing_measurement_rules():
    keys = [('a','p','r','vehicle',0.,0.), ('a','p','r','x',1.,0.),
            ('a','p','r','vehicle',0.,1.), ('a','p','r','x',1.,1.)]
    sums = np.ones((4, 2, 3))*10
    support = np.ones((4, 2), dtype=int)*10
    record = reconstruct_metrics(sums, support, keys, ['a'], ['x', 'y'])
    assert record['declared_population'] == 2 and record['supported_population'] == 1
    assert record['rows'][0]['response_half_mse'] == 0
    assert record['rows'][1]['status'] == 'unsupported'
    support[0, 1] = 9
    assert reconstruct_metrics(sums, support, keys, ['a'], ['x'])['rows'][0]['status'] == 'unsupported'


def test_recovered_membership_and_denominator_integrity():
    sums = np.ones((2, 2, 3))
    mapping = np.array([0, 0, 1, 1])
    support = np.ones((2, 2), dtype=int)
    total = np.ones(4)
    validate_aggregates(sums, support, total, mapping)
    with pytest.raises(ValueError, match='membership/support'):
        validate_aggregates(sums, support+1, total, mapping)
    with pytest.raises(ValueError, match='denominator'):
        validate_aggregates(sums, support, total*0, mapping)
