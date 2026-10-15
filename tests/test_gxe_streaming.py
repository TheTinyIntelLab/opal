import gzip
import copy
import numpy as np
import pytest
from opal.gxe_reconstruction import coordinates
from opal.gxe_streaming import coordinates_c, validate_limits, RuntimeGuard
from tools.verify_gxe_panel import panel_variance


def limits():
    return dict(cpu_seconds=300, wall_seconds=600, ram_bytes=1_000_000_000,
                retained_bytes=10_000_000, reserve_bytes=1073741824, threads=1,
                workers=1, downloads=False, installs=False, fitting=False)


def test_c_parser_preserves_rows_counts_and_normalized_aggregation(tmp_path):
    p = tmp_path/'counts.gz'
    with gzip.open(p, 'wt') as f:
        f.write('1\t1\t2\n2\t1\t3\n1\t2\t5\n2\t2\t7\n')
    slow = list(coordinates(p, (2, 2), lambda: None, chunk_lines=3))
    fast = list(coordinates_c(p, (2, 2), lambda: None, chunk_lines=2))
    for i in range(3):
        np.testing.assert_array_equal(np.concatenate([x[i] for x in slow]), np.concatenate([x[i] for x in fast]))
    sums = []
    for blocks in (slow, fast):
        total = np.zeros(2)
        for r, c, v in blocks:
            np.add.at(total, c, v)
        output = np.zeros(2)
        for r, c, v in blocks:
            np.add.at(output, r, v.astype(float)*10000/total[c])
        sums.append(output)
    np.testing.assert_array_equal(*sums)


@pytest.mark.parametrize('text', ['1\t1\t1.5\n', '1\t1\t-1\n', '3\t1\t2\n', '1\t1\t2\t4\n', '1\t1\n'])
def test_c_parser_refuses_malformed_or_incompatible_coordinates(tmp_path, text):
    p = tmp_path/'bad.gz'
    with gzip.open(p, 'wt') as f:
        f.write(text)
    with pytest.raises((ValueError, TypeError)):
        list(coordinates_c(p, (2, 2), lambda: None))


def test_limits_cannot_silently_raise_old_envelope_and_reserve_is_effective(tmp_path, monkeypatch):
    cfg = limits()
    for key in ('cpu_seconds', 'ram_bytes', 'retained_bytes', 'wall_seconds'):
        changed = copy.deepcopy(cfg)
        changed[key] += 1
        with pytest.raises(ValueError):
            validate_limits(changed)
    clock = [0.]
    monkeypatch.setattr('opal.gxe_streaming.time.process_time', lambda: clock[0])
    guard = RuntimeGuard(tmp_path, cfg)
    guard.check(phase='fixture')
    clock[0] = 295.
    with pytest.raises(RuntimeError, match='finalization reserve'):
        guard.check()
    assert (tmp_path/'progress.json').is_file()


def test_tiled_panel_variance_is_exact_and_ignores_pair_values():
    keys = [(c, 'p', 'r', drug, dose, anchor) for c in ('a', 'b', 'c')
            for drug, dose, anchor in [('vehicle', 0., 0.), ('x', 1., 0.),
                                        ('vehicle', 0., 1.), ('x', 1., 1.)]]
    rng = np.random.default_rng(73)
    sums = rng.uniform(1, 1000, (len(keys), 2, 31))
    support = np.full((len(keys), 2), 10)
    full = panel_variance(sums, support, keys, ['a', 'b', 'c'], block_genes=31)
    tiled = panel_variance(sums, support, keys, ['a', 'b', 'c'], block_genes=2)
    np.testing.assert_array_equal(full, tiled)
    sums[3::4] *= 100000
    np.testing.assert_array_equal(full, panel_variance(sums, support, keys, ['a', 'b', 'c'], block_genes=2))
