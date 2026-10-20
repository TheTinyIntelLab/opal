import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('block_study', Path(__file__).parents[1]/'tools/study_block_completion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_gate_requires_all_replicates_controls_and_robust_simple_comparison():
    pairs = np.array([(a, b) for a in range(11) for b in range(a+1, 11)])
    y = np.arange(len(pairs))/100
    screen = dict(pairs=pairs, responses=y, replicates=np.column_stack((y-.01, y+.01)))
    ids = np.arange(len(pairs))
    controls = {f'shuffled-{seed}': y+.3 for seed in m.CONFIG['control_seeds']}
    args = (screen, ids, y+.1, y+.2, controls, {'mean': y+.2}, True)
    result = m.evaluate_arrays(*args)
    assert result['passes_gate']
    assert result['relative_gain'] == pytest.approx(.75)
    assert not m.evaluate_arrays(*args[:-1], False)['passes_gate']
    assert not m.evaluate_arrays(screen, ids, y+.1, y+.2, controls, {'better_simple': y}, True)['passes_gate']
    assert not m.evaluate_arrays(screen, ids, y+.1, y+.2, {}, {'mean': y+.2}, True)['sufficient']
    assert not m.evaluate_arrays(screen, ids, y+.1, y+.2, controls, {}, True)['sufficient']
    better_controls = {name: y for name in controls}
    assert not m.evaluate_arrays(screen, ids, y+.1, y+.2, better_controls, {'mean': y+.2}, True)['passes_gate']


def test_integrity_ordering_and_partial_run_refusal(tmp_path):
    p = tmp_path/'input'
    p.write_bytes(b'original')
    frozen = {str(p): m.sha256(p)}
    m.verify_files(frozen)
    p.write_bytes(b'drift')
    with pytest.raises(ValueError, match='Frozen input differs'):
        m.verify_files(frozen)
    m.write_json(tmp_path/'protocol.json', dict(code={}, files={}))
    m.write_json(tmp_path/'prefit_tests.json', dict(code={}, returncode=0))
    with pytest.raises(ValueError, match='Passing tests'):
        m.preconditions(tmp_path)


def test_guard_checks_resource_envelope_without_fitting(tmp_path, monkeypatch):
    config = dict(wall_seconds=10, rss_bytes=1, reserve_bytes=0, artifact_bytes=10**6)
    with pytest.raises(RuntimeError, match='memory'):
        m.Guard(tmp_path, config).check()
    config['rss_bytes'] = 10**15
    guard = m.Guard(tmp_path, config, started=-10**15)
    with pytest.raises(RuntimeError, match='CPU/wall'):
        guard.check()
    config['reserve_bytes'] = 10**20
    with pytest.raises(RuntimeError, match='reserve'):
        m.Guard(tmp_path, config).check()
    config['reserve_bytes'] = 0
    with pytest.raises(RuntimeError, match='CPU/wall'):
        m.Guard(tmp_path, config, consumed={'wall_seconds': 11}).check()


def test_no_scoring_without_frozen_selections(tmp_path):
    m.write_json(tmp_path/'protocol.json', dict(code={}, files={}))
    m.write_json(tmp_path/'prefit_tests.json', dict(code={}, returncode=1))
    with pytest.raises(ValueError, match='Passing tests'):
        m.score(tmp_path)
