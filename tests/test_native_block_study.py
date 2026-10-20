"""Synthetic lifecycle and comparison prerequisites for native block execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('native_block_study', Path(__file__).parents[1]/'tools/study_native_blocks.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic():
    pairs = np.array([(a, b) for a in range(6) for b in range(a+1, 6)])
    y = pairs.sum(1)/10
    screen = dict(genes=np.array(list('ABCDEF')), pairs=pairs, singles=np.zeros(6),
                  responses=y, replicates=np.column_stack((y-.01, y+.01)))
    assignments = []
    for fold in range(5):
        split = np.zeros(len(pairs), dtype=np.int8)
        split[fold], split[fold+5] = 2, 1
        assignments.append(split)
    return screen, assignments


def q2_arrays(screen, indices):
    y = screen['responses'][indices]
    return dict(test_indices=indices, candidate=y+.1, baseline=y+.2,
                **{f'ref_{i}': y+.2+i/100 for i in range(12)})


def test_pairing_reference_menu_and_conditional_comparisons():
    screen, _ = synthetic()
    ids = np.arange(5)
    q2 = q2_arrays(screen, ids)
    result = m.evaluate_arrays(screen, ids, screen['responses'][ids]+.05, q2, draws=25)
    assert result['comparisons']['selected_simple']['relative_gain'] == pytest.approx(.9375)
    assert len(result['all_simple']) == 12 and len(result['infections']) == 2
    assert result['promotion'].startswith('none')
    for comparison in result['infections']:
        assert comparison['rank8']['paired_gene_difference']['requested_draws'] == 25
    with pytest.raises(ValueError, match='order'):
        m.evaluate_arrays(screen, ids[::-1], screen['responses'][ids]+.05, q2, draws=25)
    del q2['ref_0']
    with pytest.raises(ValueError, match='twelve'):
        m.evaluate_arrays(screen, ids, screen['responses'][ids]+.05, q2, draws=25)


def test_all_selections_and_checkpoint_tamper_are_required(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'TASKS', ('toy',))
    with pytest.raises(ValueError, match='thirty'):
        list(m.selection_records(tmp_path, {'fits': []}))
    p = tmp_path/'artifact'
    p.write_bytes(b'original')
    records = []
    for fold in range(5):
        path = tmp_path/f'{fold}.json'
        m.write_json(path, dict(task='toy', fold=fold, files={'artifact': m.sha256(p)}))
        records.append(dict(task='toy', fold=fold, path=path.name, sha256=m.sha256(path)))
    assert len(list(m.selection_records(tmp_path, {'fits': records}))) == 5
    p.write_bytes(b'tamper')
    with pytest.raises(ValueError, match='Frozen input'):
        list(m.selection_records(tmp_path, {'fits': records}))


def test_full_synthetic_lifecycle_with_mock_native_training(tmp_path, monkeypatch):
    # Native architecture training is separately tested against the real pinned
    # source. Here a small numerical stand-in tests runner ordering and custody.
    monkeypatch.setattr(m, 'TASKS', ('toy',))
    monkeypatch.setattr(m, 'load_official', lambda source: None)
    monkeypatch.setattr(m, 'dependencies', lambda: {'synthetic-native': 'test'})
    monkeypatch.setattr(m.block, 'signature', lambda: {})
    screen, splits = synthetic()
    block_run, output = tmp_path/'block', tmp_path/'native'
    block_run.mkdir()
    np.savez(block_run/'assignments.npz', **{f'toy__{i}': split for i, split in enumerate(splits)})
    bp = dict(code={}, files={}, original='unused', prepared='unused',
              panels={'toy': {'folds': [dict(eligible=True) for _ in range(5)]}})
    m.write_json(block_run/'protocol.json', bp)
    monkeypatch.setattr(m.block, 'preconditions', lambda path: bp)
    monkeypatch.setattr(m, 'load_screen_roster', lambda *args: {'toy': screen})
    monkeypatch.setattr(m, 'block_assignments', lambda s: splits)
    m.freeze(SimpleNamespace(output=output, block_run=block_run, source=tmp_path))
    assert not (output/'selection.json').exists()
    protocol = m.read_json(output/'protocol.json')
    m.write_json(output/'prefit_tests.json', dict(code=m.signature(), dependencies=m.dependencies(), returncode=0,
        protocol_sha256=m.sha256(output/'protocol.json'), started_utc=m.now()))
    with pytest.raises(FileNotFoundError):
        m.fit(output)  # A complete Q2 comparison must already exist.
    m.write_json(block_run/'selection.json', {'frozen': True})
    m.write_json(block_run/'prefit_tests.json', {'passed': True})
    ids = np.arange(5)
    np.savez(block_run/'toy_predictions.npz', **q2_arrays(screen, ids))
    m.write_json(block_run/'results.json', dict(protocol_sha256=m.sha256(block_run/'protocol.json'),
        selection_sha256=m.sha256(block_run/'selection.json'), tasks={'toy': {}},
        prediction_hashes={'toy': m.sha256(block_run/'toy_predictions.npz')}))
    calls = []
    class MockNative:
        def __init__(self, split):
            self.fit_rows = {key: np.flatnonzero(split == label).tolist() for key, label in [('train', 0), ('val', 1)]}
            self.best_epoch, self.parameter_count = 1, 1
            self.model_args, self.training_metrics = {'model_type': 'both'}, {'train_loss': [.1], 'val_loss': [.1]}
        def predict(self, pairs):
            return pairs.sum(1)/10+.05
        def save(self, dest):
            dest.mkdir(parents=True)
            m.write_json(dest/'metadata.json', dict(fit_rows=self.fit_rows))
            np.savez(dest/'state.npz', state=np.array([1.]))
    def fit_native(source, genes, pairs, singles, targets, split, config, check):
        assert config.epochs == 100
        assert np.isnan(targets[split == 2]).all() and np.isfinite(targets[split != 2]).all()
        assert not (output/'results.json').exists()
        calls.append(1)
        check()
        return MockNative(split)
    def load(source, dest):
        rows = m.read_json(dest/'metadata.json')['fit_rows']
        split = np.full(len(screen['pairs']), 2)
        for key, label in [('train', 0), ('val', 1)]:
            split[rows[key]] = label
        return MockNative(split)
    monkeypatch.setattr(m, 'fit_native', fit_native)
    monkeypatch.setattr(m.NativeNAIAD, 'load', load)
    m.fit(output)
    assert len(calls) == 5
    for record in m.read_json(output/'selection.json')['fits']:
        compact = m.read_json(output/record['path'])
        assert 'fit_rows' not in compact and 'fit_order_sha256' in compact
    with pytest.raises(FileExistsError):
        m.fit(output)
    m.score(output)
    m.score(output, replay=True)
    assert m.read_json(output/'replay.json')['status'].startswith('exact')
    with pytest.raises(FileExistsError):
        m.score(output)
    with pytest.raises(FileExistsError):
        m.score(output, replay=True)


def test_failed_attempt_cannot_be_reentered(tmp_path):
    m.write_json(tmp_path/'failure.json', {'error': 'resource cap'})
    with pytest.raises(FileExistsError, match='failed'):
        m.preconditions(tmp_path)
