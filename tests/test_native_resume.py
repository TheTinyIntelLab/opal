"""Synthetic custody and ten-fit native recovery tests; no biological fits."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from tools import resume_native_blocks as m


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    b = m.base
    # Do not leak the recovery module's runtime installation into other tests.
    monkeypatch.setattr(b, 'CODE', b.CODE)
    monkeypatch.setattr(b, 'preconditions', b.preconditions)
    monkeypatch.setattr(m, 'OLD_PRECONDITIONS', m.OLD_PRECONDITIONS)
    monkeypatch.setattr(b, 'signature', lambda: {'old': 'fixed', 'recovery': 'new'})
    monkeypatch.setattr(m, 'old_signature', lambda: {'old': 'fixed'})
    monkeypatch.setattr(b, 'dependencies', lambda: {'synthetic-native': 'test'})
    monkeypatch.setattr(b, 'load_official', lambda source: None)
    monkeypatch.setattr(b.block, 'signature', lambda: {})
    pairs = np.array([(a, c) for a in range(6) for c in range(a + 1, 6)])
    y = pairs.sum(1) / 10
    screen = dict(genes=np.array(list('ABCDEF')), pairs=pairs, singles=np.zeros(6),
                  responses=y, replicates=np.column_stack((y - .01, y + .01)))
    splits = []
    for fold in range(5):
        split = np.zeros(len(pairs), dtype=np.int8)
        split[fold], split[fold + 5] = 2, 1
        splits.append(split)
    screens = {task: screen for task in b.TASKS}
    monkeypatch.setattr(b, 'load_screen_roster', lambda *args: screens)
    monkeypatch.setattr(b, 'block_assignments', lambda _: splits)
    block_run, source, output = tmp_path / 'block', tmp_path / 'source', tmp_path / 'output'
    block_run.mkdir()
    np.savez(block_run / 'assignments.npz', **{f'{task}__{fold}': split for task in b.TASKS
                                            for fold, split in enumerate(splits)})
    bp = dict(code={}, files={}, original='unused', prepared='unused', panels={task: {
        'folds': [dict(eligible=True, train_pairs=13, validation_pairs=1) for _ in range(5)]} for task in b.TASKS})
    b.write_json(block_run / 'protocol.json', bp)
    monkeypatch.setattr(b.block, 'preconditions', lambda _: bp)
    b.write_json(block_run / 'selection.json', {'frozen': True})
    b.write_json(block_run / 'prefit_tests.json', {'passed': True})
    for task in b.TASKS:
        ids = np.arange(5)
        np.savez(block_run / (task + '_predictions.npz'), test_indices=ids, candidate=y[ids] + .1,
                 baseline=y[ids] + .2, **{f'ref_{i}': y[ids] + .2 + i / 100 for i in range(12)})
    b.write_json(block_run / 'results.json', dict(protocol_sha256=b.sha256(block_run / 'protocol.json'),
        selection_sha256=b.sha256(block_run / 'selection.json'), tasks={task: {} for task in b.TASKS},
        prediction_hashes={task: b.sha256(block_run / (task + '_predictions.npz')) for task in b.TASKS}))
    b.freeze(SimpleNamespace(output=source, block_run=block_run, source=tmp_path))
    old = b.read_json(source / 'protocol.json')
    old['code'] = m.old_signature()
    b.write_json(source / 'protocol.json', old)
    b.write_json(source / 'prefit_tests.json', dict(code=old['code'], dependencies=b.dependencies(), returncode=0,
        protocol_sha256=b.sha256(source / 'protocol.json'), started_utc=b.now(), completed_utc=b.now()))
    b.write_json(source / 'run_started.json', dict(started_utc=b.now(),
        protocol_sha256=b.sha256(source / 'protocol.json'), prefit_tests_sha256=b.sha256(source / 'prefit_tests.json'),
        block_result_files=b.bind_block_results(block_run)))

    class MockNative:
        def __init__(self, split):
            self.fit_rows = {'train': np.random.default_rng(b.CONFIG['seed']).permutation(np.flatnonzero(split == 0)).tolist(),
                             'val': np.flatnonzero(split == 1).tolist()}
            self.best_epoch, self.parameter_count = 100, 1
            self.model_args = {'d_embed': 128, 'd_pheno_hid': 256, 'model_type': 'both', 'p_dropout': .1}
            self.training_metrics = {key: np.linspace(.2, .1, 100).tolist() for key in ('train_loss', 'val_loss')}

        def predict(self, p):
            return p.sum(1) / 10 + .05

        def save(self, dest):
            dest.mkdir(parents=True)
            np.savez(dest / 'state.npz', state=np.array([1.]))
            b.write_json(dest / 'metadata.json', dict(revision=b.REVISION,
                config={key: b.CONFIG[key] for key in ('seed', 'epochs', 'batch_size')},
                fit_rows=self.fit_rows, best_epoch=self.best_epoch, parameter_count=self.parameter_count,
                model_args=self.model_args, training_metrics=self.training_metrics, state_sha256=b.sha256(dest / 'state.npz')))

    records = []
    for task in m.INHERITED_TASKS:
        for fold, split in enumerate(splits):
            model = MockNative(split)
            dest = source / 'checkpoints' / task / f'fold-{fold}'
            model.save(dest)
            selected = dict(task=task, fold=fold, checkpoint=str(dest.relative_to(source)),
                files={str((dest / name).relative_to(source)): b.sha256(dest / name) for name in ('metadata.json', 'state.npz')},
                split_sha256=b.array_hash(split), counts={k: len(v) for k, v in model.fit_rows.items()},
                fit_order_sha256={k: b.array_hash(np.asarray(v, dtype=np.int64)) for k, v in model.fit_rows.items()},
                best_epoch=model.best_epoch, model_args=model.model_args, parameter_count=model.parameter_count,
                training_metrics=model.training_metrics)
            path = source / 'selections' / task / f'fold-{fold}.json'
            b.write_json(path, selected)
            records.append(dict(task=task, fold=fold, path=str(path.relative_to(source)), sha256=b.sha256(path)))
    consumed = dict(wall_seconds=2042.0365001000027, cpu_seconds=2041.15625, peak_rss_bytes=602046464)
    b.write_json(source / 'fit_progress.json', dict(completed=records, resources=consumed))
    b.write_json(source / 'interruption.json', dict(test_scoring=False, completed_fits=20))
    calls, charges = [], []

    def fit_native(native_source, genes, p, singles, target, split, config, check):
        assert config.epochs == 100 and config.seed == 20261022 and config.batch_size == 1024
        assert np.isnan(target[split == 2]).all() and np.isfinite(target[split != 2]).all()
        assert not (output / 'results.json').exists()
        calls.append(1)
        check()
        return MockNative(split)

    def load(native_source, dest):
        rows = b.read_json(dest / 'metadata.json')['fit_rows']
        split = np.full(len(pairs), 2)
        for key, label in [('train', 0), ('val', 1)]:
            split[rows[key]] = label
        return MockNative(split)

    class MockGuard:
        def __init__(self, output, config, consumed=None):
            self.consumed = consumed or {}
            charges.append(self.consumed)

        def check(self):
            pass

        def receipt(self):
            return dict(wall_seconds=self.consumed.get('wall_seconds', 0) + 1,
                        cpu_seconds=self.consumed.get('cpu_seconds', 0) + 1,
                        peak_rss_bytes=self.consumed.get('peak_rss_bytes', 0))

    monkeypatch.setattr(b, 'fit_native', fit_native)
    monkeypatch.setattr(b.NativeNAIAD, 'load', load)
    monkeypatch.setattr(b.block, 'Guard', MockGuard)
    args = SimpleNamespace(output=output, block_run=block_run, source=tmp_path, inherit_run=source,
                           aborted_run=None, failed_run=None)
    return SimpleNamespace(args=args, source=source, output=output, block_run=block_run, bp=bp,
                           splits=splits, records=records, consumed=consumed, calls=calls, charges=charges)


def prereq(c):
    b = m.base
    p = b.read_json(c.output / 'protocol.json')
    b.write_json(c.output / 'prefit_tests.json', dict(code=p['code'], dependencies=p['dependencies'], returncode=0,
        protocol_sha256=b.sha256(c.output / 'protocol.json'), started_utc=b.now(), completed_utc=b.now()))


def test_recovery_fits_only_ten_preserves_twenty_and_replays(campaign):
    c, b = campaign, m.base
    m.freeze(c.args)
    prereq(c)
    m.fit(c.output)
    selected = b.read_json(c.output / 'selection.json')
    assert len(c.calls) == 10
    assert len(selected['fits']) == len({(r['task'], r['fold']) for r in selected['fits']}) == 30
    assert c.charges[0] == c.consumed
    for record in c.records:
        assert (c.output / record['path']).read_bytes() == (c.source / record['path']).read_bytes()
        old = b.read_json(c.source / record['path'])
        for path in old['files']:
            assert (c.output / path).read_bytes() == (c.source / path).read_bytes()
    b.score(c.output)
    b.score(c.output, replay=True)
    assert b.read_json(c.output / 'replay.json')['status'].startswith('exact')
    assert c.charges[-1]['wall_seconds'] > c.consumed['wall_seconds']
    with pytest.raises(FileExistsError):
        m.fit(c.output)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'config', 'dependencies', 'split', 'tamper', 'scored', 'test_order', 'path_alias'])
def test_recovery_rejects_invalid_inheritance(campaign, mutation):
    c, b = campaign, m.base
    if mutation in ('missing', 'duplicate', 'path_alias'):
        path = c.source / 'fit_progress.json'
        p = b.read_json(path)
        if mutation == 'missing':
            p['completed'].pop()
        elif mutation == 'duplicate':
            p['completed'][-1] = p['completed'][0]
        else:
            p['completed'][0]['path'] = str(Path('..') / c.source.name / p['completed'][0]['path'])
        b.write_json(path, p)
    elif mutation in ('config', 'dependencies'):
        path = c.source / 'protocol.json'
        p = b.read_json(path)
        p[mutation]['epochs' if mutation == 'config' else 'synthetic-native'] = 99
        b.write_json(path, p)
    elif mutation == 'split':
        # Change selection while keeping its outer checksum valid.
        path = c.source / c.records[0]['path']
        p = b.read_json(path)
        p['split_sha256'] = 'wrong'
        b.write_json(path, p)
        progress = b.read_json(c.source / 'fit_progress.json')
        progress['completed'][0]['sha256'] = b.sha256(path)
        b.write_json(c.source / 'fit_progress.json', progress)
    elif mutation == 'tamper':
        (c.source / 'checkpoints' / m.INHERITED_TASKS[0] / 'fold-0' / 'state.npz').write_bytes(b'tamper')
    elif mutation == 'scored':
        b.write_json(c.source / 'score_started.json', {})
    else:
        path = c.source / 'prefit_tests.json'
        p = b.read_json(path)
        p['started_utc'] = '1900'
        b.write_json(path, p)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.freeze(c.args)


def test_frozen_inherited_bytes_are_rechecked_before_fit(campaign):
    c = campaign
    m.freeze(c.args)
    prereq(c)
    (c.source / c.records[0]['path']).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError):
        m.fit(c.output)
    assert not c.calls


def test_operational_bom_receipt_is_bound_and_live_pid_refused(campaign, monkeypatch):
    c, b = campaign, m.base
    aborted = c.source.parent / 'aborted'
    aborted.mkdir()
    (aborted / 'operational_stop.json').write_text('{"pid": 123456, "test_scoring": false}', encoding='utf-8-sig')
    c.args.aborted_run = aborted
    monkeypatch.setattr(m.psutil, 'pid_exists', lambda pid: True)
    with pytest.raises(ValueError, match='stopped'):
        m.freeze(c.args)
    monkeypatch.setattr(m.psutil, 'pid_exists', lambda pid: False)
    m.freeze(c.args)
    manifest = b.read_json(c.output / 'inheritance_manifest.json')
    assert str((aborted / 'operational_stop.json').resolve()) in manifest['operational_receipts']
