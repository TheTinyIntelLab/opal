"""Synthetic family selection, custody, alignment and exact replay checks."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import select_block_family as m


def test_exact_tie_preference_and_validation_only_choice():
    y = np.array([1., 2.])
    predictions = {f: y.copy() for f in m.FAMILIES}
    assert m.choose_family(y, predictions)[0] == 'simple'
    predictions['simple'] += 1
    assert m.choose_family(y, predictions)[0] == 'rank8'
    predictions['rank8'] += 1
    assert m.choose_family(y, predictions)[0] == 'native'
    with pytest.raises(ValueError):
        m.choose_family(y, {'native': y})


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    b = m.native
    br, nr, out = tmp_path / 'block', tmp_path / 'native', tmp_path / 'family'
    br.mkdir()
    nr.mkdir()
    pairs = np.array([(a, c) for a in range(6) for c in range(a + 1, 6)])
    y = pairs.sum(1) / 10
    screen = dict(genes=np.array(list('ABCDEF')), pairs=pairs, singles=np.zeros(6),
                  responses=y.copy(), replicates=np.column_stack((y - .01, y + .01)))
    splits = []
    for fold in range(5):
        split = np.zeros(len(pairs), dtype=np.int8)
        split[fold], split[fold + 5] = 2, 1
        splits.append(split)
    assignments = {f'{task}__{fold}': split for task in b.TASKS for fold, split in enumerate(splits)}
    np.savez(br / 'assignments.npz', **assignments)
    bp = dict(files={}, original='unused', prepared='unused')
    b.write_json(br / 'protocol.json', bp)
    b.write_json(br / 'prefit_tests.json', {})
    np_ = dict(code={}, files={}, source=str(tmp_path), block_run=str(br.resolve()))
    b.write_json(nr / 'protocol.json', np_)
    for name in ('inheritance_manifest.json', 'prefit_tests.json', 'run_started.json'):
        b.write_json(nr / name, {})
    records = {'q2': [], 'native': []}
    for task in b.TASKS:
        for fold, split in enumerate(splits):
            qdir, ndir = br / task / f'fold-{fold}', nr / 'checkpoints' / task / f'fold-{fold}'
            qdir.mkdir(parents=True)
            ndir.mkdir(parents=True)
            for name in ('ref.npz', 'aligned.npz'):
                np.savez(qdir / name, synthetic=np.array([1.]))
            np.savez(ndir / 'state.npz', synthetic=np.array([1.]))
            b.write_json(ndir / 'metadata.json', {'synthetic': True})
            q = dict(task=task, fold=fold, chosen_reference=0, reference_checkpoints=['ref.npz'],
                     models=[dict(name='aligned', checkpoint='aligned.npz')],
                     files={name: b.sha256(qdir / name) for name in ('ref.npz', 'aligned.npz')})
            b.write_json(qdir / 'selection.json', q)
            n = dict(task=task, fold=fold, checkpoint=str(ndir.relative_to(nr)),
                files={str((ndir / name).relative_to(nr)): b.sha256(ndir / name) for name in ('metadata.json', 'state.npz')},
                best_epoch=1, model_args={}, parameter_count=1, training_metrics={}, split_sha256=b.array_hash(split))
            npath = nr / 'selections' / task / f'fold-{fold}.json'
            b.write_json(npath, n)
            for key, root, path in [('q2', br, qdir / 'selection.json'), ('native', nr, npath)]:
                records[key].append(dict(task=task, fold=fold, path=str(path.relative_to(root)), sha256=b.sha256(path)))
    for key, root in [('q2', br), ('native', nr)]:
        b.write_json(root / 'selection.json', dict(fits=records[key], protocol_sha256=b.sha256(root / 'protocol.json'),
                     run_started_sha256=b.sha256(nr / 'run_started.json')))
    ids = np.arange(5)
    for task in b.TASKS:
        np.savez(br / (task + '_predictions.npz'), test_indices=ids, candidate=y[ids] + .1, baseline=y[ids] + .2,
                 **{f'ref_{i}': y[ids] + .2 + i / 100 for i in range(12)})
    b.write_json(br / 'results.json', dict(protocol_sha256=b.sha256(br / 'protocol.json'),
        selection_sha256=b.sha256(br / 'selection.json'), tasks={t: {} for t in b.TASKS},
        prediction_hashes={t: b.sha256(br / (t + '_predictions.npz')) for t in b.TASKS}))
    monkeypatch.setattr(b, 'dependencies', lambda: {'synthetic': 'test'})
    monkeypatch.setattr(b.block, 'preconditions', lambda _: bp)
    monkeypatch.setattr(m.recovery, 'activate', lambda: None)
    monkeypatch.setattr(m.recovery, 'preconditions', lambda _: (np_, bp))
    monkeypatch.setattr(b, 'load_screen_roster', lambda *args: {task: screen for task in b.TASKS})
    monkeypatch.setattr(b, 'block_assignments', lambda _: splits)

    class Simple:
        def predict(self, p):
            return p.sum(1) / 10 + .2

    class Completion:
        def predict(self, p):
            return np.full(len(p), -.1)

    class Native:
        best_epoch, model_args, parameter_count, training_metrics = 1, {}, 1, {}

        def predict(self, p):
            return p.sum(1) / 10 + .05

    monkeypatch.setattr(m.ScalarReference, 'load', lambda _: Simple())
    monkeypatch.setattr(m.ScalarCompletion, 'load', lambda _: Completion())
    monkeypatch.setattr(b.NativeNAIAD, 'load', lambda *args: Native())

    class Guard:
        def __init__(self, output, config, consumed=None):
            self.consumed = consumed or {}

        def check(self):
            pass

        def receipt(self):
            return dict(wall_seconds=self.consumed.get('wall_seconds', 0) + 1,
                        cpu_seconds=self.consumed.get('cpu_seconds', 0) + 1, peak_rss_bytes=0)

    monkeypatch.setattr(b.block, 'Guard', Guard)
    return SimpleNamespace(args=SimpleNamespace(output=out, block_run=br, native_run=nr),
        output=out, block_run=br, native_run=nr, screen=screen, records=records, assignments=assignments)


def prereq(c):
    b = m.native
    p = b.read_json(c.output / 'protocol.json')
    b.write_json(c.output / 'prefit_tests.json', dict(started_utc=b.now(), completed_utc=b.now(),
        code=p['code'], dependencies=p['dependencies'], returncode=0,
        protocol_sha256=b.sha256(c.output / 'protocol.json')))


def native_scoring(c):
    b, nr = m.native, c.native_run
    ids = np.arange(5)
    for task in b.TASKS:
        np.savez(nr / (task + '_predictions.npz'), test_indices=ids, native=c.screen['pairs'][ids].sum(1) / 10 + .05)
    b.write_json(nr / 'score_started.json', dict(started_utc=b.now()))
    b.write_json(nr / 'results.json', dict(selection_sha256=b.sha256(nr / 'selection.json'),
        protocol_sha256=b.sha256(nr / 'protocol.json'), prediction_hashes={task: b.sha256(nr / (task + '_predictions.npz')) for task in b.TASKS}))
    b.write_json(nr / 'replay.json', dict(results_sha256=b.sha256(nr / 'results.json')))


def test_full_lifecycle_freezes_thirty_choices_before_scoring_and_replays(campaign):
    c, b = campaign, m.native
    m.freeze(c.args)
    prereq(c)
    with pytest.raises(FileNotFoundError):
        m.score(c.output)
    m.select(c.output)
    choices = b.read_json(c.output / 'choices.json')['choices']
    assert len(choices) == len({(r['task'], r['fold']) for r in choices}) == 30
    assert all(r['family'] == 'native' for r in choices)
    assert not (c.native_run / 'score_started.json').exists()
    with pytest.raises(FileNotFoundError):
        m.score(c.output)  # Fixed native test results are not yet available.
    native_scoring(c)
    m.score(c.output)
    m.score(c.output, replay=True)
    assert b.read_json(c.output / 'replay.json')['status'].startswith('exact')
    for r in b.read_json(c.output / 'results.json')['tasks'].values():
        assert r['family_fold_counts'] == dict(simple=0, rank8=0, native=5)
        assert r['comparisons']['native']['relative_gain'] == pytest.approx(0)
    with pytest.raises(FileExistsError):
        m.select(c.output)


def test_held_test_labels_cannot_change_family_choices(campaign):
    c, b = campaign, m.native
    m.freeze(c.args)
    prereq(c)
    # All five held test rows are distinct from every shared validation row.
    c.screen['responses'][:5] = np.nan
    m.select(c.output)
    before = b.read_json(c.output / 'choices.json')['choices']
    assert all(r['family'] == 'native' for r in before)
    # Direct validation calls use the same local models and val rows regardless
    # of sentinel test targets, covering actual predictor integration too.
    p = b.read_json(c.output / 'protocol.json')
    q, n, _ = m.bind_selections(p)
    for fold in range(5):
        rows, predictions = m.validation_predictions(p, c.screen, c.assignments[f'horlbeck_K562__{fold}'],
                                                    q[('horlbeck_K562', fold)], n[('horlbeck_K562', fold)])
        assert m.choose_family(c.screen['responses'][rows], predictions)[0] == 'native'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'checkpoint', 'native_scored'])
def test_invalid_checkpoint_custody_or_early_native_scoring_is_refused(campaign, mutation):
    c, b = campaign, m.native
    m.freeze(c.args)
    prereq(c)
    if mutation in ('missing', 'duplicate'):
        path = c.native_run / 'selection.json'
        selected = b.read_json(path)
        if mutation == 'missing':
            selected['fits'].pop()
        else:
            selected['fits'][-1] = selected['fits'][0]
        b.write_json(path, selected)
    elif mutation == 'checkpoint':
        (c.native_run / 'checkpoints' / 'horlbeck_K562' / 'fold-0' / 'state.npz').write_bytes(b'tamper')
    else:
        b.write_json(c.native_run / 'score_started.json', {})
    with pytest.raises(ValueError):
        m.select(c.output)


def test_fixed_array_alignment_and_duplicate_choices_are_refused(campaign):
    c, b = campaign, m.native
    m.freeze(c.args)
    prereq(c)
    m.select(c.output)
    choices = b.read_json(c.output / 'choices.json')
    choices['choices'][-1] = choices['choices'][0]
    b.write_json(c.output / 'choices.json', choices)
    with pytest.raises(ValueError, match='thirty'):
        m.score(c.output)
    fixed = dict(test_indices=np.arange(5)[::-1], simple=np.zeros(5), rank8=np.zeros(5), native=np.zeros(5))
    with pytest.raises(ValueError, match='order'):
        m.combine(fixed, {}, c.assignments, 'horlbeck_K562')


def test_saved_prediction_tamper_fails_replay(campaign):
    c, b = campaign, m.native
    m.freeze(c.args)
    prereq(c)
    m.select(c.output)
    native_scoring(c)
    m.score(c.output)
    path = c.output / 'horlbeck_K562_predictions.npz'
    np.savez(path, test_indices=np.arange(5), selector=np.zeros(5))
    with pytest.raises(ValueError, match='prediction replay'):
        m.score(c.output, replay=True)


def test_native_array_order_and_scoring_chronology_are_checked(campaign):
    c, b = campaign, m.native
    m.freeze(c.args)
    prereq(c)
    m.select(c.output)
    native_scoring(c)
    path = c.native_run / 'score_started.json'
    b.write_json(path, dict(started_utc='1900'))
    with pytest.raises(ValueError, match='precede'):
        m.score(c.output)
    b.write_json(path, dict(started_utc=b.now()))
    task = 'horlbeck_K562'
    path = c.native_run / (task + '_predictions.npz')
    fixed = b.load_npz(path)
    np.savez(path, test_indices=fixed['test_indices'][::-1], native=fixed['native'][::-1])
    result = b.read_json(c.native_run / 'results.json')
    result['prediction_hashes'][task] = b.sha256(path)
    b.write_json(c.native_run / 'results.json', result)
    b.write_json(c.native_run / 'replay.json', dict(results_sha256=b.sha256(c.native_run / 'results.json')))
    with pytest.raises(ValueError, match='alignment'):
        m.score(c.output)
