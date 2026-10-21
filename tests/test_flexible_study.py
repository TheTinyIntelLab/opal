from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from tools import study_flexible_singles as m


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    b = m.block
    br, q, out = [tmp_path/n for n in ('block', 'q1b', 'flex')]
    br.mkdir()
    q.mkdir()
    pairs = np.array([(a, c) for a in range(20) for c in range(a+1, 20)])
    np.random.default_rng(8).shuffle(pairs)
    singles = np.linspace(-1., .1, 20)
    y = singles[pairs].sum(1)+.2*np.sin(singles[pairs].prod(1)*4)
    screen = dict(genes=np.array([f'G{i}' for i in range(20)]), pairs=pairs,
        singles=singles, responses=y, replicates=np.column_stack((y-.01, y+.01)))
    screens = {t: screen for t in b.TASKS}
    splits = []
    for fold in range(5):
        split = np.zeros(len(pairs), dtype=np.int8)
        split[np.arange(len(pairs)) % 5 == fold] = 2
        non_test = np.flatnonzero(split != 2)
        split[non_test[:30]] = 1
        splits.append(split)
    np.savez_compressed(br/'assignments.npz', **{f'{t}__{f}': s for t in b.TASKS for f, s in enumerate(splits)})
    bp = dict(original='unused', prepared='unused', code={}, files={})
    for name in ('protocol.json', 'selection.json', 'replay.json'):
        b.write_json(br/name, bp if name == 'protocol.json' else {})
    hashes = {}
    ids = np.concatenate([np.flatnonzero(s == 2) for s in splits])
    for task in b.TASKS:
        p = br/(task+'_predictions.npz')
        np.savez_compressed(p, test_indices=ids, baseline=y[ids]+.2, candidate=y[ids]+.1)
        hashes[task] = b.sha256(p)
    b.write_json(br/'results.json', dict(prediction_hashes=hashes))
    for name in ('registration.json', 'prerequisites.json'):
        b.write_json(q/name, {})
    b.write_json(q/'results.json', {'integrity_gate': True})
    b.write_json(q/'replay.json', {'results_sha256': b.sha256(q/'results.json')})
    monkeypatch.setattr(b, 'preconditions', lambda path: bp)
    monkeypatch.setattr(b, 'load_screen_roster', lambda *args: screens)
    monkeypatch.setattr(m, 'signature', lambda: {'synthetic': 'fixed'})
    return SimpleNamespace(args=SimpleNamespace(block_run=br, q1b_run=q, output=out),
                           out=out, br=br, q=q, screens=screens, splits=splits)


def qualify(c):
    m.freeze(c.args)
    p = m.block.read_json(c.out/'protocol.json')
    m.block.write_json(c.out/'prefit_tests.json', dict(returncode=0, code=p['code'],
        protocol_sha256=m.block.sha256(c.out/'protocol.json'), started_utc=m.block.now()))


def test_complete_lifecycle_selection_before_scoring_exact_replay(campaign):
    c = campaign
    qualify(c)
    with threadpool_limits(limits=1):
        m.fit(c.out)
        selected = m.block.read_json(c.out/'selection.json')
        assert len(selected['fits']) == 30
        assert not (c.out/'score_started.json').exists()
        m.score(c.out)
        m.score(c.out, True)
    assert m.block.read_json(c.out/'replay.json')['status'].startswith('exact')
    for call in (lambda: m.fit(c.out), lambda: m.score(c.out), lambda: m.score(c.out, True)):
        with pytest.raises(FileExistsError):
            call()


def test_missing_or_failed_q1b_blocks_fit_without_start(campaign):
    c = campaign
    qualify(c)
    m.block.write_json(c.q/'results.json', {'integrity_gate': False})
    with pytest.raises(ValueError, match='Q1b'):
        m.fit(c.out)
    assert not (c.out/'run_started.json').exists()


def test_selection_must_be_complete_and_checkpoint_tamper_refused(campaign):
    c = campaign
    qualify(c)
    m.block.write_json(c.out/'selection.json', {'fits': []})
    with pytest.raises(ValueError, match='thirty'):
        m.score(c.out)


def test_real_saved_checkpoint_tamper_stops_before_test_scoring(campaign):
    c = campaign
    qualify(c)
    with threadpool_limits(limits=1):
        m.fit(c.out)
    record = m.block.read_json(c.out/'selection.json')['fits'][0]
    path = c.out/record['path']
    selected = m.block.read_json(path)
    file = path.parent/next(iter(selected['files']))
    file.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='differs'):
        m.score(c.out)
    assert not (c.out/'score_started.json').exists()


def test_one_fold_test_sentinels_leave_its_models_and_choices_unchanged(campaign):
    c = campaign
    qualify(c)
    with threadpool_limits(limits=1):
        m.fit(c.out)
    task = m.block.TASKS[0]
    first = m.block.read_json(c.out/task/'fold-0'/'selection.json')
    screen = c.screens[task]
    screen['responses'] = screen['responses'].copy()
    screen['responses'][c.splits[0] == 2] = 1e15
    c.args.output = c.out.parent/'sentinel'
    c.out = c.args.output
    qualify(c)
    with threadpool_limits(limits=1):
        m.fit(c.args.output)
    second = m.block.read_json(c.args.output/task/'fold-0'/'selection.json')
    assert first['validation_mse'] == second['validation_mse']
    assert first['chosen'] == second['chosen'] and first['winner'] == second['winner']
    assert first['files'] == second['files']


def test_validation_ties_and_incomplete_menu():
    chosen, winner = m.select_losses({'phenotype': [1., 1., 2., 3.], 'nodes': [1., 2., 3., 4.]})
    assert chosen == {'phenotype': 0, 'nodes': 0} and winner == 'phenotype'
    with pytest.raises(ValueError):
        m.select_losses({'phenotype': [1., 2.], 'nodes': [1., 2.]})


def test_scientific_gates_do_not_equate_absence_of_superiority_with_equivalence():
    p = np.array([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)])
    y = np.arange(6)/10
    replicate = np.column_stack((y, y))
    equal = m.acceptance(y, y+.1, y+.1, y+.1, p, replicate)
    assert equal['phenotype_conditionally_noninferior'] and not equal['completion_residual_gate']
    poor = m.acceptance(y, y+.3, y+.3, y+.1, p, replicate)
    assert not poor['phenotype_conditionally_noninferior'] and poor['completion_residual_gate']
