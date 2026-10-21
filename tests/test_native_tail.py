"""Synthetic tests for one registered four-fit native continuation."""
from types import SimpleNamespace

import pytest

from test_native_resume import campaign as recovered_campaign, prereq as recovery_prereq
from tools import resume_native_tail as m


@pytest.fixture
def campaign(recovered_campaign, monkeypatch):
    c, b = recovered_campaign, m.base
    m.recovery.freeze(c.args)
    recovery_prereq(c)
    m.recovery.fit(c.output)  # Existing stand-in fits only artificial values.
    (c.output / 'selection.json').unlink()
    progress = b.read_json(c.output / 'fit_progress.json')
    progress['completed'] = progress['completed'][:26]
    progress['resources'] = dict(wall_seconds=4187.432, cpu_seconds=4186., peak_rss_bytes=602046464)
    b.write_json(c.output / 'fit_progress.json', progress)
    b.write_json(c.output / 'interruption.json', dict(test_scoring=False, completed_fits=26))
    for task, fold in m.MISSING:
        directory = c.output / 'checkpoints' / task / f'fold-{fold}'
        for name in ('metadata.json', 'state.npz'):
            (directory / name).unlink()
        directory.rmdir()
        (c.output / 'selections' / task / f'fold-{fold}.json').unlink()
    c.calls.clear()
    c.charges.clear()
    monkeypatch.setattr(m, 'signature', lambda code: {'synthetic-tail': 'fixed'})
    amendment = c.source.parent / 'tail'
    return SimpleNamespace(native=c, args=SimpleNamespace(native_run=c.output, output=amendment), output=amendment)


def prereq(c):
    b = m.base
    p = b.read_json(c.output / 'protocol.json')
    b.write_json(c.output / 'prefit_tests.json', dict(code=p['code'], dependencies=p['dependencies'],
        returncode=0, protocol_sha256=b.sha256(c.output / 'protocol.json'), started_utc=b.now(), completed_utc=b.now()))


def test_tail_appends_exactly_four_without_changing_frozen_native_custody(campaign):
    c, b = campaign, m.base
    nr = c.native.output
    frozen = {name: (nr / name).read_bytes() for name in ('protocol.json', 'prefit_tests.json', 'run_started.json')}
    old_records = b.read_json(nr / 'fit_progress.json')['completed']
    old_bytes = {r['path']: (nr / r['path']).read_bytes() for r in old_records}
    m.freeze(c.args)
    prereq(c)
    m.fit(c.output)
    assert len(c.native.calls) == 4
    selected = b.read_json(nr / 'selection.json')
    assert len(selected['fits']) == len({(r['task'], r['fold']) for r in selected['fits']}) == 30
    assert c.native.charges[0]['wall_seconds'] == 4187.432
    assert all((nr / name).read_bytes() == data for name, data in frozen.items())
    assert all((nr / name).read_bytes() == data for name, data in old_bytes.items())
    m.score(c.output)
    m.score(c.output, replay=True)
    assert b.read_json(nr / 'replay.json')['status'].startswith('exact')
    with pytest.raises(FileExistsError):
        m.fit(c.output)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'state', 'recipe', 'split', 'scored'])
def test_tail_rejects_changed_or_incomplete_inheritance(campaign, mutation):
    c, b = campaign, m.base
    nr = c.native.output
    if mutation in ('missing', 'duplicate'):
        progress = b.read_json(nr / 'fit_progress.json')
        if mutation == 'missing':
            progress['completed'].pop()
        else:
            progress['completed'][-1] = progress['completed'][0]
        b.write_json(nr / 'fit_progress.json', progress)
    elif mutation == 'state':
        (nr / 'checkpoints' / 'parp2023' / 'fold-0' / 'state.npz').write_bytes(b'tamper')
    elif mutation == 'scored':
        b.write_json(nr / 'score_started.json', {})
    else:
        # Keep file checksums self-consistent to test semantic metadata checks.
        path = nr / 'checkpoints' / 'parp2023' / 'fold-0' / 'metadata.json'
        meta = b.read_json(path)
        if mutation == 'recipe':
            meta['config']['epochs'] = 99
        else:
            meta['fit_rows']['val'] = []
        b.write_json(path, meta)
        selection_path = nr / 'selections' / 'parp2023' / 'fold-0.json'
        r = b.read_json(selection_path)
        r['files'][str(path.relative_to(nr))] = b.sha256(path)
        b.write_json(selection_path, r)
        progress = b.read_json(nr / 'fit_progress.json')
        progress['completed'][-1]['sha256'] = b.sha256(selection_path)
        b.write_json(nr / 'fit_progress.json', progress)
    with pytest.raises(ValueError):
        m.freeze(c.args)


def test_tail_refuses_progress_drift_after_freeze(campaign):
    c, b = campaign, m.base
    m.freeze(c.args)
    prereq(c)
    path = c.native.output / 'fit_progress.json'
    progress = b.read_json(path)
    progress['resources']['wall_seconds'] += 1
    b.write_json(path, progress)
    with pytest.raises(ValueError, match='progress changed'):
        m.fit(c.output)
    assert not c.native.calls
