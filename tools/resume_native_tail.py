"""Registered sidecar continuation of the exact twenty-six native-v6 fits."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import resume_native_blocks as recovery
from tools import study_native_blocks as base

EXTRA_CODE = ('tools/resume_native_tail.py', 'tests/test_native_tail.py',
              'docs/research_reset/native_tail_protocol.md')
EXPECTED = {(task, fold) for task in base.TASKS[:5] for fold in range(5)} | {('parp2023', 0)}
MISSING = [('parp2023', fold) for fold in range(1, 5)]


def signature(native_code):
    return {name: base.sha256(ROOT / name) for name in tuple(native_code) + EXTRA_CODE}


def unscored(nr):
    if any((nr / name).exists() for name in ('score_started.json', 'results.json', 'replay.json')):
        raise ValueError('Tail continuation requires unscored native outcomes')


def validate_records(nr, progress, protocol, bp, assignments):
    records = progress['completed']
    if len(records) != 26 or {(r['task'], r['fold']) for r in records} != EXPECTED:
        raise ValueError('Exactly all twenty-six completed fits required')
    files = {}
    for record in records:
        task, fold = record['task'], record['fold']
        path = nr / 'selections' / task / f'fold-{fold}.json'
        if Path(record['path']).is_absolute() or (nr / record['path']).resolve() != path.resolve():
            raise ValueError('Tail inherited selection path mismatch')
        base.block.verify_files({str(path): record['sha256']})
        r = base.read_json(path)
        directory = nr / 'checkpoints' / task / f'fold-{fold}'
        if (r['task'], r['fold']) != (task, fold) or (nr / r['checkpoint']).resolve() != directory.resolve():
            raise ValueError('Tail checkpoint identity mismatch')
        expected_files = {str((directory / name).relative_to(nr)) for name in ('metadata.json', 'state.npz')}
        if set(r['files']) != expected_files:
            raise ValueError('Exactly two inherited checkpoint files required')
        bound = {str(nr / name): digest for name, digest in r['files'].items()}
        base.block.verify_files(bound)
        meta = base.read_json(directory / 'metadata.json')
        if (meta['revision'] != protocol['native_revision'] or
                meta['config'] != {key: protocol['config'][key] for key in ('seed', 'epochs', 'batch_size')} or
                meta['state_sha256'] != base.sha256(directory / 'state.npz')):
            raise ValueError('Tail native recipe/state mismatch')
        for key in ('best_epoch', 'model_args', 'parameter_count', 'training_metrics'):
            if meta[key] != r[key]:
                raise ValueError('Tail checkpoint metadata mismatch')
        curves = meta['training_metrics']
        if (set(curves) != {'train_loss', 'val_loss'} or
                any(len(v) != 100 or not np.isfinite(v).all() for v in curves.values()) or
                r['best_epoch'] != int(np.argmin(curves['val_loss'])) + 1):
            raise ValueError('Tail complete 100-epoch selection required')
        split = assignments[f'{task}__{fold}']
        rows = dict(train=np.random.default_rng(protocol['config']['seed']).permutation(np.flatnonzero(split == 0)),
                    val=np.flatnonzero(split == 1))
        if (r['split_sha256'] != base.array_hash(split) or meta['fit_rows'] != {k: v.tolist() for k, v in rows.items()} or
                r['fit_order_sha256'] != {k: base.array_hash(v.astype(np.int64)) for k, v in rows.items()} or
                r['counts'] != {k: len(v) for k, v in rows.items()} or
                r['counts'] != dict(train=bp['panels'][task]['folds'][fold]['train_pairs'],
                                    val=bp['panels'][task]['folds'][fold]['validation_pairs'])):
            raise ValueError('Tail inherited split/count/seeded-order mismatch')
        files[str(path.resolve())] = record['sha256']
        files.update({str(Path(name).resolve()): digest for name, digest in bound.items()})
    return files


def freeze(args):
    if args.output.exists():
        raise FileExistsError('Preserve tail amendment')
    nr = args.native_run.resolve()
    unscored(nr)
    if (nr / 'selection.json').exists():
        raise FileExistsError('Native roster already finalized')
    recovery.activate()
    p, bp = recovery.preconditions(nr)
    if p['config'] != base.CONFIG:
        raise ValueError('Unchanged native budget and recipe required')
    progress = base.read_json(nr / 'fit_progress.json')
    assignments = base.load_npz(Path(p['block_run']) / 'assignments.npz')
    files = validate_records(nr, progress, p, bp, assignments)
    actual = {path.resolve() for path in (nr / 'selections').rglob('*.json')}
    if actual != {(nr / 'selections' / task / f'fold-{fold}.json').resolve() for task, fold in EXPECTED}:
        raise ValueError('Unrecorded tail checkpoint selections exist')
    for key in ('wall_seconds', 'cpu_seconds', 'peak_rss_bytes'):
        if not np.isfinite(progress['resources'][key]) or progress['resources'][key] < 0:
            raise ValueError('Finite nonnegative cumulative resource charges required')
    if max(progress['resources'][k] for k in ('wall_seconds', 'cpu_seconds')) >= p['config']['wall_seconds']:
        raise ValueError('Native cumulative budget exhausted')
    interrupt = recovery.read_receipt(nr / 'interruption.json')
    if interrupt.get('test_scoring') is not False or interrupt.get('completed_fits') != 26:
        raise ValueError('Recorded unscored twenty-six-fit interruption required')
    for name in ('protocol.json', 'prefit_tests.json', 'run_started.json', 'inheritance_manifest.json', 'interruption.json'):
        files[str((nr / name).resolve())] = base.sha256(nr / name)
    args.output.mkdir(parents=True)
    snapshot = args.output / 'inherited_progress.json'
    snapshot.write_bytes((nr / 'fit_progress.json').read_bytes())
    files[str(snapshot.resolve())] = base.sha256(snapshot)
    base.write_json(args.output / 'protocol.json', dict(id='native-v6-tail-v1', created_utc=base.now(),
        native_run=str(nr), native_protocol_sha256=base.sha256(nr / 'protocol.json'),
        files=files, code=signature(p['code']), native_code=p['code'], config=p['config'],
        dependencies=base.dependencies(), python=sys.version, initial_progress_sha256=base.sha256(snapshot),
        consumed=progress['resources'], missing=[list(x) for x in MISSING],
        evidence='prospective operational continuation only; retain all twenty-six fits, append four; no scientific recipe/gate change'))


def custody(output):
    p = base.read_json(output / 'protocol.json')
    if (p['code'] != signature(p['native_code']) or p['config'] != base.CONFIG or
            p['dependencies'] != base.dependencies() or p['python'] != sys.version or
            p['missing'] != [list(x) for x in MISSING]):
        raise ValueError('Tail registration code/config/dependency drift')
    base.block.verify_files(p['files'])
    recovery.activate()
    native_p, bp = recovery.preconditions(Path(p['native_run']))
    snapshot = base.read_json(output / 'inherited_progress.json')
    files = validate_records(Path(p['native_run']), snapshot, native_p, bp,
                            base.load_npz(Path(native_p['block_run']) / 'assignments.npz'))
    base.block.verify_files(files)
    if p['consumed'] != snapshot['resources']:
        raise ValueError('Tail inherited resource charge changed')
    return p, native_p, bp, snapshot


def verify(output):
    p, _, _, _ = custody(output)
    if (output / 'prefit_tests.json').exists() or (output / 'tail_started.json').exists():
        raise FileExistsError('Preserve tail prerequisite ordering')
    started = base.now()
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_native_tail.py', 'tests/test_native_resume.py',
               'tests/test_naiad_reference.py', 'tests/test_native_block_study.py', 'tests/test_scalar_screen.py']
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    base.write_json(output / 'prefit_tests.json', dict(started_utc=started, completed_utc=base.now(),
        protocol_sha256=base.sha256(output / 'protocol.json'), code=p['code'], dependencies=p['dependencies'],
        command=command, returncode=run.returncode, stdout=run.stdout, stderr=run.stderr))
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)


def preconditions(output):
    if (output / 'failure.json').exists():
        raise FileExistsError('Preserve failed tail continuation')
    p, np_, bp, snapshot = custody(output)
    tests = base.read_json(output / 'prefit_tests.json')
    if (tests['returncode'] or tests['code'] != p['code'] or tests['dependencies'] != p['dependencies'] or
            tests['protocol_sha256'] != base.sha256(output / 'protocol.json') or tests['started_utc'] < p['created_utc']):
        raise ValueError('Passing postfreeze tail prerequisites required')
    return p, np_, bp, snapshot


def fit(output):
    p, np_, bp, snapshot = preconditions(output)
    nr = Path(p['native_run'])
    if (output / 'tail_started.json').exists() or (nr / 'selection.json').exists():
        raise FileExistsError('Preserve completed or partial tail execution')
    unscored(nr)
    if base.sha256(nr / 'fit_progress.json') != p['initial_progress_sha256']:
        raise ValueError('Native progress changed before registered continuation')
    base.write_json(output / 'tail_started.json', dict(started_utc=base.now(),
        protocol_sha256=base.sha256(output / 'protocol.json'), prerequisite_sha256=base.sha256(output / 'prefit_tests.json')))
    guard = base.block.Guard(nr, p['config'], consumed=p['consumed'])
    selections = list(snapshot['completed'])
    try:
        screens = base.load_screen_roster(Path(bp['original']), Path(bp['prepared']))
        assignments = base.load_npz(Path(np_['block_run']) / 'assignments.npz')
        config = base.NativeConfig(seed=p['config']['seed'], epochs=100, batch_size=p['config']['batch_size'])
        for task, fold in MISSING:
            guard.check()
            screen = screens[task]
            base.verify_assignments(screen, assignments, task)
            split = assignments[f'{task}__{fold}']
            target = screen['responses'].copy()
            target[split == 2] = np.nan
            model = base.fit_native(np_['source'], screen['genes'], screen['pairs'], screen['singles'], target, split, config, check=guard.check)
            dest = nr / 'checkpoints' / task / f'fold-{fold}'
            model.save(dest)
            selected = dict(task=task, fold=fold, completed_utc=base.now(), checkpoint=str(dest.relative_to(nr)),
                files={str((dest / name).relative_to(nr)): base.sha256(dest / name) for name in ('metadata.json', 'state.npz')},
                split_sha256=base.array_hash(split),
                fit_order_sha256={k: base.array_hash(np.asarray(v, dtype=np.int64)) for k, v in model.fit_rows.items()},
                counts={k: len(v) for k, v in model.fit_rows.items()}, best_epoch=model.best_epoch, model_args=model.model_args,
                parameter_count=model.parameter_count, training_metrics=model.training_metrics,
                checkpoint_rule='first strict native validation MSE minimum, before any native test scoring')
            path = nr / 'selections' / task / f'fold-{fold}.json'
            if path.exists():
                raise FileExistsError('Never replace an existing tail selection')
            base.write_json(path, selected)
            selections.append(dict(task=task, fold=fold, path=str(path.relative_to(nr)), sha256=base.sha256(path)))
            guard.check()
            base.write_json(nr / 'fit_progress.json', dict(completed=selections, resources=guard.receipt(),
                inherited_fits=20, tail_amendment_sha256=base.sha256(output / 'protocol.json')))
            print(task, fold, 'native tail complete', round(guard.receipt()['wall_seconds'], 2), flush=True)
            del model
        custody(output)
        unscored(nr)
        list(base.selection_records(nr, {'fits': selections}))
        guard.check()
        base.write_json(nr / 'selection.json', dict(completed_utc=base.now(), fits=selections, resources=guard.receipt(),
            protocol_sha256=base.sha256(nr / 'protocol.json'), run_started_sha256=base.sha256(nr / 'run_started.json'),
            tail_amendment_sha256=base.sha256(output / 'protocol.json'), tail_started_sha256=base.sha256(output / 'tail_started.json')))
        base.write_json(output / 'completed.json', dict(completed_utc=base.now(), selection_sha256=base.sha256(nr / 'selection.json'), resources=guard.receipt()))
    except BaseException as error:
        base.write_json(output / 'failure.json', dict(phase='fit', error=repr(error), resources=guard.receipt()))
        raise


def score(output, replay=False):
    p, _, _, _ = preconditions(output)
    nr = Path(p['native_run'])
    complete = base.read_json(output / 'completed.json')
    selected = base.read_json(nr / 'selection.json')
    if (complete['selection_sha256'] != base.sha256(nr / 'selection.json') or
            selected['tail_amendment_sha256'] != base.sha256(output / 'protocol.json') or
            selected['tail_started_sha256'] != base.sha256(output / 'tail_started.json')):
        raise ValueError('Completed registered tail continuation required')
    base.score(nr, replay=replay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'verify', 'fit', 'score', 'replay'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.action == 'freeze':
            if args.native_run is None:
                parser.error('freeze requires --native-run')
            freeze(args)
        else:
            {'verify': verify, 'fit': fit, 'score': score,
             'replay': lambda output: score(output, True)}[args.action](args.output)
