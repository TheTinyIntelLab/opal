"""Prospectively registered recovery of all twenty completed native-v2 fits."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import study_native_blocks as base

OLD_CODE = base.CODE
OLD_PRECONDITIONS = base.preconditions
EXTRA_CODE = ('tools/resume_native_blocks.py', 'tests/test_native_resume.py',
              'docs/research_reset/native_resume_protocol.md')
INHERITED_TASKS = ('horlbeck_K562', 'horlbeck_Jurkat', 'environment1', 'environment2')
MISSING_TASKS = ('parp2022', 'parp2023')


def activate():
    base.CODE = tuple(dict.fromkeys(OLD_CODE + EXTRA_CODE))
    base.preconditions = preconditions


def old_signature():
    return {name: base.sha256(ROOT / name) for name in OLD_CODE}


def read_receipt(path):
    # PowerShell operational-stop receipts may have a UTF-8 BOM. Hash their
    # original bytes; decode without rewriting the historical receipt.
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def no_scoring(path):
    if any((path / name).exists() for name in ('score_started.json', 'results.json', 'replay.json')):
        raise ValueError('Inherited or abandoned native outcomes must remain unscored')


def validate_inheritance(source, protocol, block_protocol, assignments):
    """Validate every completed fold; never choose folds using their losses."""
    source = Path(source).resolve()
    old = base.read_json(source / 'protocol.json')
    tests = base.read_json(source / 'prefit_tests.json')
    start = base.read_json(source / 'run_started.json')
    progress = base.read_json(source / 'fit_progress.json')
    no_scoring(source)
    if (source / 'selection.json').exists():
        raise ValueError('Recovery requires the exact incomplete twenty-fold source')
    if old['code'] != old_signature():
        raise ValueError('Inherited original code drift')
    for key in ('config', 'dependencies', 'python', 'native_revision', 'native_canonical_lf_hashes',
                'source', 'block_run', 'files', 'block_files', 'panels'):
        if old[key] != protocol[key]:
            raise ValueError('Inherited recipe/input/dependency mismatch: ' + key)
    if old['config'] != base.CONFIG or old['dependencies'] != base.dependencies():
        raise ValueError('Inherited config/dependencies differ from current fixed recipe')
    if (tests['code'] != old['code'] or tests['dependencies'] != old['dependencies'] or
            tests['returncode'] != 0 or tests['protocol_sha256'] != base.sha256(source / 'protocol.json') or
            tests['started_utc'] < old['created_utc'] or tests['completed_utc'] < tests['started_utc'] or
            start['started_utc'] < tests['completed_utc'] or
            start['protocol_sha256'] != base.sha256(source / 'protocol.json') or
            start['prefit_tests_sha256'] != base.sha256(source / 'prefit_tests.json')):
        raise ValueError('Inherited postfreeze prerequisite/lifecycle mismatch')
    if start['block_result_files'] != base.bind_block_results(Path(protocol['block_run'])):
        raise ValueError('Inherited Q2 comparison drift')
    base.block.verify_files(old['files'])
    base.block.verify_files(old['block_files'])
    base.block.verify_files(start['block_result_files'])
    expected = {(task, fold) for task in INHERITED_TASKS for fold in range(5)}
    records = progress['completed']
    if len(records) != 20 or {(r['task'], r['fold']) for r in records} != expected:
        raise ValueError('Exactly all twenty original completed folds required')
    actual_selections = {p.resolve() for p in (source / 'selections').rglob('*.json')}
    expected_selections = {(source / 'selections' / task / f'fold-{fold}.json').resolve()
                           for task, fold in expected}
    if actual_selections != expected_selections:
        raise ValueError('Unexpected or missing inherited selection file')
    files = {str((source / name).resolve()): base.sha256(source / name) for name in
             ('protocol.json', 'prefit_tests.json', 'run_started.json', 'fit_progress.json', 'interruption.json')}
    interruption = read_receipt(source / 'interruption.json')
    if interruption.get('test_scoring') is not False or interruption.get('completed_fits') != 20:
        raise ValueError('Inherited interruption receipt must declare twenty unscored fits')
    consumed = progress['resources']
    if any(not np.isfinite(consumed[key]) or consumed[key] < 0 for key in
           ('wall_seconds', 'cpu_seconds', 'peak_rss_bytes')):
        raise ValueError('Finite nonnegative inherited resource charges required')
    if max(consumed['wall_seconds'], consumed['cpu_seconds']) >= protocol['config']['wall_seconds']:
        raise ValueError('Inherited budget already exhausted')
    for record in records:
        task, fold = record['task'], record['fold']
        path = source / 'selections' / task / f'fold-{fold}.json'
        if (Path(record['path']) != path.relative_to(source) or
                (source / record['path']).resolve() != path.resolve()):
            raise ValueError('Inherited selection path mismatch')
        base.block.verify_files({str(path): record['sha256']})
        selected = base.read_json(path)
        dest = source / 'checkpoints' / task / f'fold-{fold}'
        if ((selected['task'], selected['fold']) != (task, fold) or
                (source / selected['checkpoint']).resolve() != dest.resolve()):
            raise ValueError('Inherited checkpoint identity mismatch')
        expected_files = {str((dest / name).relative_to(source)) for name in ('metadata.json', 'state.npz')}
        if set(selected['files']) != expected_files:
            raise ValueError('Exactly two inherited checkpoint files required')
        base.block.verify_files({str(source / name): digest for name, digest in selected['files'].items()})
        meta = base.read_json(dest / 'metadata.json')
        if meta['revision'] != protocol['native_revision'] or meta['config'] != {
                key: protocol['config'][key] for key in ('seed', 'epochs', 'batch_size')}:
            raise ValueError('Inherited checkpoint native config/revision mismatch')
        for key in ('best_epoch', 'model_args', 'parameter_count', 'training_metrics'):
            if meta[key] != selected[key]:
                raise ValueError('Inherited checkpoint/selection metadata mismatch')
        trajectories = meta['training_metrics']
        if (set(trajectories) != {'train_loss', 'val_loss'} or
                any(len(v) != protocol['config']['epochs'] or not np.isfinite(v).all()
                    for v in trajectories.values()) or
                meta['best_epoch'] != int(np.argmin(trajectories['val_loss'])) + 1 or
                meta['state_sha256'] != base.sha256(dest / 'state.npz')):
            raise ValueError('Inherited complete epoch/state/selection mismatch')
        split = assignments[f'{task}__{fold}']
        if selected['split_sha256'] != base.array_hash(split):
            raise ValueError('Inherited split signature mismatch')
        rows = {'train': np.random.default_rng(protocol['config']['seed']).permutation(np.flatnonzero(split == 0)),
                'val': np.flatnonzero(split == 1)}
        counts = {'train': block_protocol['panels'][task]['folds'][fold]['train_pairs'],
                  'val': block_protocol['panels'][task]['folds'][fold]['validation_pairs']}
        if (selected['counts'] != counts or selected['counts'] != {k: len(v) for k, v in rows.items()} or
                meta['fit_rows'] != {k: v.tolist() for k, v in rows.items()} or
                selected['fit_order_sha256'] != {k: base.array_hash(v.astype(np.int64)) for k, v in rows.items()}):
            raise ValueError('Inherited split/order/count mismatch')
        files[str(path.resolve())] = base.sha256(path)
        files.update({str((source / name).resolve()): digest for name, digest in selected['files'].items()})
    return dict(source=str(source), files=files, completed=records, consumed=consumed,
                rule='all twenty completed v2 folds, no loss-based choice; only ten PARP fits are missing')


def freeze(args):
    activate()
    if args.output.exists():
        raise FileExistsError('Preserve prior recovery registration')
    # Validate inheritance before creating a new registration directory.
    old = base.read_json(args.inherit_run / 'protocol.json')
    prototype = dict(old, code=base.signature(), config=base.CONFIG, dependencies=base.dependencies(),
                     python=sys.version, source=str(args.source.resolve()), block_run=str(args.block_run.resolve()))
    bp = base.block.preconditions(args.block_run)
    assignments = base.load_npz(args.block_run / 'assignments.npz')
    inheritance = validate_inheritance(args.inherit_run, prototype, bp, assignments)
    receipts = {}
    for directory, name in ((getattr(args, 'aborted_run', None), 'operational_stop.json'),
                            (getattr(args, 'failed_run', None), 'failure.json')):
        if directory is None:
            continue
        no_scoring(directory)
        receipt = read_receipt(directory / name)
        if name == 'operational_stop.json':
            if receipt.get('test_scoring') is not False or psutil.pid_exists(receipt['pid']):
                raise ValueError('Abandoned native attempt must be stopped and unscored')
        receipts[str((directory / name).resolve())] = base.sha256(directory / name)
    inheritance['operational_receipts'] = receipts
    base.freeze(args)
    p = base.read_json(args.output / 'protocol.json')
    # Revalidate the actual fresh base registration, not just the prototype.
    validate_inheritance(args.inherit_run, p, bp, assignments)
    base.write_json(args.output / 'inheritance_manifest.json', inheritance)
    p['files'].update(inheritance['files'])
    p['files'].update(receipts)
    p['files'][str((args.output / 'inheritance_manifest.json').resolve())] = base.sha256(args.output / 'inheritance_manifest.json')
    p['recovery_id'] = 'native-block-v5'
    p['inherit_run'] = str(args.inherit_run.resolve())
    p['recovery_rule'] = inheritance['rule']
    p['resource_accounting'] = 'guard charges all completed inherited work plus new active work; abandoned attempts are a separate recorded program lower bound'
    base.write_json(args.output / 'protocol.json', p)


def custody(output, protocol):
    base.block.verify_files(protocol['files'])
    manifest = base.read_json(output / 'inheritance_manifest.json')
    bp = base.read_json(Path(protocol['block_run']) / 'protocol.json')
    assignments = base.load_npz(Path(protocol['block_run']) / 'assignments.npz')
    # Original input-file dictionaries exclude the additional recovery bindings.
    source_protocol = base.read_json(Path(protocol['inherit_run']) / 'protocol.json')
    check_protocol = dict(protocol, files=source_protocol['files'])
    actual = validate_inheritance(protocol['inherit_run'], check_protocol, bp, assignments)
    if any(actual[key] != manifest[key] for key in actual):
        raise ValueError('Frozen inheritance manifest drift')
    base.block.verify_files(manifest['operational_receipts'])
    for path in manifest['operational_receipts']:
        no_scoring(Path(path).parent)
    if protocol['config'] != base.CONFIG or protocol['recovery_id'] != 'native-block-v5':
        raise ValueError('Recovery must retain the original fixed native recipe')
    return manifest


def verify(output):
    activate()
    p = base.read_json(output / 'protocol.json')
    if p['code'] != base.signature() or p['dependencies'] != base.dependencies():
        raise ValueError('Recovery registration code/dependency drift')
    custody(output, p)
    if (output / 'prefit_tests.json').exists() or (output / 'run_started.json').exists():
        raise FileExistsError('Preserve recovery prerequisite ordering')
    started = base.now()
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_naiad_reference.py',
               'tests/test_native_block_study.py', 'tests/test_scalar_screen.py', 'tests/test_native_resume.py']
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    base.write_json(output / 'prefit_tests.json', dict(started_utc=started, completed_utc=base.now(),
        protocol_sha256=base.sha256(output / 'protocol.json'), code=base.signature(),
        dependencies=base.dependencies(), command=command, returncode=run.returncode,
        stdout=run.stdout, stderr=run.stderr))
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)


def preconditions(output):
    p, bp = OLD_PRECONDITIONS(output)
    custody(output, p)
    return p, bp


def fit(output):
    activate()
    p, bp = preconditions(output)
    if (output / 'run_started.json').exists() or (output / 'selection.json').exists():
        raise FileExistsError('Preserve completed or partial recovery fit')
    inheritance = custody(output, p)
    comparator_files = base.bind_block_results(Path(p['block_run']))
    base.write_json(output / 'run_started.json', dict(started_utc=base.now(),
        protocol_sha256=base.sha256(output / 'protocol.json'), prefit_tests_sha256=base.sha256(output / 'prefit_tests.json'),
        block_result_files=comparator_files, inherited_fits=20,
        inheritance_manifest_sha256=base.sha256(output / 'inheritance_manifest.json')))
    guard = base.block.Guard(output, p['config'], consumed=inheritance['consumed'])
    selections = []
    try:
        guard.check()
        source = Path(inheritance['source'])
        for record in inheritance['completed']:
            selected = base.read_json(source / record['path'])
            for name in (*selected['files'], record['path']):
                dest = output / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, dest)
                if base.sha256(dest) != base.sha256(source / name):
                    raise ValueError('Inherited byte copy mismatch')
            selections.append(record.copy())
            guard.check()
        base.write_json(output / 'fit_progress.json', dict(completed=selections, resources=guard.receipt(), inherited_fits=20))
        screens = base.load_screen_roster(Path(bp['original']), Path(bp['prepared']))
        assignments = base.load_npz(Path(p['block_run']) / 'assignments.npz')
        config = base.NativeConfig(seed=p['config']['seed'], epochs=p['config']['epochs'], batch_size=p['config']['batch_size'])
        for task in MISSING_TASKS:
            screen = screens[task]
            base.verify_assignments(screen, assignments, task)
            for fold in range(5):
                guard.check()
                split = assignments[f'{task}__{fold}']
                target = screen['responses'].copy()
                target[split == 2] = np.nan
                model = base.fit_native(p['source'], screen['genes'], screen['pairs'], screen['singles'],
                                        target, split, config, check=guard.check)
                dest = output / 'checkpoints' / task / f'fold-{fold}'
                model.save(dest)
                files = {str((dest / name).relative_to(output)): base.sha256(dest / name)
                         for name in ('metadata.json', 'state.npz')}
                selected = dict(task=task, fold=fold, completed_utc=base.now(),
                    checkpoint=str(dest.relative_to(output)), files=files,
                    split_sha256=base.array_hash(split),
                    fit_order_sha256={key: base.array_hash(np.asarray(rows, dtype=np.int64)) for key, rows in model.fit_rows.items()},
                    counts={key: len(rows) for key, rows in model.fit_rows.items()},
                    best_epoch=model.best_epoch, model_args=model.model_args, parameter_count=model.parameter_count,
                    training_metrics=model.training_metrics,
                    checkpoint_rule='first strict native validation MSE minimum, before any native test scoring')
                path = output / 'selections' / task / f'fold-{fold}.json'
                base.write_json(path, selected)
                selections.append(dict(task=task, fold=fold, path=str(path.relative_to(output)), sha256=base.sha256(path)))
                guard.check()
                base.write_json(output / 'fit_progress.json', dict(completed=selections, resources=guard.receipt(), inherited_fits=20))
                print(task, fold, 'recovered native fit complete', round(guard.receipt()['wall_seconds'], 2), flush=True)
                del model
        custody(output, p)
        base.block.verify_files(comparator_files)
        list(base.selection_records(output, {'fits': selections}))
        guard.check()
        base.write_json(output / 'selection.json', dict(completed_utc=base.now(), fits=selections,
            resources=guard.receipt(), protocol_sha256=base.sha256(output / 'protocol.json'),
            run_started_sha256=base.sha256(output / 'run_started.json')))
    except BaseException as error:
        base.write_json(output / 'failure.json', dict(created_utc=base.now(), phase='fit', error=repr(error), resources=guard.receipt()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'verify', 'fit', 'score', 'replay'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block-run', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--inherit-run', type=Path)
    parser.add_argument('--aborted-run', type=Path)
    parser.add_argument('--failed-run', type=Path)
    args = parser.parse_args()
    activate()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.action == 'freeze':
            if any(getattr(args, name) is None for name in ('block_run', 'source', 'inherit_run')):
                parser.error('freeze requires --block-run, --source and --inherit-run')
            freeze(args)
        else:
            {'verify': verify, 'fit': fit, 'score': base.score,
             'replay': lambda out: base.score(out, True)}[args.action](args.output)
