"""One exploratory validation-only selector among three fixed block predictors."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import study_native_blocks as native
from tools import resume_native_blocks as recovery
from opal.block_completion import ScalarReference, ScalarCompletion

FAMILIES = ('simple', 'rank8', 'native')
EXTRA_CODE = ('tools/select_block_family.py', 'tests/test_block_family.py',
              'docs/research_reset/block_family_protocol.md')
CONFIG = dict(wall_seconds=600, rss_bytes=2 * 1024**3, artifact_bytes=128 * 1024**2,
              reserve_bytes=1024**3, threads=1, bootstrap_draws=2000, seed=20261022)


def signature(native_code):
    return {name: native.sha256(ROOT / name) for name in tuple(native_code) + EXTRA_CODE}


def unscored(path):
    if any((path / name).exists() for name in ('score_started.json', 'results.json', 'replay.json')):
        raise ValueError('Native test scoring must follow all thirty family choices')


def choose_family(target, predictions):
    target = np.asarray(target)
    if not len(target) or target.ndim != 1 or not np.isfinite(target).all() or set(predictions) != set(FAMILIES):
        raise ValueError('Finite validation targets and exactly three families required')
    if any(np.asarray(p).shape != target.shape or not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError('Aligned finite validation predictions required')
    losses = {name: float(np.mean((predictions[name] - target)**2)) for name in FAMILIES}
    return min(FAMILIES, key=lambda name: losses[name]), losses


def freeze(args):
    if args.output.exists():
        raise FileExistsError('Preserve previous family registration')
    unscored(args.native_run)
    bp = native.block.preconditions(args.block_run)
    np_ = native.read_json(args.native_run / 'protocol.json')
    if Path(np_['block_run']).resolve() != args.block_run.resolve():
        raise ValueError('Shared Q2 source required')
    native.block.verify_files({str(ROOT / k): v for k, v in np_['code'].items()})
    native.block.verify_files(np_['files'])
    files = {str((root / name).resolve()): native.sha256(root / name) for root, names in
             ((args.block_run, ('protocol.json', 'assignments.npz', 'selection.json', 'results.json')),
              (args.native_run, ('protocol.json', 'inheritance_manifest.json', 'prefit_tests.json', 'run_started.json')))
             for name in names}
    native.block.verify_files(bp['files'])
    args.output.mkdir(parents=True)
    (args.output / '.gitignore').write_text('*.npz\n**/*.npz\n', encoding='utf-8')
    native.write_json(args.output / 'protocol.json', dict(id='family-v1', created_utc=native.now(),
        config=CONFIG, code=signature(np_['code']), native_code=np_['code'], files=files,
        dependencies=native.dependencies(), python=sys.version,
        block_run=str(args.block_run.resolve()), native_run=str(args.native_run.resolve()),
        native_source=np_['source'], families=list(FAMILIES),
        evidence='exploratory partly post-Q2 development; Q2 test outcomes exposed before this selector was proposed; no promotion or new gate',
        selection='minimum equal-pair shared-validation MSE; exact ties prefer simple then rank8 then native; stored checkpoints unchanged'))


def registered(output):
    p = native.read_json(output / 'protocol.json')
    if (p['code'] != signature(p['native_code']) or p['dependencies'] != native.dependencies() or
            p['config'] != CONFIG or p['families'] != list(FAMILIES) or p['python'] != sys.version):
        raise ValueError('Family registration code/dependency/configuration drift')
    native.block.verify_files(p['files'])
    native.block.preconditions(Path(p['block_run']))
    return p


def verify(output):
    p = registered(output)
    if (output / 'prefit_tests.json').exists() or (output / 'selection_started.json').exists():
        raise FileExistsError('Preserve family prerequisite ordering')
    started = native.now()
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_block_family.py', 'tests/test_scalar_screen.py']
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    native.write_json(output / 'prefit_tests.json', dict(started_utc=started, completed_utc=native.now(),
        protocol_sha256=native.sha256(output / 'protocol.json'), code=p['code'], dependencies=p['dependencies'],
        returncode=run.returncode, command=command, stdout=run.stdout, stderr=run.stderr))
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)


def preconditions(output):
    if (output / 'failure.json').exists():
        raise FileExistsError('Preserve failed family attempt')
    p = registered(output)
    tests = native.read_json(output / 'prefit_tests.json')
    if (tests['returncode'] or tests['code'] != p['code'] or tests['dependencies'] != p['dependencies'] or
            tests['protocol_sha256'] != native.sha256(output / 'protocol.json') or
            tests['started_utc'] < p['created_utc']):
        raise ValueError('Passing postfreeze family prerequisites required')
    return p


def complete_records(root, manifest, q2=False):
    records = manifest['fits']
    expected = {(task, fold) for task in native.TASKS for fold in range(5)}
    if len(records) != 30 or {(r['task'], r['fold']) for r in records} != expected:
        raise ValueError('All thirty unique checkpoint selections required')
    answer, files = {}, {}
    for record in records:
        path = root / record['path']
        native.block.verify_files({str(path): record['sha256']})
        r = native.read_json(path)
        if (r['task'], r['fold']) != (record['task'], record['fold']):
            raise ValueError('Checkpoint selection identity mismatch')
        local = path.parent if q2 else root
        bound = {str(local / key): value for key, value in r['files'].items()}
        native.block.verify_files(bound)
        files[str(path.resolve())] = record['sha256']
        files.update({str(Path(key).resolve()): value for key, value in bound.items()})
        answer[(r['task'], r['fold'])] = (r, path.parent)
    return answer, files


def bind_selections(p):
    br, nr = Path(p['block_run']), Path(p['native_run'])
    recovery.activate()
    recovery.preconditions(nr)
    selections = {name: native.read_json(root / 'selection.json') for name, root in [('q2', br), ('native', nr)]}
    for name, root in [('q2', br), ('native', nr)]:
        if selections[name]['protocol_sha256'] != native.sha256(root / 'protocol.json'):
            raise ValueError('Source selection/protocol mismatch')
    if selections['native']['run_started_sha256'] != native.sha256(nr / 'run_started.json'):
        raise ValueError('Native selected-checkpoint lifecycle mismatch')
    q2, qfiles = complete_records(br, selections['q2'], q2=True)
    nrecords, nfiles = complete_records(nr, selections['native'])
    files = dict(qfiles, **nfiles)
    files.update({str((root / 'selection.json').resolve()): native.sha256(root / 'selection.json') for root in (br, nr)})
    return q2, nrecords, files


def validation_predictions(p, screen, split, qrecord, nrecord):
    rows = np.flatnonzero(split == 1)
    q, directory = qrecord
    n, _ = nrecord
    ref = ScalarReference.load(directory / q['reference_checkpoints'][q['chosen_reference']])
    aligned = [r for r in q['models'] if r['name'] == 'aligned']
    if len(aligned) != 1:
        raise ValueError('Exactly one fixed aligned rank8 checkpoint required')
    simple = ref.predict(screen['pairs'][rows])
    rank8 = simple + ScalarCompletion.load(directory / aligned[0]['checkpoint']).predict(screen['pairs'][rows])
    model = native.NativeNAIAD.load(p['native_source'], Path(p['native_run']) / n['checkpoint'])
    for key in ('best_epoch', 'model_args', 'parameter_count', 'training_metrics'):
        if getattr(model, key) != n[key]:
            raise ValueError('Native fixed checkpoint metadata mismatch')
    if native.array_hash(split) != n['split_sha256']:
        raise ValueError('Native/shared validation split mismatch')
    prediction = model.predict(screen['pairs'][rows])
    return rows, dict(simple=simple, rank8=rank8, native=prediction)


def select(output):
    p = preconditions(output)
    nr = Path(p['native_run'])
    if (output / 'selection_started.json').exists() or (output / 'choices.json').exists():
        raise FileExistsError('Preserve partial or completed family selection')
    unscored(nr)
    qrecords, nrecords, files = bind_selections(p)
    native.write_json(output / 'checkpoint_bindings.json', files)
    native.write_json(output / 'selection_started.json', dict(started_utc=native.now(),
        protocol_sha256=native.sha256(output / 'protocol.json'),
        checkpoint_bindings_sha256=native.sha256(output / 'checkpoint_bindings.json')))
    guard = native.block.Guard(output, p['config'])
    try:
        bp = native.read_json(Path(p['block_run']) / 'protocol.json')
        screens = native.load_screen_roster(Path(bp['original']), Path(bp['prepared']))
        assignments = native.load_npz(Path(p['block_run']) / 'assignments.npz')
        choices = []
        for task in native.TASKS:
            screen = screens[task]
            native.verify_assignments(screen, assignments, task)
            for fold in range(5):
                guard.check()
                split = assignments[f'{task}__{fold}']
                rows, predictions = validation_predictions(p, screen, split, qrecords[(task, fold)], nrecords[(task, fold)])
                choice, losses = choose_family(screen['responses'][rows], predictions)
                choices.append(dict(task=task, fold=fold, family=choice, validation_pairs=len(rows),
                    validation_mse=losses, validation_indices_sha256=native.array_hash(rows),
                    validation_prediction_sha256={name: native.array_hash(value) for name, value in predictions.items()}))
        native.block.verify_files(files)
        unscored(nr)
        guard.check()
        native.write_json(output / 'choices.json', dict(completed_utc=native.now(), choices=choices,
            resources=guard.receipt(), protocol_sha256=native.sha256(output / 'protocol.json'),
            checkpoint_bindings_sha256=native.sha256(output / 'checkpoint_bindings.json'),
            selection_started_sha256=native.sha256(output / 'selection_started.json')))
    except BaseException as error:
        native.write_json(output / 'failure.json', dict(phase='select', error=repr(error), resources=guard.receipt()))
        raise


def choices_ready(output, p):
    selected = native.read_json(output / 'choices.json')
    if (selected['protocol_sha256'] != native.sha256(output / 'protocol.json') or
            selected['checkpoint_bindings_sha256'] != native.sha256(output / 'checkpoint_bindings.json') or
            selected['selection_started_sha256'] != native.sha256(output / 'selection_started.json')):
        raise ValueError('Family choice lifecycle drift')
    expected = {(task, fold) for task in native.TASKS for fold in range(5)}
    rows = selected['choices']
    if len(rows) != 30 or {(r['task'], r['fold']) for r in rows} != expected:
        raise ValueError('All thirty unique family choices required before scoring')
    for r in rows:
        if set(r['validation_mse']) != set(FAMILIES) or r['family'] != min(FAMILIES, key=lambda f: r['validation_mse'][f]):
            raise ValueError('Frozen family choice/tie rule mismatch')
    native.block.verify_files(native.read_json(output / 'checkpoint_bindings.json'))
    return selected


def fixed_test_arrays(p):
    br, nr = Path(p['block_run']), Path(p['native_run'])
    result = native.read_json(nr / 'results.json')
    replay = native.read_json(nr / 'replay.json')
    if (result['selection_sha256'] != native.sha256(nr / 'selection.json') or
            result['protocol_sha256'] != native.sha256(nr / 'protocol.json') or
            replay['results_sha256'] != native.sha256(nr / 'results.json')):
        raise ValueError('Completed native scientific replay required')
    files = native.bind_block_results(br)
    files.update({str((nr / name).resolve()): native.sha256(nr / name) for name in ('results.json', 'replay.json', 'score_started.json')})
    arrays = {}
    for task in native.TASKS:
        path = nr / (task + '_predictions.npz')
        native.block.verify_files({str(path): result['prediction_hashes'][task]})
        files[str(path.resolve())] = result['prediction_hashes'][task]
        q2, na = native.load_npz(br / (task + '_predictions.npz')), native.load_npz(path)
        if not np.array_equal(q2['test_indices'], na['test_indices']):
            raise ValueError('Fixed family test array alignment mismatch')
        arrays[task] = dict(test_indices=q2['test_indices'], simple=q2['baseline'], rank8=q2['candidate'], native=na['native'])
    return arrays, files


def combine(arrays, choices, assignments, task):
    expected = np.concatenate([np.flatnonzero(assignments[f'{task}__{fold}'] == 2) for fold in range(5)])
    if not np.array_equal(arrays['test_indices'], expected) or len(np.unique(expected)) != len(expected):
        raise ValueError('Fixed test arrays must match exact disjoint shared fold order')
    if any(np.asarray(arrays[name]).shape != expected.shape or not np.isfinite(arrays[name]).all() for name in FAMILIES):
        raise ValueError('Aligned finite fixed test predictions required')
    result = np.empty(len(expected))
    start = 0
    for fold in range(5):
        rows = np.flatnonzero(assignments[f'{task}__{fold}'] == 2)
        family = choices[(task, fold)]['family']
        result[start:start + len(rows)] = arrays[family][start:start + len(rows)]
        start += len(rows)
    return result


def comparison(target, selected, reference, pairs, p, check):
    answer = native.compare(target, selected, reference, pairs, p['config']['bootstrap_draws'], check)
    answer['selector'] = answer.pop('native')
    return answer


def score(output, replay=False):
    p = preconditions(output)
    selected = choices_ready(output, p)
    if (output / ('replay.json' if replay else 'score_started.json')).exists():
        raise FileExistsError('Preserve family score/replay attempt')
    native_start = native.read_json(Path(p['native_run']) / 'score_started.json')
    if native_start['started_utc'] < selected['completed_utc']:
        raise ValueError('All family choices must precede native test scoring')
    arrays, files = fixed_test_arrays(p)
    if replay:
        if native.read_json(output / 'fixed_test_bindings.json') != files:
            raise ValueError('Fixed family result custody drift')
    else:
        native.write_json(output / 'fixed_test_bindings.json', files)
        native.write_json(output / 'score_started.json', dict(started_utc=native.now(), choices_sha256=native.sha256(output / 'choices.json')))
    if native.read_json(output / 'score_started.json')['choices_sha256'] != native.sha256(output / 'choices.json'):
        raise ValueError('Scored family choices changed')
    consumed = native.read_json(output / 'score_resources.json') if replay else selected['resources']
    guard = native.block.Guard(output, p['config'], consumed=consumed)
    try:
        bp = native.read_json(Path(p['block_run']) / 'protocol.json')
        screens = native.load_screen_roster(Path(bp['original']), Path(bp['prepared']))
        assignments = native.load_npz(Path(p['block_run']) / 'assignments.npz')
        choices = {(r['task'], r['fold']): r for r in selected['choices']}
        results, hashes = {}, {}
        for task in native.TASKS:
            guard.check()
            screen, fixed = screens[task], arrays[task]
            native.verify_assignments(screen, assignments, task)
            prediction = combine(fixed, choices, assignments, task)
            ids = fixed['test_indices']
            pairs, target = screen['pairs'][ids], screen['responses'][ids]
            folds, start = [], 0
            for fold in range(5):
                count = int(np.sum(assignments[f'{task}__{fold}'] == 2))
                sl = slice(start, start + count)
                losses = {f: float(np.mean((fixed[f][sl] - target[sl])**2)) for f in FAMILIES}
                family = choices[(task, fold)]['family']
                best = min(FAMILIES, key=lambda f: losses[f])
                folds.append(dict(fold=fold, pairs=count, selected=family, fixed_test_mse=losses,
                    empirical_test_best=best, selected_is_empirical_test_best=family == best,
                    regret_mse=losses[family] - losses[best], diagnostic_only=True))
                start += count
            results[task] = dict(selector=native.metrics(target, prediction),
                comparisons={f: comparison(target, prediction, fixed[f], pairs, p, guard.check) for f in FAMILIES},
                infections=[{f: comparison(screen['replicates'][ids, r], prediction, fixed[f], pairs, p, guard.check)
                             for f in FAMILIES} for r in range(2)],
                fixed_metrics={f: native.metrics(target, fixed[f]) for f in FAMILIES},
                family_fold_counts={f: sum(r['selected'] == f for r in folds) for f in FAMILIES},
                family_pair_counts={f: sum(r['pairs'] for r in folds if r['selected'] == f) for f in FAMILIES}, folds=folds)
            path = output / (task + '_predictions.npz')
            predicted = dict(test_indices=ids, selector=prediction)
            if not replay:
                np.savez_compressed(path, **predicted)
            stored = native.load_npz(path)
            if set(stored) != set(predicted) or any(not np.array_equal(stored[k], v) for k, v in predicted.items()):
                raise ValueError('Family saved prediction replay differs')
            hashes[task] = native.sha256(path)
        native.block.verify_files(files)
        guard.check()
        science = dict(label=p['evidence'], tasks=results, promotion='none; unchanged Q2/Q3 gates',
            choices_sha256=native.sha256(output / 'choices.json'), protocol_sha256=native.sha256(output / 'protocol.json'),
            fixed_test_bindings_sha256=native.sha256(output / 'fixed_test_bindings.json'), prediction_hashes=hashes)
        if replay:
            if science != native.read_json(output / 'results.json'):
                raise ValueError('Family scientific JSON replay differs')
            native.write_json(output / 'replay.json', dict(status='exact family predictions and scientific JSON replay',
                results_sha256=native.sha256(output / 'results.json'), resources=guard.receipt()))
        else:
            native.write_json(output / 'results.json', science)
            native.write_json(output / 'score_resources.json', guard.receipt())
    except BaseException as error:
        native.write_json(output / 'failure.json', dict(phase='replay' if replay else 'score', error=repr(error), resources=guard.receipt()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'verify', 'select', 'score', 'replay'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--block-run', type=Path)
    parser.add_argument('--native-run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.action == 'freeze':
            if args.block_run is None or args.native_run is None:
                parser.error('freeze requires --block-run and --native-run')
            freeze(args)
        else:
            {'verify': verify, 'select': select, 'score': score,
             'replay': lambda output: score(output, True)}[args.action](args.output)
