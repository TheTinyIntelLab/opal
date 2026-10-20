"""Frozen native NAIAD comparator on the exact Q2 six-screen block splits."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import array_hash, load_npz, read_json, sha256, write_json
from opal.naiad_reference import NativeConfig, NativeNAIAD, REVISION, SOURCE_HASHES, fit_native, load_official
from opal.scalar_screen import TASKS, load_screen_roster, block_assignments, metrics, paired_gene_uncertainty
from tools import study_block_completion as block

CONFIG = dict(seed=20261022, epochs=100, batch_size=1024, bootstrap_draws=2000,
              wall_seconds=7200, rss_bytes=2 * 1024**3, artifact_bytes=128 * 1024**2,
              reserve_bytes=1024**3, threads=1)
CODE = tuple(dict.fromkeys(block.CODE + (
    'src/naiad_reference.py', 'tools/study_native_blocks.py',
    'tests/test_naiad_reference.py', 'tests/test_native_block_study.py',
    'docs/research_reset/naiad_protocol.md', 'docs/research_reset/native_block_protocol.md')))
DEPENDENCIES = ('numpy', 'scipy', 'torch', 'pandas', 'seaborn', 'matplotlib',
                'scikit-learn', 'tqdm', 'openpyxl', 'psutil', 'threadpoolctl')


def now():
    return datetime.now(timezone.utc).isoformat()


def signature():
    return {name: sha256(ROOT / name) for name in CODE}


def dependencies():
    return {name: importlib.metadata.version(name) for name in DEPENDENCIES}


def freeze(args):
    if args.output.exists():
        raise FileExistsError('Preserve prior native registration')
    load_official(args.source)
    bp = read_json(args.block_run / 'protocol.json')
    if bp['code'] != block.signature():
        raise ValueError('Q2 registration code drift')
    block.verify_files(bp['files'])
    if set(bp['panels']) != set(TASKS) or any(len(bp['panels'][task]['folds']) != 5 for task in TASKS):
        raise ValueError('Exact six-task/five-fold Q2 roster required')
    if any(not fold['eligible'] for task in TASKS for fold in bp['panels'][task]['folds']):
        raise ValueError('All registered native folds must be eligible')
    args.output.mkdir(parents=True)
    (args.output / '.gitignore').write_text('checkpoints/\n*.npz\n**/*.npz\n', encoding='utf-8')
    write_json(args.output / 'protocol.json', dict(
        id='native-block-v1', created_utc=now(), config=CONFIG, code=signature(),
        block_run=str(args.block_run.resolve()), source=str(args.source.resolve()),
        files={str((args.block_run / name).resolve()): sha256(args.block_run / name)
               for name in ('protocol.json', 'assignments.npz')},
        block_files=bp['files'], panels=bp['panels'], native_revision=REVISION,
        native_canonical_lf_hashes=SOURCE_HASHES, dependencies=dependencies(), python=sys.version,
        information='same local singles, pair populations and fixed Q2 memberships; test labels withheld from native fit',
        recipe='official100epochs/defaultnativecapacity/Adam/summedMSE/scheduler; seededTRAIN-onlyorder; nativeVAL-min checkpoint; symmetric inference',
        evidence='exploratory reused sources; native comparator only, no model promotion or untouched-source confirmation'))


def verify(output):
    p = read_json(output / 'protocol.json')
    if p['code'] != signature() or p['dependencies'] != dependencies():
        raise ValueError('Native registration code/dependency drift')
    block.verify_files(p['files'])
    block.verify_files(p['block_files'])
    load_official(p['source'])
    if (output / 'prefit_tests.json').exists() or (output / 'run_started.json').exists():
        raise FileExistsError('Preserve postfreeze prerequisite ordering')
    started = now()
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_naiad_reference.py',
               'tests/test_native_block_study.py', 'tests/test_scalar_screen.py']
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    write_json(output / 'prefit_tests.json', dict(started_utc=started, completed_utc=now(),
        protocol_sha256=sha256(output / 'protocol.json'), code=signature(),
        dependencies=dependencies(), command=command, returncode=run.returncode,
        stdout=run.stdout, stderr=run.stderr))
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)


def preconditions(output):
    if (output / 'failure.json').exists():
        raise FileExistsError('Preserve failed native attempt; use a new registration')
    p, tests = read_json(output / 'protocol.json'), read_json(output / 'prefit_tests.json')
    if (p['code'] != signature() or tests['code'] != p['code'] or tests['returncode'] or
            p['dependencies'] != dependencies() or tests['dependencies'] != p['dependencies']):
        raise ValueError('Passing prerequisites must cover frozen native code/dependencies')
    if tests['protocol_sha256'] != sha256(output / 'protocol.json') or tests['started_utc'] < p['created_utc']:
        raise ValueError('Native tests must follow frozen registration')
    block.verify_files(p['files'])
    block.verify_files(p['block_files'])
    load_official(p['source'])
    bp = block.preconditions(Path(p['block_run']))
    return p, bp


def bind_block_results(block_run):
    """Bind an already finished Q2 evaluation, including exact prediction bytes."""
    results = read_json(block_run / 'results.json')
    if results['protocol_sha256'] != sha256(block_run / 'protocol.json'):
        raise ValueError('Q2 results/protocol mismatch')
    if results['selection_sha256'] != sha256(block_run / 'selection.json'):
        raise ValueError('Q2 results/selection mismatch')
    if set(results['tasks']) != set(TASKS) or set(results['prediction_hashes']) != set(TASKS):
        raise ValueError('Complete Q2 results required')
    files = {str((block_run / name).resolve()): sha256(block_run / name)
             for name in ('results.json', 'selection.json', 'prefit_tests.json')}
    for task, expected in results['prediction_hashes'].items():
        path = block_run / (task + '_predictions.npz')
        if sha256(path) != expected:
            raise ValueError('Q2 prediction hash mismatch')
        files[str(path.resolve())] = expected
    return files


def verify_assignments(screen, assignment, task):
    expected = block_assignments(screen)
    for fold in range(5):
        if not np.array_equal(assignment[f'{task}__{fold}'], expected[fold]):
            raise ValueError('Q2 assignment identity drift')


def fit(output):
    p, bp = preconditions(output)
    if (output / 'run_started.json').exists() or (output / 'selection.json').exists():
        raise FileExistsError('Preserve completed or partial native fits')
    comparator_files = bind_block_results(Path(p['block_run']))
    write_json(output / 'run_started.json', dict(started_utc=now(),
        protocol_sha256=sha256(output / 'protocol.json'), prefit_tests_sha256=sha256(output / 'prefit_tests.json'),
        block_result_files=comparator_files))
    guard = block.Guard(output, p['config'])
    selections = []
    try:
        guard.check()
        screens = load_screen_roster(Path(bp['original']), Path(bp['prepared']))
        assignment = load_npz(Path(p['block_run']) / 'assignments.npz')
        config = NativeConfig(seed=p['config']['seed'], epochs=p['config']['epochs'], batch_size=p['config']['batch_size'])
        for task in TASKS:
            screen = screens[task]
            verify_assignments(screen, assignment, task)
            for fold in range(5):
                guard.check()
                split = assignment[f'{task}__{fold}']
                target = screen['responses'].copy()
                target[split == 2] = np.nan
                model = fit_native(p['source'], screen['genes'], screen['pairs'], screen['singles'],
                                   target, split, config, check=guard.check)
                dest = output / 'checkpoints' / task / f'fold-{fold}'
                model.save(dest)
                files = {str((dest / name).relative_to(output)): sha256(dest / name)
                         for name in ('metadata.json', 'state.npz')}
                selection = dict(task=task, fold=fold, completed_utc=now(),
                    checkpoint=str(dest.relative_to(output)), files=files,
                    split_sha256=array_hash(split),
                    fit_order_sha256={key: array_hash(np.asarray(rows, dtype=np.int64))
                                      for key, rows in model.fit_rows.items()},
                    counts={key: len(rows) for key, rows in model.fit_rows.items()},
                    best_epoch=model.best_epoch, model_args=model.model_args,
                    parameter_count=model.parameter_count, training_metrics=model.training_metrics,
                    checkpoint_rule='first strict native validation MSE minimum, before any native test scoring')
                path = output / 'selections' / task / f'fold-{fold}.json'
                write_json(path, selection)
                selections.append(dict(task=task, fold=fold, path=str(path.relative_to(output)), sha256=sha256(path)))
                guard.check()
                write_json(output / 'fit_progress.json', dict(completed=selections, resources=guard.receipt()))
                print(task, fold, 'native fit complete', round(guard.receipt()['wall_seconds'], 2), flush=True)
                del model
        block.verify_files(comparator_files)
        guard.check()
        write_json(output / 'selection.json', dict(completed_utc=now(), fits=selections,
            resources=guard.receipt(), protocol_sha256=sha256(output / 'protocol.json'),
            run_started_sha256=sha256(output / 'run_started.json')))
    except BaseException as error:
        write_json(output / 'failure.json', dict(created_utc=now(), phase='fit', error=repr(error), resources=guard.receipt()))
        raise


def compare(target, native, reference, pairs, draws, check=None):
    if check is not None:
        check()
    native_score, reference_score = metrics(target, native), metrics(target, reference)
    delta = (native - target)**2 - (reference - target)**2
    result = dict(native=native_score, reference=reference_score,
                relative_gain=None if reference_score['mse'] == 0 else 1-native_score['mse']/reference_score['mse'],
                paired_gene_difference=paired_gene_uncertainty(delta, pairs, draws=draws, seed=20261022))
    if check is not None:
        check()
    return result


def evaluate_arrays(screen, indices, native, q2, draws=2000, check=None):
    indices = np.asarray(indices)
    if not np.array_equal(indices, q2['test_indices']):
        raise ValueError('Native/Q2 test index order mismatch')
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Disjoint native block test pairs required')
    y, pairs = screen['responses'][indices], screen['pairs'][indices]
    excluded = {'test_indices', 'candidate', 'baseline'} | {f'shuffled-{seed}' for seed in (20261023, 20261024, 20261025)}
    references = {key: value for key, value in q2.items() if key not in excluded}
    if len(references) != 12:
        raise ValueError('All twelve registered Q2 simple references required')
    main = {key: compare(y, native, q2[name], pairs, draws, check)
            for key, name in [('selected_simple', 'baseline'), ('rank8', 'candidate')]}
    infections = [{key: compare(screen['replicates'][indices, infection], native, q2[name], pairs, draws, check)
                   for key, name in [('selected_simple', 'baseline'), ('rank8', 'candidate')]}
                  for infection in range(2)]
    simple = {name: dict(metrics=metrics(y, value),
                         native_minus_reference_mse=float(np.mean((native-y)**2-(value-y)**2)))
              for name, value in references.items()}
    return dict(pairs=len(indices), genes=len(np.unique(pairs)), native=metrics(y, native),
                comparisons=main, infections=infections, all_simple=simple,
                native_no_point_loss_to_all_simple=all(v['native_minus_reference_mse'] <= 0 for v in simple.values()),
                promotion='none; descriptive native comparator on previously exposed development sources')


def selection_records(output, selection):
    records = selection['fits']
    expected = {(task, fold) for task in TASKS for fold in range(5)}
    if len(records) != len(expected) or {(r['task'], r['fold']) for r in records} != expected:
        raise ValueError('All thirty unique native selections required before scoring')
    for record in records:
        path = output / record['path']
        block.verify_files({str(path): record['sha256']})
        r = read_json(path)
        if (r['task'], r['fold']) != (record['task'], record['fold']):
            raise ValueError('Native selection identity mismatch')
        block.verify_files({str(output / name): digest for name, digest in r['files'].items()})
        yield r


def score(output, replay=False):
    p, bp = preconditions(output)
    selection = read_json(output / 'selection.json')
    if (selection['protocol_sha256'] != sha256(output / 'protocol.json') or
            selection['run_started_sha256'] != sha256(output / 'run_started.json')):
        raise ValueError('Native selection lifecycle drift')
    if not replay and ((output / 'results.json').exists() or (output / 'score_started.json').exists()):
        raise FileExistsError('Preserve native scoring attempt')
    if replay and (output / 'replay.json').exists():
        raise FileExistsError('Preserve native replay receipt')
    start = read_json(output / 'run_started.json')
    block.verify_files(start['block_result_files'])
    current = bind_block_results(Path(p['block_run']))
    if current != start['block_result_files']:
        raise ValueError('Bound Q2 comparator drift')
    # Verify the complete selected population before loading any test targets.
    records = list(selection_records(output, selection))
    consumed = read_json(output / 'score_resources.json') if replay else selection['resources']
    guard = block.Guard(output, p['config'], consumed=consumed)
    if not replay:
        write_json(output / 'score_started.json', dict(started_utc=now(), selection_sha256=sha256(output / 'selection.json')))
    try:
        guard.check()
        screens = load_screen_roster(Path(bp['original']), Path(bp['prepared']))
        assignments = load_npz(Path(p['block_run']) / 'assignments.npz')
        results, prediction_hashes = {}, {}
        for task in TASKS:
            screen = screens[task]
            verify_assignments(screen, assignments, task)
            gathered, fold_metrics = [], []
            for r in sorted((r for r in records if r['task'] == task), key=lambda r: r['fold']):
                guard.check()
                split = assignments[f"{task}__{r['fold']}"]
                if array_hash(split) != r['split_sha256']:
                    raise ValueError('Native split signature mismatch')
                indices = np.flatnonzero(split == 2)
                model = NativeNAIAD.load(p['source'], output / r['checkpoint'])
                for key in ('best_epoch', 'model_args', 'parameter_count', 'training_metrics'):
                    if getattr(model, key) != r[key]:
                        raise ValueError('Native checkpoint/selection metadata mismatch')
                if {key: array_hash(np.asarray(rows, dtype=np.int64)) for key, rows in model.fit_rows.items()} != r['fit_order_sha256']:
                    raise ValueError('Native fitting row-order mismatch')
                prediction = model.predict(screen['pairs'][indices])
                gathered.append((indices, prediction))
                fold_metrics.append(dict(fold=r['fold'], pairs=len(indices),
                    native=metrics(screen['responses'][indices], prediction), best_epoch=r['best_epoch']))
                del model
            indices = np.concatenate([item[0] for item in gathered])
            native = np.concatenate([item[1] for item in gathered])
            q2 = load_npz(Path(p['block_run']) / (task + '_predictions.npz'))
            results[task] = evaluate_arrays(screen, indices, native, q2, p['config']['bootstrap_draws'], guard.check)
            results[task]['folds'] = fold_metrics
            arrays = dict(test_indices=indices, native=native)
            path = output / (task + '_predictions.npz')
            if not replay:
                np.savez_compressed(path, **arrays)
            stored = load_npz(path)
            if set(stored) != set(arrays) or any(not np.array_equal(stored[key], value) for key, value in arrays.items()):
                raise RuntimeError('Native stored prediction replay differs')
            prediction_hashes[task] = sha256(path)
            guard.check()
        science = dict(label=p['evidence'], tasks=results, protocol_sha256=sha256(output / 'protocol.json'),
            selection_sha256=sha256(output / 'selection.json'),
            block_result_files=start['block_result_files'], prediction_hashes=prediction_hashes,
            promotion='none')
        if replay:
            if science != read_json(output / 'results.json'):
                raise RuntimeError('Native scientific JSON replay differs')
            guard.check()
            write_json(output / 'replay.json', dict(completed_utc=now(), status='exact native predictions and scientific JSON replay',
                results_sha256=sha256(output / 'results.json'), resources=guard.receipt()))
        else:
            guard.check()
            write_json(output / 'results.json', science)
            write_json(output / 'score_resources.json', guard.receipt())
    except BaseException as error:
        write_json(output / 'failure.json', dict(created_utc=now(), phase='replay' if replay else 'score',
                                               error=repr(error), resources=guard.receipt()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'verify', 'fit', 'score', 'replay'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block-run', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.action == 'freeze':
            if args.block_run is None or args.source is None:
                parser.error('freeze requires --block-run and --source')
            freeze(args)
        else:
            {'verify': verify, 'fit': fit, 'score': score, 'replay': lambda out: score(out, True)}[args.action](args.output)
