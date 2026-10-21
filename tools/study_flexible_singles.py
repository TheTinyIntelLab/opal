"""Registered Q2b functional diagnostic on unchanged six-task scalar panels."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import study_block_completion as block
from opal.flexible_singles import FlexibleSingles

CODE = ('src/flexible_singles.py', 'tools/study_flexible_singles.py',
        'tests/test_flexible_singles.py', 'tests/test_flexible_study.py',
        'docs/research_reset/flexible_singles_protocol.md')
CONFIG = dict(centers=64, seed=20261022, alphas=[1., 10., 100., 1000.],
              wall_seconds=1800, rss_bytes=2*1024**3, artifact_bytes=64*1024**2,
              reserve_bytes=1024**3, bootstrap_draws=2000,
              noninferiority_margin=.02, residual_gain_gate=.05)
FAMILIES = ('phenotype', 'nodes')


def signature():
    return {name: block.sha256(ROOT/name) for name in CODE}


def freeze(args):
    if args.output.exists():
        raise FileExistsError('Preserve prior flexible registration')
    bp = block.preconditions(args.block_run)
    # The Q1b protocol is bound before dependent outcomes, its successful result
    # and replay are additionally bound at the start of fitting.
    files = {str((args.block_run/name).resolve()): block.sha256(args.block_run/name)
             for name in ('protocol.json', 'selection.json', 'results.json', 'replay.json', 'assignments.npz')}
    files[str((args.q1b_run/'registration.json').resolve())] = block.sha256(args.q1b_run/'registration.json')
    block.verify_files(bp['files'])
    args.output.mkdir(parents=True)
    (args.output/'.gitignore').write_text('*.npz\n**/*.npz\n', encoding='utf-8')
    block.write_json(args.output/'protocol.json', dict(created_utc=block.now(), config=CONFIG,
        code=signature(), files=files, block_run=str(args.block_run.resolve()),
        q1b_run=str(args.q1b_run.resolve()), block_code=bp['code'],
        evidence='exposed development; fixed original scalar banks/targets/test populations; no molecular interaction attribution',
        selection='lowest validation MSE per family across ascending alphas; overall tie prefers phenotype then nodes',
        environment=block.environment()))


def registered(output):
    p = block.read_json(output/'protocol.json')
    if p['code'] != signature() or p['config'] != CONFIG or p['environment'] != block.environment():
        raise ValueError('Flexible registration code/config/environment drift')
    block.verify_files(p['files'])
    block.preconditions(Path(p['block_run']))
    if (output/'failure.json').exists():
        raise ValueError('Preserve failed flexible attempt')
    return p


def verify(output):
    p = registered(output)
    if (output/'prefit_tests.json').exists() or (output/'run_started.json').exists():
        raise FileExistsError('Preserve prerequisite ordering')
    started = block.now()
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_flexible_singles.py', 'tests/test_flexible_study.py']
    r = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    block.write_json(output/'prefit_tests.json', dict(started_utc=started, completed_utc=block.now(),
        protocol_sha256=block.sha256(output/'protocol.json'), code=p['code'], command=command,
        returncode=r.returncode, stdout=r.stdout, stderr=r.stderr))
    if r.returncode:
        raise RuntimeError(r.stdout+r.stderr)


def q1b_gate(path):
    result, replay = block.read_json(path/'results.json'), block.read_json(path/'replay.json')
    if not result.get('integrity_gate') or replay.get('results_sha256') != block.sha256(path/'results.json'):
        raise ValueError('Passing Q1b integrity and exact replay required before flexible fitting')
    return {str((path/name).resolve()): block.sha256(path/name)
            for name in ('registration.json', 'prerequisites.json', 'results.json', 'replay.json')}


def preconditions(output):
    p = registered(output)
    tests = block.read_json(output/'prefit_tests.json')
    if (tests['code'] != p['code'] or tests['returncode'] or
            tests['protocol_sha256'] != block.sha256(output/'protocol.json') or tests['started_utc'] < p['created_utc']):
        raise ValueError('Passing postfreeze flexible tests required')
    return p


def select_losses(losses):
    if set(losses) != set(FAMILIES) or any(len(v) != 4 or not np.isfinite(v).all() for v in losses.values()):
        raise ValueError('Two complete finite four-penalty validation menus required')
    chosen = {f: int(np.argmin(losses[f])) for f in FAMILIES}
    winner = min(FAMILIES, key=lambda f: losses[f][chosen[f]])
    return chosen, winner


def fit(output):
    p = preconditions(output)
    if (output/'run_started.json').exists():
        raise FileExistsError('Preserve completed or partial flexible fit')
    dependent = q1b_gate(Path(p['q1b_run']))
    block.write_json(output/'run_started.json', dict(started_utc=block.now(),
        protocol_sha256=block.sha256(output/'protocol.json'),
        prefit_tests_sha256=block.sha256(output/'prefit_tests.json'), q1b_files=dependent))
    bp = block.read_json(Path(p['block_run'])/'protocol.json')
    guard = block.Guard(output, p['config'])
    records = []
    try:
        screens = block.load_screen_roster(Path(bp['original']), Path(bp['prepared']))
        assignments = block.load_npz(Path(p['block_run'])/'assignments.npz')
        for task in block.TASKS:
            s = screens[task]
            for fold in range(5):
                split = assignments[f'{task}__{fold}']
                train, val = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
                dest = output/task/f'fold-{fold}'
                dest.mkdir(parents=True)
                files, losses = {}, {}
                for family in FAMILIES:
                    guard.check()
                    model = FlexibleSingles(nodes=family == 'nodes', centers=p['config']['centers'],
                        seed=p['config']['seed']).prepare(s['singles'], s['pairs'][train], s['responses'][train], check=guard.check)
                    losses[family] = []
                    for i, alpha in enumerate(p['config']['alphas']):
                        guard.check()
                        model.solve(alpha)
                        losses[family].append(float(np.mean((model.predict(s['pairs'][val])-s['responses'][val])**2)))
                        name = f'{family}-{i}.npz'
                        model.save(dest/name)
                        files[name] = block.sha256(dest/name)
                chosen, winner = select_losses(losses)
                record = dict(task=task, fold=fold, validation_mse=losses, chosen=chosen,
                              winner=winner, files=files, completed_utc=block.now())
                block.write_json(dest/'selection.json', record)
                records.append(dict(task=task, fold=fold, path=str((dest/'selection.json').relative_to(output)),
                                    sha256=block.sha256(dest/'selection.json')))
                block.write_json(output/'fit_progress.json', dict(fits=records, resources=guard.receipt()))
        block.verify_files(dependent)
        guard.check()
        block.write_json(output/'selection.json', dict(completed_utc=block.now(), fits=records,
            resources=guard.receipt(), protocol_sha256=block.sha256(output/'protocol.json')))
    except BaseException as e:
        block.write_json(output/'failure.json', dict(phase='fit', error=repr(e), resources=guard.receipt()))
        raise


def compare(target, candidate, reference, pairs):
    d = (candidate-target)**2-(reference-target)**2
    return dict(candidate=block.metrics(target, candidate), reference=block.metrics(target, reference),
        relative_gain=1-float(np.mean((candidate-target)**2))/float(np.mean((reference-target)**2)),
        uncertainty=block.paired_gene_uncertainty(d, pairs, draws=CONFIG['bootstrap_draws']))


def acceptance(target, phenotype, selected, completion, pairs, replicates):
    alone = compare(target, phenotype, completion, pairs)
    residual = compare(target, completion, selected, pairs)
    noninferior = (alone['uncertainty']['gene_conditional_95'][1] <=
                  CONFIG['noninferiority_margin']*alone['reference']['mse'] and
                  all(np.mean((phenotype-replicates[:, r])**2) <=
                      (1+CONFIG['noninferiority_margin'])*np.mean((completion-replicates[:, r])**2) for r in range(2)))
    residual_remains = (residual['relative_gain'] >= CONFIG['residual_gain_gate'] and
        residual['uncertainty']['gene_conditional_95'][1] < 0 and
        all(np.mean((completion-replicates[:, r])**2) <= np.mean((selected-replicates[:, r])**2) for r in range(2)))
    return dict(phenotype_vs_completion=alone, completion_vs_selected_flexible=residual,
                phenotype_conditionally_noninferior=bool(noninferior), completion_residual_gate=bool(residual_remains))


def score(output, replay=False):
    p = preconditions(output)
    if (output/('replay.json' if replay else 'score_started.json')).exists():
        raise FileExistsError('Preserve flexible scoring attempt')
    selection = block.read_json(output/'selection.json')
    expected = {(task, fold) for task in block.TASKS for fold in range(5)}
    if len(selection['fits']) != 30 or {(r['task'], r['fold']) for r in selection['fits']} != expected:
        raise ValueError('All thirty flexible selections required before test scoring')
    if selection['protocol_sha256'] != block.sha256(output/'protocol.json'):
        raise ValueError('Flexible selection lifecycle drift')
    start = block.read_json(output/'run_started.json')
    if q1b_gate(Path(p['q1b_run'])) != start['q1b_files']:
        raise ValueError('Q1b prerequisite drift')
    # Verify every selected model before loading test responses.
    records = {}
    for r in selection['fits']:
        path = output/r['path']
        block.verify_files({str(path): r['sha256']})
        record = block.read_json(path)
        if (record['task'], record['fold']) != (r['task'], r['fold']):
            raise ValueError('Flexible selection identity mismatch')
        if select_losses(record['validation_mse']) != (record['chosen'], record['winner']):
            raise ValueError('Flexible validation argmin drift')
        block.verify_files({str(path.parent/k): v for k, v in record['files'].items()})
        records[r['task'], r['fold']] = (record, path.parent)
    if not replay:
        block.write_json(output/'score_started.json', dict(started_utc=block.now(), selection_sha256=block.sha256(output/'selection.json')))
    if block.read_json(output/'score_started.json')['selection_sha256'] != block.sha256(output/'selection.json'):
        raise ValueError('Flexible score/selection lifecycle drift')
    guard = block.Guard(output, p['config'], consumed=block.read_json(output/'score_resources.json') if replay else selection['resources'])
    try:
        bp = block.read_json(Path(p['block_run'])/'protocol.json')
        screens = block.load_screen_roster(Path(bp['original']), Path(bp['prepared']))
        assignments = block.load_npz(Path(p['block_run'])/'assignments.npz')
        tasks, hashes = {}, {}
        for task in block.TASKS:
            s, gathered = screens[task], []
            for fold in range(5):
                guard.check()
                r, dest = records[task, fold]
                ids = np.flatnonzero(assignments[f'{task}__{fold}'] == 2)
                values = {f: FlexibleSingles.load(dest/f'{f}-{r["chosen"][f]}.npz').predict(s['pairs'][ids]) for f in FAMILIES}
                gathered.append(dict(ids=ids, **values, selected=values[r['winner']]))
            ids = np.concatenate([g['ids'] for g in gathered])
            if len(np.unique(ids)) != len(ids):
                raise ValueError('Disjoint original test blocks required')
            old_path = Path(p['block_run'])/(task+'_predictions.npz')
            block.verify_files({str(old_path): block.read_json(Path(p['block_run'])/'results.json')['prediction_hashes'][task]})
            old = block.load_npz(old_path)
            if not np.array_equal(ids, old['test_indices']):
                raise ValueError('Original full test population/order must remain identical')
            values = {f: np.concatenate([g[f] for g in gathered]) for f in (*FAMILIES, 'selected')}
            y, pairs = s['responses'][ids], s['pairs'][ids]
            tasks[task] = dict(pairs=len(ids), fixed_simple=block.metrics(y, old['baseline']),
                fixed_completion=block.metrics(y, old['candidate']),
                flexible={f: block.metrics(y, v) for f, v in values.items()},
                comparisons={f: compare(y, v, old['baseline'], pairs) for f, v in values.items()},
                acceptance=acceptance(y, values['phenotype'], values['selected'], old['candidate'], pairs, s['replicates'][ids]),
                infections=[{f: block.metrics(s['replicates'][ids, r], v) for f, v in values.items()} for r in range(2)])
            path = output/(task+'_predictions.npz')
            arrays = dict(test_indices=ids, **values)
            if not replay:
                np.savez_compressed(path, **arrays)
            stored = block.load_npz(path)
            if set(stored) != set(arrays) or any(not np.array_equal(stored[k], v) for k, v in arrays.items()):
                raise ValueError('Flexible saved prediction replay differs')
            hashes[task] = block.sha256(path)
        science = dict(tasks=tasks, protocol_sha256=block.sha256(output/'protocol.json'),
            selection_sha256=block.sha256(output/'selection.json'), prediction_hashes=hashes,
            all_task_phenotype_noninferiority=all(r['acceptance']['phenotype_conditionally_noninferior'] for r in tasks.values()),
            all_task_completion_residual_gate=all(r['acceptance']['completion_residual_gate'] for r in tasks.values()),
            evidence=p['evidence'], promotion='none; bounded function-class diagnostic')
        guard.check()
        if replay:
            if science != block.read_json(output/'results.json'):
                raise ValueError('Flexible scientific JSON replay differs')
            block.write_json(output/'replay.json', dict(results_sha256=block.sha256(output/'results.json'), resources=guard.receipt(), status='exact arrays and scientific JSON'))
        else:
            block.write_json(output/'results.json', science)
            block.write_json(output/'score_resources.json', guard.receipt())
    except BaseException as e:
        block.write_json(output/'failure.json', dict(phase='replay' if replay else 'score', error=repr(e), resources=guard.receipt()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'verify', 'fit', 'score', 'replay'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block-run', type=Path)
    parser.add_argument('--q1b-run', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {'freeze': lambda: freeze(args), 'verify': lambda: verify(args.output),
         'fit': lambda: fit(args.output), 'score': lambda: score(args.output),
         'replay': lambda: score(args.output, True)}[args.action]()
