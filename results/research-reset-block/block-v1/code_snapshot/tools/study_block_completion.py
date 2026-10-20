"""Registered block-holdout development on six exposed scalar screens."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import shutil

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import environment, load_npz, read_json, sha256, write_json
from opal.block_completion import ScalarCompletion, ScalarReference, fit_reference_grid
from opal.scalar_screen import (TASKS, load_screen_roster, block_assignments, split_report,
                                residual_permutations, paired_gene_uncertainty, metrics)

CODE = ('src/block_completion.py', 'src/scalar_screen.py', 'src/models/fitness.py',
        'tools/study_block_completion.py', 'tests/test_block_completion.py',
        'tests/test_scalar_screen.py', 'tests/test_block_study.py',
        'docs/research_reset/block_protocol.md')
CONFIG = dict(rank=8, penalty=.01, steps=500, seed=20261022,
              reference_alphas=[1., 10., 100., 1000.],
              control_seeds=[20261023, 20261024, 20261025], bootstrap_draws=2000,
              relative_gain_gate=.05, minimum_permutable_fraction=.8,
              wall_seconds=3600, rss_bytes=2*1024**3, artifact_bytes=128*1024**2,
              reserve_bytes=1024**3)


def now():
    return datetime.now(timezone.utc).isoformat()


def signature():
    return {name: sha256(ROOT/name) for name in CODE}


def verify_files(files):
    for name, expected in files.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f'Frozen input differs: {name}')


class Guard:
    def __init__(self, output, config, started=None):
        self.output, self.config = Path(output), config
        self.started = time.monotonic() if started is None else started
        self.cpu = time.process_time()
        self.peak = 0
        self.last_disk = -float('inf')

    def check(self):
        elapsed = time.monotonic()-self.started
        info = psutil.Process().memory_info()
        self.peak = max(self.peak, getattr(info, 'peak_wset', info.rss))
        if elapsed > self.config['wall_seconds'] or time.process_time()-self.cpu > self.config['wall_seconds']:
            raise RuntimeError('Registered CPU/wall cap reached')
        if self.peak > self.config['rss_bytes']:
            raise RuntimeError('Registered process-memory cap reached')
        if elapsed-self.last_disk >= 2:
            self.last_disk = elapsed
            if shutil.disk_usage(self.output).free < self.config['reserve_bytes']:
                raise RuntimeError('Registered writable-volume reserve reached')
            if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()) > self.config['artifact_bytes']:
                raise RuntimeError('Registered artifact cap reached')

    def receipt(self):
        return dict(wall_seconds=time.monotonic()-self.started,
                    cpu_seconds=time.process_time()-self.cpu, peak_rss_bytes=self.peak)


def freeze(args):
    if args.output.exists():
        raise FileExistsError('Preserve prior registrations')
    screens = load_screen_roster(args.original, args.prepared)
    args.output.mkdir(parents=True)
    (args.output/'.gitignore').write_text('*.npz\n**/*.npz\n', encoding='utf-8')
    assignments, panels, files = {}, {}, {}
    for task in TASKS:
        s = screens[task]
        files[s['bank_path']] = s['bank_sha256']
        panels[task] = dict(pairs=len(s['pairs']), genes=len(np.unique(s['pairs'])), units=s['units'],
                           source=s['source'], context=s['context'], exclusions=s['exclusions'], folds=[])
        for fold, split in enumerate(block_assignments(s)):
            assignments[f'{task}__{fold}'] = split
            controls = residual_permutations(s, split)
            record = split_report(s, split)
            record['controls'] = [dict(seed=seed, permutable_fraction=c['permutable_fraction'],
                moved_fraction=c['moved_fraction'], sufficient=c['sufficient'],
                permutation_sha256=__import__('hashlib').sha256(c['permutation'].tobytes()).hexdigest())
                for seed, c in zip(CONFIG['control_seeds'], controls)]
            panels[task]['folds'].append(record)
    np.savez_compressed(args.output/'assignments.npz', **assignments)
    files[str((args.output/'assignments.npz').resolve())] = sha256(args.output/'assignments.npz')
    write_json(args.output/'protocol.json', dict(created_utc=now(), config=CONFIG, code=signature(), files=files,
        original=str(args.original.resolve()), prepared=str(args.prepared.resolve()), panels=panels,
        evidence='exploratory reused outcomes; missing blocks among train-supported genes, not cold genes or confirmation',
        environment=environment()))
    print('frozen block study without new outcome loss scoring')


def verify(output):
    p = read_json(output/'protocol.json')
    if p['code'] != signature():
        raise ValueError('Registration code drift')
    verify_files(p['files'])
    if (output/'prefit_tests.json').exists() or (output/'run_started.json').exists():
        raise FileExistsError('Preserve prerequisite ordering')
    start = now()
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_block_completion.py',
               'tests/test_scalar_screen.py', 'tests/test_block_study.py']
    r = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    write_json(output/'prefit_tests.json', dict(started_utc=start, completed_utc=now(),
        protocol_sha256=sha256(output/'protocol.json'), code=signature(), command=command,
        returncode=r.returncode, stdout=r.stdout, stderr=r.stderr))
    if r.returncode:
        raise RuntimeError(r.stdout+r.stderr)


def preconditions(output):
    p, tests = read_json(output/'protocol.json'), read_json(output/'prefit_tests.json')
    if p['code'] != signature() or tests['code'] != p['code'] or tests['returncode']:
        raise ValueError('Passing tests must cover frozen code')
    if tests['protocol_sha256'] != sha256(output/'protocol.json') or tests['started_utc'] < p['created_utc']:
        raise ValueError('Tests must follow the registration')
    verify_files(p['files'])
    return p


def reference_label(model):
    return model.kind+(f'/{model.alpha:g}' if model.kind in {'node', 'calibration'} else '')


def fit(output):
    p = preconditions(output)
    if (output/'run_started.json').exists():
        raise FileExistsError('Preserve completed or partial study')
    write_json(output/'run_started.json', dict(started_utc=now(), protocol_sha256=sha256(output/'protocol.json'),
        prefit_tests_sha256=sha256(output/'prefit_tests.json')))
    guard = Guard(output, p['config'])
    selections = []
    try:
        screens = load_screen_roster(Path(p['original']), Path(p['prepared']))
        assignment = load_npz(output/'assignments.npz')
        for task in TASKS:
            s = screens[task]
            for fold in range(5):
                guard.check()
                split = assignment[f'{task}__{fold}']
                train, val = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
                dest = output/task/f'fold-{fold}'
                dest.mkdir(parents=True)
                refs = fit_reference_grid(s['singles'], s['pairs'][train], s['responses'][train],
                                          p['config']['reference_alphas'], check=guard.check)
                val_losses = [float(np.mean((r.predict(s['pairs'][val])-s['responses'][val])**2)) for r in refs]
                chosen = int(np.argmin(val_losses))
                reference = refs[chosen]
                files = {}
                for r in refs:
                    name = reference_label(r).replace('/', '-')+'.npz'
                    r.save(dest/name)
                    files[name] = sha256(dest/name)
                base_train, base_val = reference.predict(s['pairs'][train]), reference.predict(s['pairs'][val])
                residual = s['responses'][train]-base_train
                controls = residual_permutations(s, split)
                models = []
                for name, permutation in [('aligned', None)]+[(f'shuffled-{seed}', control['local_permutation'])
                        for seed, control in zip(p['config']['control_seeds'], controls)]:
                    y = residual if permutation is None else residual[permutation]
                    model = ScalarCompletion(p['config']['rank'], p['config']['penalty'],
                        p['config']['steps'], p['config']['seed']).fit(s['pairs'][train], y, len(s['genes']),
                        max(float(s['responses'][train].std()), 1e-6),
                        (s['pairs'][val], s['responses'][val]-base_val), check=guard.check)
                    model.save(dest/(name+'.npz'))
                    files[name+'.npz'] = sha256(dest/(name+'.npz'))
                    models.append(dict(name=name, checkpoint=name+'.npz', best_step=model.best_step,
                        parameter_count=model.parameter_count, history=model.history))
                selection = dict(task=task, fold=fold, completed_utc=now(), chosen_reference=chosen,
                    reference_labels=[reference_label(r) for r in refs], reference_validation_mse=val_losses,
                    reference_checkpoints=[reference_label(r).replace('/', '-')+'.npz' for r in refs],
                    models=models, files=files)
                write_json(dest/'selection.json', selection)
                selections.append(dict(task=task, fold=fold, path=str((dest/'selection.json').relative_to(output)),
                    sha256=sha256(dest/'selection.json')))
                write_json(output/'fit_progress.json', dict(completed=selections, resources=guard.receipt()))
                print(task, fold, 'fit complete', round(guard.receipt()['wall_seconds'], 2), flush=True)
        guard.check()
        write_json(output/'selection.json', dict(completed_utc=now(), fits=selections,
            resources=guard.receipt(), protocol_sha256=sha256(output/'protocol.json')))
    except BaseException as error:
        write_json(output/'failure.json', dict(created_utc=now(), phase='fit', error=repr(error), resources=guard.receipt()))
        raise


def evaluate_arrays(screen, indices, prediction, baseline, controls, all_references, qualifying_controls):
    y = screen['responses'][indices]
    d = (prediction-y)**2-(baseline-y)**2
    candidate_score, baseline_score = metrics(y, prediction), metrics(y, baseline)
    intervals = paired_gene_uncertainty(d, screen['pairs'][indices])
    references = {label: metrics(y, values) for label, values in all_references.items()}
    gain = 1-candidate_score['mse']/baseline_score['mse'] if baseline_score['mse'] else None
    infections = [dict(candidate=metrics(screen['replicates'][indices, r], prediction),
                       baseline=metrics(screen['replicates'][indices, r], baseline)) for r in range(2)]
    control_scores = {name: metrics(y, values) for name, values in controls.items()}
    # Missing eligibility/control information is insufficient, never an omitted task.
    sufficient = len(indices) >= 50 and len(np.unique(screen['pairs'][indices])) >= 10 and qualifying_controls
    upper = intervals['gene_conditional_95'][1]
    passed = bool(sufficient and gain is not None and gain >= .05 and upper < 0 and
                  all(r['candidate']['mse'] <= r['baseline']['mse'] for r in infections) and
                  all(candidate_score['mse'] <= r['mse'] for r in references.values()) and
                  all(candidate_score['mse'] < r['mse'] for r in control_scores.values()))
    return dict(pairs=len(indices), genes=len(np.unique(screen['pairs'][indices])), candidate=candidate_score,
        selected_simple=baseline_score, relative_gain=gain, paired_gene_difference=intervals,
        infections=infections, all_simple=references, shuffled_controls=control_scores,
        sufficient=sufficient, passes_gate=passed)


def score(output, replay=False):
    p = preconditions(output)
    selection = read_json(output/'selection.json')
    if selection['protocol_sha256'] != sha256(output/'protocol.json'):
        raise ValueError('Selection/protocol drift')
    if not replay and (output/'results.json').exists():
        raise FileExistsError('Preserve completed scores')
    guard = Guard(output, p['config'])
    screens = load_screen_roster(Path(p['original']), Path(p['prepared']))
    assignments = load_npz(output/'assignments.npz')
    results, manifests = {}, {}
    for task in TASKS:
        s, gathered = screens[task], []
        for record in (r for r in selection['fits'] if r['task'] == task):
            guard.check()
            path = output/record['path']
            verify_files({str(path): record['sha256']})
            r, dest = read_json(path), path.parent
            verify_files({str(dest/name): h for name, h in r['files'].items()})
            indices = np.flatnonzero(assignments[f"{task}__{r['fold']}"] == 2)
            refs = [ScalarReference.load(dest/name) for name in r['reference_checkpoints']]
            base = refs[r['chosen_reference']].predict(s['pairs'][indices])
            predictions = {m['name']: base+ScalarCompletion.load(dest/m['checkpoint']).predict(s['pairs'][indices])
                           for m in r['models']}
            all_refs = {label: ref.predict(s['pairs'][indices]) for label, ref in zip(r['reference_labels'], refs)}
            controls_ok = all(c['sufficient'] for c in p['panels'][task]['folds'][r['fold']]['controls'])
            gathered.append(dict(indices=indices, base=base, predictions=predictions,
                                 references=all_refs, controls_ok=controls_ok))
        if len(gathered) != 5:
            raise ValueError('All five fixed folds required')
        indices = np.concatenate([g['indices'] for g in gathered])
        if len(np.unique(indices)) != len(indices):
            raise ValueError('Block tests overlap')
        pred = np.concatenate([g['predictions']['aligned'] for g in gathered])
        base = np.concatenate([g['base'] for g in gathered])
        controls = {name: np.concatenate([g['predictions'][name] for g in gathered])
                    for name in gathered[0]['predictions'] if name != 'aligned'}
        refs = {name: np.concatenate([g['references'][name] for g in gathered]) for name in gathered[0]['references']}
        results[task] = evaluate_arrays(s, indices, pred, base, controls, refs,
                                       all(g['controls_ok'] for g in gathered))
        results[task]['folds'] = [dict(fold=i, pairs=len(g['indices']),
            candidate_mse=float(np.mean((g['predictions']['aligned']-s['responses'][g['indices']])**2)),
            baseline_mse=float(np.mean((g['base']-s['responses'][g['indices']])**2))) for i, g in enumerate(gathered)]
        if not replay:
            np.savez_compressed(output/(task+'_predictions.npz'), test_indices=indices, candidate=pred,
                                baseline=base, **controls, **{k.replace('/', '_'): v for k, v in refs.items()})
        manifests[task] = sha256(output/(task+'_predictions.npz'))
    science = dict(label='exploratory six-task block-held development; no untouched-source confirmation',
        tasks=results, all_task_gate=all(r['passes_gate'] for r in results.values()),
        all_task_sufficient=all(r['sufficient'] for r in results.values()),
        protocol_sha256=sha256(output/'protocol.json'), selection_sha256=sha256(output/'selection.json'),
        prediction_hashes=manifests)
    if replay:
        if science != read_json(output/'results.json'):
            raise RuntimeError('Scientific JSON replay differs')
        write_json(output/'replay.json', dict(created_utc=now(), status='exact scientific JSON replay',
                                            results_sha256=sha256(output/'results.json')))
    else:
        write_json(output/'results.json', science)
        write_json(output/'score_resources.json', guard.receipt())
    guard.check()
    print('replayed' if replay else 'scored', science['all_task_gate'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'verify', 'fit', 'score', 'replay'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--prepared', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.action == 'freeze':
            freeze(args)
        else:
            {'verify': verify, 'fit': fit, 'score': score, 'replay': lambda o: score(o, True)}[args.action](args.output)
