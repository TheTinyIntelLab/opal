"""Frozen nested-budget experiment, synthetic qualification and exact replay."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import load_npz, read_json, sha256, write_json
from opal.block_completion import ScalarReference, ScalarCompletion, fit_reference_grid
from opal.label_budget import (BUDGETS, budget_split, supported_queries,
                               masked_native_inputs, choose_family, scientific_gate, nominee)
from opal.naiad_reference import NativeConfig, NativeNAIAD, fit_native, load_official
from opal.scalar_screen import TASKS, load_screen_roster, residual_permutations, metrics
from tools import study_block_completion as block
from tools import study_native_blocks as native

CONFIG = dict(wall_seconds=7200, rss_bytes=2*1024**3, artifact_bytes=256*1024**2,
              reserve_bytes=1024**3, threads=1, seed=20261022, epochs=100,
              batch_size=1024, rank=8, penalty=.01, steps=500,
              reference_alphas=[1., 10., 100., 1000.],
              control_seeds=[20261023, 20261024, 20261025], bootstrap_draws=2000,
              profile_margin=1.25)
CODE = tuple(dict.fromkeys(native.CODE + ('src/label_budget.py', 'tools/study_label_budget.py',
    'tests/test_label_budget.py', 'docs/research_reset/label_budget_protocol.md')))


def signature():
    return {name: sha256(ROOT/name) for name in CODE}


def registered(output):
    p = read_json(output/'protocol.json')
    if p['code'] != signature() or p['config'] != CONFIG or p['dependencies'] != native.dependencies():
        raise ValueError('Budget registration code/configuration/dependency drift')
    if p['python'] != sys.version:
        raise ValueError('Python changed after registration')
    block.verify_files(p['files'])
    block.verify_files(p['bank_files'])
    load_official(p['source'])
    return p


def freeze(args):
    if args.output.exists():
        raise FileExistsError('Preserve prior budget registrations')
    bp = block.preconditions(args.block_run)
    load_official(args.source)
    args.output.mkdir(parents=True)
    (args.output/'.gitignore').write_text('*.npz\n**/*.npz\ncheckpoints/\nprofile/\n', encoding='utf-8')
    # Reading banks here verifies schema/eligibility; no loss or model is fitted.
    screens = load_screen_roster(Path(bp['original']), Path(bp['prepared']))
    assignments = load_npz(args.block_run/'assignments.npz')
    budgets, panels = {}, {}
    for total, nt, nv in BUDGETS:
        panels[str(total)] = {}
        for task in TASKS:
            screen, folds = screens[task], []
            for fold in range(5):
                split = budget_split(screen, assignments[f'{task}__{fold}'], nt, nv)
                budgets[f'{total}__{task}__{fold}'] = split
                q = supported_queries(screen['pairs'], split, len(screen['genes']))
                controls = controls_for(screen, split)
                folds.append(dict(fold=fold, train=len(q['train']), validation=len(q['validation']),
                    unused_validation=q['unused_validation'], supported_test=len(q['test']),
                    full_test=len(q['full_test']), supported_genes=int(q['support'].sum()),
                    controls=[dict(seed=c['seed'], sufficient=c['sufficient'],
                        permutable_fraction=c['permutable_fraction'], moved_fraction=c['moved_fraction']) for c in controls]))
            panels[str(total)][task] = folds
    np.savez_compressed(args.output/'assignments.npz', **budgets)
    files = {str((args.block_run/n).resolve()): sha256(args.block_run/n)
             for n in ('protocol.json', 'assignments.npz', 'selection.json', 'results.json')}
    files[str((args.output/'assignments.npz').resolve())] = sha256(args.output/'assignments.npz')
    write_json(args.output/'protocol.json', dict(id='q3b-label-budget-v1', created_utc=native.now(),
        config=CONFIG, code=signature(), python=sys.version, dependencies=native.dependencies(),
        source=str(args.source.resolve()), block_run=str(args.block_run.resolve()),
        original=bp['original'], prepared=bp['prepared'], files=files, bank_files=bp['files'], panels=panels,
        evidence='Previously exposed source outcomes; development only, not independent confirmation or physical assay savings'))


def verify(output):
    p = registered(output)
    if (output/'prefit_tests.json').exists() or (output/'profile_started.json').exists():
        raise FileExistsError('Preserve prerequisite ordering')
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_label_budget.py',
               'tests/test_scalar_screen.py', 'tests/test_block_completion.py', 'tests/test_naiad_reference.py']
    started = native.now()
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    write_json(output/'prefit_tests.json', dict(started_utc=started, completed_utc=native.now(),
        protocol_sha256=sha256(output/'protocol.json'), code=p['code'], returncode=run.returncode,
        command=command, stdout=run.stdout, stderr=run.stderr))
    if run.returncode:
        raise RuntimeError(run.stdout+run.stderr)


def prerequisites(output):
    p, t = registered(output), read_json(output/'prefit_tests.json')
    if (t['returncode'] or t['code'] != p['code'] or t['protocol_sha256'] != sha256(output/'protocol.json')
            or t['started_utc'] < p['created_utc']):
        raise ValueError('Passing postregistration synthetic tests required')
    if list(output.glob('failure*.json')):
        raise FileExistsError('Preserve failed experiment')
    return p


def controls_for(screen, split):
    # Original helper accepts 0/1/2 only; unavailable rows share test status
    # solely for outcome-blind training-stratum construction.
    return residual_permutations(screen, np.where(split < 0, 2, split).astype(np.int8))


def full_reference_prediction(model, pairs):
    """Nonlookup diagnostic on every test pair, without relaxing model APIs."""
    pairs = np.asarray(pairs)
    if model.kind == 'zero':
        return np.zeros(len(pairs))
    if model.kind == 'minimum':
        return model.singles[pairs].min(1)
    pred = model.singles[pairs].sum(1)
    if model.kind == 'additive':
        return pred
    if model.kind != 'calibration':
        raise ValueError('Only nonlookup diagnostic references permitted')
    from opal.models.fitness import design
    for start in range(0, len(pairs), 4096):
        sl = slice(start, start+4096)
        pred[sl] += model.offset+(design(model.singles, pairs[sl], False)-model.center)/model.scale@model.coefficient
    return pred


def fit_fold(screen, split, source, dest, guard):
    q = supported_queries(screen['pairs'], split, len(screen['genes']))
    train, val = q['train'], q['validation']
    if min(len(train), len(val), len(q['test'])) < 3:
        raise ValueError('Insufficient shared support, stop this experiment')
    refs = fit_reference_grid(screen['singles'], screen['pairs'][train], screen['responses'][train],
                              CONFIG['reference_alphas'], check=guard.check)
    losses = [float(np.mean((r.predict(screen['pairs'][val])-screen['responses'][val])**2)) for r in refs]
    chosen = int(np.argmin(losses))
    ref = refs[chosen]
    dest.mkdir(parents=True)
    ref_names = [block.reference_label(r).replace('/', '-')+'.npz' for r in refs]
    for name, r in zip(ref_names, refs):
        r.save(dest/name)
    base_train, base_val = ref.predict(screen['pairs'][train]), ref.predict(screen['pairs'][val])
    residual = screen['responses'][train]-base_train
    controls = controls_for(screen, split)
    model_records = []
    for name, c in [('aligned', None)]+[(f'shuffled-{c["seed"]}', c) for c in controls]:
        y = residual if c is None else residual[c['local_permutation']]
        m = ScalarCompletion(8, .01, 500, 20261022).fit(screen['pairs'][train], y, len(screen['genes']),
            max(float(screen['responses'][train].std()), 1e-6),
            (screen['pairs'][val], screen['responses'][val]-base_val), check=guard.check)
        m.save(dest/(name+'.npz'))
        model_records.append(dict(name=name, history=m.history, best_step=m.best_step))
    rows, y, membership = masked_native_inputs(screen, split)
    n = fit_native(source, screen['genes'], screen['pairs'][rows], screen['singles'], y, membership,
                   NativeConfig(), check=guard.check)
    n.save(dest/'native')
    predictions = dict(simple=base_val,
        rank8=base_val+ScalarCompletion.load(dest/'aligned.npz').predict(screen['pairs'][val]),
        native=n.predict(screen['pairs'][val]))
    family, family_losses = choose_family(screen['responses'][val], predictions)
    guard.check()
    files = {str(f.relative_to(dest)): sha256(f) for f in dest.rglob('*') if f.is_file()}
    selection = dict(chosen_reference=chosen, reference_files=ref_names,
        reference_labels=[block.reference_label(r) for r in refs], reference_validation_mse=losses,
        chosen_family=family, family_validation_mse=family_losses, completion=model_records,
        controls=[dict(seed=c['seed'], sufficient=c['sufficient'], permutable_fraction=c['permutable_fraction']) for c in controls],
        train_rows=train.tolist(), validation_rows=val.tolist(), native_rows_sha256=__import__('hashlib').sha256(rows.tobytes()).hexdigest(),
        files=files, completed_utc=native.now())
    write_json(dest/'selection.json', selection)
    return selection


def synthetic_screen(n_genes=600):
    rng = np.random.default_rng(20261022)
    genes = np.array([f'S{i:03}' for i in range(n_genes)])
    pairs = np.array([(a, b) for a in range(n_genes) for b in range(a+1, n_genes)], dtype=np.int64)
    pairs = pairs[rng.choice(len(pairs), 23000, replace=False)]
    singles = rng.normal(0, .2, len(genes))
    y = singles[pairs].sum(1)+.1*np.sin(pairs.sum(1))
    screen = dict(genes=genes, pairs=pairs, singles=singles, responses=y,
        replicates=np.column_stack((y, y)), counts=np.full((len(pairs), 2), 8), accession='synthetic')
    split = np.repeat([0, 1, 2], [16000, 4000, 3000]).astype(np.int8)
    return screen, split


def profile(output):
    p = prerequisites(output)
    if (output/'profile_started.json').exists():
        raise FileExistsError('Preserve synthetic profile attempts')
    write_json(output/'profile_started.json', dict(created_utc=native.now(), protocol_sha256=sha256(output/'protocol.json')))
    config = dict(CONFIG, wall_seconds=600, artifact_bytes=64*1024**2)
    guard = block.Guard(output, config)
    # Maximum lookup capacity occurs on the smallest task roster, not the
    # largest: 16000/313 selects native d_embed64, while 16000/600 selects16.
    screen, split = synthetic_screen(313)
    record = fit_fold(screen, split, p['source'], output/'profile', guard)
    # Exercise all saved numerical APIs twice, including the 4k validation and
    # 3k test workloads, and compare exact arrays without biological responses.
    for part in (1, 2):
        indices = np.flatnonzero(split == part)
        first = predictions(output/'profile', record, p['source'], screen, indices)
        second = predictions(output/'profile', record, p['source'], screen, indices)
        assert_arrays(first, second)
        guard.check()
    # Separately bound the larger full-matrix completion/ridge workload. The
    # forecast charges BOTH workloads per fit, deliberately overestimating.
    large, ls = synthetic_screen(600)
    tr, va, te = [np.flatnonzero(ls == part) for part in (0, 1, 2)]
    refs = fit_reference_grid(large['singles'], large['pairs'][tr], large['responses'][tr],
                              CONFIG['reference_alphas'], check=guard.check)
    chosen = int(np.argmin([np.mean((r.predict(large['pairs'][va])-large['responses'][va])**2) for r in refs]))
    base_train, base_val = refs[chosen].predict(large['pairs'][tr]), refs[chosen].predict(large['pairs'][va])
    residual = large['responses'][tr]-base_train
    controls = controls_for(large, ls)
    for number, control in enumerate([None]+controls):
        y = residual if control is None else residual[control['local_permutation']]
        m = ScalarCompletion(8, .01, 500, 20261022).fit(large['pairs'][tr], y, 600,
            max(float(large['responses'][tr].std()), 1e-6),
            (large['pairs'][va], large['responses'][va]-base_val), check=guard.check)
        file = output/'profile'/f'large-completion-{number}.npz'
        m.save(file)
        np.testing.assert_array_equal(m.predict(large['pairs'][te]), ScalarCompletion.load(file).predict(large['pairs'][te]))
        guard.check()
    resources = guard.receipt()
    forecast = CONFIG['profile_margin']*60*max(resources['wall_seconds'], resources['cpu_seconds'])
    write_json(output/'qualification.json', dict(completed_utc=native.now(),
        scope='Synthetic maximum-budget workload only; no biological fitting',
        native_genes=313, completion_max_genes=600,
        protocol_sha256=sha256(output/'protocol.json'), resources=resources,
        forecast_60_fit_score_replay_seconds=forecast, forecast_margin=CONFIG['profile_margin'],
        passes_forecast=forecast <= CONFIG['wall_seconds'], exact_replay=True))
    print('qualification forecast seconds', forecast, 'passes', forecast <= CONFIG['wall_seconds'], flush=True)


def fit(output):
    p = prerequisites(output)
    qualification = read_json(output/'qualification.json')
    if qualification['protocol_sha256'] != sha256(output/'protocol.json') or not qualification['passes_forecast']:
        raise ValueError('Synthetic full-program resource forecast failed; no biological fit authorized')
    if (output/'run_started.json').exists():
        raise FileExistsError('Preserve partial/completed fitting')
    write_json(output/'run_started.json', dict(started_utc=native.now(),
        protocol_sha256=sha256(output/'protocol.json'), qualification_sha256=sha256(output/'qualification.json')))
    guard = block.Guard(output, p['config'])
    screens = load_screen_roster(Path(p['original']), Path(p['prepared']))
    assignments = load_npz(output/'assignments.npz')
    records = []
    for total, _, _ in BUDGETS:
        for task in TASKS:
            for fold in range(5):
                dest = output/'checkpoints'/str(total)/task/f'fold-{fold}'
                fit_fold(screens[task], assignments[f'{total}__{task}__{fold}'], p['source'], dest, guard)
                records.append(dict(budget=total, task=task, fold=fold,
                    path=str((dest/'selection.json').relative_to(output)), sha256=sha256(dest/'selection.json')))
                write_json(output/'fit_progress.json', dict(records=records, resources=guard.receipt()))
                print(total, task, fold, 'fit complete', guard.receipt()['wall_seconds'], flush=True)
    guard.check()
    write_json(output/'selection.json', dict(records=records, resources=guard.receipt(),
        protocol_sha256=sha256(output/'protocol.json'), completed_utc=native.now()))


def predictions(dest, record, source, screen, indices):
    block.verify_files({str(dest/name): h for name, h in record['files'].items()})
    pairs = screen['pairs'][indices]
    refs = [ScalarReference.load(dest/name) for name in record['reference_files']]
    base = refs[record['chosen_reference']].predict(pairs)
    values = dict(simple=base)
    for name in ('aligned',)+tuple(f'shuffled-{s}' for s in CONFIG['control_seeds']):
        values['rank8' if name == 'aligned' else name] = base+ScalarCompletion.load(dest/(name+'.npz')).predict(pairs)
    values['native'] = NativeNAIAD.load(source, dest/'native').predict(pairs)
    values['candidate'] = values[record['chosen_family']]
    for label, ref in zip(record['reference_labels'], refs):
        values['ref_'+label.replace('/', '_')] = ref.predict(pairs)
    return values


def assert_arrays(actual, stored):
    if set(actual) != set(stored) or any(not np.array_equal(actual[k], stored[k]) for k in actual):
        raise RuntimeError('Exact prediction replay mismatch')


def require_all_choices(selection):
    expected = {(total, task, fold) for total, _, _ in BUDGETS for task in TASKS for fold in range(5)}
    keys = [(r['budget'], r['task'], r['fold']) for r in selection['records']]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('All sixty unique validation choices must precede test scoring')


def score(output, replay=False):
    p, selection = prerequisites(output), read_json(output/'selection.json')
    require_all_choices(selection)
    if selection['protocol_sha256'] != sha256(output/'protocol.json'):
        raise ValueError('Selection/protocol drift')
    if not replay and (output/'results.json').exists():
        raise FileExistsError('Preserve prior test scores')
    consumed = read_json(output/'score_resources.json') if replay else selection['resources']
    guard = block.Guard(output, p['config'], consumed=consumed)
    screens = load_screen_roster(Path(p['original']), Path(p['prepared']))
    assignments = load_npz(output/'assignments.npz')
    results, hashes = {}, {}
    for total, _, _ in BUDGETS:
        results[str(total)] = {}
        for task in TASKS:
            screen, gathered, full_diagnostics, fold_details = screens[task], [], [], []
            controls_ok, full_count = True, 0
            for rr in [r for r in selection['records'] if r['budget'] == total and r['task'] == task]:
                guard.check()
                path = output/rr['path']
                block.verify_files({str(path): rr['sha256']})
                r, dest = read_json(path), path.parent
                split = assignments[f'{total}__{task}__{rr["fold"]}']
                q = supported_queries(screen['pairs'], split, len(screen['genes']))
                values = predictions(dest, r, p['source'], screen, q['test'])
                gathered.append(dict(indices=q['test'], **values))
                refs = [ScalarReference.load(dest/name) for name in r['reference_files']]
                diagnostic = {block.reference_label(ref): full_reference_prediction(ref, screen['pairs'][q['full_test']])
                              for ref in refs if ref.kind in {'zero', 'additive', 'minimum', 'calibration'}}
                full_diagnostics.append(dict(indices=q['full_test'], **diagnostic))
                full_count += len(q['full_test'])
                controls_ok &= len(r['controls']) == 3 and all(c['sufficient'] for c in r['controls'])
                fold_details.append(dict(fold=rr['fold'], selected_family=r['chosen_family'],
                    train=len(q['train']), used_validation=len(q['validation']),
                    reserved_validation=q['reserved_validation'], unused_validation=q['unused_validation'],
                    supported_test=len(q['test']), full_test=len(q['full_test'])))
            arrays = {key: np.concatenate([g[key] for g in gathered]) for key in gathered[0]}
            if len(np.unique(arrays['indices'])) != len(arrays['indices']):
                raise ValueError('Test block overlap')
            refs = {key[4:]: value for key, value in arrays.items() if key.startswith('ref_')}
            controls = {f'shuffled-{seed}': arrays[f'shuffled-{seed}'] for seed in CONFIG['control_seeds']}
            primary = block.evaluate_arrays(screen, arrays['indices'], arrays['candidate'], arrays['simple'], controls, refs, controls_ok)
            coverage = len(arrays['indices'])/full_count
            primary.update(coverage=coverage, full_test_pairs=full_count, abstentions=full_count-len(arrays['indices']),
                           passes_budget_gate=scientific_gate(coverage, primary), folds=fold_details)
            primary['fixed_families'] = {k: metrics(screen['responses'][arrays['indices']], arrays[k]) for k in ('rank8', 'native')}
            fi = np.concatenate([d['indices'] for d in full_diagnostics])
            primary['full_test_nonlookup_diagnostics'] = {k: metrics(screen['responses'][fi], np.concatenate([d[k] for d in full_diagnostics]))
                for k in full_diagnostics[0] if k != 'indices'}
            results[str(total)][task] = primary
            name = f'{total}__{task}_predictions.npz'
            if not replay:
                np.savez_compressed(output/name, **arrays)
            assert_arrays(arrays, load_npz(output/name))
            hashes[name] = sha256(output/name)
            guard.check()
    science = dict(protocol_sha256=sha256(output/'protocol.json'), selection_sha256=sha256(output/'selection.json'),
        tasks=results, selected_budget=nominee(results, TASKS), prediction_hashes=hashes,
        evidence='Exposed-source conditional development; no independent confirmation or measured physical assay savings')
    if replay:
        if science != read_json(output/'results.json'):
            raise RuntimeError('Exact scientific JSON replay mismatch')
        guard.check()
        write_json(output/'replay.json', dict(completed_utc=native.now(), exact_replay=True,
            results_sha256=sha256(output/'results.json'), resources=guard.receipt()))
    else:
        guard.check()
        write_json(output/'results.json', science)
        write_json(output/'score_resources.json', guard.receipt())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'verify', 'profile', 'fit', 'score', 'replay'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block-run', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.action == 'freeze':
            freeze(args)
        else:
            try:
                {'verify': verify, 'profile': profile, 'fit': fit, 'score': score,
                 'replay': lambda p: score(p, True)}[args.action](args.output)
            except BaseException as error:
                if args.output.is_dir():
                    write_json(args.output/f'failure-{args.action}-{time.time_ns()}.json',
                        dict(created_utc=native.now(), phase=args.action, error=repr(error)))
                raise
