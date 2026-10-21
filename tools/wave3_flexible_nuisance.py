"""Prospective strongest-reference nuisance discriminator, exposed source only.

Reuse frozen v1 reagent extraction and ridge arithmetic without changing v1.
The historical flexible node reference is nested in the augmented reference.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import wave3_nuisance as prior
from tools.wave3_evaluator_v3 import strict_load_completion
from opal.flexible_singles import FlexibleSingles
from opal.block_completion import ScalarCompletion
from opal.artifacts import load_npz, read_json, write_json, sha256, environment
from opal.scalar_screen import block_assignments, split_report, residual_permutations, metrics

CONFIG = dict(prior.CONFIG, centers=64)
FAMILIES = ('flexible_nodes', 'flexible_nuisance')
CODE = prior.CODE + ('tools/wave3_flexible_nuisance.py',
    'tests/test_wave3_flexible_nuisance.py', 'docs/research_reset/wave3_flexible_nuisance_protocol.md',
    'src/flexible_singles.py', 'tools/wave3_evaluator_v3.py')


def signature():
    return {name: sha256(ROOT/name) for name in CODE}


def mapped_features(transform, pairs, nuisance=None):
    """Symmetric train-frozen singles map; physical-arm features are pair keyed."""
    chunks = []
    for start in range(0, len(pairs), 2048):
        chunks.append(transform._features(pairs[start:start+2048]))
    x = np.concatenate(chunks)
    return x if nuisance is None else np.column_stack((x, nuisance))


def fit_reference(transform, pairs, singles, nuisance, y, train, val, check):
    x = mapped_features(transform, pairs, nuisance)
    additive = singles[pairs].sum(1)
    model, menu = prior.ridge_grid(x, pairs, y-additive, train, val, True, len(singles), check)
    return model, menu, x


def reference_predict(model, x, pairs, singles, ids):
    return singles[pairs[ids]].sum(1)+prior.predict(model, x, pairs, ids, len(singles))


def gate(stats, comparisons, infections, informative):
    return bool(comparisons['flexible_nuisance']['relative_gain'] >= CONFIG['gate']
        and comparisons['flexible_nuisance']['uncertainty']['gene_conditional_95'][1] < 0
        and all(r['aligned']['mse'] <= r['flexible_nuisance']['mse'] for r in infections)
        and informative
        and all(stats['aligned']['mse'] < stats[f'shuffled-{s}']['mse'] for s in CONFIG['control_seeds'])
        and all(stats['aligned']['mse'] <= stats[k]['mse'] for k in
            FAMILIES+('existing_calibrated_main', 'historical_minimum', 'guide_minimum', 'zero')))


def freeze(a):
    if a.output.exists():
        raise FileExistsError('Preserve attempts')
    old = prior.registered(a.prior)
    if not read_json(a.prior/'replay.json')['passed']:
        raise ValueError('Require preserved v1 replay')
    prior.check_files(old['inputs'])
    evidence = {}
    for path in (a.prior, a.flexible):
        for f in path.rglob('*'):
            if f.is_file() and f.suffix in ('.json', '.npz'):
                evidence[str(f.resolve())] = sha256(f)
    a.output.mkdir(parents=True)
    write_json(a.output/'registration.json', dict(created_utc=prior.now(), code=signature(),
        config=CONFIG, environment=environment(), inputs=dict(old['inputs'], **evidence),
        q1b=old['q1b'], sidecar=old['sidecar'], block=old['block'],
        prior=str(a.prior.resolve()), flexible=str(a.flexible.resolve()),
        evidence='v1 and Q2b outcomes exposed before this design; exploratory only',
        selection='four existing penalties per nested family; completion baseline always flexible_nuisance',
        eligibility=old['eligibility']))


def registered(out, require_tests=True):
    r = read_json(out/'registration.json')
    if r['code'] != signature() or r['config'] != CONFIG or r['environment'] != environment():
        raise ValueError('Registration drift')
    prior.check_files(r['inputs'])
    if require_tests:
        t = read_json(out/'prerequisites.json')
        if not t['passed'] or t['registration_sha256'] != sha256(out/'registration.json'):
            raise ValueError('Passing postfreeze prerequisites required')
    if any(out.glob('*failure.json')):
        raise ValueError('Preserve failed attempt')
    return r


def verify(out):
    registered(out, False)
    if (out/'prerequisites.json').exists() or (out/'fit_started.json').exists():
        raise FileExistsError('Preserve prerequisite ordering')
    started = prior.now()
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_wave3_flexible_nuisance.py',
               'tests/test_wave3_nuisance.py', 'tests/test_wave3_evaluator_v3.py']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    write_json(out/'prerequisites.json', dict(started=started, completed=prior.now(), command=command,
        passed=result.returncode == 0, registration_sha256=sha256(out/'registration.json'),
        stdout=result.stdout, stderr=result.stderr))
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)


def fit(out):
    r = registered(out)
    if (out/'fit_started.json').exists():
        raise FileExistsError('Preserve started fit')
    write_json(out/'fit_started.json', dict(started=prior.now()))
    guard = prior.Guard(out, CONFIG)
    try:
        bank, features, audit = prior.prepare(r, guard.check)
        np.savez_compressed(out/'features.npz', features=features)
        write_json(out/'covariate_audit.json', audit)
        assignments = load_npz(Path(r['block'])/'assignments.npz')
        records = []
        for c, context in enumerate(('K562', 'Jurkat')):
            task = 'horlbeck_'+context
            pairs, y = bank['pairs'], bank['responses'][c]
            singles = bank['singles'][c].mean(0)
            screen = dict(genes=bank['genes'], pairs=pairs, accession='GSE116198', singles=singles,
                counts=bank['counts'][c].T, control_counts=bank['counts'].min(0).T,
                control_singles_summary=bank['singles'].mean((0, 1)))
            for fold, expected in enumerate(block_assignments(screen)):
                guard.check()
                split = assignments[f'{task}__{fold}']
                if not np.array_equal(split, expected) or not split_report(screen, split)['eligible']:
                    raise ValueError('Unchanged eligible blocks required')
                tr, va = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
                dest = out/task/f'fold-{fold}'
                dest.mkdir(parents=True)
                transform = FlexibleSingles(centers=CONFIG['centers'], seed=CONFIG['seed']).fit(
                    singles, pairs[tr], y[tr], check=guard.check)
                transform.save(dest/'transform.npz')
                menus = {}
                for family in FAMILIES:
                    nuisance = features[c] if family == 'flexible_nuisance' else None
                    model, menu, x = fit_reference(transform, pairs, singles, nuisance, y, tr, va, guard.check)
                    np.savez_compressed(dest/(family+'.npz'), **model)
                    menus[family] = menu
                    if family == 'flexible_nodes':
                        # The established strongest reference must remain the same function.
                        old_selection = read_json(Path(r['flexible'])/task/f'fold-{fold}'/'selection.json')
                        old = FlexibleSingles.load(Path(r['flexible'])/task/f'fold-{fold}'/
                            f"nodes-{old_selection['chosen']['nodes']}.npz")
                        if not np.allclose(reference_predict(model, x, pairs, singles, va),
                                           old.predict(pairs[va]), rtol=0, atol=1e-12):
                            raise ValueError('Strong-reference validation equivalence failed')
                base_train = reference_predict(model, x, pairs, singles, tr)
                base_val = reference_predict(model, x, pairs, singles, va)
                residual = y[tr]-base_train
                permutations = residual_permutations(screen, split)
                if not all(p['sufficient'] for p in permutations):
                    raise ValueError('Insufficient permutation support')
                checkpoints = []
                for name, order in [('aligned', None)]+[(f'shuffled-{p["seed"]}', p['local_permutation']) for p in permutations]:
                    target = residual if order is None else residual[order]
                    comp = ScalarCompletion(rank=CONFIG['rank'], penalty=CONFIG['penalty'],
                        steps=CONFIG['steps'], seed=CONFIG['seed']).fit(pairs[tr], target, len(singles),
                        float(y[tr].std()), (pairs[va], y[va]-base_val), check=guard.check)
                    comp.save(dest/(name+'.npz'))
                    checkpoints.append(dict(name=name, best_step=comp.best_step, history=comp.history))
                records.append(dict(task=task, fold=fold, menus=menus, completion=checkpoints,
                    files={f.name: sha256(f) for f in dest.glob('*.npz')}, split=split_report(screen, split),
                    controls=[{k: p[k] for k in ('seed', 'permutable_fraction', 'moved_fraction', 'sufficient')} for p in permutations]))
                write_json(out/'fit_progress.json', dict(records=records, resources=guard.receipt()))
        guard.check()
        write_json(out/'selection.json', dict(completed_utc=prior.now(), records=records,
            features_sha256=sha256(out/'features.npz'), resources=guard.receipt()))
    except BaseException as e:
        write_json(out/'fit_failure.json', dict(error=repr(e), resources=guard.receipt()))
        raise


def score(out, replay=False):
    r = registered(out)
    selection = read_json(out/'selection.json')
    prefix = 'replay' if replay else 'score'
    if (out/(prefix+'_started.json')).exists():
        raise FileExistsError('Preserve scoring attempts')
    roster = {(f'horlbeck_{c}', f) for c in ('K562', 'Jurkat') for f in range(5)}
    if len(selection['records']) != 10 or {(s['task'], s['fold']) for s in selection['records']} != roster:
        raise ValueError('Complete validation choices required')
    if selection['features_sha256'] != sha256(out/'features.npz'):
        raise ValueError('Feature drift')
    for rec in selection['records']:
        dest = out/rec['task']/f'fold-{rec["fold"]}'
        prior.check_files({str(dest/k): v for k, v in rec['files'].items()})
    if replay and read_json(out/'score_started.json')['selection_sha256'] != sha256(out/'selection.json'):
        raise ValueError('Scored selection drift')
    write_json(out/(prefix+'_started.json'), dict(started=prior.now(), selection_sha256=sha256(out/'selection.json')))
    guard = prior.Guard(out, CONFIG, consumed=read_json(out/'score_resources.json') if replay else selection['resources'])
    try:
        q = read_json(Path(r['q1b'])/'registration.json')
        bank = load_npz(q['paths'][2])
        features = load_npz(out/'features.npz')['features']
        splits = load_npz(Path(r['block'])/'assignments.npz')
        tasks = {}
        for c, context in enumerate(('K562', 'Jurkat')):
            task = 'horlbeck_'+context
            singles, pairs = bank['singles'][c].mean(0), bank['pairs']
            gathered = []
            for rec in [s for s in selection['records'] if s['task'] == task]:
                guard.check()
                ids = np.flatnonzero(splits[f'{task}__{rec["fold"]}'] == 2)
                dest = out/task/f'fold-{rec["fold"]}'
                transform = FlexibleSingles.load(dest/'transform.npz')
                predictions = {}
                for family in FAMILIES:
                    x = mapped_features(transform, pairs, features[c] if family == 'flexible_nuisance' else None)
                    predictions[family] = reference_predict(load_npz(dest/(family+'.npz')), x, pairs, singles, ids)
                for name in ['aligned']+[f'shuffled-{s}' for s in CONFIG['control_seeds']]:
                    comp = strict_load_completion(dest/(name+'.npz'), rec['files'][name+'.npz'])
                    predictions[name] = predictions['flexible_nuisance']+comp.predict(pairs[ids])
                gathered.append(dict(ids=ids, predictions=predictions))
            ids = np.concatenate([g['ids'] for g in gathered])
            predictions = {k: np.concatenate([g['predictions'][k] for g in gathered]) for k in gathered[0]['predictions']}
            old = load_npz(Path(r['block'])/(task+'_predictions.npz'))
            if not np.array_equal(ids, old['test_indices']) or len(ids) != len(np.unique(ids)):
                raise ValueError('Identical paired evaluation required')
            flexible = load_npz(Path(r['flexible'])/(task+'_predictions.npz'))
            if not np.array_equal(ids, flexible['test_indices']) or not np.allclose(
                    predictions['flexible_nodes'], flexible['nodes'], rtol=0, atol=1e-12):
                raise ValueError('Strong-reference test equivalence failed')
            predictions.update(existing_completion=old['candidate'], existing_calibrated_main=old['baseline'],
                historical_minimum=singles[pairs[ids]].min(1), guide_minimum=features[c, ids, 3], zero=np.zeros(len(ids)))
            y, test_pairs = bank['responses'][c, ids], pairs[ids]
            stats = {k: metrics(y, v) for k, v in predictions.items()}
            comparisons = {k: prior.comparison(y, predictions['aligned'], predictions[k], test_pairs) for k in
                FAMILIES+('existing_completion',)+tuple(f'shuffled-{s}' for s in CONFIG['control_seeds'])}
            nuisance_effect = prior.comparison(y, predictions['flexible_nuisance'], predictions['flexible_nodes'], test_pairs)
            infections = [{k: metrics(bank['replicates'][c, infection, ids], v) for k, v in predictions.items()} for infection in range(2)]
            informative = all(p['sufficient'] for rec in selection['records'] if rec['task'] == task for p in rec['controls'])
            tasks[task] = dict(pairs=len(ids), metrics=stats, comparisons=comparisons, nuisance_effect=nuisance_effect,
                infections=infections, controls_informative=informative,
                completion_beyond_nuisance_gate=gate(stats, comparisons, infections, informative))
            arrays = dict(test_indices=ids, **predictions)
            path = out/(task+'_predictions.npz')
            if replay:
                saved = load_npz(path)
                if saved.keys() != arrays.keys() or any(not np.array_equal(v, saved[k]) for k, v in arrays.items()):
                    raise ValueError('Array replay differs')
            else:
                np.savez_compressed(path, **arrays)
            guard.check()
        science = dict(tasks=tasks, all_context_gate=all(t['completion_beyond_nuisance_gate'] for t in tasks.values()),
            registration_sha256=sha256(out/'registration.json'), selection_sha256=sha256(out/'selection.json'),
            label='exposed development; finite nuisance representation; pair-technical versus biology not identified')
        if replay:
            if science != read_json(out/'results.json'):
                raise ValueError('Scientific JSON replay differs')
            write_json(out/'replay.json', dict(passed=True, results_sha256=sha256(out/'results.json'), resources=guard.receipt()))
        else:
            write_json(out/'results.json', science)
            write_json(out/'score_resources.json', guard.receipt())
    except BaseException as e:
        write_json(out/(prefix+'_failure.json'), dict(error=repr(e), resources=guard.receipt()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze', 'verify', 'fit', 'score', 'replay'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior', type=Path)
    p.add_argument('--flexible', type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with threadpool_limits(1):
        if a.action == 'freeze':
            freeze(a)
        elif a.action == 'replay':
            score(a.output, True)
        else:
            {'verify': verify, 'fit': fit, 'score': score}[a.action](a.output)
