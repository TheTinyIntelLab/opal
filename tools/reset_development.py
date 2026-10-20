"""Frozen exploratory cross-infection-replicate growth diagnostic, one CPU worker."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import environment, load_npz, read_json, sha256, write_json
from opal.models.fitness import FitnessCompletion, GrowthReference

CODE = ['tools/reset_development.py', 'tests/test_reset_development.py',
        'src/models/fitness.py', 'docs/research_reset/development_protocol.md']
CONFIG = dict(seed=20261022, rank=8, penalty=.01, steps=500,
              bootstrap_draws=2000, minimum_relative_gain=.02,
              references=['zero', 'additive', 'minimum', 'calibration', 'node'],
              reference_alpha=100., wall_seconds=1200)


def now():
    return datetime.now(timezone.utc).isoformat()


def signature():
    return {name: sha256(ROOT / name) for name in CODE}


def validate(data, split):
    pairs, reps, singles = data['pairs'], data['replicates'], data['singles']
    n = len(data['genes'])
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError('Integer pair matrix required')
    if np.any(pairs < 0) or np.any(pairs >= n) or np.any(pairs[:, 0] >= pairs[:, 1]):
        raise ValueError('Distinct ordered canonical pairs required')
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError('Duplicate pairs leak across splits')
    if len(np.unique(data['genes'])) != n or reps.shape != (2, 2, len(pairs)) or singles.shape != (2, 2, n):
        raise ValueError('Gene or response schema mismatch')
    if not np.isfinite(reps).all() or not np.isfinite(singles).all():
        raise ValueError('Missing outcomes/singles cannot be silently imputed')
    if split.shape != (len(pairs),) or set(np.unique(split)) != {0, 1, 2}:
        raise ValueError('Three disjoint nonempty splits required')
    support = np.bincount(pairs[split == 0].ravel(), minlength=n) > 0
    if not support[pairs].all():
        raise ValueError('Completion requires train-supported genes')
    if np.any(data['counts'] < 4) or np.any(data['single_counts'][:, :, pairs.ravel()] < 20):
        raise ValueError('Frozen coverage eligibility violated')
    if not np.allclose(data['responses'], reps.mean(1), rtol=0, atol=1e-15):
        raise ValueError('Replicate aggregation mismatch')


def mse(prediction, target):
    p, y = np.asarray(prediction), np.asarray(target)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Finite aligned predictions and targets required')
    return np.mean((p-y)**2, axis=-1)


def shuffle_residuals(residuals, seed):
    """One shared pair permutation preserves cross-context residual covariance."""
    return np.asarray(residuals)[:, np.random.default_rng(seed).permutation(residuals.shape[1])]


def node_bootstrap(delta, pairs, genes, draws, seed):
    """Positive node multipliers retain dependence among pairs sharing a gene.

    This is a descriptive dyadic sensitivity interval, not a culture-level CI.
    Both endpoints share the same Exp(1) node weight because pairs are symmetric.
    """
    delta, pairs = np.asarray(delta), np.asarray(pairs)
    if delta.shape != (len(pairs),) or draws < 1:
        raise ValueError('One paired loss difference per canonical pair required')
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for i in range(draws):
        w = rng.exponential(size=genes)
        weights = w[pairs[:, 0]] * w[pairs[:, 1]]
        values[i] = weights @ delta / weights.sum()
    return np.quantile(values, [.025, .975]).tolist()


def selected_reference(singles, pairs, y, train, val, config):
    models, scores = [], []
    for kind in config['references']:
        model = GrowthReference(kind, config['reference_alpha']).fit(singles, pairs[train], y[:, train])
        models.append(model)
        scores.append(mse(model.predict(pairs[val]), y[:, val]))
    scores = np.asarray(scores)
    choices = np.argmin(scores, axis=0)
    prediction = np.asarray([models[i].predict(pairs)[c] for c, i in enumerate(choices)])
    return prediction, choices, scores, models


def freeze(data, prior, output):
    if output.exists():
        raise FileExistsError('Never overwrite an existing registration')
    original = read_json(ROOT / 'results/fitness-20261020/protocol.json')
    if sha256(data / 'profiles.npz') != original['profiles_sha256'] or sha256(prior) != original['assignment_sha256']:
        raise ValueError('Reuse exactly the deposited development population and original split')
    d, split = load_npz(data / 'profiles.npz'), load_npz(prior)['split']
    validate(d, split)
    output.mkdir(parents=True)
    (output / '.gitignore').write_text('*.npz\n', encoding='utf-8')
    np.savez_compressed(output / 'assignment.npz', split=split)
    write_json(output / 'protocol.json', dict(created_utc=now(), config=CONFIG, code=signature(),
        data_root=str(data.resolve()), profiles_sha256=sha256(data / 'profiles.npz'),
        original_protocol_sha256=sha256(ROOT / 'results/fitness-20261020/protocol.json'),
        assignment_sha256=sha256(output / 'assignment.npz'),
        split_counts={name: int(np.sum(split == i)) for i, name in enumerate(['train', 'validation', 'test'])},
        environment=environment(), evidence='exploratory reused development source, all responses previously exposed',
        source='GSE116198 author deposit; existing acquisition and licensing receipts remain authoritative'))


def verify(output):
    p = read_json(output / 'protocol.json')
    if signature() != p['code']:
        raise ValueError('Registration code drift')
    if (output / 'prefit_tests.json').exists() or (output / 'run_started.json').exists():
        raise FileExistsError('Preserve ordering receipts')
    started = now()
    command = [sys.executable, '-m', 'pytest', 'tests/test_reset_development.py', '-q']
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    write_json(output / 'prefit_tests.json', dict(started_utc=started, completed_utc=now(),
        command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        protocol_sha256=sha256(output / 'protocol.json'), code=signature()))
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def run(output):
    p, tests = read_json(output / 'protocol.json'), read_json(output / 'prefit_tests.json')
    if signature() != p['code'] or tests['code'] != p['code'] or tests['returncode'] != 0:
        raise ValueError('Passing tests must cover frozen code')
    if tests['protocol_sha256'] != sha256(output / 'protocol.json') or tests['started_utc'] < p['created_utc']:
        raise ValueError('Tests must follow protocol registration')
    if (output / 'run_started.json').exists():
        raise FileExistsError('Completed and partial runs are immutable')
    data = Path(p['data_root']) / 'profiles.npz'
    if sha256(data) != p['profiles_sha256'] or sha256(output / 'assignment.npz') != p['assignment_sha256']:
        raise ValueError('Data/split drift')
    d, split = load_npz(data), load_npz(output / 'assignment.npz')['split']
    validate(d, split)
    write_json(output / 'run_started.json', dict(started_utc=now(),
        protocol_sha256=sha256(output / 'protocol.json'), prefit_tests_sha256=sha256(output / 'prefit_tests.json')))
    started = time.perf_counter()
    cfg, pairs, genes = p['config'], d['pairs'], len(d['genes'])
    train, val, test = [np.flatnonzero(split == i) for i in range(3)]
    selections, fits = [], []
    for source_rep in [0, 1]:
        y, singles = d['replicates'][:, source_rep], d['singles'][:, source_rep]
        base, choices, scores, references = selected_reference(singles, pairs, y, train, val, cfg)
        scales = np.maximum(y[:, train].std(1), 1e-6)
        residual = y[:, train] - base[:, train]
        selection = dict(source_replicate=source_rep + 1, choices=choices.tolist(),
            chosen_references=[cfg['references'][i] for i in choices], validation_mse=scores.tolist())
        for control in [False, True]:
            name = f'rep{source_rep+1}-' + ('shuffled' if control else 'aligned')
            target = shuffle_residuals(residual, cfg['seed']) if control else residual
            model = FitnessCompletion('independent', cfg['rank'], cfg['penalty'], cfg['steps'], cfg['seed']).fit(
                pairs[train], target, genes, scales, validation=(pairs[val], y[:, val]-base[:, val]))
            model.save(output / (name + '.npz'))
            residual_test = model.predict(pairs[test])
            np.testing.assert_array_equal(residual_test, FitnessCompletion.load(output / (name + '.npz')).predict(pairs[test]))
            fits.append(dict(name=name, source_rep=source_rep, negative_control=control,
                checkpoint_sha256=sha256(output / (name + '.npz')), best_step=model.best_step,
                history=model.history, baseline=base[:, test], prediction=base[:, test] + residual_test,
                references=np.asarray([ref.predict(pairs[test]) for ref in references])))
            if time.perf_counter() - started > cfg['wall_seconds']:
                raise RuntimeError('Registered wall-time cap reached; preserve partial attempt')
        selections.append(selection)
    # All model/reference choices precede test metrics, including opposite-replica scoring.
    write_json(output / 'selection.json', dict(completed_utc=now(), selections=selections,
        fits=[{k: v for k, v in fit.items() if k not in {'baseline', 'prediction', 'references'}} for fit in fits]))
    rows, arrays = [], dict(test_indices=test, test_pairs=pairs[test])
    for fit in fits:
        source_rep, name = fit['source_rep'], fit['name']
        arrays[name + '_baseline'], arrays[name + '_prediction'] = fit['baseline'], fit['prediction']
        for c, context in enumerate(d['contexts']):
            target = d['replicates'][c, 1-source_rep, test]
            loss = (fit['prediction'][c]-target)**2
            base_loss = (fit['baseline'][c]-target)**2
            ci = node_bootstrap(loss-base_loss, pairs[test], genes, cfg['bootstrap_draws'], cfg['seed'])
            rows.append(dict(name=name, context=str(context), source_replicate=source_rep+1,
                target_replicate=2-source_rep, negative_control=fit['negative_control'],
                candidate_cross_replicate_mse=float(loss.mean()), baseline_cross_replicate_mse=float(base_loss.mean()),
                relative_gain=float(1-loss.mean()/base_loss.mean()), gene_multiplier_95_loss_difference=ci,
                same_replicate_candidate_mse=float(mse(fit['prediction'][c], d['replicates'][c, source_rep, test])),
                same_replicate_baseline_mse=float(mse(fit['baseline'][c], d['replicates'][c, source_rep, test])),
                target_infection_minimum_single_mse=float(mse(d['singles'][c, 1-source_rep, pairs[test]].min(-1), target)),
                target_infection_additive_single_mse=float(mse(d['singles'][c, 1-source_rep, pairs[test]].sum(-1), target)),
                all_reference_cross_replicate_mse={kind: float(mse(pred[c], target))
                    for kind, pred in zip(cfg['references'], fit['references'])}))
    residual_agreement = []
    bases = [next(f for f in fits if f['source_rep'] == r and not f['negative_control'])['baseline'] for r in [0, 1]]
    for c, context in enumerate(d['contexts']):
        a, b = [d['replicates'][c, r, test] - bases[r][c] for r in [0, 1]]
        residual_agreement.append(dict(context=str(context), residual_pearson=float(np.corrcoef(a, b)[0, 1]),
            raw_replicate_pearson=float(np.corrcoef(d['replicates'][c, 0, test], d['replicates'][c, 1, test])[0, 1]),
            residual_cross_product=float(np.mean(a*b)), replicate_difference_mse=float(np.mean((a-b)**2))))
    aligned = [r for r in rows if not r['negative_control']]
    passed = all(r['relative_gain'] >= cfg['minimum_relative_gain'] and r['gene_multiplier_95_loss_difference'][1] < 0 for r in aligned)
    control_pass = all(r['candidate_cross_replicate_mse'] < next(x for x in rows if x['negative_control'] and x['context'] == r['context'] and x['source_replicate'] == r['source_replicate'])['candidate_cross_replicate_mse'] for r in aligned)
    np.savez_compressed(output / 'predictions.npz', **arrays)
    write_json(output / 'results.json', dict(completed_utc=now(), wall_seconds=time.perf_counter()-started,
        protocol_sha256=sha256(output / 'protocol.json'), selection_sha256=sha256(output / 'selection.json'),
        predictions_sha256=sha256(output / 'predictions.npz'), contexts=rows, residual_agreement=residual_agreement,
        development_gate=bool(passed and control_pass), beats_shuffled_control_all_directions=control_pass,
        scope='exploratory within-source held-pair cross-infection-replicate diagnosis; no external confirmation',
        uncertainty='dyadic node-multiplier sensitivity conditional on two deposited infections, not independent-culture uncertainty'))


def replay(output):
    p, result = read_json(output / 'protocol.json'), read_json(output / 'results.json')
    if sha256(output / 'protocol.json') != result['protocol_sha256'] or sha256(output / 'predictions.npz') != result['predictions_sha256']:
        raise ValueError('Receipt/array drift')
    predictions, selection = load_npz(output / 'predictions.npz'), read_json(output / 'selection.json')
    maximum = 0.
    for fit in selection['fits']:
        name = fit['name']
        if sha256(output / (name + '.npz')) != fit['checkpoint_sha256']:
            raise ValueError('Checkpoint drift')
        prediction = predictions[name + '_baseline'] + FitnessCompletion.load(output / (name + '.npz')).predict(predictions['test_pairs'])
        maximum = max(maximum, float(np.max(np.abs(prediction-predictions[name + '_prediction']))))
    if maximum != 0:
        raise ValueError('Prediction replay differs')
    write_json(output / 'replay.json', dict(completed_utc=now(), maximum_absolute_error=maximum,
        models=4, pairs=len(predictions['test_pairs']), protocol_sha256=sha256(output / 'protocol.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'verify', 'run', 'replay'])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--prior-assignment', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.action == 'freeze':
            freeze(args.data, args.prior_assignment, args.output)
        else:
            {'verify': verify, 'run': run, 'replay': replay}[args.action](args.output)
