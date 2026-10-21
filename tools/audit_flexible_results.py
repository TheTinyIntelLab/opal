"""Read-only Q2b checkpoint/selection/statistic audit; performs no fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import load_npz, read_json, sha256, write_json
from opal.flexible_singles import FlexibleSingles
from opal.scalar_screen import load_screen_roster, TASKS


def verify_hashes(files):
    for path, digest in files.items():
        assert sha256(path) == digest, path


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-15)


def metric(y, p, expected):
    d = p-y
    actual = dict(mse=float(np.average(d*d)), mae=float(np.average(np.abs(d))),
                  bias=float(np.average(d)), target_variance=float(np.var(y)),
                  pearson=float(np.corrcoef(y, p)[0, 1]), pairs=len(y))
    assert set(actual) == set(expected)
    for k in actual:
        close(actual[k], expected[k])


def uncertainty(delta, pairs, expected):
    # Independent arithmetic, with the registered RNG and endpoint weights.
    active = np.unique(pairs)
    e = np.searchsorted(active, pairs)
    g = len(active)
    rng = np.random.default_rng(20261022)
    means = []
    for _ in range(2000):
        counts = np.bincount(rng.integers(0, g, g), minlength=g)
        w = np.prod(counts[e], axis=1)
        if np.sum(w):
            means.append(np.sum(w*delta)/np.sum(w))
    bounds = np.quantile(means, [.025, .975])
    close(bounds, expected['gene_conditional_95'])
    close(bounds, expected['interval'])
    close(np.mean(delta), expected['mean'])
    assert expected['active_genes'] == g and expected['nonempty_draws'] == len(means)
    assert expected['discarded_empty_draws'] == 2000-len(means)
    for gene in active:
        keep = (pairs[:, 0] != gene) & (pairs[:, 1] != gene)
        close(np.average(delta[keep]), expected['leave_one_gene_out'][str(gene)])
    return bounds.tolist()


def audit(run, output):
    if output.exists():
        raise FileExistsError('Preserve prior independent audit')
    started = time.monotonic()
    p = read_json(run/'protocol.json')
    verify_hashes({str(ROOT/k): v for k, v in p['code'].items()})
    verify_hashes({str(ROOT/k): v for k, v in p['block_code'].items()})
    verify_hashes(p['files'])
    br, q = Path(p['block_run']), Path(p['q1b_run'])
    bp = read_json(br/'protocol.json')
    verify_hashes(bp['files'])
    qreg = read_json(q/'registration.json')
    verify_hashes(qreg['inputs'])
    verify_hashes({str(ROOT/k): v for k, v in qreg['code'].items()})
    prereq, qr = read_json(q/'prerequisites.json'), read_json(q/'replay.json')
    assert prereq['passed'] and qr['passed']
    assert prereq['registration_sha256'] == sha256(q/'registration.json')
    assert prereq['junit_sha256'] == sha256(q/'prerequisites.xml')
    assert prereq['log_sha256'] == sha256(q/'prerequisites.log')
    assert qr['results_sha256'] == sha256(q/'results.json')
    assert qr['arrays_sha256'] == sha256(q/'prepared.npz')
    assert read_json(q/'results.json')['integrity_gate']
    assert not any((q/n).exists() for n in ('failure.json', 'replay_failure.json'))
    start, tests = read_json(run/'run_started.json'), read_json(run/'prefit_tests.json')
    verify_hashes(start['q1b_files'])
    assert tests['returncode'] == 0 and tests['code'] == p['code']
    assert tests['protocol_sha256'] == sha256(run/'protocol.json')
    assert p['created_utc'] <= tests['started_utc'] <= tests['completed_utc'] <= start['started_utc']
    assert start['prefit_tests_sha256'] == sha256(run/'prefit_tests.json')
    result, replay = read_json(run/'results.json'), read_json(run/'replay.json')
    selection = read_json(run/'selection.json')
    assert result['protocol_sha256'] == sha256(run/'protocol.json')
    assert replay['results_sha256'] == sha256(run/'results.json')
    assert result['selection_sha256'] == sha256(run/'selection.json')
    assert read_json(run/'score_started.json')['selection_sha256'] == sha256(run/'selection.json')
    assert len(selection['fits']) == 30
    screens = load_screen_roster(Path(bp['original']), Path(bp['prepared']))
    assignments = load_npz(br/'assignments.npz')
    records, winners, penalties = {}, Counter(), Counter()
    max_val_difference = 0.
    for r in selection['fits']:
        path = run/r['path']
        verify_hashes({str(path): r['sha256']})
        rec = read_json(path)
        verify_hashes({str(path.parent/k): v for k, v in rec['files'].items()})
        s = screens[r['task']]
        split = assignments[f"{r['task']}__{r['fold']}"]
        tr, va = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
        losses = {}
        for family in ('phenotype', 'nodes'):
            losses[family] = []
            for i, alpha in enumerate(p['config']['alphas']):
                model = FlexibleSingles.load(path.parent/f'{family}-{i}.npz')
                assert model.alpha == alpha and model.nodes == (family == 'nodes')
                np.testing.assert_array_equal(model.singles, s['singles'])
                training_coordinates = np.sort(s['singles'][s['pairs'][tr]], axis=1)
                close(model.input_center, np.average(training_coordinates, axis=0))
                close(model.input_scale, np.maximum(np.std(training_coordinates, axis=0), 1e-8))
                expected_rows = np.random.default_rng(20261022).choice(len(tr), 64, replace=False)
                np.testing.assert_array_equal(model.center_rows, expected_rows)
                close(model.centers, ((training_coordinates-model.input_center)/model.input_scale)[expected_rows])
                loss = float(np.average((model.predict(s['pairs'][va])-s['responses'][va])**2))
                max_val_difference = max(max_val_difference, abs(loss-rec['validation_mse'][family][i]))
                close(loss, rec['validation_mse'][family][i])
                losses[family].append(loss)
        chosen = {f: int(np.argmin(v)) for f, v in losses.items()}
        winner = min(('phenotype', 'nodes'), key=lambda f: losses[f][chosen[f]])
        assert chosen == rec['chosen'] and winner == rec['winner']
        winners[winner] += 1
        for family in chosen:
            penalties[f'{family}/{p["config"]["alphas"][chosen[family]]:g}'] += 1
        records[r['task'], r['fold']] = (rec, path.parent)
    task_results = {}
    for task in TASKS:
        s = screens[task]
        ids, values = [], {f: [] for f in ('phenotype', 'nodes', 'selected')}
        for fold in range(5):
            rec, dest = records[task, fold]
            index = np.flatnonzero(assignments[f'{task}__{fold}'] == 2)
            ids.append(index)
            pred = {f: FlexibleSingles.load(dest/f'{f}-{rec["chosen"][f]}.npz').predict(s['pairs'][index])
                    for f in ('phenotype', 'nodes')}
            pred['selected'] = pred[rec['winner']]
            for f in values:
                values[f].append(pred[f])
        ids = np.concatenate(ids)
        values = {f: np.concatenate(v) for f, v in values.items()}
        stored, old = load_npz(run/(task+'_predictions.npz')), load_npz(br/(task+'_predictions.npz'))
        assert sha256(run/(task+'_predictions.npz')) == result['prediction_hashes'][task]
        assert sha256(br/(task+'_predictions.npz')) == read_json(br/'results.json')['prediction_hashes'][task]
        np.testing.assert_array_equal(ids, old['test_indices'])
        np.testing.assert_array_equal(ids, stored['test_indices'])
        assert len(np.unique(ids)) == len(ids)
        y, pairs = s['responses'][ids], s['pairs'][ids]
        tr = result['tasks'][task]
        metric(y, old['baseline'], tr['fixed_simple'])
        metric(y, old['candidate'], tr['fixed_completion'])
        for f, v in values.items():
            np.testing.assert_array_equal(v, stored[f])
            metric(y, v, tr['flexible'][f])
            for r in range(2):
                metric(s['replicates'][ids, r], v, tr['infections'][r][f])
        a, b, c = values['phenotype'], values['selected'], old['candidate']
        da, dr = (a-y)**2-(c-y)**2, (c-y)**2-(b-y)**2
        pa = tr['acceptance']['phenotype_vs_completion']
        pr = tr['acceptance']['completion_vs_selected_flexible']
        ci_a, ci_r = uncertainty(da, pairs, pa['uncertainty']), uncertainty(dr, pairs, pr['uncertainty'])
        gain = 1-np.average((c-y)**2)/np.average((b-y)**2)
        close(gain, pr['relative_gain'])
        noninferior = (ci_a[1] <= .02*np.average((c-y)**2) and all(
            np.average((a-s['replicates'][ids, r])**2) <= 1.02*np.average((c-s['replicates'][ids, r])**2) for r in range(2)))
        residual = (gain >= .05 and ci_r[1] < 0 and all(
            np.average((c-s['replicates'][ids, r])**2) <= np.average((b-s['replicates'][ids, r])**2) for r in range(2)))
        assert bool(noninferior) == tr['acceptance']['phenotype_conditionally_noninferior']
        assert bool(residual) == tr['acceptance']['completion_residual_gate']
        task_results[task] = dict(pairs=len(ids), phenotype_mse=np.mean((a-y)**2), selected_flexible_mse=np.mean((b-y)**2),
            completion_mse=np.mean((c-y)**2), completion_gain_percent=100*gain,
            completion_minus_flexible_ci=ci_r, phenotype_noninferiority=bool(noninferior), residual_gate=bool(residual))
    assert all(t['residual_gate'] for t in task_results.values()) == result['all_task_completion_residual_gate']
    assert all(t['phenotype_noninferiority'] for t in task_results.values()) == result['all_task_phenotype_noninferiority']
    receipt = dict(status='passed', fits_performed=0, source_run=str(run.resolve()),
        checks=['all original input/code hashes', 'Q1b success/prerequisite/replay/array hash chain and no failure',
                'postfreeze passing Q2b tests', 'all 240 validation losses and 30 argmins/policies',
                'training moments and seeded center rows', 'all selected checkpoint test predictions bit-exact',
                'all original test populations/order', 'all primary/secondary metrics',
                '12 primary paired conditional intervals and leave-gene-out arithmetic', 'all primary scientific gate flags'],
        protocol_sha256=sha256(run/'protocol.json'), results_sha256=sha256(run/'results.json'),
        replay_sha256=sha256(run/'replay.json'), code_sha256=sha256(__file__),
        validation_loss_max_abs_difference=max_val_difference, family_selection=dict(winners),
        family_penalty_counts=dict(penalties), tasks=task_results,
        wall_seconds=time.monotonic()-started,
        command=f'.venv/Scripts/python.exe tools/audit_flexible_results.py --run "{run}" --output <new-json-path>')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        audit(args.run, args.output)
