"""Read-only independent numerical/custody audit of completed Q3b artifacts.

Requires completed scoring AND replay before opening either result. This tool
fits no models, writes only its separate audit receipt, and preserves all runs.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import load_npz, read_json, sha256, write_json
from opal.block_completion import ScalarReference, ScalarCompletion
from opal.naiad_reference import NativeNAIAD
from opal.scalar_screen import TASKS, load_screen_roster


def close(actual, expected, tolerance=1e-14):
    if actual is None or expected is None:
        assert actual is expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def metrics(y, pred):
    difference = pred-y
    return dict(mse=float(np.square(difference).mean()), mae=float(abs(difference).mean()),
        bias=float(difference.mean()), pearson=None if y.std() == 0 or pred.std() == 0 else float(np.corrcoef(y, pred)[0, 1]),
        target_variance=float(y.var()), pairs=len(y))


def check_metrics(y, pred, recorded):
    calculated = metrics(y, pred)
    for key, value in calculated.items():
        close(value, recorded[key])
    return calculated


def uncertainty(delta, pairs):
    active, inverse = np.unique(pairs, return_inverse=True)
    edges, g = inverse.reshape(-1, 2), len(active)
    rng, samples = np.random.default_rng(20261022), []
    for _ in range(2000):
        multiplicities = np.bincount(rng.integers(g, size=g), minlength=g)
        weights = np.prod(multiplicities[edges], axis=1)
        if weights.sum():
            samples.append(float(np.dot(weights, delta)/weights.sum()))
    influence = {}
    for gene in active:
        keep = np.all(pairs != gene, axis=1)
        influence[str(int(gene))] = float(delta[keep].mean()) if keep.any() else None
    return np.quantile(samples, [.025, .975]), influence, len(samples)


def verify_hashes(files):
    for path, expected in files.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f'Artifact hash changed: {path}')


def run(source, destination):
    wall, cpu, peak = time.monotonic(), time.process_time(), [0]
    def check():
        info = psutil.Process().memory_info()
        peak[0] = max(peak[0], getattr(info, 'peak_wset', info.rss))
        if max(time.monotonic()-wall, time.process_time()-cpu) > 600 or peak[0] > 2*1024**3:
            raise RuntimeError('Read-only audit resource cap exceeded')
        if shutil.disk_usage(destination.parent).free < 1024**3:
            raise RuntimeError('Read-only audit writable-volume reserve reached')
    check()
    if destination.exists():
        raise FileExistsError('Preserve previous audit receipts')
    # This is deliberately checked before read_json(results.json).
    if not all((source/name).is_file() for name in ('results.json', 'replay.json')):
        raise ValueError('Complete scoring and exact replay required before opening scores')
    protocol = read_json(source/'protocol.json')
    results, replay = read_json(source/'results.json'), read_json(source/'replay.json')
    assert replay['exact_replay'] and replay['results_sha256'] == sha256(source/'results.json')
    assert results['protocol_sha256'] == sha256(source/'protocol.json')
    verify_hashes({str(ROOT/name): digest for name, digest in protocol['code'].items()})
    verify_hashes(protocol['files'])
    verify_hashes(protocol['bank_files'])
    selection = read_json(source/'selection.json')
    assert results['selection_sha256'] == sha256(source/'selection.json')
    score_start, replay_start = [read_json(source/name) for name in ('score_started.json', 'replay_started.json')]
    assert score_start['files'] == replay_start['files']
    verify_hashes(score_start['files'])
    assert selection['completed_utc'] <= score_start['started_utc'] <= replay_start['started_utc'] <= replay['completed_utc']
    tests, qualification, fit_start = [read_json(source/name) for name in ('prefit_tests.json', 'qualification.json', 'run_started.json')]
    assert tests['returncode'] == 0 and qualification['passes_forecast']
    assert protocol['created_utc'] <= tests['started_utc'] <= qualification['completed_utc'] <= fit_start['started_utc']
    assert fit_start['qualification_sha256'] == sha256(source/'qualification.json')
    expected_keys = {(budget, task, fold) for budget in (5000, 20000) for task in TASKS for fold in range(5)}
    keys = [(r['budget'], r['task'], r['fold']) for r in selection['records']]
    assert len(keys) == 60 and set(keys) == expected_keys
    screens = load_screen_roster(Path(protocol['original']), Path(protocol['prepared']))
    base = load_npz(Path(protocol['block_run'])/'assignments.npz')
    assignments = load_npz(source/'assignments.npz')
    fold_summaries, union_costs, checkpoint_predictions = [], {}, {}
    family_counts = {str(b): Counter() for b in (5000, 20000)}
    for budget, nt, nv in ((5000, 4000, 1000), (20000, 16000, 4000)):
        union_costs[str(budget)] = {}
        for task in TASKS:
            check()
            s = screens[task]
            checkpoint_predictions[(budget, task)] = []
            train_union, validation_union = set(), set()
            hashes = [hashlib.sha256(f'opal-budget-v1|{s["accession"]}|{s["genes"][a]}|{s["genes"][b]}'.encode()).hexdigest()
                      for a, b in s['pairs']]
            for fold in range(5):
                check()
                split = assignments[f'{budget}__{task}__{fold}']
                original = base[f'{task}__{fold}']
                expected = np.full(len(split), -1, dtype=np.int8)
                expected[original == 2] = 2
                for part, count in ((0, nt), (1, nv)):
                    rows = np.flatnonzero(original == part)
                    rows = sorted(rows.tolist(), key=hashes.__getitem__)[:count]
                    expected[rows] = part
                np.testing.assert_array_equal(split, expected)
                if budget == 20000:
                    smaller = assignments[f'5000__{task}__{fold}']
                    for part in (0, 1):
                        assert set(np.flatnonzero(smaller == part)) <= set(np.flatnonzero(split == part))
                train, val, test = [np.flatnonzero(split == part) for part in (0, 1, 2)]
                support = np.bincount(s['pairs'][train].ravel(), minlength=len(s['genes'])) > 0
                val = val[support[s['pairs'][val]].all(1)]
                supported_test = test[support[s['pairs'][test]].all(1)]
                rr = next(r for r in selection['records'] if (r['budget'], r['task'], r['fold']) == (budget, task, fold))
                path = source/rr['path']
                assert sha256(path) == rr['sha256']
                r = read_json(path)
                assert r['completed_utc'] <= selection['completed_utc']
                np.testing.assert_array_equal(r['train_rows'], train)
                np.testing.assert_array_equal(r['validation_rows'], val)
                dest = path.parent
                refs = [ScalarReference.load(dest/name) for name in r['reference_files']]
                losses = [float(np.mean((ref.predict(s['pairs'][val])-s['responses'][val])**2)) for ref in refs]
                np.testing.assert_array_equal(losses, r['reference_validation_mse'])
                assert r['chosen_reference'] == int(np.argmin(losses))
                base_val = refs[r['chosen_reference']].predict(s['pairs'][val])
                completion = ScalarCompletion.load(dest/'aligned.npz')
                native = NativeNAIAD.load(protocol['source'], dest/'native')
                val_predictions = dict(simple=base_val,
                    rank8=base_val+completion.predict(s['pairs'][val]), native=native.predict(s['pairs'][val]))
                family_losses = {name: float(np.mean((pred-s['responses'][val])**2)) for name, pred in val_predictions.items()}
                for name, value in family_losses.items():
                    close(value, r['family_validation_mse'][name], tolerance=0)
                chosen = min(('simple', 'rank8', 'native'), key=family_losses.__getitem__)
                assert r['chosen_family'] == chosen
                family_counts[str(budget)][chosen] += 1
                rows = np.r_[train, val, supported_test]
                assert hashlib.sha256(rows.tobytes()).hexdigest() == r['native_rows_sha256']
                native_train = np.random.default_rng(20261022).permutation(np.arange(len(train))).tolist()
                assert native.fit_rows['train'] == native_train
                assert native.fit_rows['val'] == list(range(len(train), len(train)+len(val)))
                assert native.config.epochs == 100 and native.config.seed == 20261022
                assert native.best_epoch == int(np.argmin(native.training_metrics['val_loss']))+1
                assert len(native.training_metrics['val_loss']) == 100
                test_pair = s['pairs'][supported_test]
                test_base = refs[r['chosen_reference']].predict(test_pair)
                test_predictions = dict(indices=supported_test, simple=test_base,
                    native=native.predict(test_pair))
                for model_record in r['completion']:
                    loaded = ScalarCompletion.load(dest/(model_record['name']+'.npz'))
                    assert loaded.steps == 500 and loaded.rank == 8 and loaded.penalty == .01 and loaded.seed == 20261022
                    assert loaded.best_step == min(model_record['history'], key=lambda h: h['validation_mse'])['step']
                    label = 'rank8' if model_record['name'] == 'aligned' else model_record['name']
                    test_predictions[label] = test_base+loaded.predict(test_pair)
                test_predictions['candidate'] = test_predictions[chosen]
                for label, ref in zip(r['reference_labels'], refs):
                    test_predictions['ref_'+label.replace('/', '_')] = ref.predict(test_pair)
                checkpoint_predictions[(budget, task)].append(test_predictions)
                train_union.update(train.tolist())
                validation_union.update(val.tolist())
                fold_summaries.append(dict(budget=budget, task=task, fold=fold, selected_family=chosen,
                    selected_reference=r['reference_labels'][r['chosen_reference']], validation_mse=family_losses,
                    train_labels=len(train), validation_labels=len(val), supported_test=len(supported_test), full_test=len(test),
                    native_embedding_dimension=native.model_args['d_embed'], native_best_epoch=native.best_epoch,
                    selection_sha256=sha256(path), checkpoint_hashes=r['files']))
            assert not train_union & validation_union
            union_costs[str(budget)][task] = dict(train=len(train_union), validation=len(validation_union),
                total=len(train_union | validation_union), per_fit_nominal=budget,
                five_fits_nominal=5*budget, singles_genes=int(np.isfinite(s['singles']).sum()),
                information='Singles, controls and repeated construct/infection measurements cost extra; not a physical assay count')
    task_summaries = {}
    for budget in (5000, 20000):
        task_summaries[str(budget)] = {}
        for task in TASKS:
            check()
            s = screens[task]
            name = f'{budget}__{task}_predictions.npz'
            assert sha256(source/name) == results['prediction_hashes'][name]
            arrays = load_npz(source/name)
            from_checkpoints = checkpoint_predictions[(budget, task)]
            assert set(arrays) == set(from_checkpoints[0])
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, np.concatenate([fold[key] for fold in from_checkpoints]))
            indices = arrays['indices']
            assert len(np.unique(indices)) == len(indices)
            expected_indices = []
            full_count = 0
            controls_sufficient = True
            full_diagnostics = {}
            full_diagnostic_indices = []
            for fold in range(5):
                split = assignments[f'{budget}__{task}__{fold}']
                support = np.bincount(s['pairs'][split == 0].ravel(), minlength=len(s['genes'])) > 0
                test = np.flatnonzero(split == 2)
                full_count += len(test)
                expected_indices.extend(test[support[s['pairs'][test]].all(1)].tolist())
                rr = next(r for r in selection['records'] if (r['budget'], r['task'], r['fold']) == (budget, task, fold))
                record_path = source/rr['path']
                fold_record = read_json(record_path)
                controls_sufficient &= (len(fold_record['controls']) == 3 and
                    all(c['sufficient'] and c['permutable_fraction'] >= .8 for c in fold_record['controls']))
                full_diagnostic_indices.extend(test.tolist())
                for label, file in zip(fold_record['reference_labels'], fold_record['reference_files']):
                    ref = ScalarReference.load(record_path.parent/file)
                    if ref.kind not in ('zero', 'additive', 'minimum', 'calibration'):
                        continue
                    pair = s['pairs'][test]
                    if ref.kind == 'zero':
                        prediction = np.zeros(len(pair))
                    elif ref.kind == 'minimum':
                        prediction = ref.singles[pair].min(1)
                    else:
                        prediction = ref.singles[pair].sum(1)
                        if ref.kind == 'calibration':
                            a, b = ref.singles[pair].T
                            x = np.column_stack((a+b, abs(a-b), a*b))
                            prediction += ref.offset+(x-ref.center)/ref.scale@ref.coefficient
                    full_diagnostics.setdefault(label, []).append(prediction)
            np.testing.assert_array_equal(indices, expected_indices)
            recorded = results['tasks'][str(budget)][task]
            y = s['responses'][indices]
            candidate = check_metrics(y, arrays['candidate'], recorded['candidate'])
            simple = check_metrics(y, arrays['simple'], recorded['selected_simple'])
            for key in ('rank8', 'native'):
                check_metrics(y, arrays[key], recorded['fixed_families'][key])
            for key, values in recorded['all_simple'].items():
                check_metrics(y, arrays['ref_'+key], values)
            for key, values in recorded['shuffled_controls'].items():
                check_metrics(y, arrays[key], values)
            fi = np.asarray(full_diagnostic_indices)
            for key, values in recorded['full_test_nonlookup_diagnostics'].items():
                check_metrics(s['responses'][fi], np.concatenate(full_diagnostics[key]), values)
            intervals, influence, draws = uncertainty((arrays['candidate']-y)**2-(arrays['simple']-y)**2, s['pairs'][indices])
            close(intervals, recorded['paired_gene_difference']['gene_conditional_95'])
            assert draws == recorded['paired_gene_difference']['nonempty_draws']
            for gene, value in influence.items():
                close(value, recorded['paired_gene_difference']['leave_one_gene_out'][gene])
            for rep in range(2):
                target = s['replicates'][indices, rep]
                check_metrics(target, arrays['candidate'], recorded['infections'][rep]['candidate'])
                check_metrics(target, arrays['simple'], recorded['infections'][rep]['baseline'])
            gain = 1-candidate['mse']/simple['mse']
            close(gain, recorded['relative_gain'])
            coverage = len(indices)/full_count
            close(coverage, recorded['coverage'])
            sufficient = len(indices) >= 50 and len(np.unique(s['pairs'][indices])) >= 10 and controls_sufficient
            assert sufficient == recorded['sufficient']
            conditions = dict(coverage=coverage >= .9, gain=gain >= .05,
                negative_upper=bool(intervals[1] < 0),
                both_infections=all(r['candidate']['mse'] <= r['baseline']['mse'] for r in recorded['infections']),
                all_references=all(candidate['mse'] <= r['mse'] for r in recorded['all_simple'].values()),
                all_controls=all(candidate['mse'] < r['mse'] for r in recorded['shuffled_controls'].values()),
                sufficient=sufficient)
            assert all(conditions.values()) == recorded['passes_budget_gate']
            task_summaries[str(budget)][task] = dict(conditions=conditions, gain=gain,
                candidate_mse=candidate['mse'], simple_mse=simple['mse'], interval=intervals.tolist(),
                coverage=coverage, passes_budget_gate=recorded['passes_budget_gate'])
    nominee = next((budget for budget in (5000, 20000) if all(
        r['passes_budget_gate'] for r in task_summaries[str(budget)].values())), None)
    assert nominee == results['selected_budget']
    resources = replay['resources']
    assert max(resources['wall_seconds'], resources['cpu_seconds']) <= 7200
    assert resources['peak_rss_bytes'] <= 2*1024**3
    check()
    write_json(destination, dict(status='passed', protocol_sha256=sha256(source/'protocol.json'),
        results_sha256=sha256(source/'results.json'), selection_sha256=sha256(source/'selection.json'),
        replay_sha256=sha256(source/'replay.json'), audit_code_sha256=sha256(Path(__file__)),
        validation_choices_verified=60, task_assessments_verified=12,
        family_counts={key: dict(value) for key, value in family_counts.items()},
        union_label_costs=union_costs, tasks=task_summaries, selected_budget=nominee,
        fold_summaries=fold_summaries, final_resources=resources,
        audit_resources=dict(wall_seconds=time.monotonic()-wall, cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=peak[0], wall_cpu_cap_seconds=600, memory_cap_bytes=2*1024**3,
            reserve_bytes=1024**3), direct_test_checkpoint_replay=True,
        limitation='Read-only numerical and custody audit on exposed sources, not independent biological evidence'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args.run, args.output)
