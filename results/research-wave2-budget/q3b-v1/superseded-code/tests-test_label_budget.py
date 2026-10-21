"""Outcome-independent prerequisites for Q3b budgets, support and custody."""
import copy
import hashlib
import itertools

import numpy as np
import pytest

from opal.label_budget import (BUDGETS, budget_split, supported_queries, masked_native_inputs,
                               choose_family, scientific_gate, nominee)
from opal.block_completion import ScalarReference, ScalarCompletion, fit_reference_grid
from opal.artifacts import sha256
from tools import study_label_budget as study


def fixture(n=30):
    genes = np.array([f'G{i:03}' for i in range(n)])
    pairs = np.array(list(itertools.combinations(range(n), 2)), dtype=np.int64)
    singles = np.linspace(-.8, .2, n)
    y = singles[pairs].sum(1)+.03*np.sin(pairs.sum(1))
    return dict(genes=genes, pairs=pairs, singles=singles, responses=y,
        replicates=np.column_stack((y-.01, y+.01)), counts=np.full((len(pairs), 2), 8), accession='SYNTH')


def test_exact_full_sha_prefixes_nested_and_test_unchanged():
    s = fixture()
    original = np.arange(len(s['pairs'])) % 3
    small = budget_split(s, original, 20, 10)
    large = budget_split(s, original, 80, 30)
    for part, count in ((0, 20), (1, 10)):
        rows = np.flatnonzero(original == part)
        expected = sorted(rows, key=lambda i: hashlib.sha256(
            f'opal-budget-v1|SYNTH|{s["genes"][s["pairs"][i,0]]}|{s["genes"][s["pairs"][i,1]]}'.encode()).hexdigest())[:count]
        assert set(np.flatnonzero(small == part)) == set(expected)
        assert set(np.flatnonzero(small == part)) <= set(np.flatnonzero(large == part))
    np.testing.assert_array_equal(small == 2, original == 2)
    np.testing.assert_array_equal(large == 2, original == 2)
    changed = dict(s, responses=np.full(len(original), np.nan), counts=np.zeros((len(original), 2)))
    np.testing.assert_array_equal(budget_split(changed, original, 20, 10), small)


def test_unavailable_partitions_and_invalid_budgets_refused():
    s = fixture(6)
    with pytest.raises(ValueError):
        budget_split(s, np.zeros(len(s['pairs']), dtype=int), 4, 3)
    with pytest.raises(ValueError):
        budget_split(s, np.full(len(s['pairs']), 3), 4, 3)
    with pytest.raises(ValueError):
        budget_split(s, np.arange(len(s['pairs'])) % 3, 0, 3)


def test_support_never_adds_edges_or_drops_original_test_accounting():
    s = fixture(6)
    split = np.full(len(s['pairs']), -1, dtype=np.int8)
    split[[0, 1, 5]] = 0  # only genes 0,1,2
    split[[2, 6]] = 1
    split[[7, 14]] = 2
    q = supported_queries(s['pairs'], split, 6)
    np.testing.assert_array_equal(q['train'], [0, 1, 5])
    np.testing.assert_array_equal(q['full_test'], [7, 14])
    assert not len(q['validation']) and not len(q['test'])
    assert q['unused_validation'] == 2
    assert (split == 0).sum() == 3
    with pytest.raises(ValueError):
        masked_native_inputs(s, split)


def test_native_targets_exclude_unused_and_test_labels():
    s = fixture()
    split = np.full(len(s['pairs']), -1, dtype=np.int8)
    split[:100], split[100:130], split[130:180] = 0, 1, 2
    rows, y, member = masked_native_inputs(s, split)
    assert np.isnan(y[member == 2]).all()
    assert not set(rows) & set(np.flatnonzero(split == -1))
    changed = copy.deepcopy(s)
    changed['responses'][split != 0] = 1e30
    changed['responses'][split == 1] = s['responses'][split == 1]
    rows2, y2, member2 = masked_native_inputs(changed, split)
    np.testing.assert_array_equal(rows, rows2)
    np.testing.assert_array_equal(member, member2)
    np.testing.assert_array_equal(y, y2)


def test_family_validation_ties_and_missing_values():
    y = np.array([1., 2., 3.])
    predictions = {name: y.copy() for name in ('native', 'rank8', 'simple')}
    assert choose_family(y, predictions)[0] == 'simple'
    predictions['simple'] += 1
    assert choose_family(y, predictions)[0] == 'rank8'
    predictions['rank8'] += 2
    assert choose_family(y, predictions)[0] == 'native'
    with pytest.raises(ValueError):
        choose_family(y, dict(simple=y))
    predictions['native'] = np.full(3, np.nan)
    with pytest.raises(ValueError):
        choose_family(y, predictions)


def test_nonlookup_baselines_and_serialization_are_exact(tmp_path):
    s = fixture()
    train = np.arange(200)
    models = fit_reference_grid(s['singles'], s['pairs'][train], s['responses'][train])
    assert len(models) == 12
    query = s['pairs'][100:200]
    for i, model in enumerate(models):
        file = tmp_path/f'{i}.npz'
        model.save(file)
        loaded = ScalarReference.load(file)
        np.testing.assert_array_equal(model.predict(query), loaded.predict(query))
        if model.kind in ('zero', 'minimum', 'additive', 'calibration'):
            np.testing.assert_allclose(study.full_reference_prediction(model, query), model.predict(query), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(study.full_reference_prediction(models[0], query), np.zeros(len(query)))
    np.testing.assert_array_equal(study.full_reference_prediction(models[3], query), s['singles'][query].min(1))
    with pytest.raises(ValueError):
        study.full_reference_prediction(models[-1], query)


def test_full_calibration_predicts_unsupported_test_without_relaxing_model():
    s = fixture(8)
    train = np.flatnonzero((s['pairs'] < 6).all(1))
    model = ScalarReference('calibration', 10).fit(s['singles'], s['pairs'][train], s['responses'][train])
    query = s['pairs'][-1:]
    with pytest.raises(ValueError):
        model.predict(query)
    assert np.isfinite(study.full_reference_prediction(model, query)).all()
    assert not model.support[-1]


def test_completion_exact_replay_and_leakage_refusal(tmp_path):
    s = fixture()
    train, val = np.arange(200), np.arange(200, 240)
    m = ScalarCompletion(8, .01, 25, 20261022).fit(s['pairs'][train], s['responses'][train], 30,
        1., (s['pairs'][val], s['responses'][val]))
    path = tmp_path/'model.npz'
    m.save(path)
    np.testing.assert_array_equal(m.predict(s['pairs'][val]), ScalarCompletion.load(path).predict(s['pairs'][val]))
    with pytest.raises(ValueError, match='leakage'):
        ScalarCompletion(8, .01, 25).fit(s['pairs'][train], s['responses'][train], 30, 1.,
            (s['pairs'][train[:5]], s['responses'][train[:5]]))


def test_coverage_gate_all_tasks_and_smaller_budget_only():
    assert not scientific_gate(.89999, {'passes_gate': True})
    assert scientific_gate(.90, {'passes_gate': True})
    assert not scientific_gate(1., {'passes_gate': False})
    tasks = ('a', 'b')
    r = {str(total): {task: {'passes_budget_gate': True} for task in tasks} for total, _, _ in BUDGETS}
    assert nominee(r, tasks) == 5000
    r['5000']['b']['passes_budget_gate'] = False
    assert nominee(r, tasks) == 20000
    r['20000'].pop('a')
    assert nominee(r, tasks) is None


def test_prior_scientific_gate_retains_all_vetoes():
    s = fixture()
    indices = np.arange(len(s['pairs']))
    y = s['responses']
    baseline = y+.2
    controls = {f'shuffled-{seed}': y+.3 for seed in study.CONFIG['control_seeds']}
    refs = {'minimum': y+.2}
    assessment = study.block.evaluate_arrays(s, indices, y, baseline, controls, refs, True)
    assert assessment['passes_gate']
    assert study.block.evaluate_arrays(s, indices, y, baseline, controls, {'better': y.copy()}, True)['passes_gate']
    # A genuinely better fixed reference vetoes an otherwise improved candidate.
    assert not study.block.evaluate_arrays(s, indices, y+.01, baseline, controls, {'better': y.copy()}, True)['passes_gate']
    assert not study.block.evaluate_arrays(s, indices, y, baseline, controls, refs, False)['passes_gate']


def test_all_sixty_choices_and_checkpoint_tamper_guard(tmp_path):
    records = [dict(budget=b, task=t, fold=f) for b, _, _ in BUDGETS for t in study.TASKS for f in range(5)]
    study.require_all_choices(dict(records=records))
    with pytest.raises(ValueError):
        study.require_all_choices(dict(records=records[:-1]))
    with pytest.raises(ValueError):
        study.require_all_choices(dict(records=records+[records[0]]))
    file = tmp_path/'checkpoint'
    file.write_bytes(b'original')
    hashes = {str(file): sha256(file)}
    study.block.verify_files(hashes)
    file.write_bytes(b'tampered')
    with pytest.raises(ValueError):
        study.block.verify_files(hashes)


def test_prediction_replay_detects_alignment_and_values():
    values = dict(indices=np.array([1, 2]), candidate=np.array([.1, .2]))
    study.assert_arrays(values, copy.deepcopy(values))
    for bad in (dict(values, indices=np.array([2, 1])), dict(values, candidate=np.array([.1, .3])), dict(candidate=values['candidate'])):
        with pytest.raises(RuntimeError):
            study.assert_arrays(values, bad)


def test_resource_guard_charges_consumed_time_memory_disk_and_artifacts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    guard = study.block.Guard(tmp_path, study.CONFIG, consumed={'wall_seconds': 7201})
    with pytest.raises(RuntimeError, match='CPU/wall'):
        guard.check()
    monkeypatch.setattr(study.block.psutil, 'Process', lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=3*1024**3)))
    with pytest.raises(RuntimeError, match='memory'):
        study.block.Guard(tmp_path, study.CONFIG).check()
    monkeypatch.setattr(study.block.psutil, 'Process', lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=1024)))
    monkeypatch.setattr(study.block.shutil, 'disk_usage', lambda p: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match='reserve'):
        study.block.Guard(tmp_path, study.CONFIG).check()
    monkeypatch.setattr(study.block.shutil, 'disk_usage', lambda p: SimpleNamespace(free=2*1024**3))
    (tmp_path/'artifact').write_bytes(b'1234')
    with pytest.raises(RuntimeError, match='artifact'):
        study.block.Guard(tmp_path, dict(study.CONFIG, artifact_bytes=3)).check()


def test_prefit_forecast_failure_stops_before_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'prerequisites', lambda output: {})
    monkeypatch.setattr(study, 'read_json', lambda path: {'protocol_sha256': 'same', 'passes_forecast': False})
    monkeypatch.setattr(study, 'sha256', lambda path: 'same')
    monkeypatch.setattr(study, 'load_screen_roster', lambda *a: pytest.fail('Biological bank loaded after failed forecast'))
    with pytest.raises(ValueError, match='forecast failed'):
        study.fit(tmp_path)
