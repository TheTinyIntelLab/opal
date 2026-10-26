"""Fixture checks for the growth adapter. Registered profile and biological fit stay with root."""
import numpy as np
import pytest

from opal.scalar_screen import block_assignments
import tools.frontier_growth as growth


def test_storage_amendment_and_fixed_query_budget():
    assert growth.synthetic_holdout_count(600) == 53910
    assert growth.synthetic_holdout_count(467) == 32644
    assert growth.RAW_COMPLETION_BYTES == 4 * 600 * 600 * 8 * 30 == 345_600_000
    assert growth.RAW_COMPLETION_BYTES > 256 * 1024**2
    assert growth.PROGRAM_CAP['artifact_bytes'] == 512 * 1024**2
    assert growth.PROFILE_CAP['artifact_bytes'] == 64 * 1024**2
    assert growth.PROGRAM_CAP['wall_seconds'] == 7200
    assert growth.PROGRAM_CAP['reserve_bytes'] == 1024**3 + 32 * 1024**2
    assert growth.PROFILE_QUERY_COUNT == 10000


def test_profile_queries_are_a_fixed_metadata_prefix():
    pairs = np.array([(a, b) for a in range(200) for b in range(a + 1, 200)], dtype=np.int64)
    data = dict(pairs=pairs, test=np.arange(len(pairs), dtype=np.int64))
    chosen = growth.profile_test_queries(data)
    ordered = data['test'][growth.pair_order(pairs, 'frontier-budget-test')]
    assert len(chosen) == 10000
    assert np.array_equal(chosen, ordered[:10000])
    assert ordered[10000] not in chosen
    with pytest.raises(ValueError, match='fixed at 10000'):
        growth.profile_test_queries(data, 53910)


def test_biological_test_bound_does_not_truncate_or_extend():
    growth.require_biological_test_bound([('task', 0, 10000), ('task', 1, 1)])
    with pytest.raises(growth.QualificationError, match='not truncated'):
        growth.require_biological_test_bound([('task', 0, 10000), ('task', 1, 10001)])
    assert growth.PROFILE_QUERY_COUNT == 10000


def test_unused_nan_gene_preserves_every_pair_name():
    genes = np.array(['aaa', 'unused', 'bbb', 'ccc'])
    pairs = np.array([[0, 2], [0, 3], [2, 3]], dtype=np.int64)
    singles = np.array([0.2, np.nan, -0.4, 0.5])
    collapsed = growth.collapse_eligible_genes(genes, pairs, singles)
    assert collapsed['dropped_names'] == ['unused']
    assert np.array_equal(collapsed['pair_names_before'], collapsed['pair_names_after'])
    assert list(collapsed['pair_names_after'][2]) == ['bbb', 'ccc']
    assert np.array_equal(collapsed['singles'], np.array([0.2, -0.4, 0.5]))
    assert len(collapsed['pairs']) == len(pairs)
    screen = dict(genes=genes, pairs=pairs, singles=singles, accession='synthetic-growth')
    screen2 = dict(genes=collapsed['genes'], pairs=collapsed['pairs'], accession=screen['accession'])
    for before, after in zip(block_assignments(screen), block_assignments(screen2), strict=True):
        assert np.array_equal(before, after)


def test_nonfinite_eligible_single_fails_without_dropping_the_pair():
    with pytest.raises(growth.QualificationError, match='without imputation or pair deletion'):
        growth.collapse_eligible_genes(np.array(['aaa', 'bbb']), np.array([[0, 1]]),
                                       np.array([np.nan, 1.0]))


def _folds(offset_policy=0.2, reference_offset=2.0, perfect_reference=False):
    pairs = np.array([(a, b) for a in range(15) for b in range(a + 1, 15)], dtype=np.int64)
    assert len(pairs) == 105
    reference_names = [f'B{i}--fixed' for i in range(16)]
    simple_name = reference_names[0]
    folds = []
    for fold in range(5):
        rows = pairs[fold * 21:(fold + 1) * 21]
        y = np.linspace(-1, 1, len(rows))
        aligned = {family: f'{family}--fold{fold}' for family in ('I0', 'I1', 'I2')}
        shuffles = {f'shuffle{i}': f'I0-shuffle{i}--fold{fold}' for i in range(3)}
        predictions = {simple_name: y + 1, 'policy--fold': y + offset_policy}
        predictions.update({aligned[family]: y + 0.4 + 0.1 * index for index, family in enumerate(('I0', 'I1', 'I2'))})
        predictions.update({shuffles[key]: y + 1.5 for key in shuffles})
        for index, name in enumerate(reference_names):
            predictions[name] = y + (0 if perfect_reference and index == 3 else reference_offset + index * 0.01)
        predictions[simple_name] = y + 1
        folds.append(dict(pairs=rows, support=np.ones(len(rows), dtype=bool), mean=y, infection0=y,
            infection1=y, predictions=predictions, selected_policy='policy--fold',
            selected_simple=simple_name, aligned_names=aligned, shuffled_names=shuffles,
            permutable={f'shuffle{i}': 1.0 for i in range(3)}))
    return folds, reference_names


def test_score_ledger_keeps_every_baseline_and_refuses_a_test_nominee():
    folds, names = _folds()
    report = growth.evaluate_task(folds, names)
    assert report['passes'] is True
    assert report['test_best_diagnostic']['nominee'] is False
    assert report['test_best_diagnostic']['selection'] == 'test'
    assert 'not a nominee' in report['test_best_diagnostic']['label']
    assert report['models']['selected_policy']['selection'] == 'validation'
    assert report['models']['selected_policy']['nominee'] is False
    for slot in ('selected_policy', 'selected_simple', 'I0', 'I1', 'I2', 'shuffle0', 'shuffle1', 'shuffle2'):
        block = report['models'][slot]
        for population in ('common_support', 'full_population'):
            for target in ('mean', 'infection0', 'infection1'):
                metrics = block[population][target]
                assert set(metrics) == {'mse', 'mae', 'bias', 'correlation', 'pairs'}
    assert len(report['fixed_simple_references']) == 16
    assert report['models']['I0']['names'] == [f'I0--fold{fold}' for fold in range(5)]
    assert report['policy_mse'] == pytest.approx(0.04)
    assert set(report['uncertainty']) >= {'interval', 'leave_one_gene_out', 'mean'}
    assert len(report['uncertainty']['leave_one_gene_out']) == 15
    poisoned, names = _folds(perfect_reference=True)
    failed = growth.evaluate_task(poisoned, names)
    assert failed['passes'] is False
    assert 'reference_veto' in failed['reasons']
    assert failed['test_best_diagnostic']['nominee'] is False
    assert failed['test_best_diagnostic']['slot'] == 'reference:' + names[3]
    assert failed['models']['selected_policy']['names'] == ['policy--fold'] * 5


def test_forecast_uses_the_512_mib_cap_without_declaring_a_pass():
    measurements = [dict(world=world, genes=genes, wall_seconds=10, cpu_seconds=9,
        artifact_bytes=1000, peak_rss_bytes=100, persisted_models=28, threads=1)
        for world in ('additive', 'lowrank') for genes in (467, 600)]
    qualified = growth.forecast_program(measurements, profile_bytes=100)
    assert qualified['qualified'] is True
    assert qualified['artifact_cap_bytes'] == 512 * 1024**2
    measurements[0]['wall_seconds'] = 200
    refused = growth.forecast_program(measurements, profile_bytes=100)
    assert refused['qualified'] is False
    with pytest.raises(RuntimeError, match='biological fitting is stopped'):
        growth.require_qualified_forecast(refused)


def test_freeze_does_not_load_the_roster(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError('roster loaded')
    monkeypatch.setattr('opal.scalar_screen.load_screen_roster', boom)
    growth.freeze(tmp_path / 'out')
    assert growth.read(tmp_path / 'out' / 'bank-expectations.json')['banks_opened'] is False
    assert 'load_screen_roster' not in growth.profile.__code__.co_names
    assert 'load_screen_roster' in growth.fit.__code__.co_names


def test_installed_replay_rejects_the_checkout(tmp_path):
    with pytest.raises(ValueError, match='outside the checkout'):
        growth.active_predictor(growth.ROOT)
    with pytest.raises(ValueError, match='checkout'):
        growth.active_predictor(tmp_path)
