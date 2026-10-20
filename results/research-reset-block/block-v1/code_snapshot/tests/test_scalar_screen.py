"""Synthetic prerequisites for scalar-bank and missing-block contracts."""
import copy
import itertools
import json

import numpy as np
import pytest

from opal import scalar_screen as scalar
from opal.artifacts import sha256


def fixture(n=30):
    genes = np.array([f'G{i:03}' for i in range(n)])
    pairs = np.array(list(itertools.combinations(range(n), 2)), dtype=np.int64)
    singles = np.linspace(-1, 1, n)
    reps = np.column_stack((singles[pairs].sum(1), singles[pairs].sum(1)+.2))
    return dict(genes=genes, pairs=pairs, singles=singles, replicates=reps,
        responses=reps.mean(1), counts=np.full((len(pairs), 2), 8), units='gamma',
        source='synthetic', accession='GSE-test', context='fixture', exclusions={})


def test_schema_refuses_alias_orientation_missingness_counts_and_aggregation():
    good = fixture()
    scalar.validate_screen(good)
    assert scalar.canonical_pair(' B ', 'A') == ('A', 'B')
    assert scalar.canonical_pair('A', 'B') == scalar.canonical_pair('B', 'A')
    for pair in [('A', 'A'), (' ', 'B'), ('A|B', 'C')]:
        with pytest.raises(ValueError):
            scalar.canonical_pair(*pair)
    changes = [
        ('genes', lambda a: a.__setitem__(1, a[0])),
        ('pairs', lambda a: a.__setitem__(0, a[0, ::-1])),
        ('pairs', lambda a: a.__setitem__(1, a[0])),
        ('singles', lambda a: a.__setitem__(0, np.nan)),
        ('replicates', lambda a: a.__setitem__((0, 0), np.nan)),
        ('counts', lambda a: a.__setitem__((0, 1), 1)),
        ('responses', lambda a: a.__setitem__(0, 99)),
    ]
    for key, mutate in changes:
        bad = copy.deepcopy(good)
        mutate(bad[key])
        with pytest.raises(ValueError):
            scalar.validate_screen(bad)
    bad = dict(good, units='LFC')
    with pytest.raises(ValueError, match='units'):
        scalar.validate_screen(bad)
    bad = dict(good, counts=np.full(good['counts'].shape, 2.5))
    with pytest.raises(ValueError, match='integer'):
        scalar.validate_screen(bad)


def test_missing_block_hash_contract_and_all_context_pair_disjointness():
    d = fixture(6)
    d['genes'] = np.array(list('ABCDEF'))
    d['pairs'] = np.array([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [4, 5]])
    splits = scalar.block_assignments(d)
    expected = np.array([0, 1, 0, 0, 0, 1], dtype=np.int8)
    for fold in range(5):
        correct = expected.copy()
        if fold == 2:
            correct[3] = 2
        np.testing.assert_array_equal(splits[fold], correct)
        assert splits[fold].dtype == np.int8
    assert np.sum(np.array(splits) == 2) == 1
    shifted = copy.deepcopy(d)
    shifted['context'] = 'another context'
    shifted['responses'] = np.full(6, 1e40)
    shifted['replicates'] = np.full((6, 2), -1e30)
    for a, b in zip(splits, scalar.block_assignments(shifted), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(scalar.pair_assignments(d), scalar.pair_assignments(shifted))


def test_split_support_and_insufficient_panels_are_reported_without_exclusion():
    d = fixture(6)
    split = np.zeros(len(d['pairs']), dtype=np.int8)
    split[(d['pairs'] == 0).any(1)] = 2
    report = scalar.split_report(d, split)
    assert report['unsupported_query_genes'] == ['G000']
    assert not report['eligible']
    assert report['test_pairs'] == 5
    assert scalar.split_report(d, np.zeros(len(split), dtype=np.int8))['test_pairs'] == 0
    with pytest.raises(ValueError):
        scalar.split_report(d, np.full(len(split), 3))


def test_strata_and_permutations_never_access_held_pair_outcomes():
    d = fixture()
    split = scalar.block_assignments(d)[0]
    stats = scalar.training_statistics(d, split)
    assert len(stats['train_indices']) == (split == 0).sum()
    assert stats['strata'].shape[1] == 5
    changed = copy.deepcopy(d)
    changed['responses'][:] = np.nan
    changed['replicates'][:] = np.nan
    changed['counts'][split != 0] = 100000
    a = scalar.residual_permutations(d, split)
    b = scalar.residual_permutations(changed, split)
    train = np.flatnonzero(split == 0)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x['permutation'], y['permutation'])
        np.testing.assert_array_equal(np.sort(x['permutation']), train)
        np.testing.assert_array_equal(x['permutation'], train[x['local_permutation']])
        np.testing.assert_array_equal(x['strata'], x['strata'][x['local_permutation']])
        assert 0 <= x['moved_fraction'] <= x['permutable_fraction'] <= 1
        # Joint-context arrays retain cross-context covariance under one mapping.
        values = np.vstack((np.arange(len(train)), np.arange(len(train))+1000))
        shuffled = values[:, x['local_permutation']]
        np.testing.assert_array_equal(shuffled[1]-shuffled[0], 1000)
    assert [x['seed'] for x in a] == list(scalar.CONTROL_SEEDS)
    assert not np.array_equal(a[0]['permutation'], a[1]['permutation'])


def test_active_training_gene_cutpoints_and_joint_single_summary():
    d = fixture(8)
    split = np.full(len(d['pairs']), 2, dtype=np.int8)
    split[np.max(d['pairs'], axis=1) < 6] = 0
    a = scalar.training_statistics(d, split)
    changed = copy.deepcopy(d)
    changed['singles'][6:] = 1e100
    changed['counts'][split != 0] = 2
    b = scalar.training_statistics(changed, split)
    np.testing.assert_array_equal(a['single_cutpoints'], b['single_cutpoints'])
    np.testing.assert_array_equal(a['strata'], b['strata'])
    summary = np.arange(8, dtype=float)
    a = scalar.residual_permutations(d, split, singles_summary=summary)
    b = scalar.residual_permutations(changed, split, singles_summary=summary)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x['permutation'], y['permutation'])
    # Singleton strata remain fixed and make the control insufficient.
    split[:] = 2
    split[0] = 0
    control = scalar.residual_permutations(d, split)[0]
    assert control['permutable_fraction'] == control['moved_fraction'] == 0
    assert not control['sufficient']


def test_joint_context_controls_share_count_and_single_strata():
    a, b = fixture(), fixture()
    b['singles'] *= 4
    b['counts'][:] = 2
    shared_singles = (a['singles']+b['singles'])/2
    shared_counts = np.minimum(a['counts'], b['counts'])
    for screen in (a, b):
        screen['control_singles_summary'] = shared_singles
        screen['control_counts'] = shared_counts
    split = scalar.block_assignments(a)[0]
    for x, y in zip(scalar.residual_permutations(a, split), scalar.residual_permutations(b, split), strict=True):
        np.testing.assert_array_equal(x['strata'], y['strata'])
        np.testing.assert_array_equal(x['permutation'], y['permutation'])


def test_metric_hand_calculation_missingness_and_paired_gene_multiplicity():
    m = scalar.metrics(np.array([1., 3.]), np.array([2., 1.]))
    assert m['mse'] == 2.5 and m['mae'] == 1.5 and m['bias'] == -.5
    assert m['target_variance'] == 1 and m['pearson'] == pytest.approx(-1.)
    assert scalar.metrics(np.zeros(2), np.zeros(2))['pearson'] is None
    for y, p in [(np.zeros(2), np.zeros(3)), (np.array([np.nan]), np.zeros(1)),
                 (np.zeros((1, 2)), np.zeros((1, 2))), (np.zeros(0), np.zeros(0))]:
        with pytest.raises(ValueError):
            scalar.metrics(y, p)
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    a = scalar.paired_gene_uncertainty(np.full(3, -.2), pairs, draws=200)
    np.testing.assert_allclose(a['interval'], [-.2, -.2], atol=1e-15)
    assert a['discarded_empty_draws'] > 0
    assert a['discarded_empty_draws'] + a['nonempty_draws'] == 200
    assert a['leave_one_gene_out'] == {'0': -.2, '1': -.2, '2': -.2}
    delta = np.array([1., 2., -3.])
    a = scalar.paired_gene_uncertainty(delta, pairs, n_genes=3, draws=200)
    assert a == scalar.paired_gene_uncertainty(delta, pairs, n_genes=30, draws=200)
    assert a['leave_one_gene_out'] == {'0': -3., '1': 2., '2': 1.}
    with pytest.raises(ValueError):
        scalar.paired_gene_uncertainty(delta, pairs, n_genes=2)


def test_loader_requires_historical_hashes_before_deserializing(tmp_path):
    (tmp_path/'profiles.npz').write_bytes(b'not an archive')
    with pytest.raises(ValueError, match='Historical bank hash'):
        scalar.load_screen_roster(tmp_path, tmp_path)


def test_external_full_local_population_retains_original_unsupported_genes(tmp_path, monkeypatch):
    # Loader integration is synthetic: patch only expected historical identities.
    root, ext = tmp_path/'original', tmp_path/'external'
    root.mkdir()
    ext.mkdir()
    d = fixture(467)
    pairs = d['pairs'][:81466]
    reps = np.zeros((2, 2, 81466))
    np.savez(root/'profiles.npz', genes=d['genes'], pairs=pairs,
        singles=np.zeros((2, 2, 467)), single_counts=np.full((2, 2, 467), 20),
        replicates=reps, responses=reps.mean(1), counts=np.full(reps.shape, 4),
        contexts=np.array(['K562', 'Jurkat']))
    digest = sha256(root/'profiles.npz')
    monkeypatch.setattr(scalar, 'ORIGINAL_HASH', digest)
    (root/'preparation.json').write_text(json.dumps(dict(source_sha256=scalar.ORIGINAL_SOURCE_HASH, profiles_sha256=digest)))
    metadata = {}
    for task in scalar.EXTERNAL_HASHES:
        genes = np.array(['UNSEEN_A', 'UNSEEN_B', 'UNSEEN_C', 'UNSEEN_D'])
        np.savez(ext/(task+'.npz'), genes=genes,
            pair_names=np.array([['UNSEEN_A', 'UNSEEN_B'], ['UNSEEN_A', 'UNSEEN_C'], ['UNSEEN_B', 'UNSEEN_D']]),
            gamma=np.zeros((3, 2)), counts=np.full((3, 2), 2),
            single_genes=genes[:3], singles=np.zeros((3, 2)), single_counts=np.full((3, 2), 2))
        digest = sha256(ext/(task+'.npz'))
        monkeypatch.setitem(scalar.EXTERNAL_HASHES, task, digest)
        metadata[task] = dict(output_sha256=digest, source_sha256='test', source_path='unavailable.gz')
    (ext/'preparation.json').write_text(json.dumps(dict(cohorts=metadata)))
    monkeypatch.setattr(scalar, 'EXTERNAL_PREPARATION_HASH', sha256(ext/'preparation.json'))
    roster = scalar.load_screen_roster(root, ext)
    assert tuple(roster) == scalar.TASKS
    for task in scalar.EXTERNAL_HASHES:
        screen = roster[task]
        assert len(screen['pairs']) == 2
        assert screen['exclusions']['missing_or_low_count_local_single_pairs'] == 1
        assert not screen['raw_source_hash_verified']
        assert screen['genes'][screen['pairs'][0]].tolist() == ['UNSEEN_A', 'UNSEEN_B']
        scalar.validate_screen(screen)
