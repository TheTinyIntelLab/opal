"""Outer isolation and explicit retrospective inference limits."""
import itertools
import json

import numpy as np
import pytest

from opal.replicated.retrospective import nested_pair_benchmark, pair_folds, reference_gene_panel


def bank():
    pairs = [json.dumps(p, separators=(',', ':')) for p in itertools.combinations('ABCDE', 2)]
    rng = np.random.default_rng(31)
    profiles = dict(zip('ABCDE', rng.normal(size=(5, 4))))
    singles = np.asarray([[profiles[a], profiles[b]] for a, b in map(json.loads, pairs)] * 3)
    return {'pair_ids': np.tile(pairs, 3), 'culture_ids': np.repeat(['r1', 'r2', 'r3'], len(pairs)),
            'singles': singles, 'responses': singles.sum(1) + rng.normal(size=(30, 4)) * .1}


MENU = [{'id': 'add', 'kind': 'additive', 'role': 'reference', 'alpha': 1},
        {'id': 'offset', 'kind': 'shared_offset', 'role': 'reference', 'alpha': 1},
        {'id': 'product', 'kind': 'global_plus_pair', 'role': 'candidate', 'alpha': 1}]


def test_global_pair_folds_and_outer_response_isolation():
    original = bank()
    folds = pair_folds(original['pair_ids'])
    np.testing.assert_array_equal(folds[:10], folds[10:20])
    first, predictions = nested_pair_benchmark(original, MENU, shuffle_seeds=())
    modified = {k: v.copy() for k, v in original.items()}
    modified['responses'][folds == 0] += 100
    second, altered = nested_pair_benchmark(modified, MENU, shuffle_seeds=())
    for role in ('reference', 'candidate'):
        np.testing.assert_array_equal(predictions['aligned'][role][folds == 0],
                                      altered['aligned'][role][folds == 0])
    for a, b in zip(first['fold_records'], second['fold_records']):
        if a['outer_fold'] == 0:
            assert a['selection'] == b['selection']
    assert not first['primary_claim_pass']
    assert not first['new_confirmation_performed']
    assert not first['confidence_intervals_computed']


def test_reference_endpoint_ignores_degree_duplicates_and_breaks_ties():
    x = bank()['singles'][:10][None, ...]
    chosen = reference_gene_panel(x, ['a', 'b', 'c', 'd'], 2)
    repeated = np.concatenate([x, x[:, :1]], axis=1)
    np.testing.assert_array_equal(chosen, reference_gene_panel(repeated, ['a', 'b', 'c', 'd'], 2))
    tied = np.zeros((1, 2, 2, 3))
    np.testing.assert_array_equal(reference_gene_panel(tied, ['c', 'a', 'b'], 2), [1, 2])


def test_shuffles_reselect_and_technical_rows_are_rejected():
    b = bank()
    report, _ = nested_pair_benchmark(b, MENU, shuffle_seeds=(11,))
    assert len(report['fold_records']) == 18
    assert all(r['selection']['search_count'] == 3 for r in report['fold_records'])
    b = {k: np.concatenate([v, v[:1]]) for k, v in b.items()}
    with pytest.raises(ValueError, match='technical'):
        nested_pair_benchmark(b, MENU)


def test_saved_prediction_verifier_detects_loss_and_gate_tampering(tmp_path):
    import importlib.util
    from pathlib import Path
    from opal.replicated.contracts import sha, write_json
    spec = importlib.util.spec_from_file_location('retrospective_verifier',
             Path(__file__).parents[1] / 'tools/verify_retrospective_combi.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    b = bank()
    report, predictions = nested_pair_benchmark(b, MENU, shuffle_seeds=())
    frozen = tmp_path / 'code_snapshot' / 'example.py'
    frozen.parent.mkdir()
    frozen.write_text('version = 1\n')
    write_json(tmp_path / 'scope_before_fit.json', {'menu': MENU,
               'code_sha256': {'example.py': sha(frozen)}})
    np.savez_compressed(tmp_path / 'predictions.npz', pair_ids=b['pair_ids'],
                        run_ids=b['culture_ids'], responses=b['responses'],
                        **{f'{label}_{role}': value for label, values in predictions.items()
                           for role, value in values.items()})
    report.update({'scope_sha256': sha(tmp_path / 'scope_before_fit.json'),
                   'prediction_archive_sha256': sha(tmp_path / 'predictions.npz'),
                   'source_rows': 30, 'nominal_pair_count': 10, 'endpoint_genes': list('abcd')})
    write_json(tmp_path / 'report.json', report)
    assert module.verify(tmp_path)['checked_prediction_losses'] == 24
    frozen.write_text('version = 2\n')
    with pytest.raises(ValueError, match='Frozen code hash'):
        module.verify(tmp_path)
    frozen.write_text('version = 1\n')
    report['fold_records'][0]['losses']['candidate'] += 1
    write_json(tmp_path / 'report.json', report)
    with pytest.raises(ValueError, match='Fold loss'):
        module.verify(tmp_path)
    report['primary_claim_pass'] = True
    write_json(tmp_path / 'report.json', report)
    with pytest.raises(ValueError, match='scientific gates'):
        module.verify(tmp_path)
