"""Failure-oriented model-series access, restoration and evaluator tests."""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from opal.series import (MODEL_REGISTRY, OpalBase, Unavailable, acquisition_order,
                         canonicalize, evaluate_predictions, freeze_predictions, subset)
from opal.replicated.models import balanced_weights
from opal.replicated.retrospective import pair_folds


def data():
    rng = np.random.default_rng(17)
    profiles = dict(zip('ABCDEFGH', rng.normal(size=(8, 5))))
    pairs = list(itertools.combinations(profiles, 2))
    singles = np.asarray([[profiles[a], profiles[b]] for a, b in pairs] * 2)
    bank = {'singles': singles, 'pair_ids': np.asarray([json.dumps(p, separators=(',', ':')) for p in pairs] * 2),
            'culture_ids': np.repeat(['r1', 'r2'], len(pairs)),
            'responses': singles.sum(1) + 0.2 + .1 * singles[:, 0] * singles[:, 1]}
    config = json.loads(Path('configs/experiments/opal_model_series_v1.json').read_text())
    config.update(epoch_cap=3, gate_epoch_cap=3, regularization=[.01], gate_regularization=[.01])
    menu = [{'id': 'add', 'kind': 'additive', 'role': 'reference', 'alpha': 1},
            {'id': 'component', 'kind': 'component_additive', 'role': 'reference', 'alpha': .1},
            {'id': 'product', 'kind': 'global_plus_pair', 'role': 'candidate', 'alpha': 1}]
    contract = {'genes': list('abcde'), 'units': 'synthetic_expression_change', 'source': 'test',
                'assay': 'test', 'reference_provenance': 'original synthetic singles', 'split_lineage': 'test-17'}
    torch.set_num_threads(1)
    return bank, config, menu, contract


@pytest.mark.parametrize('name', list(MODEL_REGISTRY))
def test_actual_fit_reload_symmetry_and_support(tmp_path, name):
    bank, config, menu, contract = data()
    model = MODEL_REGISTRY[name](config, menu).fit(bank, contract)
    query = {k: v[:4].copy() for k, v in bank.items() if k != 'responses'}
    first = model.predict(query, contract)
    reversed_query = copy.deepcopy(query)
    reversed_query['singles'] = reversed_query['singles'][:, ::-1]
    reversed_query['pair_ids'] = np.asarray([json.dumps(json.loads(p)[::-1]) for p in query['pair_ids']])
    np.testing.assert_allclose(first['predictions'], model.predict(reversed_query, contract)['predictions'], atol=1e-12)
    model.save(tmp_path / name)
    restored = OpalBase.load(tmp_path / name)
    np.testing.assert_allclose(first['predictions'], restored.predict(query, contract)['predictions'], atol=1e-12)
    unsupported = copy.deepcopy(query)
    unsupported['pair_ids'][0] = '["A","Z"]'
    prediction = restored.predict(unsupported, contract)
    assert prediction['gate'][0] == 0
    assert prediction['status'][0].startswith('base_fallback')
    np.testing.assert_allclose(prediction['predictions'][0], restored._base(canonicalize(unsupported))[0])
    if name == 'Opal-Guard':
        for record in model.crossfit_records:
            assert not set(record['train_pair_ids']) & set(record['validation_pair_ids'])
            assert not set(record['expert_access']['fitting_and_selection_pair_ids']) & set(record['validation_pair_ids'])
        assert model.gate_selection['expert_predictions'].startswith('nested grouped')
    with pytest.raises(ValueError, match='integrity'):
        (tmp_path / name / 'parameters.npz').write_bytes(b'changed')
        OpalBase.load(tmp_path / name)


@pytest.mark.parametrize('name', list(MODEL_REGISTRY))
def test_held_outcome_mutation_does_not_change_fit(tmp_path, name):
    bank, config, menu, contract = data()
    held = set(bank['pair_ids'][:3])
    train = ~np.isin(bank['pair_ids'], list(held))
    contract.update(held_pair_ids=sorted(held), acquired_pair_ids=sorted(set(bank['pair_ids'][train])))
    model = MODEL_REGISTRY[name](config, menu).fit(subset(bank, train), contract)
    changed = copy.deepcopy(bank)
    changed['responses'][~train] += 100000
    other = MODEL_REGISTRY[name](config, menu).fit(subset(changed, train), contract)
    query = {k: v[~train] for k, v in bank.items() if k != 'responses'}
    assert model.selection == other.selection
    assert model.fit_hash == other.fit_hash
    np.testing.assert_array_equal(model.input_scale, other.input_scale)
    np.testing.assert_array_equal(model.predict(query, contract)['predictions'], other.predict(query, contract)['predictions'])
    with pytest.raises(ValueError, match='held pair'):
        MODEL_REGISTRY[name](config, menu).fit(changed, contract)


def test_alias_grouping_and_unacquired_validation_rejection():
    bank, config, menu, contract = data()
    query = {k: v[:1].copy() for k, v in bank.items() if k != 'responses'}
    query['pair_ids'] = np.asarray(['["guide_B","guide_A"]'])
    query['singles'] = query['singles'][:, ::-1]
    normalized = canonicalize(query, {'guide_B': 'B', 'guide_A': 'A'})
    assert normalized['pair_ids'][0] == '["A","B"]'
    np.testing.assert_array_equal(normalized['singles'], bank['singles'][:1])
    contract['acquired_pair_ids'] = sorted(set(bank['pair_ids']))[1:]
    with pytest.raises(ValueError, match='Unacquired'):
        OpalBase(config, menu).fit(bank, contract)
    bank['partitions'] = np.tile(np.asarray(['train'] * 14 + ['validation'] * 14), 2)
    bank['pair_ids'][14] = bank['pair_ids'][0]
    contract.pop('acquired_pair_ids')
    with pytest.raises(ValueError, match='leakage'):
        OpalBase(config, menu).fit(bank, contract)


def test_global_alias_occurrences_cannot_bypass_holdout():
    bank, config, menu, contract = data()
    bank['pair_ids'] = bank['pair_ids'].astype('U100')
    bank['pair_ids'][0] = '["guide_B","guide_A"]'
    bank['singles'][0] = bank['singles'][0, ::-1]
    contract.update(aliases={'guide_A': 'A', 'guide_B': 'B'}, held_pair_ids=['["A","B"]'])
    with pytest.raises(ValueError, match='held pair'):
        OpalBase(config, menu).fit(bank, contract)
    contract['held_pair_ids'] = ['["guide_B","guide_A"]']
    with pytest.raises(ValueError, match='canonical resolved'):
        OpalBase(config, menu).fit(bank, contract)


def test_missing_reference_scale_outcome_and_schema_refusals():
    bank, config, menu, contract = data()
    model = OpalBase(config, menu).fit(bank, contract)
    query = {k: v[:2] for k, v in bank.items() if k != 'responses'}
    for key in ('reference_provenance', 'units', 'genes'):
        bad = copy.deepcopy(contract)
        bad.pop(key)
        with pytest.raises(ValueError):
            model.predict(query, bad)
    for key, value in [('units', 'counts_inverted_from_log'), ('source', 'another'), ('genes', list('edcba'))]:
        bad = dict(contract, **{key: value})
        with pytest.raises(ValueError):
            model.predict(query, bad)
    with pytest.raises(ValueError, match='outcomes'):
        model.predict(bank, contract)
    with pytest.raises(Unavailable):
        OpalBase(config, menu).fit(subset(bank, np.arange(0)), contract)


@pytest.mark.parametrize('method', ['uniform', 'target-balanced'])
def test_deterministic_nested_prefixes_zero_budget_and_fixed_outer_population(method):
    bank, _, _, _ = data()
    folds = pair_folds(bank['pair_ids'])
    for outer in range(3):
        test = set(bank['pair_ids'][folds == outer])
        dev = bank['pair_ids'][folds != outer]
        order = acquisition_order(dev, method, 11)
        assert order == acquisition_order(dev[::-1], method, 11)
        previous = set()
        for budget in [0, .25, .5, 1]:
            acquired = set(order[:int(budget * len(order))])
            assert previous <= acquired
            assert not acquired & test
            assert set(bank['pair_ids'][folds == outer]) == test
            if budget == 0:
                assert not acquired
            previous = acquired


def test_equal_stratum_weights_ignore_technical_duplicates():
    pairs = np.asarray(['p', 'q', 'p', 'q'])
    strata = np.asarray(['a', 'a', 'b', 'b'])
    losses = np.asarray([1., 3., 10., 20.])
    expected = balanced_weights(pairs, strata) @ losses
    indices = np.asarray([0, 1, 2, 3, 0, 0, 1])
    assert balanced_weights(pairs[indices], strata[indices]) @ losses[indices] == pytest.approx(expected)


def test_freeze_refuses_in_sample_pairs_masquerading_as_held(tmp_path):
    bank, config, menu, contract = data()
    held = set(bank['pair_ids'][:3])
    train = ~np.isin(bank['pair_ids'], list(held))
    contract['held_pair_ids'] = sorted(held)
    model = OpalBase(config, menu).fit(subset(bank, train), contract)
    query = {k: v[train][:2] for k, v in bank.items() if k != 'responses'}
    with pytest.raises(ValueError, match='frozen globally held'):
        freeze_predictions(tmp_path / 'freeze', model, query, contract)


@pytest.mark.parametrize('changed', ['genes', 'units', 'inputs', 'predictions'])
def test_evaluator_rejects_tampering(tmp_path, changed):
    bank, config, menu, contract = data()
    model = OpalBase(config, menu).fit(bank, contract)
    query = {k: v[:2].copy() for k, v in bank.items() if k != 'responses'}
    freeze_predictions(tmp_path / 'freeze', model, query, contract)
    assert evaluate_predictions(tmp_path / 'freeze', query, contract, bank['responses'][:2])['mse'] >= 0
    bad = copy.deepcopy(contract)
    if changed == 'genes':
        bad['genes'].reverse()
    elif changed == 'units':
        bad['units'] = 'deposited_log_change'
    elif changed == 'inputs':
        query['singles'][0, 0, 0] += 1
    else:
        (tmp_path / 'freeze' / 'predictions.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Evaluator rejected'):
        evaluate_predictions(tmp_path / 'freeze', query, bad, bank['responses'][:2])
