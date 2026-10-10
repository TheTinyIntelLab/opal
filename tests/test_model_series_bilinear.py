"""Failure-oriented tests of the finite bilinear/nested-memory architecture."""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from opal.series import MODEL_REGISTRY, OpalBase, Unavailable, subset
from opal.series_bilinear import (FIT_CACHE, base_oof, cross_features, fit_branch,
                                fit_memory, fit_reference, partition, pca,
                                predict_branch, predict_reference, reference_menu,
                                retrieve, expert_oof, weights)


def fixture():
    rng = np.random.default_rng(381)
    profiles = rng.normal(size=(18, 12))
    pairs = list(itertools.combinations(range(len(profiles)), 2))
    x = np.asarray([profiles[list(p)] for p in pairs])
    ids = np.asarray([json.dumps([f't{a:02}', f't{b:02}'], separators=(',', ':')) for a, b in pairs])
    y = .6 * x.sum(1) + .15 * (x[:, 0, 0] * x[:, 1, 1] + x[:, 0, 1] * x[:, 1, 0])[:, None] * rng.normal(size=(1, 12))
    bank = {'singles': x, 'responses': y, 'pair_ids': ids, 'culture_ids': np.repeat('r', len(x))}
    settings = json.loads(Path('configs/experiments/opal_model_series_v2.json').read_text())
    menu = json.loads(Path('configs/experiments/replicated_fixed_state_v1.json').read_text())['menu']
    contract = {'genes': [f'g{i}' for i in range(12)], 'units': 'synthetic_expression_change', 'source': 'controlled',
                'assay': 'controlled', 'reference_provenance': 'generated', 'split_lineage': '381', 'context_contract': 'fixed'}
    return bank, settings, menu, contract


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        FIT_CACHE.clear()
        yield


def test_exact_menus_and_rank_refusal():
    bank, settings, menu, _ = fixture()
    assert len(reference_menu(menu)) == 24
    assert len(settings['input_ranks']) * len(settings['output_ranks']) * len(settings['penalties']) * len(settings['residual_routes']) == 24
    with pytest.raises(Unavailable, match='rank'):
        pca(np.ones_like(bank['singles']), 4)


@pytest.mark.parametrize('name', list(MODEL_REGISTRY))
def test_fit_safe_replay_symmetry_global_exclusions(tmp_path, name):
    bank, settings, menu, contract = fixture()
    held = bank['pair_ids'][::7]
    mask = ~np.isin(bank['pair_ids'], held)
    contract.update(held_pair_ids=held.tolist(), acquired_pair_ids=bank['pair_ids'][mask].tolist())
    model = MODEL_REGISTRY[name](settings, menu).fit(subset(bank, mask), contract)
    query = {k: v[~mask] for k, v in bank.items() if k != 'responses'}
    original = model.predict(query, contract)
    reverse = copy.deepcopy(query)
    reverse['singles'] = reverse['singles'][:, ::-1]
    reverse['pair_ids'] = np.asarray([json.dumps(json.loads(p)[::-1]) for p in query['pair_ids']])
    np.testing.assert_allclose(model.predict(reverse, contract)['predictions'], original['predictions'], atol=1e-12, rtol=0)
    model.save(tmp_path / name)
    restored = OpalBase.load(tmp_path / name)
    np.testing.assert_array_equal(original['predictions'], restored.predict(query, contract)['predictions'])
    mutated = copy.deepcopy(bank)
    mutated['responses'][~mask] += 100000
    other = MODEL_REGISTRY[name](settings, menu).fit(subset(mutated, mask), contract)
    assert other.fit_hash == model.fit_hash
    np.testing.assert_array_equal(other.predict(query, contract)['predictions'], original['predictions'])
    assert np.allclose(original['expert_weights'].sum(1), 1)
    if name == 'Opal-Guard' and 'router_validation_pair_ids' in model.gate_selection:
        gv = set(model.gate_selection['router_validation_pair_ids'])
        for attempt in model.gate_selection['attempts']:
            for fold in attempt.get('nested_records', []):
                assert not gv & set(fold['train_pair_ids'])
                assert not gv & set(fold['validation_pair_ids'])
        assert not original['expert_weights'][:, ~model.fitted['expert_mask']].any()


def test_router_training_outcome_mutation_invariance():
    bank, settings, menu, _ = fixture()
    train, validation = partition(bank, settings)
    before = expert_oof(train, settings, menu, 4)
    mutated = copy.deepcopy(bank)
    mutated['responses'][np.isin(bank['pair_ids'], validation['pair_ids'])] += 1e5
    after_train, _ = partition(mutated, settings)
    FIT_CACHE.clear()
    after = expert_oof(after_train, settings, menu, 4)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    assert before[3] == after[3]


def test_training_transforms_retrieval_scales_and_alias_occurrences():
    bank, settings, menu, _ = fixture()
    train, validation = partition(bank, settings)
    memory = fit_memory(train, settings, menu, 4)
    first, _, neighbors = retrieve(memory, validation)
    changed = copy.deepcopy(validation)
    changed['responses'] *= 1e5
    second, _, _ = retrieve(memory, changed)
    np.testing.assert_array_equal(first, second)
    repeated = {k: np.concatenate((v, v[:2])) for k, v in train.items()}
    repeated_memory = fit_memory(repeated, settings, menu, 4)
    _, _, repeated_neighbors = retrieve(repeated_memory, {k: v[:1] for k, v in train.items()})
    assert train['pair_ids'][0] not in {n['pair_id'] for n in repeated_neighbors[0]}
    assert all(len({n['pair_id'] for n in group}) == 4 for group in neighbors)
    assert memory['temperature'] > 0


def test_cross_factor_capability_on_held_pairs():
    bank, settings, menu, _ = fixture()
    rng = np.random.default_rng(44)
    # Generate cross-factor targets in a training-only encoder, then hold pairs.
    held = np.arange(len(bank['pair_ids'])) % 5 == 0
    train, validation = subset(bank, ~held), subset(bank, held)
    encoder = pca(train['singles'], 4)
    z = (bank['singles'] - encoder['mean']) @ encoder['basis']
    direction = rng.normal(size=(4, 12))
    bank['responses'] = 2 * cross_features(z)[:, [1, 2, 3, 5]] @ direction
    train, validation = subset(bank, ~held), subset(bank, held)
    config = {'kind': 'no_change', 'id': 'null'}
    reference = fit_reference(config, train)
    branch = fit_branch(train, {'input_rank': 4, 'output_rank': 4, 'alpha': .1, 'route': 'in-sample'}, config, reference, settings, menu)
    prediction = predict_branch(branch, config, reference, validation)
    assert np.mean((prediction - validation['responses']) ** 2) < .05 * np.mean(validation['responses'] ** 2)


def test_zero_unacquired_and_same_context_refusal():
    bank, settings, menu, contract = fixture()
    with pytest.raises(Unavailable, match='Zero'):
        OpalBase(settings, menu).fit(subset(bank, np.asarray([], dtype=int)), contract)
    contract['acquired_pair_ids'] = bank['pair_ids'][1:].tolist()
    with pytest.raises(ValueError, match='Unacquired'):
        OpalBase(settings, menu).fit(bank, contract)
    contract.pop('acquired_pair_ids')
    model = OpalBase(settings, menu).fit(bank, contract)
    q = {k: v[:2] for k, v in bank.items() if k != 'responses'}
    with pytest.raises(ValueError, match='context'):
        model.predict(q, dict(contract, transport_mode='same-context', context_contract='TNF-changed'))


def test_main_effect_capability_and_equal_stratum_weights():
    bank, settings, menu, contract = fixture()
    bank['responses'] = .7 * bank['singles'].sum(1) + .2
    held = np.arange(len(bank['pair_ids'])) % 5 == 0
    train, query = subset(bank, ~held), subset(bank, held)
    reference = {'kind': 'affine', 'id': 'affine-test', 'alpha': .00001}
    fitted = fit_reference(reference, train)
    prediction = predict_reference(reference, fitted, query)
    assert np.mean((prediction-query['responses'])**2) < 1e-6
    # Unequal numbers of pairs and repeated technical rows cannot give a
    # stratum more total weight or a canonical identity more stratum weight.
    small = subset(bank, np.arange(10))
    small['culture_ids'][:3] = 's'
    repeated = {k: np.concatenate((v, v[:1], v[:1])) for k,v in small.items()}
    w = weights(repeated)
    assert np.isclose(w[repeated['culture_ids']=='s'].sum(), .5)
    assert np.isclose(w[repeated['pair_ids']==small['pair_ids'][0]].sum(), .5/3)


def test_local_memory_capability_on_unseen_pairs():
    # A local smooth residual has no global bilinear generating equation.
    # The expert sees acquired residuals and training-only distance scales.
    rng = np.random.default_rng(198)
    profiles = rng.normal(size=(40,6))
    pairs = list(itertools.combinations(range(40),2))
    x = np.asarray([profiles[list(p)] for p in pairs])
    ids = np.asarray([json.dumps(list(map(str,p))) for p in pairs])
    y = np.exp(-.2 * (x[:,0]**2+x[:,1]**2).sum(1))[:,None]
    held = np.arange(len(x)) % 7 == 0
    train = {'singles':x[~held], 'pair_ids':ids[~held], 'culture_ids':np.repeat('r',sum(~held))}
    # Known zero Base makes acquired residuals equal to acquired outcomes;
    # the test isolates retrieval capability from reference selection.
    mean = np.unique(x[~held].reshape(-1,6),axis=0).mean(0)
    scale = np.unique(x[~held].reshape(-1,6),axis=0).std(0)
    memory = dict(train,mean=mean,scale=scale,residuals=y[~held],temperature=.2,k=4)
    query = {'singles':x[held], 'pair_ids':ids[held], 'culture_ids':np.repeat('r',sum(held))}
    prediction, _, neighbors = retrieve(memory,query)
    assert np.mean((prediction-y[held])**2) < .5*np.mean((y[held]-y[~held].mean())**2)
    assert all(ids[i] not in {n['pair_id'] for n in neighbors[j]} for j,i in enumerate(np.flatnonzero(held)))


def test_shared_safe_arrays_replay_relocation_and_tampering(tmp_path):
    import shutil
    bank, settings, menu, contract = fixture()
    bank['singles'] = np.tile(bank['singles'], (1, 1, 100))
    bank['responses'] = np.tile(bank['responses'], (1, 100))
    contract['genes'] = [f'g{i}' for i in range(1200)]
    settings['checkpoint_array_store'] = str(tmp_path / 'arrays')
    held = np.arange(len(bank['pair_ids'])) % 7 == 0
    contract.update(held_pair_ids=bank['pair_ids'][held].tolist(), acquired_pair_ids=bank['pair_ids'][~held].tolist())
    model = OpalBase(settings, menu).fit(subset(bank, ~held), contract)
    query = {k:v[held] for k,v in bank.items() if k != 'responses'}
    expected = model.predict(query, contract)['predictions']
    checkpoint = model.save(tmp_path / 'first')
    rows = list((tmp_path / 'arrays').glob('*.npz'))
    assert rows
    model.save(tmp_path / 'second')
    assert len(list((tmp_path / 'arrays').glob('*.npz'))) == len(rows)
    np.testing.assert_array_equal(OpalBase.load(checkpoint).predict(query, contract)['predictions'], expected)
    relocated = tmp_path / 'relocated'
    shutil.copytree(tmp_path / 'arrays', relocated)
    np.testing.assert_array_equal(OpalBase.load(checkpoint, array_store=relocated).predict(query, contract)['predictions'], expected)
    with pytest.raises(ValueError, match='reference provenance'):
        model.predict(query, dict(contract, reference_provenance='changed source references'))
    np.savez_compressed(rows[0], value=np.zeros(1200))
    with pytest.raises(ValueError, match='hash mismatch'):
        OpalBase.load(checkpoint)


def test_resource_ledger_counts_other_results_without_double_counting(tmp_path, monkeypatch):
    import importlib.util
    tool = Path('tools/run_model_series_bilinear.py').resolve()
    spec = importlib.util.spec_from_file_location('series_resource_test', tool)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    local = tmp_path / 'local'
    compact = tmp_path / 'results/opal_model_series_v2'
    output = compact / 'current'
    output.mkdir(parents=True)
    local.mkdir()
    (local / 'checkpoint').write_bytes(b'1234567')
    (compact / 'other-result').write_bytes(b'12345')
    (output / 'current-result').write_bytes(b'1234')
    settings = dict(artifact_roots=[str(local), str(compact), str(output)],
                    reserve_bytes={}, new_artifact_bytes_cap=16)
    assert runner.resource_check(local, output, settings) == 16
    settings['new_artifact_bytes_cap'] = 15
    with pytest.raises(RuntimeError, match='artifact cap'):
        runner.resource_check(local, output, settings)


def test_deployment_batch_inference_cannot_be_scored_as_evaluation(tmp_path):
    from opal.series import freeze_predictions, evaluate_predictions
    bank, settings, menu, contract = fixture()
    contract.update(checkpoint_role='all-development deployment refit after scoring',
                    held_pair_ids=[], acquired_pair_ids=bank['pair_ids'].tolist())
    model = OpalBase(settings, menu).fit(bank, contract)
    query = {k: v[:2] for k, v in bank.items() if k != 'responses'}
    result = freeze_predictions(tmp_path / 'inference', model, query, contract)
    assert result['provenance']['checkpoint_role'] == contract['checkpoint_role']
    with pytest.raises(ValueError, match='Deployment predictions'):
        evaluate_predictions(tmp_path / 'inference', query, contract, bank['responses'][:2])
    manifest = tmp_path / 'inference/freeze.json'
    changed = json.loads(manifest.read_text())
    changed.pop('prediction_role')
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='Deployment predictions'):
        evaluate_predictions(tmp_path / 'inference', query, contract, bank['responses'][:2])
