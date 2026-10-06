import copy
import numpy as np
import pytest
from opal.contextual.synthetic import generate
from opal.state.contracts import InformationContract, group_folds, validate_isolation, error_decomposition, registered_split
from opal.state.models import MainReference, FactorizedResponse, SymmetricResidual
from opal.state.inference import export_state, export_ensemble, StatePredictor
from opal.state.reliability import evaluate_policy


@pytest.fixture
def fitted():
    bank, pairs, ci, responses = generate(targets=6, genes=12, contexts=6)
    a = bank.arrays
    model = FactorizedResponse(epochs=35, width=8, rank=4).fit(a['controls'][:5], a['singles'][:5], a['available'][:5])
    return bank, pairs, ci, responses, model


def test_information_budgets_and_group_isolation():
    main = InformationContract('main_context', 'log2', 'RNA', 'development').table()
    pair = InformationContract('pair_context_observed', 'log2', 'RNA', 'development').table()
    assert not main['inference']['singles'] and pair['inference']['singles']
    groups = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
    for _, train, test in group_folds(groups):
        validate_isolation(train, test, groups)
    with pytest.raises(ValueError, match='Biological unit'):
        validate_isolation([0], [1], groups)
    with pytest.raises(ValueError, match='pretraining'):
        validate_isolation([0], [2], groups, pretraining_interventions=['A'], forbidden_interventions=['A'])
    with pytest.raises(ValueError, match='Pair identity'):
        validate_isolation([0], [2], groups, pairs=np.array([[0, 1]] * 6), held_pair=True)


def test_joint_holdout_removes_pair_globally_and_keeps_guard_population():
    groups = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
    pairs = np.array([[0, 1], [0, 2]] * 3)
    train, query, unused = registered_split('joint_pair_context', groups, pairs=pairs, held_groups=['c'], held_pairs=[(1, 0)])
    assert train.tolist() == [1, 3] and query.tolist() == [4] and unused.tolist() == [0, 2, 5]
    train, query, _ = registered_split('unseen_pair_observed', groups, pairs=pairs, held_pairs=[(1, 0)])
    assert train.tolist() == [1, 3, 5] and query.tolist() == [0, 2, 4]
    train, query, _ = registered_split('held_partner', groups, pairs=pairs, held_interventions=[1])
    assert train.tolist() == [1, 3, 5]
    with pytest.raises(ValueError, match='qualified'):
        registered_split('higher_order', groups)


def test_known_state_generator_beats_intervention_average():
    bank, _, _, _ = generate(targets=6, genes=12, contexts=8)
    a = bank.arrays
    mean = MainReference().fit(a['controls'][:6], a['singles'][:6], a['available'][:6])
    linear = MainReference('ridge', alpha=1e-6).fit(a['controls'][:6], a['singles'][:6], a['available'][:6])
    targets = np.arange(6); controls = np.repeat(a['controls'][6:7], 6, 0)
    y = a['singles'][6]
    assert np.mean((linear.predict(controls, targets) - y) ** 2) < np.mean((mean.predict(controls, targets) - y) ** 2) * .01
    # Query profiles cannot change fold-fitted normalization or latent bases.
    assert np.allclose(linear.center, a['controls'][:6].mean(0))


def test_main_error_cancellation_is_not_interaction_recovery():
    measured = np.ones((3, 4)); main = measured + 2
    response = measured + .5; prediction = response + .1
    result = error_decomposition(prediction, main, measured, response)
    assert result['main_effect_mse'] == 4
    assert result['error_cross_term'] < 0
    assert result['total_mse'] == pytest.approx(.01)


def test_descriptor_mask_prevents_hidden_values_entering_encoder():
    bank, _, _, _ = generate(targets=6, genes=12, contexts=4)
    a = bank.arrays; mask = np.zeros((6, 2), bool)
    models = [FactorizedResponse(epochs=5).fit(a['controls'], a['singles'], a['available'], d, mask)
              for d in [np.zeros((6, 2)), np.full((6, 2), 10000.)]]
    np.testing.assert_array_equal(models[0].inputs['descriptor'], models[1].inputs['descriptor'])
    np.testing.assert_allclose(models[0].predict(a['controls'][:1], [0]), models[1].predict(a['controls'][:1], [0]))


def test_new_context_export_and_single_baseline_reconstruction(tmp_path, fitted):
    bank, _, _, _, model = fitted; a = bank.arrays
    export_state(model, tmp_path / 'export', genes=a['genes'], targets=a['targets'], assay='RNA', scale='log2', intervention_records=bank.metadata['intervention_records'])
    restored = StatePredictor(tmp_path / 'export')
    from opal import load_model
    assert isinstance(load_model(tmp_path / 'export'), StatePredictor)
    kwargs = dict(baseline=a['controls'][5], genes=a['genes'], assay='RNA', scale='log2', context='new_context_without_id')
    result = restored.predict(str(a['targets'][0]), **kwargs)
    np.testing.assert_allclose(result['effect'], model.predict(a['controls'][5:6], [0])[0], atol=1e-6)
    np.testing.assert_allclose(result['absolute_response'], kwargs['baseline'] + result['effect'])
    assert result['single_provenance'][0]['kind'] == 'predicted_exact_intervention'
    control = restored.predict(**kwargs)
    np.testing.assert_array_equal(control['effect'], np.zeros(12))
    observed = restored.predict(str(a['targets'][0]), measured_singles={str(a['targets'][0]): a['singles'][5, 0]}, **kwargs)
    assert observed['route'] == 'observed_single'
    with pytest.raises(ValueError, match='Incompatible'):
        restored.predict(str(a['targets'][0]), **{**kwargs, 'assay': 'ATAC'})
    with pytest.raises(ValueError, match='Unknown'):
        restored.predict('unknown', **kwargs)
    with pytest.raises(ValueError, match='Combination model unavailable'):
        restored.predict(*map(str, a['targets'][:2]), **kwargs)
    with pytest.raises(ValueError, match='exactly'):
        restored.predict(str(a['targets'][0]), measured_singles={}, **kwargs)


def test_staged_residual_symmetry_and_joint_validation(tmp_path, fitted):
    bank, pairs, ci, responses, main = fitted; a = bank.arrays
    train = ci < 5; query = ci == 5
    before = {k: v.copy() for k, v in main.numpy_state().items()}
    residual = SymmetricResidual(main, epochs=20).fit(a['controls'][ci[train]], pairs[train], responses[train], a['singles'][ci[train, None], pairs[train]])
    for k, value in before.items():
        np.testing.assert_array_equal(main.numpy_state()[k], value)
    direct = residual.predict_residual(a['controls'][ci[query]], pairs[query])
    np.testing.assert_allclose(direct, residual.predict_residual(a['controls'][ci[query]], pairs[query, ::-1]), atol=1e-10)
    export_state(main, tmp_path / 'export', genes=a['genes'], targets=a['targets'], assay='RNA', scale='log2', intervention_records=bank.metadata['intervention_records'], residual=residual)
    restored = StatePredictor(tmp_path / 'export'); pair = pairs[query][0]
    kwargs = dict(baseline=a['controls'][5], genes=a['genes'], assay='RNA', scale='log2')
    result = restored.predict(*map(str, a['targets'][pair]), **kwargs)
    expected_main = sum(main.predict(a['controls'][5:6], [t])[0] for t in pair)
    np.testing.assert_allclose(result['effect'], expected_main + direct[0], atol=1e-6)
    np.testing.assert_allclose(result['effect'], restored.predict(*map(str, a['targets'][pair[::-1]]), **kwargs)['effect'])
    with pytest.raises(ValueError, match='repeated'):
        restored.predict(str(a['targets'][0]), str(a['targets'][0]), **kwargs)


def test_training_resume_validates_data_and_protocol(tmp_path):
    bank, _, _, _ = generate(targets=6, genes=12, contexts=4); a = bank.arrays
    args = (a['controls'], a['singles'], a['available'])
    destination = tmp_path / 'resume.pt'
    model = FactorizedResponse(epochs=5).fit(*args, resume=destination, protocol_fingerprint='registered')
    restored = FactorizedResponse(epochs=5).fit(*args, resume=destination, protocol_fingerprint='registered')
    np.testing.assert_allclose(model.predict(a['controls'][:1], [0]), restored.predict(a['controls'][:1], [0]))
    with pytest.raises(ValueError, match='fingerprint'):
        FactorizedResponse(epochs=5).fit(*args, resume=destination, protocol_fingerprint='changed')


def test_risk_partitions_and_unusable_radius():
    groups = np.repeat(['a', 'b', 'c', 'd'], 4)
    x = np.arange(16.)[:, None]; y = np.linspace(.1, .8, 16)
    result = evaluate_policy(x, y, groups, fit_groups=['a', 'b'], calibration_groups=['c'], evaluation_groups=['d'])
    assert result['interval_status'] == 'finite_group_radius_unavailable'
    assert result['reliability_gate'] == 'failed'
    assert result['policies'][0]['accepted_conditions'] == 0
    assert result['policies'][1]['accepted_fraction'] == 1
    with pytest.raises(ValueError, match='overlap'):
        evaluate_policy(x, y, groups, fit_groups=['a', 'b'], calibration_groups=['b'], evaluation_groups=['d'])


def test_ensemble_restores_mean_and_detects_member_tampering(tmp_path, fitted):
    bank, _, _, _, model = fitted; a = bank.arrays
    second = FactorizedResponse(epochs=35, width=8, rank=4, seed=73).fit(a['controls'][:5], a['singles'][:5], a['available'][:5])
    export_ensemble([model, second], tmp_path / 'ensemble', genes=a['genes'], targets=a['targets'], assay='RNA', scale='log2', intervention_records=bank.metadata['intervention_records'])
    restored = StatePredictor(tmp_path / 'ensemble')
    result = restored.predict(str(a['targets'][0]), baseline=a['controls'][5], genes=a['genes'], assay='RNA', scale='log2')
    expected = (model.predict(a['controls'][5:6], [0]) + second.predict(a['controls'][5:6], [0]))[0] / 2
    np.testing.assert_allclose(result['effect'], expected, atol=1e-6)
    assert result['ensemble_members'] == 2
    assert result['model_variation_status'] == 'uncalibrated_fitted_model_variation'
    with (tmp_path / 'ensemble/member-1/manifest.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='checksum'):
        StatePredictor(tmp_path / 'ensemble')
