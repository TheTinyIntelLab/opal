import numpy as np
import pytest
from opal.models import ResidualNetwork
from opal.state.contracts import validate_response_sources, error_decomposition
from tools.run_stage1 import scale_features, candidates, input_features
import torch


@pytest.mark.parametrize('stage', ['fitting', 'preprocessing', 'pretraining', 'calibration'])
def test_joint_exclusions_follow_every_response_stage(stage):
    row = dict(biological_group='source', interventions=['B', 'A'], dose='1uM', response_kind='pair')
    with pytest.raises(ValueError, match='Globally held pair'):
        validate_response_sources('joint_pair_context', {stage: [row]}, held_groups=['query'], held_pairs=[('A', 'B')])
    row['interventions'] = ['C', 'D']; row['biological_group'] = 'query'
    with pytest.raises(ValueError, match='Held biological state'):
        validate_response_sources('joint_pair_context', {stage: [row]}, held_groups=['query'], held_pairs=[('A', 'B')])


def test_query_single_access_distinguishes_observed_and_baseline_only():
    row = dict(biological_group='query', interventions=['A'], dose='1uM', response_kind='single')
    validate_response_sources('joint_pair_context', {'query_references': [row]}, held_groups=['query'])
    with pytest.raises(ValueError, match='Query response forbidden'):
        validate_response_sources('main_context', {'query_references': [row]}, held_groups=['query'])


def test_intervention_and_exact_dose_exclusion():
    row = dict(biological_group='source', interventions=['A'], dose='1uM', response_kind='single')
    with pytest.raises(ValueError, match='Held intervention'):
        validate_response_sources('intervention_identity', {'pretraining': [row]}, held_interventions=['A'])
    validate_response_sources('dose', {'fitting': [row]}, held_doses=['10uM'])
    with pytest.raises(ValueError, match='Held exact dose'):
        validate_response_sources('dose', {'fitting': [row]}, held_doses=['1uM'])


def test_shared_neural_really_sees_sum_only_and_pair_swap_is_exact():
    embedding = np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])
    model = ResidualNetwork(embedding, 3, dict(family='sum_network', shared_only=True, width=4, rank=2, random_init=True))
    pairs = torch.tensor([[0, 1], [2, 3], [1, 0]])
    result = model(pairs).detach().numpy()
    np.testing.assert_array_equal(result[0], result[1])
    np.testing.assert_array_equal(result[0], result[2])


def test_outer_features_cannot_change_source_preprocessing():
    source = np.array([[1., 2.], [3., 4.], [100., 200.]])
    first = scale_features(source, np.array([0, 1]))
    source[2] *= 100
    np.testing.assert_array_equal(first[:2], scale_features(source, np.array([0, 1]))[:2])


def test_cross_reference_component_accounting():
    rng = np.random.default_rng(42)
    main, measured, response, residual = rng.normal(size=(4, 5, 3))
    r = error_decomposition(main + residual, main, measured, response)
    assert r['total_mse'] == pytest.approx(r['main_effect_mse'] + r['residual_mse'] + r['error_cross_term'])


def test_shared_only_export_restores_fitted_weights(tmp_path):
    from opal.training import fit, predict
    from opal.inference import export_members, Predictor
    rng = np.random.default_rng(8)
    singles = rng.normal(size=(5, 7))
    pairs = np.array([[0, 1], [0, 2], [1, 3], [2, 4]])
    response = singles[pairs].sum(1) + .1
    model = fit(dict(family='sum_network', shared_only=True, width=4, rank=2, embedding=3, epochs=3), singles, pairs, response)
    data = dict(singles=singles, pairs=pairs, targets=np.array(list('ABCDE')), genes=np.array([f'g{i}' for i in range(7)]), control=np.zeros(7))
    export_members([model], data, tmp_path, dict(context='source', scale='log2'))
    restored = Predictor(tmp_path)
    np.testing.assert_array_equal(restored.predict('A', 'B')['effect'], predict(model, pairs[:1])[0])
    np.testing.assert_array_equal(restored.predict('A', 'B')['effect'], restored.predict('B', 'A')['effect'])
