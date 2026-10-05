import numpy as np
import pytest
from opal.contextual.contracts import ReferenceBank
from opal.contextual.synthetic import generate
from opal.contextual.model import ContextualKernel
from opal.contextual.main_effects import MainEffectImputer, MixedSinglePredictor, export_estimated_checkpoint, load_estimated_checkpoint
from opal.contextual.benchmark import select, run_benchmark
from opal.contextual.contracts import prepare_dataset
from opal.contextual.inference import export_checkpoint, ContextPredictor


def test_estimated_route_preserves_observed_predictions_and_order():
    bank, pairs, contexts, y = generate(targets=6, genes=12, contexts=4)
    train = contexts < 3
    for kind in ContextualKernel.FAMILIES:
        model = ContextualKernel(bank, {'kind': kind}).fit(pairs[train], contexts[train], y[train])
        profiles = bank.arrays['singles'][contexts[:, None], pairs]
        np.testing.assert_allclose(model.predict_profiles(profiles, contexts), model.predict(pairs, contexts), atol=1e-10)
        np.testing.assert_allclose(model.predict_profiles(profiles[:, ::-1], contexts), model.predict(pairs, contexts), atol=1e-10)


def test_checkpoint_preserves_fitted_reference_bandwidth(tmp_path):
    bank, pairs, contexts, y = generate(targets=6, genes=12, contexts=4)
    model = ContextualKernel(bank, {'kind': 'pooled_sum'})
    # Simulate a bandwidth fitted under a different numerical environment.
    # Restoration must use that value instead of recalculating it from refs.
    model.reference_distance *= 1.123
    model.fit(pairs, contexts, y)
    export_checkpoint(model, tmp_path / 'checkpoint')
    restored = ContextPredictor(tmp_path / 'checkpoint').model
    assert float(restored.reference_distance) == model.reference_distance
    np.testing.assert_allclose(restored.predict(pairs, contexts), model.predict(pairs, contexts), atol=1e-10)


def test_missing_single_sources_are_explicit_and_unseen_targets_abstain(tmp_path):
    bank, pairs, contexts, y = generate(targets=6, genes=12, contexts=4)
    arrays = {k: v.copy() for k, v in bank.arrays.items()}
    arrays['singles'][3] = 0
    arrays['available'][3] = False
    masked = ReferenceBank(arrays, bank.metadata)
    train = contexts < 3
    model = ContextualKernel(masked).fit(pairs[train], contexts[train], y[train])
    predictor = MixedSinglePredictor(model, MainEffectImputer(masked, 'control_kernel'))
    a, b = masked.arrays['targets'][:2]
    result = predictor.predict(str(a), str(b), context=str(masked.arrays['contexts'][3]))
    assert result['route'] == 'estimated_single_pair'
    assert all(not item['observed'] and 3 not in item['source_contexts'] for item in result['single_provenance'])
    np.testing.assert_allclose(result['effect'], result['additive'] + result['residual'])
    assert not masked.arrays['available'][3].any()
    export_estimated_checkpoint(model, predictor.imputer, tmp_path / 'checkpoint')
    restored = load_estimated_checkpoint(tmp_path / 'checkpoint')
    np.testing.assert_allclose(restored.predict(str(a), str(b), context=str(masked.arrays['contexts'][3]))['effect'], result['effect'])
    arrays['available'][:, 0] = False
    arrays['singles'][:, 0] = 0
    with pytest.raises(ValueError, match='must abstain'):
        MainEffectImputer(ReferenceBank(arrays, bank.metadata)).predict(0, 3)


def test_selection_holds_entire_lineage_and_confirmation_keeps_frozen_model(tmp_path):
    bank, pairs, contexts, y = generate(targets=6, genes=12, contexts=5)
    metadata = bank.metadata.copy()
    for c in range(4):
        metadata['context_records'][str(bank.arrays['contexts'][c])]['selection_group'] = 'lineage_' + str(c // 2)
    bank = ReferenceBank(bank.arrays, metadata)
    fit = contexts < 4
    candidate = {'kind': 'shared_offset'}
    _, rows = select(bank, {'pairs': pairs[fit], 'contexts': contexts[fit], 'responses': y[fit]}, [candidate])
    assert len(rows[0]['folds']) == 2
    assert all(len(fold['held_contexts']) == 2 for fold in rows[0]['folds'])
    dataset = prepare_dataset(tmp_path / 'dataset', bank, pairs, contexts, y,
                              held_contexts=[str(bank.arrays['contexts'][4])])
    frozen = {'kind': 'pooled_sum', 'alpha': .01, 'bandwidth': 4.}
    result = run_benchmark(dataset, tmp_path / 'run', candidates=[candidate], frozen_config=frozen)
    assert result['selected_config'] == frozen
    assert result['frozen_procedure']
