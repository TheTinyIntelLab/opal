"""Retrospective missing-single study with source-only main-effect selection."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import load_npz, write_json, sha256
from opal.contextual.contracts import load_dataset, ReferenceBank
from opal.contextual.main_effects import MainEffectImputer, MixedSinglePredictor, export_estimated_checkpoint, load_estimated_checkpoint
from opal.contextual.model import ContextualKernel


def mask(bank, contexts):
    arrays = {k: v.copy() for k, v in bank.arrays.items()}
    arrays['available'][contexts] = False
    arrays['singles'][contexts] = 0
    return ReferenceBank(arrays, bank.metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT / 'data/prepared/emeraldbay-pilot-v1')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    bank, fit, query, protocol = load_dataset(args.dataset)
    source = np.unique(fit['contexts'])
    held = np.unique(query['contexts'])
    masked = mask(bank, held)
    candidates = [{'kind': 'target_mean'}] + [{'kind': 'control_kernel', 'alpha': a, 'bandwidth': b}
                                            for a in [.01, 1., 100.] for b in [.25, 1., 4.]]
    frozen_residual = {'kind': 'pooled_sum', 'alpha': .01, 'bandwidth': 4.}
    write_json(args.output / 'protocol.json', {
        'dataset_protocol_sha256': sha256(args.dataset / 'protocol.json'),
        'main_candidates': candidates, 'residual_config': frozen_residual,
        'selection': 'source-only leave-context-out measured single error',
        'query_single_access': 'removed from model bank, revealed only after predictions',
        'scope': 'retrospective after EmeraldBay pilot; no independent confirmation',
        'separate_schema': 'opal-contextual-estimated-singles-v1', 'material_main_effect_gain': .05})
    records = []
    for config in candidates:
        errors, abstentions = [], 0
        for ci in source:
            training = mask(masked, [ci])
            imputer = MainEffectImputer(training, **config)
            for ti in np.flatnonzero(masked.arrays['available'][ci]):
                try:
                    pred, _ = imputer.predict(int(ti), int(ci))
                except ValueError:
                    abstentions += 1
                    continue
                errors.append(float(np.mean((pred - bank.arrays['singles'][ci, ti]) ** 2)))
        records.append({'config': config, 'mse': float(np.mean(errors)) if errors else None,
                        'predicted_singles': len(errors), 'unsupported_singles': abstentions})
    eligible = [r for r in records if r['mse'] is not None]
    winner = min(eligible, key=lambda r: r['mse'])
    write_json(args.output / 'selection.json', {'winner': winner, 'search': records, 'query_singles_used': False})
    imputer = MainEffectImputer(masked, **winner['config'])
    model = ContextualKernel(masked, frozen_residual).fit(fit['pairs'], fit['contexts'], fit['responses'])
    predictor = MixedSinglePredictor(model, imputer)
    predictions, profiles, supported, unsupported = [], [], [], []
    for i, (pair, ci) in enumerate(zip(query['pairs'], query['contexts'])):
        try:
            result = predictor.predict(*masked.arrays['targets'][pair], context=str(masked.arrays['contexts'][ci]))
        except ValueError as error:
            unsupported.append({'identity': int(query['identities'][i]), 'reason': str(error)})
            continue
        predictions.append(result['effect']); profiles.append(result['single_profiles']); supported.append(i)
    if not supported:
        raise ValueError('No supported missing-single queries')
    predictions, profiles = np.asarray(predictions), np.asarray(profiles)
    np.savez_compressed(args.output / 'predictions.npz', prediction=predictions, estimated_singles=profiles,
                        identities=query['identities'][supported])
    export_estimated_checkpoint(model, imputer, args.output / 'checkpoint', {'selection_sha256': sha256(args.output / 'selection.json')})
    restored = load_estimated_checkpoint(args.output / 'checkpoint')
    restored_predictions = np.array([restored.predict(*masked.arrays['targets'][query['pairs'][i]],
                                  context=str(masked.arrays['contexts'][query['contexts'][i]]))['effect'] for i in supported])
    np.testing.assert_allclose(predictions, restored_predictions, atol=1e-10)
    write_json(args.output / 'test_access.json', {'previously_exposed_pilot': True,
                'predictions_sha256': sha256(args.output / 'predictions.npz'), 'held_singles_revealed_after_predictions': True})
    evaluator = load_npz(args.dataset / 'evaluator/profiles.npz')
    if not np.array_equal(evaluator['identities'], query['identities']):
        raise ValueError('Evaluator identity mismatch')
    y = evaluator['responses'][supported]
    true_singles = bank.arrays['singles'][query['contexts'][supported, None], query['pairs'][supported]]
    main_error = profiles.sum(1) - true_singles.sum(1)
    residual_error = (predictions - profiles.sum(1)) - (y - true_singles.sum(1))
    total = float(np.mean((predictions - y) ** 2))
    main = float(np.mean(main_error ** 2)); residual = float(np.mean(residual_error ** 2))
    cross = float(2 * np.mean(main_error * residual_error))
    np.testing.assert_allclose(total, main + residual + cross, atol=1e-10)
    baseline = next(r for r in records if r['config']['kind'] == 'target_mean')
    write_json(args.output / 'results.json', {'status': 'complete', 'retrospective': True,
        'mse': total, 'additive_estimated_single_mse': float(np.mean((profiles.sum(1) - y) ** 2)),
        'main_effect_mse': main, 'residual_mse': residual, 'error_cross_term': cross,
        'oracle_single_mse': float(np.mean((model.predict_profiles(true_singles, query['contexts'][supported]) - y) ** 2)),
        'source_selected_imputer': winner, 'skill_over_source_target_mean': 1 - winner['mse'] / baseline['mse'],
        'supported_queries': len(supported), 'unsupported_queries': unsupported,
        'export_maximum_absolute_error': float(np.max(np.abs(predictions - restored_predictions))),
        'learned_representation_gate': 'not established by this simple-model study',
        'formal_coverage_guarantee': False})


if __name__ == '__main__':
    main()
