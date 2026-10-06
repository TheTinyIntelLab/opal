"""Restore the declared two-seed prediction procedures in portable NumPy exports."""
import argparse
import copy
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, load_npz, write_json, sha256
from opal.contextual.contracts import ReferenceBank
from opal.state.benchmark import fit_model, groups_for, select
from opal.state.models import SymmetricResidual
from opal.state.inference import StatePredictor, export_ensemble


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--pair-fold', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new export destination')
    bank = ReferenceBank.load(args.dataset); a = bank.arrays
    config = read_json(ROOT / 'configs/experiments/state-development.json')
    groups = groups_for(bank); residuals = []
    with threadpool_limits(limits=1):
        if args.pair_fold:
            data = load_npz(args.dataset / 'profiles.npz'); source_contexts = np.unique(data['contexts'])
            held = np.unique(groups[source_contexts])[-1]
            source = source_contexts[groups[source_contexts] != held]
            fit = np.isin(data['contexts'], source); query = ~fit
            selected, _ = select(bank, source, groups, config['learned_candidates'], args.device)
            base = fit_model(bank, source, selected, device=args.device)
            pairs, ci = data['pairs'][fit], data['contexts'][fit]
            sc, st = np.nonzero(a['available'][source]); single_ci = source[sc]
            models = []
            for seed in config['seeds']:
                model = copy.deepcopy(base)
                residual = SymmetricResidual(model, epochs=config['combination_epochs'], seed=seed).fit(
                    a['controls'][ci], pairs, data['responses'][fit], a['singles'][ci[:, None], pairs], joint=True,
                    single_controls=a['controls'][single_ci], single_targets=st, single_responses=a['singles'][single_ci, st])
                residual.training_records = [{'identity': int(identity), 'context': str(a['contexts'][c]),
                    'interventions': a['targets'][pair].tolist()} for identity, pair, c in zip(data['identities'][fit], pairs, ci)]
                models.append(model); residuals.append(residual)
            saved = load_npz(args.run / 'predictions.npz')
            rows = np.flatnonzero(groups[saved['contexts']] == held)
            query_pairs, query_ci = data['pairs'][query], data['contexts'][query]
            expected = saved['joint'][rows]
            if not np.array_equal(saved['identities'][rows], data['identities'][query]):
                raise ValueError('Fold query identity mismatch')
            provenance = {'scope': 'scored two-seed last outer fold, not a selected best model', 'held_group': str(held)}
        else:
            original = read_json(args.run / 'checkpoint/manifest.json')
            selected = min(original['provenance']['full_source_search'], key=lambda row: row['mse'])['config']
            models = [fit_model(bank, np.arange(len(groups)), selected, seed=seed, device=args.device) for seed in config['seeds']]
            qc, qt = np.nonzero(a['available']); query_ci, query_pairs = qc[:12], qt[:12, None]
            expected = np.mean([m.predict(a['controls'][query_ci], query_pairs[:, 0]) for m in models], 0)
            provenance = {'scope': 'two-seed full-development refit of source-selected procedure; no independent confirmation'}
        assay = next(iter(bank.metadata['context_records'].values()))['assay']
        export_ensemble(models, args.output, residuals=residuals if args.pair_fold else None,
            genes=a['genes'], targets=a['targets'], assay=assay, scale=bank.metadata['scale'],
            intervention_records=bank.metadata['intervention_records'],
            provenance={**provenance, 'source_run_results_sha256': sha256(args.run / 'results.json')})
        restored = StatePredictor(args.output)
        actual = np.array([restored.predict(*map(str, a['targets'][pair]), baseline=a['controls'][c],
                           genes=a['genes'], assay=assay, scale=bank.metadata['scale'])['effect']
                           for pair, c in zip(query_pairs, query_ci)])
        error = float(np.max(np.abs(actual - expected)))
        write_json(args.output / 'restoration.json', {'maximum_absolute_error': error, 'tolerance': 1e-5,
            'scored_pair_fold_reproduced': args.pair_fold, 'queries': len(actual), 'evaluation_labels_included': False})
        if error > 1e-5:
            raise ValueError('Ensemble export differs from the recorded prediction procedure')
        print('Ensemble restoration:', error, flush=True)


if __name__ == '__main__':
    main()
