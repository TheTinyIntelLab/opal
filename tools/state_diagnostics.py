"""Risk partitioning and subset-retuned context efficiency from development data."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import load_npz, read_json, write_json, sha256
from opal.contextual.contracts import ReferenceBank
from opal.state.benchmark import groups_for, select, fit_model, query_rows
from opal.state.models import MainReference
from opal.state.reliability import evaluate_policy


def diagnostics(bank, predictions, config, output):
    a = bank.arrays; groups = groups_for(bank)
    ci, ti = predictions['contexts'], predictions['targets']
    features = np.empty((len(ci), 3))
    for group in np.unique(groups):
        source = np.flatnonzero(groups != group); rows = np.flatnonzero(groups[ci] == group)
        model = MainReference().fit(a['controls'][source], a['singles'][source], a['available'][source])
        query_z = model.encode_control(a['controls'][ci[rows]])
        features[rows, 0] = np.min(np.linalg.norm(query_z[:, None] - model.z[None], axis=-1), axis=1)
        features[rows, 1] = a['available'][source][:, ti[rows]].sum(0)
        features[rows, 2] = np.sqrt(np.mean(predictions['selected_simple'][rows] ** 2, 1))
    error = np.sqrt(np.mean((predictions['selected_simple'] - a['singles'][ci, ti]) ** 2, 1))
    unique = np.unique(groups)
    calibration = unique[-3:-1] if len(unique) >= 6 else unique[-2:-1]
    evaluation = unique[-1:] if len(unique) < 6 else unique[-2:]
    if len(unique) >= 6:
        calibration = unique[-4:-2]
    risk_fit = [g for g in unique if g not in set(calibration) | set(evaluation)]
    reliability = evaluate_policy(features, error, groups[ci], fit_groups=risk_fit,
        calibration_groups=calibration.tolist(), evaluation_groups=evaluation.tolist(),
        nominal=config['reliability']['nominal'], maximum_rmse=config['reliability']['maximum_rmse'],
        minimum_accepted_fraction=config['reliability']['minimum_accepted_fraction'])
    write_json(output / 'reliability.json', reliability)
    write_json(output / 'acquisition.json', {'status': 'not_extended',
        'reason': 'No successful state representation gate or independently useful risk diagnostic. Retain prior retrospective replays.',
        'existing_evidence': ['results/next/cycle-v7', 'configs/experiments/acquisition-v1.json'],
        'prospective_value': 'unestablished', 'measurement_cost': 'No source assay costs supplied; cost scenarios must be declared before acquisition'})
    # Fixed outer query population, retuned separately at each context budget.
    if len(unique) < 5:
        write_json(output / 'context_efficiency.json', {'status': 'insufficient_independent_context_groups',
            'groups': len(unique), 'reason': 'Three cell lines do not support separate multi-budget nested selection and a fixed outer population'})
        return
    source = np.flatnonzero(~np.isin(groups, unique[-2:])); query = np.flatnonzero(np.isin(groups, unique[-2:]))
    qc, qt = query_rows(a['available'], source, query)
    curves = []
    for seed in [41, 73, 101]:
        rng = np.random.default_rng(seed); order = rng.permutation(source)
        # Require one fixed population at all budgets; do not discard weakly covered queries.
        if not a['available'][order[:2]][:, qt].any(0).all():
            complete = [c for c in source if a['available'][c, np.unique(qt)].all()]
            if not complete:
                raise ValueError('Cannot preserve evaluation population at the smallest budget')
            order = np.array([complete[0]] + [c for c in order if c != complete[0]])
        for budget in sorted(set([2, 4, len(source)])):
            subset = order[:budget]
            start = time.perf_counter()
            selected, search = select(bank, subset, groups, config['main_references'], 'cpu')
            model = fit_model(bank, subset, selected)
            prediction = model.predict(a['controls'][qc], qt)
            curves.append({'seed': seed, 'source_contexts': len(subset), 'source_single_conditions': int(a['available'][subset].sum()),
                'selected': selected, 'search': search, 'mse': float(np.mean((prediction - a['singles'][qc, qt]) ** 2)),
                'seconds': time.perf_counter() - start, 'evaluation_conditions': len(qc),
                'measurement_cost_units': int(a['available'][subset].sum()) + len(subset)})
    write_json(output / 'context_efficiency.json', {'status': 'complete_development', 'curves': curves,
        'scope': 'Retrospective nested context budgets with fixed query population; orders share labels and are dependent',
        'cost_units': 'one condition assay equals one unit, including each control; hypothetical, not observed monetary costs',
        'cells_per_condition': 'not varied; independent measurement precision unavailable',
        'pair_label_efficiency': 'existing retuned curves preserved, not rerun'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True); p.add_argument('--main-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    config = read_json(ROOT / 'configs/experiments/state-development.json')
    write_json(args.output / 'protocol.json', {'main_results_sha256': sha256(args.main_run / 'results.json'),
        'selection_policy': 'post-model development diagnostic, partitions fixed by sorted biological groups',
        'fresh_confirmation': False, 'features': ['source-fold baseline distance', 'exact-intervention coverage', 'predicted response magnitude']})
    with threadpool_limits(limits=1):
        diagnostics(ReferenceBank.load(args.dataset), load_npz(args.main_run / 'predictions.npz'), config, args.output)


if __name__ == '__main__':
    main()
