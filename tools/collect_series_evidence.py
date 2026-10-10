"""Collect compact histories and comparisons without fitting or opening protected queries."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, sha256, write_json
from report_model_series_bilinear import aggregate


def main():
    evidence = ROOT / 'results/opal_model_series_v2'
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    roots = [Path(p) for p in settings['artifact_roots']]
    out = evidence / 'report'
    checkpoints, histories, receipts = {}, [], {}
    for root in roots:
        for path in root.rglob('receipt.json'):
            record = read_json(path)
            receipts[sha256(path)] = dict(path=str(path), cpu_seconds=record.get('cpu_seconds'),
                wall_seconds=record.get('wall_seconds'), process_peak_rss_bytes=record.get('process_peak_rss_bytes'))
        for path in root.rglob('model.json'):
            record = read_json(path)
            if record.get('architecture_id') != settings['architecture_id']:
                continue
            state = record['state']['fitted']
            def expand(value, archive):
                if isinstance(value, dict) and set(value) == {'json_array'}:
                    return json.loads(str(archive[value['json_array']].item()))
                if isinstance(value, dict):
                    return {k: expand(v, archive) for k, v in value.items()}
                if isinstance(value, list):
                    return [expand(v, archive) for v in value]
                return value
            with np.load(path.parent / 'parameters.npz', allow_pickle=False) as archive:
                selection = {k: expand(state[k], archive) for k in ['selection', 'branch_selection', 'gate_selection'] if k in state}
            histories.append(dict(checkpoint=str(path.parent), model=record['model'], selection=selection,
                                  model_sha256=sha256(path)))
            checkpoints[str(path.parent)] = dict(model=record['model'], model_sha256=sha256(path),
                parameters_sha256=record['parameters_sha256'], parameter_counts=record.get('parameter_counts'),
                checkpoint_bytes=sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file()),
                array_store=record.get('array_store'))
    with gzip.open(out / 'all_selection_histories.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(histories, handle, sort_keys=True)
    write_json(out / 'checkpoint_manifest.json', checkpoints)
    write_json(out / 'retained_receipt_costs.json', {'receipts': list(receipts.values()),
        'cpu_seconds': sum(r['cpu_seconds'] or 0 for r in receipts.values()),
        'accounting': 'identical receipt copies deduplicated by content hash; repeated fits with different receipts retained',
        'incomplete_costs': 'failed processes and some compatibility probes have no retained process CPU counter; not assigned zero'})
    comparisons, curves, stability = {}, {}, {}
    for source in ['combi-seq', 'TNFA-positive', 'TNFA-negative']:
        rows = read_json(evidence / 'primary' / source / 'conditions.json')
        old = []
        for row in rows:
            receipt = read_json(Path(row['local_fit_directory']) / 'receipt.json')
            old.append(dict(row, metrics={m: {'mse': v, 'retrieval_pool': row['metrics']['no-change']['retrieval_pool']}
                        for m, v in receipt.get('preserved_v1_metrics', {}).items()}))
        comparisons[source] = old
        stability[source] = {str(seed): {m: aggregate([r for r in rows if r['budget'] == 1 and r['order'] == 'uniform'
            and r['acquisition_seed'] == 11 and r['training_seed'] == seed], m)
            for m in ['Opal-Base', 'Opal-Interaction', 'Opal-Guard']} for seed in [settings['training_seed']] + settings['stability_seeds']}
    for source in ['norman-log2', 'wessels-log2', 'norman-count', 'wessels-count']:
        rows = read_json(evidence / 'retrospective-budget-completion' / source / 'conditions.json')
        curves[source] = [dict(order=order, budget=budget, seed=seed, metrics={m: aggregate([r for r in rows
            if r['order'] == order and r['budget'] == budget and r['acquisition_seed'] == seed
            and r['training_seed'] == settings['training_seed']], m) for m in ['no-change', 'additivity',
            'training-mean', 'historical-reference', 'historical-candidate', 'Opal-Base', 'Opal-Interaction', 'Opal-Guard']})
            for order in settings['orders'] for seed in settings['sampling_seeds'] for budget in settings['budgets']]
    write_json(out / 'preserved_v1_comparisons.json', comparisons)
    write_json(out / 'retrospective_learning_curves.json', curves)
    write_json(out / 'training_seed_diagnostics.json', {'scores': stability, 'selection_seed_changed': False})
    extras = {}
    for source in ['TNFA-positive', 'TNFA-negative']:
        rows = [read_json(p) for p in (evidence / 'native-extra-single-controls').glob(source + '-*.json')]
        extras[source] = aggregate([dict(r, metrics={'extra-single-linear': r['metrics']}) for r in rows], 'extra-single-linear')
    write_json(out / 'extra_single_reference.json', extras)
    print('Collected', len(histories), 'safe checkpoint histories and all fixed budget summaries')


if __name__ == '__main__':
    main()
