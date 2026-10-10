"""Generate source-qualified tables, diagnostics and figures from saved evidence."""
import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, sha256, write_json
from opal.series import OpalBase
from opal.series_bilinear import parameter_counts
from run_model_series_bilinear import source_banks, resource_check

OUT = ROOT / 'results/opal_model_series_v2/report'
PAPER = ROOT / 'paper/evidence'
NAMES = ['Opal-Base', 'Opal-Interaction', 'Opal-Guard']
COLORS = ['#35495e', '#bd6f2a', '#54866a']


def aggregate(rows, method, endpoint='mse'):
    if any(method not in r['metrics'] or r['metrics'][method].get(endpoint) is None for r in rows):
        return None
    strata = {}
    for row in rows:
        metric = row['metrics'][method]
        count = row.get('held_rows', row.get('held_pairs', metric['retrieval_pool']))
        total, n = strata.get(str(row.get('stratum', 'aggregate')), (0, 0))
        strata[str(row.get('stratum', 'aggregate'))] = (total + count * metric[endpoint], n + count)
    return float(np.mean([total / n for total, n in strata.values()]))


def main():
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    resource_check(Path(settings['artifact_roots'][1]), OUT, settings)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    inputs, primary, retrospective, costs, checkpoints = {}, {}, {}, {}, {}
    diagnostics, paired = [], {}
    def read(path):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
        return read_json(path)
    def save_figure(figure, name):
        figure.savefig(PAPER / name, bbox_inches='tight')
        plt.close(figure)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for col, (source, (bank_path, _)) in enumerate(source_banks().items()):
        rows = read(ROOT / 'results/opal_model_series_v2/primary' / source / 'conditions.json')
        full = [r for r in rows if r['budget'] == 1 and r['order'] == 'uniform'
                and r['acquisition_seed'] == 11 and r['training_seed'] == 20261017]
        methods = sorted(set().union(*(set(r['metrics']) for r in full)))
        primary[source] = {m: {endpoint: aggregate(full, m, endpoint) for endpoint in
                              ['mse', 'mae', 'centered_pearson_mean', 'retrieval_mean_rank']}
                           for m in methods}
        unique = {r['fit_id']: r for r in rows}
        costs[source] = {'unique_fit_receipts': len(unique), 'conditions': len(rows),
                         'cpu_seconds': sum(r['cpu_seconds'] for r in unique.values()),
                         'wall_seconds_sum': sum(r['wall_seconds'] for r in unique.values())}
        for order_index, order in enumerate(settings['orders']):
            axis = axes[order_index, col]
            for name, color in zip(NAMES, COLORS):
                values = []
                for budget in settings['budgets']:
                    per_seed = []
                    for seed in settings['sampling_seeds']:
                        selected = [r for r in rows if r['budget'] == budget and r['order'] == order
                                    and r['acquisition_seed'] == seed and r['training_seed'] == 20261017]
                        value = aggregate(selected, name)
                        if value is not None:
                            per_seed.append(value)
                    values.append(float(np.mean(per_seed)) if per_seed else np.nan)
                axis.plot(settings['budgets'], values, 'o-', label=name, color=color, markersize=3)
            axis.set_title(source + '\n' + order, fontsize=9)
            axis.set_xlabel('Acquired pair fraction')
            axis.set_ylabel('Primary MSE')
            axis.grid(alpha=.2)
            zero_control = [r for r in rows if r['budget'] == 0 and r['order'] == order and r['acquisition_seed'] == 11]
            axis.axhline(aggregate(zero_control, 'no-change'), color='#777777', linestyle=':', label='No change')
        with np.load(bank_path, allow_pickle=False) as z:
            bank = {k: z[k] for k in ['singles', 'pair_ids', 'culture_ids', 'responses']}
        pair_changes = {m: [] for m in NAMES[1:]}
        for row in full:
            contract = read_json(Path(row['checkpoints'][0]['path']).parent.parent / 'contract.json')
            held = (bank['culture_ids'] == row['stratum']) & np.isin(bank['pair_ids'], contract['held_pair_ids'])
            query = {k: v[held] for k, v in bank.items() if k != 'responses'}
            predictions = {}
            for cp in row['checkpoints']:
                checkpoint = Path(cp['path'])
                checkpoints[str(checkpoint)] = {'model_sha256': sha256(checkpoint / 'model.json'),
                    'parameters_sha256': sha256(checkpoint / 'parameters.npz')}
                model = OpalBase.load(checkpoint)
                result = model.predict(query, contract)
                prediction = result['predictions']
                with np.load(checkpoint.parent / 'freeze/predictions.npz', allow_pickle=False) as z:
                    if not np.array_equal(prediction, z['predictions']):
                        raise ValueError('Saved primary prediction changed during figure generation')
                predictions[cp['model']] = prediction
                delta = prediction - model._base(query)
                diagnostics.append(dict(source=source, stratum=row['stratum'], outer=row['outer'], model=cp['model'],
                    held_rows=int(held.sum()), correction_rms=float(np.sqrt(np.mean(delta ** 2))),
                    correction_to_base_norm=float(np.linalg.norm(delta) / max(np.linalg.norm(model._base(query)), 1e-30)),
                    correction_rank=int(np.linalg.matrix_rank(delta)), unsupported_fraction=float(np.mean(~result['support'])),
                    mean_expert_weights=result['expert_weights'].mean(0).tolist(), parameter_counts=parameter_counts(model.fitted),
                    reference_family=model.config['kind'], reference_configuration=model.config,
                    affine_gamma=model.fitted['reference'].get('gamma'),
                    checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.iterdir() if p.is_file()),
                    shared_array_store=model.settings.get('checkpoint_array_store')))
            base_loss = np.mean((predictions[NAMES[0]] - bank['responses'][held]) ** 2, axis=1)
            for name in NAMES[1:]:
                loss = np.mean((predictions[name] - bank['responses'][held]) ** 2, axis=1)
                pair_changes[name].extend((loss - base_loss).tolist())
        paired[source] = {name: {'rows': len(values), 'quantiles': np.quantile(values, [0, .25, .5, .75, 1]).tolist(),
                                'better_fraction': float(np.mean(np.asarray(values) < 0)), 'independent_pair_inference': False}
                          for name, values in pair_changes.items()}
    axes[0, 0].legend(fontsize=8)
    save_figure(fig, 'model-series-current-learning.pdf')
    for source in ['norman-log2', 'wessels-log2', 'norman-count', 'wessels-count']:
        rows = [read(ROOT / 'results/opal_model_series_v2/retrospective' / source / f'fold-{outer}.json') for outer in range(3)]
        ablations = [read(ROOT / 'results/opal_model_series_v2/retrospective-ablations' / f'{source}-{outer}.json') for outer in range(3)]
        methods = sorted(set().union(*(set(r['metrics']) for r in rows)))
        retrospective[source] = {m: aggregate(rows, m) for m in methods}
        retrospective[source]['constant-mixture'] = aggregate(ablations, 'constant-mixture')
        retrospective[source]['matched-main-effect'] = aggregate(ablations, 'matched-main-effect')
    native = {}
    for method, directory in [('PertSpectra', 'native-results'), ('GEARS', 'GEARS-aggregate-amended-results'), ('CPA', 'CPA-aggregate-amended-results')]:
        rows = [read(p) for p in (ROOT / 'results/opal_model_series_v2/native-completed' / directory).glob('TNFA-*.json')]
        native[method] = {}
        for source in ['TNFA-positive', 'TNFA-negative']:
            selected = [r for r in rows if r['source'] == source]
            adapted = [dict(r, metrics={method: {'mse': r['mse'], 'retrieval_pool': r['held_rows']}}) for r in selected]
            native[method][source] = dict(mse=aggregate(adapted, method), folds=len(selected),
                cpu_seconds=sum(r['cpu_seconds'] for r in selected), wall_seconds_sum=sum(r['wall_seconds'] for r in selected),
                parameter_count_range=[min(r['parameter_count'] for r in selected), max(r['parameter_count'] for r in selected)],
                classification='protocol adaptation; not full author configuration', training_ram=None)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for axis, source in zip(axes, primary):
        labels = NAMES + ['constant-mixture', 'matched-main-effect']
        values = [primary[source].get(label, {}).get('mse') for label in labels]
        axis.bar(range(len(labels)), [np.nan if v is None else v for v in values], color=COLORS + ['#8b7898', '#83909a'])
        axis.set_xticks(range(len(labels)), ['Base', 'Interaction', 'Guard', 'Constant', 'Main effect'], rotation=35, ha='right', fontsize=8)
        axis.set_title(source, fontsize=10)
        axis.set_ylabel('Primary MSE')
    save_figure(fig, 'model-series-current-comparison.pdf')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for axis, source in zip(axes, primary):
        for name, color in zip(NAMES[1:], COLORS[1:]):
            # Full distribution displayed as five-number summaries; no independent-pair intervals.
            q = paired[source][name]['quantiles']
            axis.plot([0, .25, .5, .75, 1], q, 'o-', label=name, color=color)
        axis.axhline(0, color='black', linewidth=.6)
        axis.set_title(source, fontsize=10)
        axis.set_xlabel('Empirical pair-stratum loss quantile')
        axis.set_ylabel('MSE change from Base')
    axes[0].legend(fontsize=8)
    save_figure(fig, 'model-series-current-paired.pdf')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for axis, source in zip(axes, primary):
        selected = [r for r in diagnostics if r['source'] == source and r['model'] == 'Opal-Guard']
        values = np.asarray([r['mean_expert_weights'] for r in selected])
        for i, (label, color) in enumerate(zip(['Base', 'Bilinear', 'Memory'], COLORS)):
            axis.bar(range(len(values)), values[:, i], bottom=values[:, :i].sum(1), label=label, color=color)
        axis.set_title(source, fontsize=10)
        axis.set_ylabel('Mean convex expert weight')
        axis.set_xlabel('Prespecified stratum / outer fold')
    axes[0].legend(fontsize=8)
    save_figure(fig, 'model-series-current-routing.pdf')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for name, color in zip(NAMES[1:], COLORS[1:]):
        for source_index, source in enumerate(primary):
            selected = [r for r in diagnostics if r['source'] == source and r['model'] == name]
            x = source_index + (-.12 if name == NAMES[1] else .12)
            axes[0].scatter(np.repeat(x, len(selected)), [r['correction_rms'] for r in selected], color=color, alpha=.7, s=15)
            axes[1].scatter(np.repeat(x, len(selected)), [r['correction_rank'] for r in selected], color=color, alpha=.7, s=15)
    for axis in axes:
        axis.set_xticks(range(3), list(primary), rotation=15, fontsize=8)
    axes[0].set_ylabel('Correction RMS in source units')
    axes[1].set_ylabel('Numerical correction rank')
    save_figure(fig, 'model-series-current-branches.pdf')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for axis, source in zip(axes, ['TNFA-positive', 'TNFA-negative']):
        for method, color in zip(['PertSpectra', 'GEARS', 'CPA'], ['#9a779d', '#5284a5', '#a57a54']):
            value = native[method][source]
            axis.scatter(value['cpu_seconds'], value['mse'], color=color, label=method)
        axis.axhline(primary[source]['Opal-Base']['mse'], color=COLORS[0], label='Opal-Base MSE')
        axis.set_xlabel('Total CPU seconds, 12 upstream-adapted folds')
        axis.set_ylabel('Primary MSE')
        axis.set_title(source)
    axes[0].legend(fontsize=8)
    save_figure(fig, 'model-series-current-native-cost.pdf')
    write_json(OUT / 'summary.json', {'primary': primary, 'retrospective': retrospective, 'native_adapted': native,
        'primary_costs': costs, 'paired_changes': paired, 'checkpoint_diagnostics': diagnostics,
        'default_promotion_passed': False, 'frontier_competitiveness_passed': False,
        'unequal_search': 'Base 24 references; Interaction 24 plus zero; router 2 k values x 2 penalties and nested refits; native 2 configs plus selected refit',
        'missing_native_training_ram_prevents_complete_efficiency_claim': True})
    # Emit tables directly from the saved summary and hash all empirical inputs.
    table = [r'\begin{tabular}{lrrr}', r'\toprule', r'Method & Combi-seq & TNF-positive & TNF-negative \\', r'\midrule']
    for method in ['no-change', 'additivity', 'historical-reference'] + NAMES + ['constant-mixture', 'matched-main-effect']:
        values = [primary[s].get(method, {}).get('mse') for s in ['combi-seq', 'TNFA-positive', 'TNFA-negative']]
        table.append(method.replace('constant-mixture', 'Constant mixture').replace('matched-main-effect', 'Matched main effect') + ' & ' +
            ' & '.join('Unavailable' if v is None else f'{v:.6f}' for v in values) + r' \\')
    table += [r'\bottomrule', r'\end{tabular}']
    (PAPER / 'model_series_current_table.tex').write_text('\n'.join(table) + '\n', encoding='utf-8')
    claims = {'primary_interaction_gain_percent': {s: 100 * (1 - primary[s]['Opal-Interaction']['mse'] / primary[s]['Opal-Base']['mse']) for s in primary},
              'retrospective_guard_gain_percent': {s: 100 * (1 - retrospective[s]['Opal-Guard'] / retrospective[s]['Opal-Base']) for s in retrospective}}
    write_json(OUT / 'claims.json', claims)
    write_json(OUT / 'figure_manifest.json', {'inputs': inputs, 'checkpoint_inputs': checkpoints,
        'figures': {p.name: sha256(p) for p in PAPER.glob('model-series-current-*.pdf')},
        'table_sha256': sha256(PAPER / 'model_series_current_table.tex'), 'claims_sha256': sha256(OUT / 'claims.json'),
        'primary_scores_replaced': False})
    print('Saved-evidence report and six figures generated', flush=True)


if __name__ == '__main__':
    main()
