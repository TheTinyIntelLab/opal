"""Build source-qualified model comparisons and figures from saved metrics."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, sha256, write_json

MODELS = ['no-change', 'additivity', 'historical-reference', 'historical-candidate',
          'Opal-Base', 'Opal-Interaction', 'Opal-Guard', 'fixed-half-gate']


def summarize(directory):
    conditions = read_json(directory / 'conditions.json')
    grouped = defaultdict(list)
    for row in conditions:
        grouped[(row['variant'], row['order'], row['sampling_seed'], row['budget'], row['run'])].append(row)
    strata = []
    for (variant, order, seed, budget, run), rows in sorted(grouped.items()):
        if len(rows) != 3:
            raise ValueError('Incomplete outer fold population')
        metrics = {}
        unavailable = {}
        for name in MODELS:
            if all(name in r['metrics'] for r in rows):
                metrics[name] = float(np.average([r['metrics'][name] for r in rows],
                                                 weights=[r['test_pair_count'] for r in rows]))
            else:
                unavailable[name] = 'At least one required fold unavailable; no favorable-subset average'
        strata.append({'variant': variant, 'order': order, 'seed': seed, 'budget': budget, 'stratum': run,
                       'pairs': sum(r['test_pair_count'] for r in rows), 'metrics': metrics, 'unavailable': unavailable,
                       'label_counts_by_outer': [r['labels_acquired'] for r in rows],
                       'full_development_counts_by_outer': [r['full_development_pairs'] for r in rows]})
    equal = defaultdict(list)
    for row in strata:
        equal[(row['variant'], row['order'], row['seed'], row['budget'])].append(row)
    summaries = []
    for (variant, order, seed, budget), rows in sorted(equal.items()):
        metrics = {name: float(np.mean([r['metrics'][name] for r in rows])) for name in MODELS
                   if all(name in r['metrics'] for r in rows)}
        summaries.append({'variant': variant, 'order': order, 'seed': seed, 'budget': budget,
                          'metrics': metrics, 'strata': len(rows),
                          'mean_acquired_pairs': float(np.mean([r['label_counts_by_outer'] for r in rows])),
                          'minimum_acquired_pairs': min(n for r in rows for n in r['label_counts_by_outer']),
                          'maximum_acquired_pairs': max(n for r in rows for n in r['label_counts_by_outer'])})
    full = next(r for r in summaries if r['variant'] == 'aligned' and r['order'] == 'uniform' and r['seed'] == 11 and r['budget'] == 1)
    comparisons = {name: {'mse': value, 'delta_vs_base': value - full['metrics']['Opal-Base'],
                         'relative_gain_vs_base': 1 - value / full['metrics']['Opal-Base'],
                         'delta_vs_additivity': value - full['metrics']['additivity']}
                   for name, value in full['metrics'].items()}
    curves, marginal = [], []
    for method in ['uniform', 'target-balanced']:
        for budget in [0., .25, .5, 1.]:
            rows = [r for r in summaries if r['variant'] == 'aligned' and r['order'] == method and r['budget'] == budget]
            curves.append({'order': method, 'budget': budget, 'seeds': [r['seed'] for r in rows],
                           'mean_labels': float(np.mean([r['mean_acquired_pairs'] for r in rows])),
                           'metrics': {name: float(np.mean([r['metrics'][name] for r in rows])) for name in MODELS
                                       if all(name in r['metrics'] for r in rows)}})
        rows = [r for r in curves if r['order'] == method]
        for left, right in zip(rows, rows[1:]):
            for name in set(left['metrics']) & set(right['metrics']):
                marginal.append({'order': method, 'model': name, 'from_budget': left['budget'], 'to_budget': right['budget'],
                                 'added_pair_labels': right['mean_labels'] - left['mean_labels'],
                                 'mse_change_per_added_label': (right['metrics'][name] - left['metrics'][name]) /
                                                             (right['mean_labels'] - left['mean_labels'])})
    reference_default = min(('Opal-Base', 'additivity', 'no-change'), key=lambda n: full['metrics'][n])
    record = {'conditions_sha256': sha256(directory / 'conditions.json'), 'per_stratum': strata,
              'equal_stratum_summaries': summaries, 'full_budget_comparison': comparisons,
              'learning_curves': curves, 'marginal_error_changes': marginal,
              'default': reference_default, 'series_default': 'Opal-Base' if reference_default == 'Opal-Base' else None,
              'experimental': ['Opal-Interaction', 'Opal-Guard'] + ([] if reference_default == 'Opal-Base' else ['Opal-Base']),
              'selection_rule': 'Retrospective recommendation compares the fitted reference, additivity and no-change. Nonlinear variants have not earned replacement status on these sources. Complexity and tiny exposed-source differences do not warrant promotion; a zero predictor beating supervised models is disclosed.',
              'weights': 'Equal canonical pairs within each stratum, then equal source strata; descriptive seed averages.',
              'claims': 'Retrospective source-conditional prediction only; no biological interaction discovery, noise ceiling, prospective assay savings, calibrated uncertainty or independent culture confirmation.',
              'gate_ablation': 'Base is zero interaction; Interaction is gate one; fixed-half-gate is constant 0.5; Guard is learned OOF gate.',
              'optimization_budget': 'Base reuses the 13 reference entries in the historical 16-entry selection. Interaction has two penalties and development-selected epochs. Guard has three nested OOF expert fits, two gate penalties and gate-selected epochs. Final experts shared across variants. Budgets are unequal and disclosed.'}
    write_json(directory / 'summary.json', record)
    figures = directory / 'figures'
    figures.mkdir(exist_ok=True)
    colors = {'additivity': '#999999', 'historical-reference': '#5c7885', 'historical-candidate': '#ab755d',
              'Opal-Base': '#15394c', 'Opal-Interaction': '#8872a5', 'Opal-Guard': '#438270'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, method in zip(axes, ['uniform', 'target-balanced']):
        rows = [r for r in curves if r['order'] == method]
        for name, color in colors.items():
            available = [r for r in rows if name in r['metrics']]
            ax.plot([r['mean_labels'] for r in available], [r['metrics'][name] for r in available],
                    marker='o', markersize=4, linewidth=1.4, color=color, label=name,
                    linestyle='--' if name.startswith('historical') else '-')
        ax.set_title(method)
        ax.set_xlabel('Acquired canonical pair labels, validation included')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Equal-stratum MSE')
    axes[1].legend(fontsize=8)
    fig.suptitle('Retrospective label coverage, fixed outer pair populations', fontsize=12)
    fig.tight_layout()
    fig.savefig(figures / 'learning_curves.png', dpi=180)
    fig.savefig(figures / 'learning_curves.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = ['historical-reference', 'historical-candidate', 'Opal-Base', 'Opal-Interaction', 'Opal-Guard', 'fixed-half-gate']
    names = [n for n in names if n in comparisons]
    values = [comparisons[n]['mse'] for n in names]
    ax.barh(names, values, color=[colors.get(n, '#79908b') for n in names])
    ax.axvline(full['metrics']['Opal-Base'], color='#15394c', linewidth=1, linestyle='--')
    ax.set_xlabel('Full-budget equal-stratum MSE; lower is better')
    ax.spines[['top', 'right']].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(figures / 'model_comparison.png', dpi=180)
    fig.savefig(figures / 'model_comparison.pdf')
    plt.close(fig)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(summarize(args.directory)['full_budget_comparison'])
