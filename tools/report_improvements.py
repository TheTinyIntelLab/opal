"""Aggregate every planned seed and retain exploratory scope in the report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from opal.artifacts import read_json, write_json


def summarize():
    root = Path('results/improvements')
    studies = ['residual-kernel-v1', 'residual-kernel-seed-20261006', 'residual-kernel-seed-20261007']
    rows = []
    for study in studies:
        results = read_json(root / study / 'expanded_linear_controls.json')
        for name, protocols in results['datasets'].items():
            for kind, values in protocols.items():
                rows.append({'study': study, 'dataset': name, 'protocol': kind,
                    **{k: v for k, v in values.items() if k != 'records'}})
    groups = []
    for name in ['norman-log2', 'wessels-log2']:
        for kind in ['random', 'partner']:
            values = [r['kernel_relative_mse_reduction'] for r in rows if r['dataset'] == name and r['protocol'] == kind]
            groups.append({'dataset': name, 'protocol': kind, 'minimum_relative_gain': min(values),
                          'maximum_relative_gain': max(values), 'mean_relative_gain': float(np.mean(values)),
                          'positive_studies': sum(v > 0 for v in values), 'studies': len(values)})
    write_json(root / 'summary.json', {'scope': 'Exploratory nested development, not fresh external confirmation',
        'historical_test_scores_known': True, 'new_evaluator_access': False, 'groups': groups, 'rows': rows})
    lines = ['# Residual kernel improvement experiments', '',
        'The new model combines a shared residual offset, a reference-scaled linear correction and symmetric',
        'single-profile kernels. Inner development folds choose between the shared correction, summed-single',
        'kernel, pair-product kernel and mixtures. The old neural architecture is not mandatory in this search.', '',
        'All fitting and selection use the original development profiles. No historical evaluator responses',
        'were loaded by these studies. Historical test findings informed the decision to investigate shared',
        'corrections, so these results remain exploratory and require external confirmation.', '',
        '## Design and results', '',
        'Three seeds use five outer random folds and five outer partner folds per dataset. Each outer fit',
        'uses three inner folds of the same protocol. The kernel procedure searches 67 configurations.',
        'An expanded linear comparator searches 25 configurations: a shared offset plus global, ridge and',
        'rank-four reduced-rank models at eight penalties from 0.01 through 100,000. Its feature scaling is',
        'fitted within each training fold. Kernel scaling uses singles only. Both searches select by inner',
        'equal-pair all-gene MSE. These are different model spaces and budgets, not a capacity-matched test.', '',
        '| Dataset | Holdout | Relative MSE reduction across three seeds | Positive studies |',
        '| --- | --- | --- | --- |']
    for g in groups:
        lines.append(f"| {g['dataset']} | {g['protocol']} | {100*g['minimum_relative_gain']:.2f}% to {100*g['maximum_relative_gain']:.2f}% | {g['positive_studies']}/3 |")
    lines += ['', 'Partner holdouts overlap. The primary nested score weights each validation occurrence equally;',
        'the artifact also reports scores after averaging each pair\'s repeated predictions. On Wessels,',
        'averaging can reverse the comparison with the reference-scaled shared global control. The reported',
        'gain is specifically against the separately tuned expanded linear search; it is not a universal',
        'win over every linear variant or every aggregation rule.', '',
        'No native published-method scores were added. Kernels, residual learning and symmetry are established',
        'methods; this implementation alone does not establish methodological novelty or frontier status.', '',
        '## Reproduction and checkpoints', '',
        'Run `python main.py improve --seed 20261005 --output results/improvements/new-study` followed by',
        '`python tools/compare_improvement_baselines.py --study-root results/improvements/new-study`.',
        'Repeat with seeds 20261006 and 20261007. Existing study directories are immutable and refuse reuse.',
        'The original seed\'s `norman-log2/checkpoint` and `wessels-log2/checkpoint` load with the existing',
        '`Predictor` and `main.py predict` interfaces. They contain development-only weights and observed',
        'single profiles. Predictions on registered historical test identities are stored unscored.',
        'They remain research candidates, and zero ensemble spread for a deterministic fit is not calibrated',
        'uncertainty. `export_verification.json` records numerical agreement and CPU measurements.', '',
        '## PertAdapt resources', '',
        'The supplied checkpoint loaded with `weights_only=True` and memory mapping. The archive contains',
        'three backbone tasks with 119,354,430 tensor elements each. The gene list has 19,264 unique entries;',
        'the finite GO mask is 19,264 by 19,264. Its checksum manifest is in',
        '`data/public_methods/pertadapt_resource_audit.json`. These structural checks do not establish',
        'native training compatibility or provide a PertAdapt benchmark. The published loader returns CUDA',
        'models and the training script uses CUDA distributed execution; this installation has CPU PyTorch.', '',
        'The next decisive evidence is an information-aligned native comparison and an independent',
        'confirmation dataset or genuinely untouched experimental conditions. Additional seeds drawn from',
        'these same development pairs do not supply that independence.', '']
    Path('results/improvements/report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(str(g) for g in groups))



def plot():
    record = read_json('results/improvements/summary.json')
    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    labels = []
    for i, group in enumerate(record['groups']):
        values = [100 * r['kernel_relative_mse_reduction'] for r in record['rows']
                  if r['dataset'] == group['dataset'] and r['protocol'] == group['protocol']]
        ax.scatter([i-.12, i, i+.12], values, s=35, color='#22577A')
        labels.append(group['dataset'].replace('-log2', '').capitalize() + '\n' + group['protocol'])
    ax.axhline(0, color='#888888', linewidth=.8)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel('Relative MSE reduction (%)')
    ax.set_title('Nested development results, three seeds per protocol', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig('results/improvements/relative_gain.pdf')
    fig.savefig('results/improvements/relative_gain.png', dpi=180)
    fig.savefig('paper/evidence/improvement-relative-gain.pdf')



if __name__ == '__main__':
    summarize()
    plot()
