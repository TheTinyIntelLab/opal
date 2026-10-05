"""Exploratory point estimates; no confidence interval from correlated seeds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opal.artifacts import read_json


def main():
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
    main()
