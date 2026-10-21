"""Render aggregate registered development results; never fit or select models."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
a = parser.parse_args()
r = json.loads(a.results.read_text())
tasks = ['horlbeck_K562', 'horlbeck_Jurkat', 'environment1', 'environment2', 'parp2022', 'parp2023']
labels = ['K562', 'Jurkat', 'Environment 1', 'Environment 2', 'PARPi 2022', 'PARPi 2023']
fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
x = np.arange(6)
for offset, budget, color in [(-.18, '5000', '#677487'), (.18, '20000', '#177d8d')]:
    values = [100*r['tasks'][budget][t]['relative_gain'] for t in tasks]
    bars = ax.bar(x+offset, values, width=.34, color=color, label=f'{int(budget):,} labels / model')
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
ax.axhline(5, color='#ab5b26', linestyle='--', linewidth=1, label='Required 5% margin, plus other gates')
ax.axhline(0, color='#aaa', linewidth=.6)
ax.set_xticks(x, labels, rotation=15)
ax.set_ylabel('MSE reduction vs same-budget selected simple (%)')
ax.set_ylim(-3, 36)
ax.set_title('Neither label budget passes the six-task gate', loc='left')
ax.legend(frameon=False, fontsize=8)
fig.text(.01, -.04, 'Exposed-source development. Conditional uncertainty is not source replication. All test coverage: 100%.', fontsize=8)
a.output.parent.mkdir(parents=True, exist_ok=True)
for suffix in ('.svg', '.png'):
    fig.savefig(a.output.with_suffix(suffix), dpi=160, bbox_inches='tight', metadata={'Creator': 'Opal aggregate results'})
svg = a.output.with_suffix('.svg')
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
