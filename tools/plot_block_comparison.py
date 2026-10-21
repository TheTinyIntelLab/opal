"""Export a descriptive, within-task MSE comparison from completed receipts."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'opal-block-comparison-v1'
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--block-results', type=Path, required=True)
parser.add_argument('--native-results', type=Path)
parser.add_argument('--family-results', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError('Preserve figure artifacts')
paths = {'block': args.block_results, 'native': args.native_results, 'family': args.family_results}
data = {key: json.loads(path.read_text(encoding='utf-8')) for key, path in paths.items() if path}
tasks = ('horlbeck_K562', 'horlbeck_Jurkat', 'environment1', 'environment2', 'parp2022', 'parp2023')
labels = ('Horlbeck K562', 'Horlbeck Jurkat', 'Environmental 1', 'Environmental 2', 'PARPi 2022', 'PARPi 2023')
if any(set(result['tasks']) != set(tasks) for result in data.values()):
    raise ValueError('Complete six-task receipts required')
baseline = np.array([data['block']['tasks'][task]['selected_simple']['mse'] for task in tasks])
series = {'Rank-eight completion': np.array([data['block']['tasks'][t]['candidate']['mse'] for t in tasks])}
if 'native' in data:
    series['Native NAIAD, symmetric inference'] = np.array([data['native']['tasks'][t]['native']['mse'] for t in tasks])
if 'family' in data:
    series['Validation family selector'] = np.array([data['family']['tasks'][t]['selector']['mse'] for t in tasks])
if not np.isfinite(baseline).all() or np.any(baseline <= 0) or any(not np.isfinite(v).all() or np.any(v < 0) for v in series.values()):
    raise ValueError('Finite nonnegative MSE and positive reference required')
args.output.mkdir(parents=True)
fig, ax = plt.subplots(figsize=(8.5, 4.8), layout='constrained')
colors = ('#177d80', '#b56625', '#435bb0')
height = .72 / len(series)
for i, (name, values) in enumerate(series.items()):
    positions = np.arange(6) + (i - (len(series)-1)/2)*height
    ax.barh(positions, values/baseline, height=height*.88, label=name, color=colors[i])
ax.axvline(1, color='#555555', linewidth=1, linestyle='--', label='Validation-selected simple reference')
ax.set(yticks=np.arange(6), yticklabels=labels, xlabel='Held-block MSE / selected-simple MSE within the same task')
ax.invert_yaxis()
ax.set_xlim(0, max(1.15, max(float(np.max(v/baseline)) for v in series.values())*1.08))
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left', bbox_to_anchor=(0, -.19), frameon=False, fontsize=9)
fig.savefig(args.output/'comparison.png', dpi=180, bbox_inches='tight')
fig.savefig(args.output/'comparison.svg', bbox_inches='tight', metadata={'Date': None})
svg = args.output/'comparison.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n', encoding='utf-8', newline='\n')
plt.close(fig)
note = ('Exposed development sources; local pair supervision and measured singles. '
        'Point estimates, no pooled cross-source score. Native and selector are shown only when their complete receipts are supplied. '
        'The selector was proposed after Q2 test exposure. This is not independent confirmation.')
image = base64.b64encode((args.output/'comparison.png').read_bytes()).decode()
rows = ''.join('<tr><th>'+label+'</th>'+''.join('<td>'+format(values[j]/baseline[j], '.4f')+'</td>' for values in series.values())+'</tr>' for j, label in enumerate(labels))
html = ('<!doctype html><html><head><meta charset="utf-8"><style>'
        'body{margin:0;font-family:var(--font-sans,system-ui);color:var(--foreground)}'
        '.figure{background:white;padding:16px;border-radius:var(--radius,8px)}img{width:100%;height:auto}'
        'p{font-size:13px;line-height:1.5;color:var(--muted-foreground)}'
        'table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:right;padding:7px;border-bottom:1px solid var(--border,#ccc)}'
        'th:first-child{text-align:left}details{margin-top:12px}summary{cursor:pointer}</style></head><body>'
        '<div class="figure"><img alt="Within-task missing-block error relative to the selected simple reference" src="data:image/png;base64,'+image+'"></div>'
        '<p>'+note+'</p><details><summary>Error ratios, lower is better</summary><table><thead><tr><th>Task</th>'
        +''.join('<th>'+name+'</th>' for name in series)+' </tr></thead><tbody>'+rows+'</tbody></table></details></body></html>')
(args.output/'comparison.html').write_text(html, encoding='utf-8')
(args.output/'provenance.json').write_text(json.dumps(dict(note=note, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), inputs={k: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items() if p}, baseline_mse=baseline.tolist(), mse={k:v.tolist() for k,v in series.items()}), indent=2)+'\n', encoding='utf-8')
