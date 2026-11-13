"""Render README and manuscript figures from frozen compact result records.

No fitting or evaluator outcomes are opened. The public extracted record is a
fallback when original local score JSON is unavailable in a source-only checkout.
"""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / 'images'
EVIDENCE = ROOT / 'paper/evidence'
COLORS = ['#b5b0a9', '#d7a550', '#397c82']


def records():
    sources = {}

    def read(name):
        path = ROOT / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding='utf-8'))

    data = {'primary': {}, 'state': {}}
    for name in ['norman-log2', 'wessels-log2']:
        run = f'results/runs/{name}-primary'
        result, selection = read(run + '/test_results.json'), read(run + '/selection.json')
        additive = next(r['config']['id'] for r in selection['comparisons'] if r['config']['family'] == 'additive')
        data['primary'][name] = {k: result['models'][identifier]['mse'] for k, identifier in
            [('Additive', additive), ('Selected neural', result['selected_id']), ('Selected simple', result['simpler_id'])]}
    dspin = read('results/next/dspin-confirmation-v1/results.json')
    data['dspin'] = {'Source-selected pooled-pair': dspin['mse'][dspin['development_selected_simple_reference']],
                     'Frozen pooled-sum procedure': dspin['mse']['selected']}
    for name, path in [('EmeraldBay', 'results/state/emeraldbay-main-v2/main/results.json'),
                       ('sci-Plex 3', 'results/state/sciplex-state-v1/main/results.json')]:
        result = read(path)
        data['state'][name] = {'Selected simple': result['models']['selected_simple']['mse'],
                               'Learned state': result['models']['learned']['mse']}
    data['scope'] = 'Historical primary, independent procedure evaluation, and state development are separate populations. No new selection or fitting.'
    data['source_sha256'] = sources
    return data


def save(fig, name):
    fig.savefig(IMAGES / (name + '.svg'), bbox_inches='tight', facecolor='white')
    svg = IMAGES / (name + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    fig.savefig(EVIDENCE / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(10, 3.3))
    ax.set(xlim=(0, 10), ylim=(0, 3.3)); ax.axis('off')
    boxes = [(0.1, 1.8, 2.1, .75, 'Measured single A'), (.1, .6, 2.1, .75, 'Measured single B'),
             (3.2, 1.8, 2.5, .75, 'Additive reference\nδA + δB'),
             (3.2, .6, 2.5, .75, 'Symmetric correction\nf(δA, δB) = f(δB, δA)'),
             (7.1, 1.15, 2.65, 1, 'Population-mean prediction\nδA + δB + f(δA, δB)')]
    for x, y, w, h, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=0.08',
                     linewidth=.8, edgecolor='#a6a098', facecolor='#faf8f4'))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10)
    for start, end in [((2.3,2.17),(3.1,2.17)), ((2.3,.97),(3.1,.97)),
                       ((2.3,2),(3.1,1.25)), ((2.3,1.15),(3.1,2)),
                       ((5.8,2.17),(7,1.8)), ((5.8,.97),(7,1.45))]:
        ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle':'->','color':'#666057','lw':1.1})
    ax.text(5, 3.05, 'Opal: observed-single residual prediction', ha='center', fontsize=14, fontweight='bold')
    ax.text(5, .13, 'Both singles must be measured in the same biological setting. Correction is a predictive residual.',
            ha='center', fontsize=9, color='#605a52')
    save(fig, 'opal_overview')


def results(data):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, key, label in zip(axes[:2], ['norman-log2','wessels-log2'], ['Norman: 38 held pairs','Wessels: 18 held pairs']):
        values = data['primary'][key]
        baseline = values['Additive']
        ax.bar(range(3), [v/baseline for v in values.values()], color=COLORS, width=.6)
        ax.set_xticks(range(3), ['Additive','Neural','Simple'], fontsize=9)
        ax.set(title=label, ylim=(0,1.18), ylabel='MSE / additive MSE')
        for i, v in enumerate(values.values()): ax.text(i, v/baseline+.035, f'{v:.6f}', ha='center', fontsize=8)
    vals=data['dspin'];ref=next(iter(vals.values()))
    axes[2].bar(range(2), [v/ref for v in vals.values()], color=[COLORS[2],COLORS[1]], width=.55)
    axes[2].set_xticks(range(2), ['Source-selected\npair kernel','Frozen sum\nprocedure'], fontsize=9)
    axes[2].set(title='D-SPIN: independent transfer', ylim=(0,1.18), ylabel='MSE / source-reference MSE')
    for i,v in enumerate(vals.values()):axes[2].text(i,v/ref+.035,f'{v:.6f}',ha='center',fontsize=8)
    for ax in axes:
        ax.axhline(1,color='#6f6860',lw=.7,ls='--');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Additive gains and independent superiority are different tests', fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'opal_results')
    fig,axes=plt.subplots(1,2,figsize=(8,3.3))
    for ax,(name,values) in zip(axes,data['state'].items()):
        ref=values['Selected simple'];ratio=values['Learned state']/ref
        ax.bar([0,1],[1,ratio],color=[COLORS[2],COLORS[1]],width=.55)
        ax.set_xticks([0,1],['Selected simple','Learned state'])
        ax.set(title=name,ylim=(0,1.4),ylabel='MSE / selected-simple MSE')
        ax.text(1,ratio+.035,f'{100*(ratio-1):.2f}% worse',ha='center',fontsize=9)
        ax.axhline(1,color='#6f6860',lw=.7,ls='--');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Baseline-only state development: tested encoders lose',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'opal_state')


def badges():
    """Cache genuine Shields.io badges locally, using the DeepSeek README style."""
    from urllib.request import urlopen, Request
    from urllib.parse import quote
    specs = [('lab', 'Website', 'The Tiny Intelligence Lab', '536af5'),
             ('status', 'Version', '1.0.0', '7289da'),
             ('license', 'Code_License', 'MIT', 'f5de53'),
             ('paper', 'Paper', 'Preprint draft', '536af5')]
    sources = {}
    for name, label, value, color in specs:
        url = ('https://img.shields.io/badge/' + quote(label, safe='') + '-' +
               quote(value, safe='') + '-' + color + '?style=flat&logoColor=white')
        target = IMAGES / (name + '_badge.svg')
        # Existing caches make figure regeneration possible without networking.
        if not target.exists() or b'Generated by Shields.io' not in target.read_bytes():
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
                svg = response.read()
            if b'<svg' not in svg:
                raise ValueError('Shields.io did not return an SVG')
            target.write_bytes(svg + b'\n<!-- Generated by Shields.io -->\n')
        sources[target.name] = url
    (IMAGES / 'badge_sources.json').write_text(json.dumps(sources, indent=2) + '\n', encoding='utf-8')


def main():
    IMAGES.mkdir(exist_ok=True);EVIDENCE.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    try:data=records()
    except FileNotFoundError:
        data=json.loads((IMAGES/'publication_data.json').read_text(encoding='utf-8'))
    (IMAGES/'publication_data.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    overview();results(data);badges()
    print('Generated README images and manuscript PDFs from frozen score records.')


if __name__=='__main__': main()
