"""Regenerate joint development evidence from saved scores, without fitting."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from opal.artifacts import read_json, write_json, sha256, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    source = args.cycle / 'run-v1/results.json'; result = read_json(source)
    models = result['models']
    with (args.output / 'comparisons.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'mse', 'relative_gain', 'paired_lower', 'paired_upper', 'gate'])
        writer.writeheader()
        for model, values in models.items():
            writer.writerow(dict(model=model, **values))
    with (args.cycle / 'run-v1/condition-errors.csv').open(newline='') as f:
        conditions = list(csv.DictReader(f))
    for grouping in ['state', 'canonical_pair']:
        with (args.output / f'{grouping}-errors.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[grouping, 'conditions', *models])
            writer.writeheader()
            for key in sorted({row[grouping] for row in conditions}):
                rows = [r for r in conditions if r[grouping] == key]
                writer.writerow({grouping:key, 'conditions':len(rows), **{m:np.mean([float(r[m]) for r in rows]) for m in models}})
    names = ['additive', 'mean_residual', 'global_correction', 'single_sum', 'pair_product', 'mixed']
    labels = ['additivity', 'mean residual', 'global correction', 'single sum', 'pair product', 'mixed']
    fig, ax = plt.subplots(figsize=(7.7, 3.7))
    ax.bar(range(len(names)), [100 * models[n]['relative_gain'] for n in names], color=['#99958e']*4+['#167d8d']*2)
    ax.axhline(5, color='#aa5835', linestyle='--', linewidth=1)
    ax.axhline(0, color='#333333', linewidth=.6)
    ax.set_xticks(range(len(names)), labels, rotation=20)
    ax.set_ylabel('MSE improvement over selected simple (%)')
    ax.set_title('Global chemical-pair / whole-state development')
    ax.spines[['top','right']].set_visible(False); fig.tight_layout()
    for extension in ['svg', 'pdf']:
        fig.savefig(args.output / f'joint-results.{extension}')
    plt.close(fig)
    tex = [r'\subsection{Globally held chemical pairs and whole-state development}',
        r'A separately registered metadata-balanced partition closes an exposed-development coverage gap without reopening the historical evaluator. Seven canonical chemical pairs, defined by unordered names across all doses, are greedily balanced into two global panels and crossed with six whole cell lines. Twelve outer folds query each of the 54 fitting conditions once; guard rows are unused. Source sizes range from 19 to 26. Inner selection also holds a whole state and global chemical pair, retains only preregistered metadata-feasible folds, and uses no outcome-driven redraw. Exact-dose measured query singles and the legacy query-control gene-panel adaptation are explicitly permitted \citep{opaljoint2026}.',
        r'\begin{center}\small\begin{tabular}{lrrr}\toprule',
        r'Correction & Total MSE & Gain & Paired reduction interval\\\midrule']
    for name in ['selected_simple', *names]:
        m = models[name]
        label = name.replace('_', ' ')
        tex.append(f"{label} & {m['mse']:.6f} & {100*m['relative_gain']:.2f}\\% & [{m['paired_lower']:.6f}, {m['paired_upper']:.6f}]\\\\")
    tex += [r'\bottomrule\end{tabular}\end{center}',
        f"The source-selected simple reference chooses mean residual, tuned global correction or summed-single kernels by inner MSE. Pair-product and mixed kernels worsen total error by {100*-models['pair_product']['relative_gain']:.2f}\\% and {100*-models['mixed']['relative_gain']:.2f}\\%; neither passes the registered 5\\% materiality gate. Both intervals span zero. The registered 2,000 crossed state/intervention-weight bootstrap draws have {result['diagnostic_bootstrap_nonempty_draws']:,} nonempty draws; their intervals condition on shared reference profiles and selection, exclude biological/refitting uncertainty, and remain diagnostic. The fixed search takes {result['elapsed_seconds']:.2f} elapsed seconds on one CPU thread, with {result['completion_rss_bytes']/1e6:.1f} MB completion RSS, not a measured peak.",
        r'\begin{figure}[ht]\centering\includegraphics[width=.82\linewidth]{evidence/joint_results.pdf}\caption{Pair-specific kernels fail the joint development gate against a source-selected strong simple reference. The dashed line marks the registered 5\% screen.}\end{figure}',
        r'Metadata-balanced joint development is feasible here, superseding the earlier single-row lexicographic feasibility check as a development assessment. All outcomes remain exposed. This result does not supply independent joint confirmation, new independently initiated cultures, or enough sealed states to estimate biological uncertainty. No pair head, learned state encoder, risk policy, acquisition or scaling component earns advancement.']
    (args.output / 'joint_results.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')
    write_json(args.output / 'decision.json', dict(status='completed_exposed_joint_development',
        pair_product='failed: 0.87% worsening', mixed='failed: 0.16% worsening',
        supported='Retain strong shared corrections and the previously earned narrow fixed dose interpolation',
        independent_confirmation='unevaluable: all outcomes exposed; independent cultures and multiple sealed states/global pair panels missing',
        next='Qualify a licensed independent crossed pair/state corpus or obtain the specified prospective screen, estimate biological covariance before powering confirmation',
        all_plan_scientific_requirements_complete=False,
        sources={str(source):sha256(source), str(args.cycle/'qualification-v1/qualification.json'):sha256(args.cycle/'qualification-v1/qualification.json')},
        environment=environment(), resource_accounting=dict(elapsed_seconds=result['elapsed_seconds'], completion_rss_bytes=result['completion_rss_bytes'], peak_measured=False)))


if __name__ == '__main__':
    main()
