"""Source-backed identity results and final remaining-gate ledger."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opal.artifacts import read_json, write_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--cycle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--provenance', type=Path, help='Reuse a preserved metadata audit when regenerating a score report')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    studies = {t: read_json(args.cycle / f'run-v1/{t}/results.json') for t in ['identity', 'target_class']}
    provenance = read_json(args.provenance or args.cycle / 'provenance-v2/audit.json')
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    for ax, (track, record) in zip(axes, studies.items()):
        labels = ['simple', 'annotation', 'shuffled']; metrics = record['comparisons']
        ax.bar(range(3), [100 * metrics[k]['relative_gain'] for k in labels], color=['#99958e', '#167d8d', '#b57956'])
        ax.axhline(5, color='#aa5835', linewidth=1, linestyle='--'); ax.axhline(0, color='#333333', linewidth=.6)
        ax.set_xticks(range(3), ['selected reference', 'annotations', 'shuffled'], rotation=18)
        ax.set_title('Identity holdout' if track == 'identity' else 'Shared-target holdout', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Relative MSE improvement (%)'); fig.tight_layout()
    for ext in ['svg', 'pdf']:
        fig.savefig(args.output / f'identity-results.{ext}')
    plt.close(fig)
    tex = [r'\subsection{Publisher annotation transfer and recovered provenance}',
        r'A subsequent metadata qualification tests inherited publisher target/pathway annotations from the corrected CC BY 4.0 release. Four identity folds globally exclude every held compound response at all doses and states. A separate four-fold shared-target track takes transitive connected components of comma-separated target classes before balancing folds, preventing a multi-target compound from bridging source and held target labels. Source-only categorical vocabularies retain explicit unknown/missing masks. The common population has 288 profiles from 24 compounds, three known lines and four doses. These annotations are permitted external information, not chemical structures or independently verified mechanisms \citep{opalidentity2026}.',
        r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',
        r'Track & Reference MSE & Annotation MSE & Gain & Paired reduction interval\\\midrule']
    for track, record in studies.items():
        a = record['comparisons']['annotation']; b = record['comparisons']['simple']
        label = 'Identity' if track == 'identity' else 'Shared target'
        tex.append(f"{label} & {b['mse']:.6f} & {a['mse']:.6f} & {a['relative_gain']*100:.2f}\\% & [{a['paired_lower']:.6f}, {a['paired_upper']:.6f}]\\\\")
    tex += [r'\bottomrule\end{tabular}\end{center}',
        f"Neither track passes the 5\\% screen with a positive conditional lower bound. Shuffled annotations have MSE {studies['identity']['comparisons']['shuffled']['mse']:.6f} and {studies['target_class']['comparisons']['shuffled']['mse']:.6f}, respectively. Each fold selects among nine simple/class-mean references, 24 ridge/kernel annotation maps and the same 24-map shuffled search, using whole source identity/component folds only. Both tracks together use {sum(r['elapsed_seconds'] for r in studies.values()):.2f} CPU seconds, with completion working set at most {max(r['completion_rss_bytes'] for r in studies.values())/1e6:.1f} MB. Whole-compound bootstrap draws condition on shared controls and known lines; biological, refitting and multiplicity uncertainty are excluded. No annotation encoder or unseen-compound production route earns advancement.",
        r'\begin{figure}[ht]\centering\includegraphics[width=.82\linewidth]{evidence/identity_results.pdf}\caption{Publisher annotations fail both identity and shared-target development gates; the dashed line marks the registered 5\% screen.}\end{figure}',
        f"A separate primary-metadata audit aligns all {provenance['matched_cells']:,} cell identities with no condition or annotation mismatches, without reading expression outcomes. It recovers one original sample identifier, two replicate labels per line, and {provenance['line_culture_control_strata']} line/culture control strata. Original culture labels alias existing assigned plate strata, supplying no independently initiated culture evidence beyond the randomized treatment wells already documented. The count matrix is not refitted. Pair responses, multiple sealed states, biological covariance for empirical power, and independent risk evaluation remain absent. Recovery of metadata nomenclature does not create fresh confirmation."]
    (args.output / 'identity_results.tex').write_text('\n'.join(tex) + '\n')
    write_json(args.output / 'decision.json', dict(status='completed_supported_annotation_and_provenance_work',
        identity='failed: 2.11% gain, diagnostic interval crosses zero', target_class='failed: 3.21% worsening',
        model_advancement='none; retain simple references and previously earned narrow dose interpolation',
        independent_culture_pair_attribution='unevaluable; restored primary metadata has no pair measurements and does not document independent culture initiation',
        sealed_joint_confirmation='unevaluable; exposed sources and no multiple sealed states/global pairs',
        structural_descriptor_scaffold_ood='unqualified; publisher annotations do not substitute for structure/scaffold provenance',
        reliability_acquisition_scaling='not earned under the plan gates',
        next='Qualify an independent crossed pair/state corpus or obtain the specified prospective screen; estimate biological covariance before powering confirmation',
        sources={str(args.cycle / f'run-v1/{t}/results.json'): sha256(args.cycle / f'run-v1/{t}/results.json') for t in studies},
        all_plan_scientific_requirements_complete=False))


if __name__ == '__main__':
    main()
