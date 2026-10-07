"""Generate the UPR transfer report from preserved frozen predictions and scores."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opal.artifacts import read_json, write_json, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=read_json(args.cycle/'evaluation-v1/results.json')
    freeze=read_json(args.cycle/'freeze-v1/freeze.json')
    prep=read_json(args.prepared/'preparation.json')
    models=result['models'];pair=models['selected_pair'];simple=models['selected_simple'];additive=models['additive']
    with (args.output/'comparisons.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','mse','relative_gain','paired_lower','paired_upper','diagnostic_materiality'])
        writer.writeheader()
        for name,record in models.items():writer.writerow(dict(model=name,**{k:record[k] for k in writer.fieldnames[1:]}))
    names=['additive','mean_residual','global_correction','single_sum','pair_product','mixed']
    fig,ax=plt.subplots(figsize=(7,3.8));ax.barh(names,[models[k]['mse'] for k in names],color=['#657b83' if k=='additive' else '#377c95' for k in names])
    ax.set_xlabel('Mean squared error, matched log2 effects');ax.invert_yaxis()
    ax.set_title('Norman to Adamson: additivity is strongest on these nine conditions')
    fig.tight_layout();fig.savefig(args.output/'adamson_transfer_results.svg');fig.savefig(args.output/'adamson_transfer_results.pdf');plt.close(fig)
    text=[r'\subsection{Frozen external UPR transfer feasibility}',
        r'A metadata-qualified local source extends the audit beyond the exposed development banks. Original Adamson UPR counts and full guide barcodes provide matched controls, three exact genetic singles and three pair identities in each of three chemical backgrounds \citep{adamson2016}. Publisher methods map capture groups to tunicamycin, thapsigargin and DMSO. The ATF6/PERK/IRE1 triple is genuine, but one triple identity supports inventory rather than a general higher-order benchmark. The backgrounds alias capture libraries, and independent culture initiation is unverified.',
        f"The fixed observed-single experiment aligns {result['aligned_genes']:,} genes using source-selected Norman genes and exact query Ensembl metadata. It freezes the Norman library normalizer and source-only three-fold selection over the existing mean, global, summed-single, pair-product and mixed menus. Source-selected {freeze['selected_simple_family'].replace('_',' ')} and {freeze['selected_pair_family']} predictions are saved before evaluator access. All nine pair/background outcomes are scored once locally. This is a public-source feasibility pilot, not sealed, powered joint pair/state confirmation.",
        r'\begin{center}\small\begin{tabular}{lr}\toprule Model & MSE\\\midrule',
        *[f"{name.replace('_',' ')} & {models[name]['mse']:.8f}\\\\" for name in names],
        r'\bottomrule\end{tabular}\end{center}',
        f"The frozen mixed candidate gains {100*pair['relative_gain']:.2f}\\% relative to the source-selected correction, with diagnostic paired MSE-reduction interval $[{pair['paired_lower']:.8f},{pair['paired_upper']:.8f}]$. Its lower bound includes zero, so the materiality gate fails. Additivity has MSE {additive['mse']:.8f}, lower than the source-selected correction's {simple['mse']:.8f}; a favorable mixed-versus-correction point estimate cannot establish value over this simpler retained competitor. No post-query family replacement or architecture advancement occurs.",
        f"Preparation takes {prep['elapsed_seconds']:.2f} seconds with {prep['peak_working_set_bytes']/1e9:.2f} GB process peak working set. Frozen source fitting takes {freeze['elapsed_seconds']:.2f} seconds with {freeze['completion_rss_bytes']/1e6:.1f} MB completion RSS. Both use one CPU thread and remain below the registered budgets. The crossed target/background bootstrap retains {result['bootstrap_nonempty_draws']} nonempty draws from 2,000; it holds reference means and selection fixed and supplies no biological confidence interval. CRISPRa-to-CRISPRi, study and chemical/capture shifts remain confounded.",
        r'\begin{figure}[ht]\centering\includegraphics[width=.9\linewidth]{evidence/adamson_transfer_results.pdf}\caption{All retained competitors on the same nine external conditions. Additivity is stronger; the frozen mixed candidate fails the paired-uncertainty gate.}\end{figure}']
    (args.output/'adamson_transfer_results.tex').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    write_json(args.output/'decision.json',dict(status='completed_external_feasibility',primary_gate=pair['diagnostic_materiality'],
        selected_simple=freeze['selected_simple_family'],selected_pair=freeze['selected_pair_family'],pair_comparison=pair,
        additive_is_stronger=additive['mse']<simple['mse'],components_advanced=[],independent_joint_confirmation=False,
        interpretation='No transfer component advances. Additivity is stronger; source-selected mixed point gain fails diagnostic uncertainty. One genuine triple is inventory only.',
        costs={k:prep[k] for k in ['elapsed_seconds','peak_working_set_bytes']},source_fitting_seconds=freeze['elapsed_seconds'],
        next='Independent cultures and multiple sealed pretreatment states, matched controls/singles and a globally held pair panel with enough identities for state-level power.',
        source_hashes={'results':sha256(args.cycle/'evaluation-v1/results.json'),'freeze':sha256(args.cycle/'freeze-v1/freeze.json'),'preparation':sha256(args.prepared/'preparation.json')},
        report_code_sha256=sha256(__file__)))


if __name__=='__main__':main()
