"""Report every registered scale/panel comparison, including favorable subsets."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opal.artifacts import read_json, write_json, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cycle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=read_json(args.cycle/'run-v1/results.json');rows=result['comparisons']
    get=lambda study,panel,model:next(r for r in rows if r['study']==study and r['panel']==panel and r['model']==model and r['subset']=='all_occurrences')
    tracks=['norman-count/random','norman-count/partner','wessels-count/random','wessels-count/partner']
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for ax,model in zip(axes,['pair_product','mixed']):
        values=[100*get(t,'all_genes',model)['relative_gain'] for t in tracks]
        ax.bar(range(4),values,color=['#167d8d' if x>0 else '#b57956' for x in values]);ax.axhline(5,color='#aa5835',linestyle='--',linewidth=1);ax.axhline(0,color='#333333',linewidth=.6)
        ax.set_xticks(range(4),['Norman random','Norman partner','Wessels random','Wessels partner'],rotation=25,ha='right');ax.set_title(model.replace('_',' ').capitalize());ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('All-gene count-scale MSE improvement (%)');fig.tight_layout()
    for ext in ['svg','pdf']:fig.savefig(args.output/f'panel-scale-results.{ext}')
    plt.close(fig)
    coverage=read_json(args.cycle/'run-v1/partner-coverage.json')
    concentration={}
    for t in tracks:
        panel=read_json(args.cycle/f"run-v1/{t.replace('/','-')}-panels.json")
        g=len(next(r['gene_indices'] for r in panel if r['panel']=='all_genes'));n=len(next(r['gene_indices'] for r in panel if r['panel']=='strong'))
        concentration[t]=get(t,'strong','selected_simple')['mse']*n/g/get(t,'all_genes','selected_simple')['mse']
    tex=[r'\subsection{Matched scale sensitivity and source-selected reporting panels}',
        r'A further registered development supplement addresses scale and reporting gaps without enlarging the neural menu. Metadata qualification verifies that count/log2 banks share raw-source hashes, genes, target/pair identities, original development/query roles, normalizers and QC. Count models reuse the exact three random and three partner outer folds per source, with the original nested source partitions. The fixed menu retains additivity, mean residual, 24 tuned global ridge/RBF settings and 18 settings each for summed-single, pair-product and mixed kernels. The simple reference is source-selected, and neither historical evaluator nor calibration outcomes are accessed \citep{opalpanel2026}.',
        r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',r'Count-scale track & Simple MSE & Pair MSE & Mixed MSE & Pair / mixed gain\\\midrule']
    for t in tracks:
        a,b,c=[get(t,'all_genes',m) for m in ['selected_simple','pair_product','mixed']]
        tex.append(f"{t.replace('-count','').replace('/',' ')} & {a['mse']:.6f} & {b['mse']:.6f} & {c['mse']:.6f} & {100*b['relative_gain']:.2f} / {100*c['relative_gain']:.2f}\\%\\\\")
    tex += [r'\bottomrule\end{tabular}\end{center}',
        r'Norman count-scale gains are positive, unlike its log2 results, so direction is scale-sensitive. The primary gates still fail: Norman random mixed correction gains 9.19\%, with paired MSE-reduction interval $[-0.011700,0.273082]$; Norman partner pair correction gains 9.18\%, with interval $[-0.113894,0.496702]$. The other count pair/mixed intervals also span zero. Wessels count corrections worsen the reference. Raw count and log2 MSE units cannot be compared directly, and neither scale replaces the original endpoint.',
        r'Each outer source defines a strong panel by its top 10\% mean absolute pair effects. A separate low-single residual-enriched proxy ranks source residual magnitude among genes whose source single magnitudes do not exceed the median. Query pair labels cannot select either mask. All models use the same fold masks; these are diagnostic source-derived panels, not validated emergent gene sets. All nine protocols, all three panels and one/two-held-target subsets are retained.',
        f"On Norman count partner folds, the residual proxy gives pair-product and mixed gains of {100*get('norman-count/partner','residual_proxy','pair_product')['relative_gain']:.2f}\\% and {100*get('norman-count/partner','residual_proxy','mixed')['relative_gain']:.2f}\\%, with positive conditional lower bounds. These favorable subsets do not override failed all-gene gates. The top source-selected 10\\% strong genes contribute {100*concentration['norman-count/random']:.2f}\\% and {100*concentration['norman-count/partner']:.2f}\\% of Norman count reference error in random and partner scoring, illustrating endpoint concentration.",
        r'\par\begin{figure}[ht]\centering\includegraphics[width=.9\linewidth]{evidence/panel_scale_results.pdf}\caption{Count-scale point gains on Norman do not pass the paired-uncertainty gate. Wessels remains unfavorable. Full panels, intervals and target subsets are retained.}\end{figure}\par',
        r'Identity-only recovery of the original partner assignments separates 120 one-held-target and 28 two-held-target Norman occurrences, versus 58 and 11 on Wessels. These correspond to only 88 and 40 unique conditions: 60 Norman pairs and 29 Wessels pairs recur in two folds. Pairwise intersections and every coverage count are saved; repeated occurrences do not supply independent replication. Opposite technical-half diagnostics retain complete shared reference matrices and report residual MSE, direction cosine and the main/residual/cross-term identity on all panels. Frozen QC, full-reference configuration selection and source-selected masks prevent a biological-independence claim.',
        f"Count fitting takes {sum(read_json(args.cycle/f'run-v1/{o}-count/execution.json')['elapsed_seconds'] for o in ['norman','wessels']):.2f} elapsed seconds; the complete fitting and diagnostic runner takes {result['elapsed_seconds']:.2f} seconds on one CPU thread, with {result['completion_rss_bytes']/1e6:.1f} MB completion RSS, not a measured peak. No pair component advances on these primary gates. Independent cultures and sealed state/pair confirmation remain missing."]
    (args.output/'panel_scale_results.tex').write_text('\n\n'.join(tex)+'\n',encoding='utf-8')
    favorable=[r for r in rows if r['model'] in ['pair_product','mixed'] and r['subset']=='all_occurrences' and r['panel']!='all_genes' and r['relative_gain']>=.05 and r['lower']>0]
    write_json(args.output/'decision.json',dict(status='completed_scale_panels_and_partner_reporting',
        primary_count_gates=result['primary_count_pair_gates'],favorable_diagnostic_subsets=favorable,
        interpretation='Norman point gains change with scale; all count primary gates fail uncertainty. Favorable residual-proxy subsets remain diagnostic and cannot replace primary failures.',
        concentration=concentration,partner_coverage={k:{n:v[n] for n in ['occurrences','unique_conditions','coverage_histogram','intersections']} for k,v in coverage.items()},
        components_advanced=[],independent_confirmation=False,all_plan_scientific_requirements_complete=False,
        next='Independent culture/reference measurements and sealed crossed state/pair data; no neural enlargement from panel sensitivities',
        elapsed_seconds=result['elapsed_seconds'],completion_rss_bytes=result['completion_rss_bytes'],
        result_sha256=sha256(args.cycle/'run-v1/results.json'),report_code_sha256=sha256(__file__)))


if __name__=='__main__':main()
