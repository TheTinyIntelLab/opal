"""Add only evidence-backed development results, preserving the previous PDF."""
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import read_json,sha256,write_json
from prepare_information import OUT,LOCAL,resource_check


def run():
    resource_check(6000000)
    original=LOCAL/'paper-before-information';original.mkdir(exist_ok=False)
    for name in ['main.pdf','main.tex','build_manifest.json']:
        shutil.copy2(ROOT/'paper'/name,original/name)
    summary=read_json(OUT/'development/summary.json')
    lines=[r'\section{Measured-single variance information: bounded development study}',
        r'A subsequent frozen experiment asks whether projected variance of measured single-cell singles adds held-partner mean-response information. Both exposed PertSpectra contexts retain their original ordered 512 genes and log2 library-10000 aggregate endpoint. Control-cell-only projections, training-only output bases and grouped nested ridge selection compare means, capacity-matched squared means, real variance and compatible-stratum shuffled variance, each with twelve rank/penalty configurations plus zero correction. A shared ten-cell reference draw is a separate precision sensitivity. No pair cells enter information-model features; query singles remain permitted, so partner holdout is not unseen-intervention prediction.',
        r'\begin{table}[htbp]\centering\small',
        r'\begin{tabular}{llrrrrr}\toprule',
        r'Context & Split & Base & Mean & Capacity & Variance & Shuffle \\ \midrule']
    for row in summary['aggregates']:
        if row['precision']:continue
        context='TNF-positive' if row['source']=='TNFA-positive' else 'TNF-negative'
        split='Pair' if row['split']=='global_pair' else 'Partner'
        m=row['mse'];values=' & '.join(f'{m[a]:.8f}' for a in ['Base','mean','capacity_mean','variance','shuffle'])
        lines.append(context+' & '+split+' & '+values+r' \\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{Exposed development only, equal-pair within stratum and equal-stratum MSE. Original populations are retained; technical sample labels are not independent cultures. TNF-negative partner heads all use the declared Base fallback because nested selection lacks support.}\end{table}',
        r'The primary TNF-positive partner variance gain over Base is 0.231163\%, while both equally tuned mean controls perform better. Ten-cell partner variance gains are 0.451677\% and zero in the separate contexts. The prespecified administrative gate requires at least 2\% improvement over Base and every matched control in both contexts and the precision sensitivity; it fails. Base is retained, without additional models, seeds, endpoints or sources. TNF-negative partner information is inconclusive because support is insufficient, not because variance information is absent. No fresh-source or independent-culture confirmation was executed.',
        r'The study retains 96 fold records, 160 canonical pairs and 640 pair-stratum profiles, with 2,560 repeated evaluation events. It summarizes 23,883 control/single cells, not independent cultures. All evaluation heads replay within $10^{-10}$ and the existing evaluator rejects changed inputs, units, deployment roles and predictions. The full local Python 3.14 suite passes 282 tests. Fresh training and separate replay profiles are available for Opal and three upstream adaptations; historical native training RAM remains unknown. Zero fully eligible native comparisons are established, and CPU/GPU/input-tier differences preclude an equivalent-workload Pareto claim. Complete CPU certification remains unavailable because bootstrap/tool work and polling losses are not measured. Qualified cultures, dose/reference lineage, custody, empirical biological power and exact-revision remote CI remain dependencies.',
        r'The previous completed model-series evidence and negative results are unchanged. The new information study does not establish biological interaction recovery, calibrated uncertainty, context transfer, prospective acquisition savings or frontier superiority. Compact provenance and every arm/failure are preserved in \texttt{results/opal\_information\_v1}; detailed eligibility and resource scopes accompany its local report.']
    evidence=ROOT/'paper/evidence/information_development.tex';evidence.write_text('\n\n'.join(lines)+'\n')
    main=ROOT/'paper/main.tex';text=main.read_text()
    if '\\input{evidence/information_development}' in text:raise FileExistsError('Addendum already exists')
    main.write_text(text.replace('\\end{document}','\\clearpage\n\\input{evidence/information_development}\n\\end{document}'))
    write_json(OUT/'verification/paper_preservation.json',{'old_pdf':str(original/'main.pdf'),'old_pdf_sha256':sha256(original/'main.pdf'),
        'old_tex':str(original/'main.tex'),'summary_sha256':sha256(OUT/'development/summary.json'),
        'addendum_sha256':sha256(evidence),'manuscript_change':'New development addendum only; prior empirical records unchanged'})
    print('Preserved old manuscript; generated source-hashed development addendum')


if __name__=='__main__':run()
