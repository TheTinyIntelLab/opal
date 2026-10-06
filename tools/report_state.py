"""Generate tables, figures and capability status from compact saved evidence only."""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, write_json, sha256, source_hash

MAIN_RUNS = {'EmeraldBay': 'results/state/emeraldbay-main-v2/main/results.json',
             'sci-Plex 3': 'results/state/sciplex-state-v1/main/results.json'}
PAIR = 'results/state/emeraldbay-pairs-v3/combinations/results.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'paper/evidence')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = {name: read_json(ROOT / path) for name, path in MAIN_RUNS.items()}
    pair = read_json(ROOT / PAIR)
    risk = read_json(ROOT / 'results/state/emeraldbay-diagnostics-v1/reliability.json')
    source = read_json(ROOT / 'results/state/sciplex-source-v1/source_audit.json')
    measurement = read_json(ROOT / 'results/state/sciplex-measurement-v1/results.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), constrained_layout=True)
    for ax, (name, record) in zip(axes, records.items()):
        values = [record['models'][k]['mse'] for k in ['mean', 'selected_simple', 'learned', 'no_state']]
        ax.bar(['Target mean', 'Selected simple', 'Learned state', 'No state'], values,
               color=['#87919a', '#36586d', '#b46b49', '#bbb3aa'])
        ax.set_title(name + ': main-effect context development'); ax.set_ylabel('Condition-mean gene MSE')
        ax.tick_params(axis='x', rotation=25)
        ax.set_ylim(0, max(values) * 1.25)
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * .035, f'{v:.5f}', ha='center', fontsize=8)
    fig.savefig(args.output / 'state_main.pdf'); fig.savefig(args.output / 'state_main.png', dpi=180); plt.close(fig)
    names = ['estimated_additive', 'selected_kernel', 'staged_sum_difference_state', 'frozen_encoder_joint', 'joint', 'no_pretraining_joint']
    labels = ['Estimated additive', 'Selected kernel', 'Staged state', 'Frozen-encoder joint', 'Fine-tuned joint', 'No neural pretraining']
    fig, ax = plt.subplots(figsize=(8.8, 3.5), constrained_layout=True)
    values = [pair['models'][k]['total_mse'] for k in names]
    ax.barh(labels[::-1], values[::-1], color=['#b46b49'] * 4 + ['#36586d', '#87919a'])
    ax.set_xlabel('Pair MSE on the same 54 source-development rows'); ax.set_xlim(0, max(values) * 1.22)
    for i, v in enumerate(values[::-1]):
        ax.text(v + .002, i, f'{v:.5f}', va='center')
    fig.savefig(args.output / 'state_pairs.pdf'); fig.savefig(args.output / 'state_pairs.png', dpi=180); plt.close(fig)
    # LaTeX contains no evaluator arrays. Every number comes from saved JSON.
    lines = [r'\section{Baseline-only state-response development}',
        r'This build tests main-effect transfer before expanding the architecture. Matched-control-relative effects, source-only inner selection and whole-biological-group outer folds define the task. Query treatment outcomes never fit control PCA, target means or response bases. The source control panels permit explicitly declared query-control gene selection; fitted state encodings remain source-only.',
        r'The tested learned response model combines a source-average intervention anchor with a projected source-response head. Source-only control PCA represents the baseline; exact intervention one-hot identity, source single-effect features, and numeric dose are separate input paths. Numeric dose is the only descriptor used here, rather than a chemical structure representation. Both the state width and response rank are 16. Two registered training settings use 80 epochs with regularization 0.01 and 160 epochs with regularization 0.1, each with seeds 41 and 73. Simple state references and the learned heads share source intervention means and permitted query-control gene-panel adaptation, so their comparison tests added representation value within this information contract.',
        r'The corrected scPerturb 1.4 sci-Plex 3 release is publisher-MD5 verified \citep{srivatsan2020state,scperturb2024state}. Its full matrix has 799,317 rows, including unassigned cells and a separate 72-hour A549 block. Restricting to 24-hour human-cell-line conditions, 24 shared exact 1,000-nM intervention identities yield 144 single condition means across 24 line/plate strata and three biological line groups. All plates and replicates of a line remain on one side of each holdout. The 512 human-gene panel uses controls only; count-library normalization uses human genes. Wells receive equal weight, and each effect subtracts its own line/plate-matched vehicle. Plate count is not independent state replication.',
        r'\begin{table}[ht]\centering\normalsize\begin{tabular}{lrrr}\toprule Source & Target mean & Selected simple & Learned state \\ \midrule']
    for name, r in records.items():
        lines.append(f"{name} & {r['models']['mean']['mse']:.6f} & {r['models']['selected_simple']['mse']:.6f} & {r['models']['learned']['mse']:.6f} " + r'\\')
    lines += [r'\bottomrule\end{tabular}\caption{Nested main-effect context development. Each source has its own scale/vocabulary and fixed evaluation population; MSE is not compared between sources. Neither simple-state nor learned-representation comparison passes the registered 5\% gate.}\end{table}',
              r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{evidence/state_main.pdf}\caption{Source-average anchors remain strong. Removed-state and shuffled-input controls use the same supervision and output anchor; they are encoder ablations, not unseen-intervention methods.}\end{figure}']
    lines.append(f"State-conditioned simple references improve on target averaging by {100 * records['EmeraldBay']['state_reference_skill_over_mean']:.3f}\\% on EmeraldBay and {100 * records['sci-Plex 3']['state_reference_skill_over_mean']:.3f}\\% on sci-Plex 3. The learned candidates are respectively {-100 * records['EmeraldBay']['learned_skill_over_selected_simple']:.2f}\\% and {-100 * records['sci-Plex 3']['learned_skill_over_selected_simple']:.2f}\\% worse than selected simple references. These negative development results close the scaling/routing gate for this build; three sci-Plex lines do not support a broad state-generalization conclusion.")
    lines.append(r'All selected sci-Plex 3 interventions use 1,000 nM, so shuffling the numeric dose descriptor is a no-op and yields identical saved learned scores. This control is uninformative about descriptor specificity. These results reject adoption or scaling of the tested state build; they do not establish that all learned state representations, biological context information, or context-dependent interactions are unhelpful.')
    lines.append(f"Nominal sci-Plex duplicate-well profiles have mean replicate-difference MSE {measurement['mean_replicate_difference_mse']:.6f}, compared with a mean selected-simple improvement of {measurement['mean_target_average_minus_selected_simple_mse']:.8f}. The replicate contrast includes plate and biological variation, shared controls and estimation error; it is a measurement-sensitivity comparison, without an identified biological accuracy ceiling or independent-culture precision claim.")
    lines += [r'\subsection{State-dependent residual and training-stage comparisons}',
        r'Only EmeraldBay source-development pair outcomes enter this new experiment; historical pair calibration and evaluator outcomes are not reopened. Inner kernel selection masks query singles and uses the same estimated-single route as outer evaluation. Symmetric sum/difference and product heads train measured-single residuals with the main model fixed. Joint variants retain source-single supervision and independent held-single diagnostics. Frozen and fine-tuned encoders use two residual seeds and equal pair epochs; removing neural single pretraining retains the source-average anchor and bases.',
        r'\begin{table}[ht]\centering\normalsize\begin{tabular}{lrrrr}\toprule Model & Total & Main & Residual & Cross term \\ \midrule']
    for label, name in zip(labels, names):
        v = pair['models'][name]
        lines.append(f"{label} & {v['total_mse']:.5f} & {v['main_effect_mse']:.5f} & {v['residual_mse']:.5f} & {v['error_cross_term']:.5f} " + r'\\')
    lines += [r'\bottomrule\end{tabular}\caption{Pair/context development on one fixed population. Total error equals main plus residual error plus the cross term. Large negative cross terms prevent interpreting good total error as accurate interaction recovery.}\end{table}',
        r'\begin{figure}[ht]\centering\includegraphics[width=0.98\linewidth]{evidence/state_pairs.pdf}\caption{State heads improve their pooled neural counterparts, but none beats the information-matched selected residual kernel. No winner is selected retrospectively from these outer scores.}\end{figure}',
        r'\subsection{Reliability, resources and remaining evidence}',
        r'A risk diagnostic fits separately from two-group calibration and two-group evaluation. At nominal 90\%, two calibration groups cannot produce a finite group-rank radius. Accept-none and accept-all comparisons remain visible; the registered accepted-risk/coverage criterion also fails. This is shared-study development, not fresh risk confirmation. D-SPIN remains a failed transfer of the earlier source radius. Inference returns unavailable intervals and diagnostic support rather than a confidence percentage.',
        r'Three subset orders retune simple references at each context budget with fixed query conditions. These dependent retrospective curves and hypothetical condition-assay costs do not establish prospective acquisition value. A new uncertainty-based acquisition extension is withheld after failed gates. Native CPA/GEARS results remain information-scoped historical comparisons; full-gene PertAdapt remains unscored. A separately identified 256-token synthetic adapter probe passes float32, float16 and bfloat16 forward/backward at batches one and two, without proving full-gene memory fit. A Linux/NCCL system with full gene alignment and an initial 48-GB float32 profiling budget is a documented profiling target rather than an established full-gene resource requirement.',
        r'Two-seed portable NumPy exports preserve normalization, gene order, exact-condition records, supporting reference identities and all heads. At the state-experiment cutoff, the suite passed 115 tests. Python and CLI routes validate compatible previously unregistered baselines, pair swapping and one-time baseline reconstruction outside the checkout in an isolated installation. Engineering completion does not establish independent confirmation or reliable inference. A replicated, randomized shared-intervention screen with separate sealed confirmation and risk-calibration blocks remains required.']
    (args.output / 'state_results.tex').write_text('\n\n'.join(lines) + '\n', encoding='utf-8')
    ledger = {'schema': 'opal-capability-ledger-v1', 'public_name': 'Opal',
        'main_context_transfer': {'implementation': 'complete', 'source_qualification': 'chemical development qualified',
            'development_evaluation': 'complete; 5% main-state and learned representation gates failed on both sources',
            'independent_confirmation': 'not established', 'reliable_inference': 'unavailable'},
        'pair_context_transfer': {'implementation': 'complete; measured/estimated routes and symmetric residuals',
            'source_qualification': 'EmeraldBay development; genetic crossed-context confirmation missing',
            'development_evaluation': 'complete; neural state heads lose to matched selected residual kernel',
            'independent_confirmation': 'earlier D-SPIN frozen procedure failed; no new qualified confirmation',
            'reliable_inference': 'unavailable'},
        'representation': {'implementation': 'complete; separate identity, dose/mask, source-effect and baseline paths',
            'development_evaluation': 'failed material contribution; no scaling or mixture-of-experts experiment authorized by evidence'},
        'training_stages': {'implementation': 'complete', 'development_evaluation': 'staged/frozen-encoder/fine-tuned/no-neural-pretraining compared; no universal pretraining benefit'},
        'reliability': {'implementation': 'complete risk diagnostic with isolated partitions and empty-accept policy',
            'development_evaluation': 'failed finite radius/usefulness criteria', 'independent_confirmation': 'absent', 'reliable_inference': 'unavailable'},
        'reproducibility': {'implementation': 'safe single and two-seed NumPy exports, resumable training, isolated wheel validation',
            'scientific_scope': 'engineering restoration only'},
        'efficiency': {'implementation': 'subset-retuned context budgets plus preserved pair-label curves',
            'development_evaluation': 'dependent retrospective curves; cells/culture budgets and monetary costs unresolved'},
        'acquisition': {'implementation': 'prior label-isolated retrospective policies retained',
            'development_evaluation': 'new risk-based acquisition not extended after failed gates', 'independent_confirmation': 'no prospective selection experiment'},
        'conditional_extensions': {k: 'unsupported; source qualification and separately registered evaluation required'
            for k in ['unknown_intervention_descriptor_only', 'unseen_dose', 'higher_order', 'sequential_treatment', 'single_cell_distributions']},
        'fresh_confirmation_design': 'configs/datasets/crossed-context-screen-spec.json',
        'preserved_failures': ['results/next/cycle-v6/status.json', 'results/state/amendments.json', 'results/state/sciplex-preparation-v1/status.json', 'results/state/method-resources-v1/status.json', 'results/state/portable-v1/failed_pair_reconstruction.json'],
        'historical_pair_evaluator_reopened': False, 'private_website_included': False, 'master_plan_included': False}
    write_json(ROOT / 'results/state/capability_ledger.json', ledger)
    inputs = list(MAIN_RUNS.values()) + [PAIR, 'results/state/emeraldbay-diagnostics-v1/reliability.json',
        'results/state/sciplex-source-v1/source_audit.json', 'results/state/method-resources-v2/results.json',
        'results/state/sciplex-measurement-v1/results.json', 'results/state/amendments.json']
    write_json(args.output / 'state_evidence_manifest.json', {'source_hash': source_hash(), 'evaluator_access': False,
        'inputs': {p: sha256(ROOT / p) for p in inputs},
        'outputs': {p.name: sha256(p) for p in args.output.glob('state_*') if p.suffix in {'.tex', '.pdf', '.png'}},
        'public_name': 'Opal'})


if __name__ == '__main__':
    main()
