"""Generate dose evidence and restore registered interpolation references."""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opal.artifacts import read_json, write_json, sha256
from opal.contextual.contracts import ReferenceBank
from opal.state.models import DoseReference

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cycle', type=Path, required=True); p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--exports', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False); args.exports.mkdir(parents=True, exist_ok=False)
    study = read_json(args.cycle / 'dose-v2/results.json'); precision = read_json(args.cycle / 'precision-v1/precision-design.json')
    rows = study['comparisons']; fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.bar(range(4), [r['relative_gain'] * 100 for r in rows], color=['#777777', '#167d8d', '#167d8d', '#777777'])
    ax.axhline(5, color='#aa5835', linestyle='--', linewidth=1, label='5% development screen')
    ax.axhline(0, color='#333333', linewidth=.6)
    ax.set_xticks(range(4), [f"{r['held_dose_nM']:,} nM\n{r['shift']}" for r in rows])
    ax.set_ylabel('Relative MSE improvement (%)'); ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False); fig.tight_layout()
    for ext in ['svg', 'pdf']:
        fig.savefig(args.output / f'dose-results.{ext}')
    plt.close(fig)
    with (args.cycle / 'precision-v1/precision-sensitivity.csv').open() as f:
        sensitivity = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    for scenario in ['low', 'high']:
        subset = [r for r in sensitivity if r['scenario'] == scenario and r['cultures_per_state'] == '3'
                  and r['held_pairs'] == '24' and r['cells_per_condition'] == '200' and r['assumed_relative_gain'] == '0.1']
        ax.plot([int(r['sealed_states']) for r in subset], [float(r['gate_probability']) for r in subset], marker='o', label=f'{scenario} assumed variance')
    ax.set(xlabel='Sealed states', ylabel='Hypothetical gate probability', ylim=(0, 1.02))
    ax.set_title('Assumed 10% gain, 3 cultures/state, 24 held pairs, 200 cells/condition', fontsize=9)
    ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False); fig.tight_layout()
    for ext in ['svg', 'pdf']:
        fig.savefig(args.output / f'precision-sensitivity.{ext}')
    plt.close(fig)
    bank = ReferenceBank.load(args.prepared); a = bank.arrays
    groups = np.array([bank.metadata['context_records'][str(c)]['biological_state'] for c in a['contexts']])
    names = np.array([str(t).split('@')[0] for t in a['targets']]); doses = np.array([float(str(t).split('@')[1].split('nM')[0]) for t in a['targets']])
    selections = read_json(args.cycle / 'dose-v2/selection.json'); verification = []
    for held in [100, 1000]:
        choice = next(r for r in selections if r['held_dose_nM'] == held)['smooth']['config']
        states, compounds, source_doses, effects = [], [], [], []
        for group in np.unique(groups):
            for target in range(len(names)):
                if doses[target] == held:
                    continue
                ci = np.flatnonzero((groups == group) & a['available'][:, target])
                if not len(ci):
                    raise ValueError('Unavailable source dose')
                states.append(group); compounds.append(names[target]); source_doses.append(doses[target]); effects.append(a['singles'][ci, target].mean(0))
        provenance = dict(endpoint_hours=24, assay='sciPlex3_scRNAseq', normalization=bank.metadata['normalization'],
            source_sha256=bank.metadata['source_sha256'], license=bank.metadata['license'], protocol_sha256=study['protocol_sha256'],
            amendment_sha256=study['amendment_sha256'], evidence_level='exposed development; engineering replay of outer fold, no new validation')
        m = DoseReference(choice).fit(states, compounds, source_doses, effects, a['genes'], provenance)
        export = args.exports / f'dose-{held}'; m.save(export); restored = DoseReference.load(export)
        query_states = np.repeat(np.unique(groups), len(set(names))); query_compounds = np.tile(sorted(set(names)), len(set(groups)))
        started = time.perf_counter()
        prediction = np.array([restored.predict(s, c, held, endpoint_hours=24)['effect'] for s, c in zip(query_states, query_compounds)])
        seconds = time.perf_counter() - started
        with np.load(args.cycle / f'dose-v2/dose-{held}-predictions.npz', allow_pickle=False) as f:
            drift = float(np.max(np.abs(prediction - f['selected_smooth'])))
        if drift > 1e-10:
            raise ArithmeticError('Scored dose predictions did not restore')
        np.savez_compressed(args.exports / f'dose-{held}-expected.npz', states=query_states, compounds=query_compounds, prediction=prediction)
        verification.append(dict(held_dose_nM=held, config=choice, profiles=len(prediction), maximum_absolute_error=drift,
                                 warm_batch_seconds=seconds, reference_bytes=sum(f.stat().st_size for f in export.iterdir())))
    write_json(args.output / 'export-verification.json', verification)
    tex = [r'\subsection{Registered dose transfer and prospective precision design}',
        r'A metadata-qualified extension uses 24 lexicographically selected compounds at four exact doses across A549, K562 and MCF7, at 24 hours. All outcomes are exposed development. An initial plate-specific run stopped on missingness before scoring; an explicit amendment pools matched-control-subtracted plate means within each line. Four global dose holdouts retain 72 profiles each and 512 control-selected genes. Nested source-dose selection compares zero, mean and nearest-dose references with 30 fixed ridge/kernel curves. Query controls may select the gene panel; held-dose effects never enter fitting, scaling, selection or pretraining \citep{opaldose2026}.',
        r'\begin{center}\small\begin{tabular}{rrrrr}\toprule',
        r'Dose (nM) & Reference MSE & Smooth MSE & Gain & Paired reduction interval\\\midrule']
    for r in rows:
        tex.append(f"{r['held_dose_nM']} & {r['reference_mse']:.6f} & {r['smooth_mse']:.6f} & {r['relative_gain']*100:.2f}\\% & [{r['paired_lower']:.6f}, {r['paired_upper']:.6f}]\\\\")
    tex += [r'\bottomrule\end{tabular}\end{center}',
        f"Both interpolation doses pass the 5\\% development screen with positive conditional compound-bootstrap lower bounds; both extrapolation doses fail. Fitting takes {study['elapsed_seconds']:.2f} CPU seconds with final working set {study['rss_bytes']/1e6:.1f} MB. Bootstrap intervals condition on shared controls and three cell lines, excluding biological and selection uncertainty. An experimental known-compound/known-state interpolation reference preserves the selected source-dose curves and rejects extrapolation, unknown identities and endpoint mismatches; restored scored predictions agree within {max(v['maximum_absolute_error'] for v in verification):.2g}. No pair/context or chemical identity transfer claim follows.",
        r'\begin{figure}[ht]\centering\includegraphics[width=.78\linewidth]{evidence/dose_results.pdf}\caption{Source-selected smooth dose curves improve interior doses and fail extrapolation on exposed sci-Plex development.}\end{figure}',
        f"A separate prospective sensitivity calculation evaluates {precision['scenarios']} hypothetical designs with 2,000 Gaussian draws each. Crossed state, culture, target, pair and shared-control effects remain correlated; cell noise alone decreases with cell count. Assumed variance is not estimated from technical halves. The known-variance lower bound is optimistic and does not establish a powered sample size. Four development states with three independently initiated cultures per state form a variance pilot; broad transfer requires multiple sealed states and globally held pairs. The pilot must estimate biological covariance and mean precision before an empirical power calculation.",
        r'\begin{figure}[ht]\centering\includegraphics[width=.78\linewidth]{evidence/precision_sensitivity.pdf}\caption{Hypothetical design sensitivity under declared low and high variance assumptions. These are simulated design probabilities, not empirical confidence in Opal.}\end{figure}']
    (args.output / 'dose_results.tex').write_text('\n'.join(tex) + '\n')
    write_json(args.output / 'decision.json', dict(dose_interpolation='earned narrow experimental fixed-curve route on both interior doses',
        dose_extrapolation='failed', identity_ood='unevaluable: no qualified inductive descriptors', pair_context_confirmation='unevaluable: sealed independent culture/state/pair provenance absent',
        prospective_power='sensitivity complete; empirical variance and sample-size decision missing',
        new_neural_state_or_scaling='not justified', sources={str(f): sha256(f) for f in [args.cycle / 'dose-v2/results.json', args.cycle / 'precision-v1/precision-design.json']},
        next='Recover primary culture/sample mapping and qualify multiple sealed crossed states/global pairs; independently evaluate the narrow dose procedure before use outside development'))
    print(verification)


if __name__ == '__main__':
    main()
