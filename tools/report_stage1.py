"""Generate successor tables/figures/decision from saved development artifacts."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opal.artifacts import read_json, write_json, sha256
from tools.run_stage1 import ROOT, BASELINE, load_data
from opal.state.contracts import registered_split, validate_response_sources


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cycle', type=Path, required=True); p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args(); args.cycle = args.cycle.resolve()
    output = args.output or args.cycle / 'report-v2'; output.mkdir(parents=True, exist_ok=False)
    records = {}
    for file in sorted((args.cycle / 'run-v1').glob('*/*/results.json')):
        records[file.parent.parent.name + '/' + file.parent.name] = (file, read_json(file))
    for file in sorted((args.cycle / 'main-v1').glob('*/main/results.json')):
        records[file.parent.parent.name + '/main'] = (file, read_json(file))
    if len(records) != 7:
        raise ValueError('All five pair protocols and two baseline-only main studies must finish')
    rows, decisions, neural_diagnosis, measurement = [], {}, {}, []
    for name, (file, record) in records.items():
        models = record['models']
        for label, result in models.items():
            if label == 'baseline_only_mean_plus_offset':
                # Original saved arithmetic ratio crosses information contracts.
                # Keep raw artifacts; explicitly exclude it from model ranking.
                rows.append(dict(study=name, model=label, mse=result['mse'], relative_gain='', lower='', upper='', gate='not_comparable_information_contract'))
                continue
            lower, upper = result['diagnostic_95_interval']
            rows.append(dict(study=name, model=label, mse=result['mse'], relative_gain=result['relative_gain'], lower=lower, upper=upper,
                             gate='pass_development_only' if result['development_gate'] else 'fail_development_screen'))
        eligible = {k: v for k, v in models.items() if k not in {'additive', 'selected_simple', 'baseline_only_mean_plus_offset', 'mean_residual',
                    'shuffled_identity', 'shuffled_context', 'shuffled_descriptor', 'shuffled_control', 'mean', 'global_correction', 'single_sum', 'measured_effect_only', 'no_context'}}
        decisions[name] = {'continuation_candidates': [k for k, v in eligible.items() if v['development_gate']],
                           'seconds': record['seconds'], 'peak_working_set_bytes': record['process_peak_working_set_bytes'],
                           'selected_simple_mse': models['selected_simple']['mse'], 'status': record['status']}
        if 'neural_pair' in models:
            diagnostic = []
            for fold in record['folds']:
                for family in ['shared', 'pair', 'original_network', 'original_sum_network']:
                    r = fold['selection'][family]; trajectory = r['trajectories']
                    diagnostic.append(dict(fold=fold['fold'], family=family, config=r['config'],
                        outer_training_pairs=len(fold['train']), inner_training_pairs=[len(f['train']) for f in fold['inner']],
                        mean_member_training_mse=float(np.mean([t['training_mse'] for t in trajectory])),
                        inner_ensemble_mse=r['mse'], refit_epoch=r['refit_epochs'],
                        best_epoch_at_cap=sum(t['stopped_epoch'] == r['config']['epochs'] for t in trajectory),
                        members=len(trajectory)))
            neural_diagnosis[name] = diagnostic
        for fold in record.get('folds', []):
            for r in fold.get('technical_cross_reference') or []:
                measurement.append(dict(study=name, fold=fold['fold'], **r))
            r = fold.get('estimated_single_decomposition')
            if r:
                measurement.append(dict(study=name, fold=fold['fold'], status='baseline-only source-mean pair accounting; query singles evaluator-only', **r))
    supplements = {}
    for file in sorted((args.cycle / 'identity-control-v1').glob('*/*/results.json')):
        name = file.parent.parent.name + '/' + file.parent.name; r = read_json(file); supplements[name] = r
        lower, upper = r['diagnostic_95_interval']
        rows.append(dict(study=name, model='shuffled_identity_rows', mse=r['mse'], relative_gain=r['relative_gain'],
                         lower=lower, upper=upper, gate='amended_negative_control'))
    if len(supplements) != 5:
        raise ValueError('Genuine shuffled-identity controls must finish')
    with (output / 'comparisons.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Only supported biological OOD isolation manifests, not invented confirmation.
    data = load_data(args.data_root, 'emeraldbay-pilot-v1')
    pair_keys = sorted(set(map(tuple, np.sort(data['pairs'], 1))))
    held_pairs = pair_keys[::4]; held_group = str(sorted(set(data['groups']))[-1])
    train, query, guard = registered_split('joint_pair_context', data['groups'], pairs=data['pairs'],
                                          held_groups=[held_group], held_pairs=held_pairs)
    stage_rows = [dict(biological_group=str(data['groups'][i]), interventions=[str(t) for t in data['targets'][data['pairs'][i]]],
                      dose='exact dose encoded in intervention', response_kind='pair') for i in train]
    held_names = [[str(t) for t in data['targets'][list(pair)]] for pair in held_pairs]
    validate_response_sources('joint_pair_context', {'fitting': stage_rows, 'preprocessing': stage_rows,
                              'pretraining': [], 'calibration': []}, held_groups=[held_group], held_pairs=held_names)
    write_json(output / 'ood-isolation.json', {'status': 'metadata_qualified_exploratory_manifest_only', 'task': 'joint_pair_context',
        'source_rows': train.tolist(), 'query_rows': query.tolist(), 'guard_rows': guard.tolist(),
        'held_groups': [held_group], 'globally_held_pair_identities': held_names,
        'outcome_access': False, 'fit_executed': False, 'sealed': False,
        'reason': 'Exposed one-study source; sparse joint query. This is software isolation, not independent confirmation.',
        'query_reference_adaptation': 'Legacy gene panel used all declared control contexts; query controls and exact singles permitted',
        'missing_for_confirmation': ['multiple sealed pretreatment states', 'globally held pair outcomes', 'independent culture provenance',
                                     'declared query reference adaptation', 'power at biological state/culture units']})
    sources = {str(file.relative_to(ROOT)): sha256(file) for file, _ in records.values()}
    sources.update({str(file.relative_to(ROOT)): sha256(file) for file in (args.cycle / 'identity-control-v1').glob('*/*/results.json')})
    all_candidates = {k: v['continuation_candidates'] for k, v in decisions.items() if v['continuation_candidates']}
    summary = {'baseline': BASELINE, 'version': '0.1.2.dev1', 'evidence_level': 'exposed development',
        'studies': decisions, 'continuation_candidates': all_candidates,
        'stage2': 'experimental development only if a registered candidate earned its gate; no independent confirmation',
        'independent_joint_confirmation': 'unevaluable', 'biological_interaction_recovery': 'unevaluable',
        'chemical_identity_descriptor_information': 'unevaluable; only numeric doses qualified',
        'reliability_active_scaling': 'not advanced; no useful independently evaluated calibration or acquisition policy',
        'source_sha256': sources,
        'amendments': ['Retained invariant one-hot renaming; added row permutation after explicit registration.',
                       'Baseline-only pair arithmetic ratios against observed-single references are excluded from ranking.',
                       'Shuffled-control main comparator is a fixed-selection sensitivity, not separately retuned.',
                       'Export-v1 preserved: float32 PCA replay drift on Norman. Export-v2 uses registered float64 conversion; scored predictions replay exactly.',
                       'Report-v1 interrupted on relative path bookkeeping; preserved and superseded by report-v2, without refitting.'],
        'total_study_seconds': sum(v['seconds'] for v in decisions.values()) + sum(v['seconds'] for v in supplements.values()),
        'resource_limits': 'CPU float64, one thread per runner, reused dependencies; no paid compute',
        'next_data': 'Crossed screen with exact matched singles/controls, globally withheld pairs, multiple sealed pretreatment states and independently initiated culture IDs'}
    write_json(output / 'decision.json', summary)
    write_json(output / 'neural-diagnosis.json', {'records': neural_diagnosis,
        'interpretation': 'Compare restored member training error with inner ensemble validation error cautiously: ensemble averaging differs. Epoch-cap hits do not prove convergence; this closes only the registered 120-update menu. Inner training counts are recorded, not an independently varied sample-size experiment.'})
    write_json(output / 'measurement-diagnostics.json', {'records': measurement,
        'independent_culture_validation': None,
        'missing': 'Independent cultures with disjoint reference/evaluation estimates and retained culture/state/condition provenance',
        'interpretation': 'Main + residual + cross term equals total MSE. Technical halves share fixed QC and preprocessing, and preserve whole shared-control/single matrices; no biological ceiling is claimed.'})
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for ax, (name, (_, record)) in zip(axes.flat, [(k, v) for k, v in records.items() if not k.endswith('/main')]):
        labels = ['context', 'identity', 'descriptor', 'pair_product', 'mixed', 'neural_shared', 'neural_pair']
        labels = [k for k in labels if k in record['models']]
        values = [100 * record['models'][k]['relative_gain'] for k in labels]
        ax.barh(labels, values, color=['#397c82' if v >= 5 else '#87919a' for v in values])
        ax.axvline(5, color='#b46b49', linestyle='--', label='5% screen'); ax.axvline(0, color='black', linewidth=.5)
        ax.set_title(name.replace('-pilot-v1', '')); ax.set_xlabel('Relative MSE improvement over selected simple (%)')
    axes.flat[-1].axis('off'); axes.flat[-1].text(0, .7, 'Exposed development only\n\nWhole-state or crossed-target diagnostic uncertainty.\nNo independent cultures, sealed confirmation,\nor validated OOD risk.', va='top')
    fig.savefig(ROOT / 'images/stage1_results.svg'); fig.savefig(ROOT / 'paper/evidence/stage1_results.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for ax, (name, (_, record)) in zip(axes, [(k, v) for k, v in records.items() if k.endswith('/main')]):
        labels = ['mean', 'selected_simple', 'learned', 'shuffled_control']
        ax.bar(labels, [record['models'][k]['mse'] for k in labels], color=['#aaa59e', '#397c82', '#b46b49', '#87919a'])
        ax.set_title(name.split('/')[0]); ax.tick_params(axis='x', rotation=20); ax.set_ylabel('Baseline-only single MSE')
    fig.savefig(ROOT / 'images/stage1_main.svg'); fig.savefig(ROOT / 'paper/evidence/stage1_main.pdf'); plt.close(fig)
    # The short manuscript amendment is generated from actual saved model scores.
    tex = [r'\section{Registered Stage 1 successor development}',
           r'The local preregistration and evidence archive are identified separately from historical findings \citep{opalstage12026}.',
           'The successor preserves commit ' + r'\texttt{84de7e3}' + ' as the historical preparation. All reused outcomes are exposed development; no independent confirmation is claimed. Feature preprocessing and model selection use only source rows and declared measured references. A two-seed ensemble is the neural selection unit, with inner-only stopping, 16 configurations per shared/pair family, 120 updates, and unchanged original 80-update schedule controls.',
           r'\begin{table}[htbp]\centering\small\begin{tabular}{llrr}\toprule',
           r'Source / split & Candidate & MSE & Gain (\%)\\\midrule']
    for name, (_, record) in records.items():
        selected = record['models']['selected_simple']; tex.append(name.replace('_', r'\_') + r' & Selected simple & ' + f"{selected['mse']:.6f}" + r' & 0.00\\')
        for label in (['context', 'identity', 'pair_product', 'mixed', 'neural_shared', 'neural_pair'] if not name.endswith('/main') else ['learned']):
            if label in record['models']:
                r = record['models'][label]
                tex.append(name.replace('_', r'\_') + ' & ' + label.replace('_', r'\_') + f" & {r['mse']:.6f} & {100*r['relative_gain']:.2f}" + r'\\')
    tex += [r'\bottomrule\end{tabular}\caption{New exposed development. Gains are against inner-selected simple competitors on the same information contract. Baseline-only main prediction is separate from observed-single pair correction. Full paired uncertainty and every control are in the saved comparison CSV.}\end{table}',
            'The initial materiality rule requires five percent relative improvement, with a positive paired diagnostic lower bound for continuation. These fixed-profile state/target resampling intervals do not establish independent biological precision. One-hot identity renaming is mathematically invariant; an explicitly registered supplementary row permutation breaks alignment. Numeric dose descriptors do not test chemical identity.',
            'Disjoint technical-cell halves fit shared correction with whole shared-reference matrices retained. Opposite-half residual diagnostics and main/residual/cross-term accounting are measurement sensitivities, not independent cultures or biological interaction recovery. Baseline-only pair ratios against observed-single references are excluded from rankings. Original results remain intact.',
            r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{evidence/stage1_results.pdf}\caption{Information, pair attribution and neural changes against retained simple competitors. The dashed line is the registered development materiality screen.}\end{figure}',
            r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{evidence/stage1_main.pdf}\caption{Separate baseline-only main prediction with fixed source identity support. Shuffled-control results are fixed-selection sensitivities.}\end{figure}',
            'Independent joint transfer remains unevaluable. It requires globally held pair identities, multiple sealed pretreatment states, independently initiated cultures with matched controls and singles, and declared query-reference adaptation. Reliability, acquisition and scaling remain conditional.']
    (ROOT / 'paper/evidence/stage1_results.tex').write_text('\n\n'.join(tex) + '\n', encoding='utf-8')
    write_json(ROOT / 'paper/evidence/stage1_manifest.json', {'sources': sources, 'decision_sha256': sha256(output / 'decision.json'),
               'figure_sha256': {name: sha256(ROOT / name) for name in ['images/stage1_results.svg', 'images/stage1_main.svg', 'paper/evidence/stage1_results.pdf', 'paper/evidence/stage1_main.pdf']}})
    print(summary)


if __name__ == '__main__':
    main()
