"""Nominal-identity public processing diagnostics, without biological inference."""
import itertools
import json

import numpy as np


def nominal_panel(rows, counts, genes, library_target=10000., pseudocount=1.):
    """Keep run labels and nominal references; never assert dose qualification.

    Input counts have one row per original treatment barcode. Technical variants
    are averaged within a run/pair before descriptive agreement calculations.
    No model selection, held-out prediction, power, or confidence interval is run.
    """
    ids = [r[''] for r in rows]
    if not rows or len(set(ids)) != len(ids) or any(not v for v in ids):
        raise ValueError('Unique source treatment IDs are required')
    if not genes or len(set(genes)) != len(genes) or any(not str(g).strip() for g in genes):
        raise ValueError('Unique nonempty source genes are required')
    x = np.asarray(counts, dtype=float)
    if x.shape != (len(rows), len(genes)) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Invalid aggregate read matrix')
    if not np.equal(x, np.floor(x)).all():
        raise ValueError('Expected uncorrected integer aggregate reads')
    if not np.isfinite([library_target, pseudocount]).all() or min(library_target, pseudocount) <= 0:
        raise ValueError('Positive normalization constants are required')
    runs = sorted({r['Run'] for r in rows})
    if len(runs) < 3 or any(not v for v in runs):
        raise ValueError('At least three named processing runs are required')
    totals = x.sum(axis=1)
    valid = totals > 0
    normalized = np.zeros_like(x)
    normalized[valid] = x[valid] / totals[valid, None] * library_target
    all_pairs = {tuple(sorted((r['Drug1'], r['Drug2']))) for r in rows
                 if 'DMSO' not in (r['Drug1'], r['Drug2']) and r['Drug1'] != r['Drug2']}
    effects, singles, references, lineage, exclusions = {}, {}, {}, {}, []
    for run in runs:
        index = [i for i, r in enumerate(rows) if r['Run'] == run]
        controls = [i for i in index if rows[i]['Drug1'] == rows[i]['Drug2'] == 'DMSO' and valid[i]]
        refs = {}
        for i in index:
            a, b = rows[i]['Drug1'], rows[i]['Drug2']
            if (a == 'DMSO') != (b == 'DMSO') and valid[i]:
                refs.setdefault(b if a == 'DMSO' else a, []).append(i)
        effects[run], singles[run], references[run], lineage[run] = {}, {}, {}, {}
        for pair in sorted(all_pairs):
            pair_rows = [i for i in index if tuple(sorted((rows[i]['Drug1'], rows[i]['Drug2']))) == pair and valid[i]]
            reason = None
            if not controls:
                reason = 'no positive-library nominal DMSO controls'
            elif not all(p in refs for p in pair):
                reason = 'missing positive-library nominal monotherapy'
            elif not pair_rows:
                reason = 'missing positive-library pair barcode'
            if reason:
                exclusions.append({'run': run, 'pair': list(pair), 'reason': reason})
                continue
            control = np.log2(pseudocount + normalized[controls].mean(axis=0))
            a, b = [np.log2(pseudocount + normalized[refs[p]].mean(axis=0)) - control for p in pair]
            # Transform each original pair row first, then collapse its variants.
            effect = (np.log2(pseudocount + normalized[pair_rows]) - control).mean(axis=0)
            effects[run][pair] = effect
            singles[run][pair] = a + b
            references[run][pair] = np.stack([a, b])
            lineage[run][json.dumps(pair, separators=(',', ':'))] = {
                'pair_rows': [ids[i] for i in pair_rows],
                'controls': [ids[i] for i in controls],
                'single_a': [ids[i] for i in refs[pair[0]]],
                'single_b': [ids[i] for i in refs[pair[1]]],
                'exact_physical_dose_and_vehicle_match_verified': False}
    complete = sorted(set.intersection(*(set(effects[r]) for r in runs)))
    if not complete:
        raise ValueError('No complete nominal-reference pair panel across runs')
    y = np.asarray([[effects[r][p] for p in complete] for r in runs])
    additive = np.asarray([[singles[r][p] for p in complete] for r in runs])
    return {'responses': y, 'singles': np.asarray([[references[r][p] for p in complete] for r in runs]),
            'runs': runs, 'pairs': complete, 'lineage': lineage, 'exclusions': exclusions,
            'all_pairs': all_pairs, 'zero_library_ids': [ids[i] for i in np.flatnonzero(~valid)]}


def processing_diagnostic(rows, counts, genes, library_target=10000., pseudocount=1.):
    """Describe processing agreement without model selection or biological inference."""
    panel = nominal_panel(rows, counts, genes, library_target, pseudocount)
    y, runs, complete = panel['responses'], panel['runs'], panel['pairs']
    additive = panel['singles'].sum(axis=2)
    lineage, exclusions, all_pairs = panel['lineage'], panel['exclusions'], panel['all_pairs']
    residual = y - additive
    departures, per_run = [], []
    for i, run in enumerate(runs):
        others = np.delete(residual, i, axis=0)
        shared = others.mean(axis=(0, 1))
        departures.append(residual[i] - shared)
        repeated = others.mean(axis=0)
        per_run.append({'run': run,
                        'zero_effect_mse': float(np.mean(y[i] ** 2)),
                        'additive_mse': float(np.mean(residual[i] ** 2)),
                        'other_run_shared_offset_mse': float(np.mean((residual[i] - shared) ** 2)),
                        'forbidden_repeated_pair_other_run_mse': float(np.mean((residual[i] - repeated) ** 2))})
    agreement = []
    departures = np.asarray(departures)
    for i, j in itertools.combinations(range(len(runs)), 2):
        for label, values in [('raw_nominal_residual', residual), ('after_other_run_shared_offset', departures)]:
            a, b = values[i], values[j]
            denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            usable = denominator > 0
            cosine = np.einsum('ij,ij->i', a[usable], b[usable]) / denominator[usable]
            agreement.append({'runs': [runs[i], runs[j]], 'definition': label,
                              'usable_nonzero_pairs': int(usable.sum()),
                              'zero_norm_pairs': int((~usable).sum()),
                              'cosine_mean': float(cosine.mean()) if len(cosine) else None,
                              'cosine_median': float(np.median(cosine)) if len(cosine) else None,
                              'positive_cosine_fraction': float((cosine > 0).mean()) if len(cosine) else None})
    return {'role': 'exploratory_nominal_identity_processing_diagnostic',
            'independent_culture_initiation_verified': False,
            'exact_physical_dose_and_vehicle_match_verified': False,
            'primary_claim_pass': False, 'biological_measurement_gate_pass': False,
            'model_selection_performed': False, 'confidence_intervals_computed': False,
            'new_confirmation_performed': False,
            'gene_count': len(genes), 'source_rows': len(rows), 'runs': runs,
            'nominal_pair_count_before_read_qc': len(all_pairs),
            'complete_nominal_pair_count': len(complete), 'complete_pairs': [list(p) for p in complete],
            'excluded_from_complete_panel': [list(p) for p in sorted(all_pairs - set(complete))],
            'zero_library_source_rows': panel['zero_library_ids'],
            'per_run_losses': per_run, 'cross_run_cosines': agreement,
            'reference_matches': lineage, 'exclusions': exclusions,
            'scope': 'Reported processing runs, source-provided gene universe and nominal drug identities only. '
                     'Repeated-pair predictions use forbidden outcome information and are measurement controls, '
                     'not deployable pair predictions. Shared offsets use the same pair panel in other runs. '
                     'No biological interval, superiority claim, power qualification or model advancement.'}
