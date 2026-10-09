"""Descriptive continuous-assay diagnostics without matching repeat numbers."""
from collections import defaultdict
import re

import numpy as np

from .donor_diagnostic import parse_brief_samples


def lncap_array_design(text):
    rows = parse_brief_samples(text)
    arms = {'vehicle control': (), 'bicalutamide': ('bicalutamide',),
            'SAHA': ('vorinostat',), '17-AAG': ('17-AAG',),
            '17-AAG+bicalutamide': ('17-AAG', 'bicalutamide'),
            '17-AAG+SAHA': ('17-AAG', 'vorinostat'),
            'bicalutamide+SAHA': ('bicalutamide', 'vorinostat')}
    seen = defaultdict(set)
    for row in rows:
        match = re.fullmatch(r'LNCaP_(.+)_biological replicate ([1-6])', row.get('title', ''))
        if (not match or match[1] not in arms
                or row['characteristics'].get('cell line') != 'LNCaP'):
            raise ValueError('Unexpected cell line or treatment title')
        treatment, repeat = match.groups()
        if repeat in seen[treatment]:
            raise ValueError('Duplicate treatment repeat label')
        seen[treatment].add(repeat)
        row['components'] = arms[treatment]
    if set(seen) != set(arms) or any(v != set('123456') for v in seen.values()):
        raise ValueError('Complete seven-arm six-repeat metadata required')
    if len({r['accession'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample identity')
    return rows


def unmatched_replicate_diagnostic(rows, columns, probes, values):
    """Pool only control/single references; never pair arms by repeat numbering."""
    by_id = {r['accession']: r for r in rows}
    x = np.asarray(values, dtype=float)
    if (len(by_id) != len(rows) or len(set(columns)) != len(columns) or set(by_id) != set(columns)
            or not probes or len(set(probes)) != len(probes) or any(not str(p).strip() for p in probes)
            or x.shape != (len(columns), len(probes)) or not np.isfinite(x).all()):
        raise ValueError('Finite complete matrix with unique sample and probe identities required')
    groups = defaultdict(list)
    # Stable accession order makes report and floating-point summation reproducible.
    for accession in sorted(columns):
        components = tuple(sorted(by_id[accession]['components']))
        if len(components) > 2 or len(set(components)) != len(components):
            raise ValueError('Require unique single or two-component treatment labels')
        groups[components].append(columns.index(accession))
    pairs = sorted(key for key in groups if len(key) == 2)
    if () not in groups or not pairs:
        raise ValueError('Control and observed pairs required')
    control = np.mean(x[groups[()]], axis=0)
    residuals, observed = {}, {}
    for pair in pairs:
        if any((component,) not in groups for component in pair) or len(groups[pair]) < 2:
            raise ValueError('Pair requires both singles and at least two observed samples')
        observed[pair] = x[groups[pair]] - control
        additive = sum(np.mean(x[groups[(c,)]], axis=0) - control for c in pair)
        residuals[pair] = observed[pair] - additive
    records = []
    for pair in pairs:
        residual = residuals[pair]
        same_pair = np.asarray([np.mean(np.delete(residual, i, axis=0), axis=0)
                                for i in range(len(residual))])
        losses = {'zero': np.mean(observed[pair] ** 2, axis=1),
                  'additive': np.mean(residual ** 2, axis=1),
                  'same_pair_other_samples': np.mean((residual - same_pair) ** 2, axis=1)}
        if len(pairs) > 1:
            # Hold this whole canonical pair out, including every repeat, at once.
            correction = np.mean([np.mean(residuals[p], axis=0) for p in pairs if p != pair], axis=0)
            losses['other_pair_mean_offset'] = np.mean((residual - correction) ** 2, axis=1)
        cosines = []
        for i in range(len(residual)):
            for j in range(i):
                denominator = np.linalg.norm(residual[i]) * np.linalg.norm(residual[j])
                if denominator > 0:
                    cosines.append(float(np.dot(residual[i], residual[j]) / denominator))
        records.append({'components': list(pair), 'joint_samples': len(residual),
                        'mean_mse': {k: float(np.mean(v)) for k, v in losses.items()},
                        'sample_loss_values': {k: v.tolist() for k, v in losses.items()},
                        'residual_cosine_count': len(cosines),
                        'residual_cosine_median': float(np.median(cosines)) if cosines else None,
                        'undefined_residual_cosines': len(residual) * (len(residual) - 1) // 2 - len(cosines)})
    return {'role': 'exposed_unmatched_repeat_processed_assay_diagnostic',
            'samples': len(columns), 'source_probe_count': len(probes), 'canonical_pairs': len(pairs),
            'endpoint': 'all deposited finite unique source probes',
            'reference_policy': 'all control and constituent-single means, no cross-arm repeat matching',
            'equal_pair_mean_mse': {k: float(np.mean([r['mean_mse'][k] for r in records]))
                                    for k in records[0]['mean_mse']},
            'pair_diagnostics': records, 'same_pair_transfer_query_sample_excluded': True,
            'same_pair_transfer_eligible_for_original_held_pair_protocol': False,
            'other_pair_offset_excludes_all_query_pair_repeats': True,
            'shared_reference_dependence_present': True, 'source_normalization_refit': False,
            'feature_selection_performed': False, 'model_menu_searched': False,
            'confidence_intervals_computed': False, 'independent_culture_blocks_inferred': False,
            'biological_measurement_gate_pass': False, 'fresh_confirmation': False}
