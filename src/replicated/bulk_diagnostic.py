"""Source-processed log-CPM measurement checks, separate from count inference."""
import itertools
import re

import numpy as np


def validate_lncap_samples(rows, columns):
    """Require literal source IDs, constituent doses and same-label references."""
    selected = {row['title'].removeprefix('LNCaP '): row for row in rows
                if row['title'].startswith('LNCaP ')}
    if len(selected) != sum(row['title'].startswith('LNCaP ') for row in rows):
        raise ValueError('Duplicate source sample titles')
    if len(set(columns)) != len(columns) or set(columns) != set(selected):
        raise ValueError('Matrix columns must match source LNCaP sample IDs exactly')
    design = {}
    expected_doses = {'T': 20., 'M': 10., 'W': 5.}
    names = {'T': 'Tamoxifen', 'M': 'Mefloquine', 'W': 'Withaferin'}
    for name in columns:
        match = re.fullmatch(r'(DMSO|T|M|W|TM|TW|MW|TMW)_(0|3|6|9|12|24)([ABC])', name)
        if not match:
            raise ValueError('Unsupported source condition identifier')
        treatment, hour, repeat = match.groups()
        row = selected[name]
        if (row['cell line'] != 'LNCaP cells' or row['hours of treatment'] != hour
                or row['replicate'] != repeat):
            raise ValueError('Source title/characteristics disagree')
        if 'in 0.3% DMSO for the viability time courses' not in row.get('treatment_protocol', ''):
            raise ValueError('Same-vehicle source protocol required')
        if treatment == 'DMSO':
            if row['treatment'] != '0.3% DMSO':
                raise ValueError('Control vehicle mismatch')
        else:
            doses = [float(v) for v in re.findall(r'\d+(?:\.\d+)?', row['treatment concentration (um)'])]
            if doses != [expected_doses[c] for c in treatment] or row.get('solvent') != 'DMSO':
                raise ValueError('Constituent dose or vehicle mismatch')
            if any(names[c] not in row['treatment'] for c in treatment):
                raise ValueError('Treatment identity mismatch')
        design[(int(hour), repeat, treatment)] = {'column': columns.index(name),
                                                   'accession': row['accession'], 'source_id': name}
    expected = set(itertools.product((0, 3, 6, 9, 12, 24), 'ABC',
                                    ('DMSO', 'T', 'M', 'W', 'TM', 'TW', 'MW', 'TMW')))
    if set(design) != expected:
        raise ValueError('Incomplete matched time/repeat/condition panel')
    return design


def processed_bulk_diagnostic(rows, columns, values, genes):
    """Keep all six time strata and reported repeats, without biological CIs."""
    design = validate_lncap_samples(rows, columns)
    x = np.asarray(values, float)
    if (not genes or len(set(genes)) != len(genes) or any(not str(g).strip() for g in genes)
            or x.shape != (len(columns), len(genes)) or not np.isfinite(x).all()):
        raise ValueError('Finite log-CPM matrix with unique genes required')
    strata = []
    for hour in (0, 3, 6, 9, 12, 24):
        y, additive, lineage = [], [], []
        for repeat in 'ABC':
            control = x[design[(hour, repeat, 'DMSO')]['column']]
            single = {c: x[design[(hour, repeat, c)]['column']] - control for c in 'TMW'}
            y.append([x[design[(hour, repeat, pair)]['column']] - control for pair in ('TM', 'TW', 'MW')])
            additive.append([sum(single[c] for c in pair) for pair in ('TM', 'TW', 'MW')])
            lineage.append({'repeat': repeat, 'sources': {kind: design[(hour, repeat, kind)]
                            for kind in ('DMSO', 'T', 'M', 'W', 'TM', 'TW', 'MW')}})
        y, additive = np.asarray(y), np.asarray(additive)
        residual = y - additive
        losses = []
        for i, repeat in enumerate('ABC'):
            others = np.delete(y, i, axis=0).mean(axis=0)
            repeated_residual = np.delete(residual, i, axis=0).mean(axis=0)
            losses.append({'repeat': repeat, 'zero_mse': float(np.mean(y[i] ** 2)),
                           'additive_mse': float(np.mean(residual[i] ** 2)),
                           'forbidden_other_repeat_pair_effect_mse': float(np.mean((y[i] - others) ** 2)),
                           'forbidden_other_repeat_residual_mse': float(np.mean((residual[i] - repeated_residual) ** 2))})
        agreement = []
        for i, j in itertools.combinations(range(3), 2):
            for p, pair in enumerate(('TM', 'TW', 'MW')):
                norm = np.linalg.norm(residual[i, p]) * np.linalg.norm(residual[j, p])
                agreement.append({'repeats': ['ABC'[i], 'ABC'[j]], 'pair': pair,
                                  'residual_cosine': float(residual[i, p] @ residual[j, p] / norm) if norm else None})
        strata.append({'nominal_hour': hour, 'per_repeat': losses, 'residual_agreement': agreement,
                       'reference_lineage': lineage,
                       'equal_repeat_mean_mse': {key: float(np.mean([r[key] for r in losses]))
                                               for key in losses[0] if key != 'repeat'}})
    return {'role': 'source_processed_bulk_repeat_measurement_diagnostic',
            'source': 'GSE149428', 'state': 'LNCaP', 'source_samples': len(columns),
            'analyzed_samples': 126, 'excluded_triple_samples': 18, 'gene_count': len(genes),
            'strata': strata, 'primary_claim_pass': False,
            'independent_culture_initiation_verified': False,
            'biological_measurement_gate_pass': False, 'new_confirmation_performed': False,
            'model_selection_performed': False, 'confidence_intervals_computed': False,
            'processing': 'Author log2 CPM with plate covariate adjustment; no count inversion or renormalization',
            'scope': 'Same-source, same-time, same-reported-repeat controls/singles; published dose and vehicle metadata. '
                     'Other-repeat pair predictions use held-pair outcomes and are forbidden prediction controls. '
                     'Time strata are separate; repeat letters are not verified independent culture initiations. '
                     'Upstream gene filtering/plate adjustment may use all source samples; no fresh-confirmation claim.'}
