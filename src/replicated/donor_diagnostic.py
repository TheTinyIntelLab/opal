"""Descriptive, single-pair donor transfer on a source-processed assay scale."""
import csv
import gzip
import re

import numpy as np

ARMS = ('E', 'V', 'EL', 'VL')
TREATMENTS = {'E': 'vehicle (EtOH)', 'V': '1,25 dihydroxyvitamin D3',
              'EL': 'lipopolysaccharide', 'VL': '1,25 dihydroxyvitamin D3 + lipopolysaccharide'}


def parse_brief_samples(text):
    """Read requested brief metadata, rejecting expression tables."""
    if '!sample_table_begin' in text.lower():
        raise ValueError('Brief sample metadata must not contain expression tables')
    rows, row = [], None
    for line in text.splitlines():
        if line.startswith('^SAMPLE = '):
            row = {'accession': line.split(' = ', 1)[1], 'characteristics': {}}
            rows.append(row)
        elif row is not None and line.startswith('!Sample_title = '):
            row['title'] = line.split(' = ', 1)[1]
        elif row is not None and line.startswith('!Sample_characteristics_ch1 = '):
            key, value = line.split(' = ', 1)[1].split(': ', 1)
            if key in row['characteristics']:
                raise ValueError('Ambiguous duplicate sample characteristic')
            row['characteristics'][key] = value
    if not rows:
        raise ValueError('No sample metadata')
    return rows


def matched_donors(rows, columns):
    if len(set(columns)) != len(columns):
        raise ValueError('Duplicate matrix sample columns')
    by_id = {r['accession']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(columns):
        raise ValueError('Matrix sample identity coverage mismatch')
    design = {}
    for column, accession in enumerate(columns):
        row = by_id[accession]
        match = re.fullmatch(r'(AA|EA)_(\d+)_(E|V|EL|VL)', row.get('title', ''))
        if not match:
            raise ValueError('Unrecognized donor/arm title')
        group, subject, arm = match.groups()
        chars = row['characteristics']
        if (chars.get('treated with') != TREATMENTS[arm]
                or chars.get('cell type') != 'Primary monocytes'
                or chars.get('subject status') != 'healthy individual'
                or chars.get('ancestry') != {'AA': 'African-American', 'EA': 'European-American'}[group]):
            raise ValueError('Sample title and source characteristics disagree')
        donor = group + '_' + subject
        if arm in design.setdefault(donor, {}):
            raise ValueError('Duplicate donor treatment arm')
        design[donor][arm] = column
    if len(design) < 3 or any(set(arms) != set(ARMS) for arms in design.values()):
        raise ValueError('Complete four-arm panel required for every donor')
    return design


def read_series_matrix(path):
    """Retain every deposited finite probe, without gene or outcome selection."""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip() == '!series_matrix_table_begin':
                break
        else:
            raise ValueError('Missing series matrix table')
        header = next(csv.reader([next(stream)], delimiter='\t'))
        if header[0] != 'ID_REF':
            raise ValueError('Invalid series matrix header')
        probes, profiles = [], []
        for line in stream:
            if line.strip() == '!series_matrix_table_end':
                break
            fields = next(csv.reader([line], delimiter='\t'))
            if len(fields) != len(header):
                raise ValueError('Ragged series matrix row')
            probes.append(fields[0])
            profiles.append([float(value) for value in fields[1:]])
        else:
            raise ValueError('Missing series matrix terminator')
    return header[1:], probes, np.asarray(profiles, dtype=float).T


def single_pair_donor_diagnostic(rows, columns, probes, values):
    """No held-pair claim: transfer uses this same pair in other donors."""
    design = matched_donors(rows, columns)
    x = np.asarray(values, dtype=float)
    if (not probes or len(set(probes)) != len(probes) or any(not str(p).strip() for p in probes)
            or x.shape != (len(columns), len(probes)) or not np.isfinite(x).all()):
        raise ValueError('Finite processed matrix with unique probe IDs required')
    donors = sorted(design)
    bank = np.asarray([[x[design[d][arm]] for arm in ARMS] for d in donors])
    observed = bank[:, 3] - bank[:, 0]
    additive = bank[:, 1] + bank[:, 2] - 2 * bank[:, 0]
    residual = observed - additive
    # Exclude the complete query donor, including its observed pair, from transfer.
    correction = np.asarray([np.mean(np.delete(residual, i, axis=0), axis=0)
                             for i in range(len(donors))])
    losses = {'zero': np.mean(observed ** 2, axis=1),
              'additive': np.mean(residual ** 2, axis=1),
              'same_pair_other_donors': np.mean((residual - correction) ** 2, axis=1)}
    cosines = []
    for i in range(len(donors)):
        for j in range(i):
            denom = np.linalg.norm(residual[i]) * np.linalg.norm(residual[j])
            if denom > 0:
                cosines.append(float(np.dot(residual[i], residual[j]) / denom))
    return {'role': 'exposed_single_pair_donor_measurement_diagnostic',
            'donors': len(donors), 'samples': len(columns), 'source_probe_count': len(probes),
            'canonical_pair_count': 1, 'endpoint': 'all deposited finite source probes',
            'scale': 'deposited LUMI variance-transformed quantile-normalized intensity',
            'normalization_refit': False, 'feature_selection_performed': False,
            'equal_donor_mean_mse': {name: float(np.mean(value)) for name, value in losses.items()},
            'donor_loss_values': {name: value.tolist() for name, value in losses.items()},
            'same_pair_transfer_query_donor_excluded': True,
            'same_pair_transfer_eligible_for_original_held_pair_protocol': False,
            'residual_pairwise_cosine_count': len(cosines),
            'residual_pairwise_cosine_median': float(np.median(cosines)) if cosines else None,
            'undefined_residual_pairwise_cosines': len(donors) * (len(donors) - 1) // 2 - len(cosines),
            'confidence_intervals_computed': False, 'biological_measurement_gate_pass': False,
            'fresh_confirmation': False, 'model_menu_searched': False}
