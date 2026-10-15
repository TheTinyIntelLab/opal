"""Source-labelled repeat diagnostics, without inferring biological independence."""
from collections import Counter
import gzip
import io
import re
import numpy as np


def soft_samples(path):
    raw = path.read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError('Compressed metadata cap exceeded')
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        data = stream.read(20_000_001)
    if len(data) > 20_000_000 or b'!sample_table_begin' in data.lower():
        raise ValueError('Metadata expansion/table refusal')
    samples, current = [], None
    for line in data.decode('utf-8').splitlines():
        if line.startswith('^SAMPLE = '):
            current = {'accession': line.split(' = ', 1)[1], 'characteristics': {}}
            samples.append(current)
        elif current is not None and ' = ' in line:
            name, value = line.split(' = ', 1)
            if name == '!Sample_title':
                current['title'] = value
            elif name == '!Sample_characteristics_ch1':
                key, separator, entry = value.partition(': ')
                if separator:
                    current['characteristics'].setdefault(key.casefold(), []).append(entry)
    ids = [s['accession'] for s in samples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Missing/duplicate source samples')
    return samples


def metadata_summary(samples):
    keys = sorted({k for s in samples for k in s['characteristics']})
    return dict(sample_count=len(samples), characteristics={k: dict(Counter(
        v for s in samples for v in s['characteristics'].get(k, []))) for k in keys},
        independent_initiation_verified=False, pairing_inferred=False,
        role='metadata counts, not source or biological qualification')


TREATMENTS = {'untr': 'untreated', 'DMSO': 'DMSO', 'JQ1': 'JQ-1 (BET inhibitor)',
              'PD901': 'PD901 (MEK inhibitor)', 'JQPD': 'JQ-1+PD901'}


def bind_columns(columns, samples):
    expected = {f'{role}.{r}' for role in TREATMENTS for r in (1, 2, 3)}
    if len(columns) != len(expected) or set(columns) != expected:
        raise ValueError('Exact 15 source repeat columns required')
    by_condition = {}
    for sample in samples:
        chars = sample['characteristics']
        if len(chars.get('treatment', [])) != 1 or len(chars.get('replicate', [])) != 1:
            raise ValueError('Ambiguous literal sample characteristics')
        key = (chars['treatment'][0], chars['replicate'][0])
        if key in by_condition:
            raise ValueError('Duplicate treatment/repeat metadata')
        by_condition[key] = sample['accession']
    keys = {(v, str(r)) for v in TREATMENTS.values() for r in (1, 2, 3)}
    if set(by_condition) != keys:
        raise ValueError('Incomplete or unexpected source conditions')
    return {c: by_condition[(TREATMENTS[c.rsplit('.', 1)[0]], c.rsplit('.', 1)[1])] for c in columns}


def repeat_diagnostic(columns, genes, values, samples):
    mapping = bind_columns(columns, samples)
    x = np.asarray(values, dtype=np.float64)
    if (not genes or len(genes) > 100_000 or len(genes) != len(set(genes))
            or any(not re.fullmatch(r'ENSG[0-9]+(?:\.[0-9]+)?', g) for g in genes)
            or x.shape != (len(columns), len(genes)) or not np.isfinite(x).all()
            or np.any(x < 0)):
        raise ValueError('Finite nonnegative source counts and unique gene axes required')
    totals = x.sum(1)
    if np.any(totals <= 0):
        raise ValueError('Positive whole-column library totals required')
    logged = np.log2(1 + x * 1_000_000 / totals[:, None])
    conditions = {}
    for role, treatment in TREATMENTS.items():
        indices = [columns.index(f'{role}.{r}') for r in (1, 2, 3)]
        z = logged[indices]
        conditions[role] = dict(treatment=treatment, reported_repeats=3,
            samples=[mapping[columns[i]] for i in indices],
            mean_gene_sample_variance=float(z.var(axis=0, ddof=1).mean()),
            within_condition_pairwise_mse=[dict(repeats=[i+1, j+1],
                mse=float(np.mean((z[i]-z[j])**2))) for i in range(3) for j in range(i+1, 3)])
    return dict(source='GSE95153', gene_count=len(genes), source_sample_count=len(columns),
        all_zero_gene_rows_retained=int(np.count_nonzero(~np.any(x, axis=0))),
        conditions=conditions, column_to_sample=mapping,
        endpoint='Gene-average sample variance of log2(1+CPM), ddof=1, all source genes',
        biological_confirmation=False, independent_cultures_verified=False,
        paired_cross_condition_analysis=False, predictor_fit=False,
        biological_confidence_intervals=False, useful_margin_selected=False,
        status='descriptive_source_labelled_repeat_variation_only')
