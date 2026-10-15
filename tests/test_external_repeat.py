import gzip
import numpy as np
import pytest
from opal.external_repeat import soft_samples, repeat_diagnostic, TREATMENTS


def fixture():
    columns = [f'{role}.{r}' for role in TREATMENTS for r in (1, 2, 3)]
    samples = [dict(accession=f'GSM{i}', characteristics=dict(treatment=[TREATMENTS[c.split('.')[0]]],
        replicate=[c.split('.')[1]])) for i, c in enumerate(columns)]
    values = np.arange(1., 46.).reshape(15, 3)
    return columns, ['ENSG1', 'ENSG2', 'ENSG3'], values, samples


def test_repeat_statistics_preserve_fractional_counts_and_do_not_infer_pairing():
    columns, genes, x, samples = fixture()
    x[0, 0] += .25
    r = repeat_diagnostic(columns, genes, x, samples)
    z = np.log2(1+x[:3]*1_000_000/x[:3].sum(1)[:, None])
    assert r['conditions']['untr']['mean_gene_sample_variance'] == z.var(0, ddof=1).mean()
    for row in r['conditions'].values():
        assert np.mean([a['mse'] for a in row['within_condition_pairwise_mse']]) == pytest.approx(2*row['mean_gene_sample_variance'])
    assert not r['paired_cross_condition_analysis'] and not r['independent_cultures_verified']
    assert not r['biological_confirmation'] and not r['predictor_fit']


def test_exact_axes_missingness_and_zero_library_fail_closed():
    columns, genes, x, samples = fixture()
    for xx in (x*0, np.where(x == 1, np.nan, x), -x):
        with pytest.raises(ValueError): repeat_diagnostic(columns, genes, xx, samples)
    with pytest.raises(ValueError): repeat_diagnostic(columns[:-1], genes, x[:-1], samples)
    with pytest.raises(ValueError): repeat_diagnostic(columns, [genes[0]]*3, x, samples)
    with pytest.raises(ValueError): repeat_diagnostic(columns, genes, x, samples+[samples[0]])


def test_reordering_is_identity_bound_and_all_zero_genes_are_retained():
    columns, genes, x, samples = fixture()
    x[:, -1] = 0
    original = repeat_diagnostic(columns, genes, x, samples)
    order = np.arange(15)[::-1]
    changed = repeat_diagnostic([columns[i] for i in order], genes, x[order], samples)
    assert changed['conditions'] == original['conditions']
    assert changed['all_zero_gene_rows_retained'] == 1 and changed['gene_count'] == 3


def test_metadata_does_not_copy_contacts_or_accept_expression_tables(tmp_path):
    p = tmp_path/'source.gz'
    with gzip.open(p, 'wt') as f:
        f.write('^SAMPLE = GSM1\n!Sample_title = label\n!Sample_contact_email = private@example.org\n!Sample_characteristics_ch1 = replicate: 1\n')
    r = soft_samples(p)
    assert r == [dict(accession='GSM1', title='label', characteristics={'replicate':['1']})]
    with gzip.open(p, 'wt') as f: f.write('^SAMPLE = GSM1\n!sample_table_begin\n')
    with pytest.raises(ValueError, match='table'): soft_samples(p)
