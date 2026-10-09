"""Metadata leads cannot silently qualify a source or swallow query errors."""
import importlib.util
import gzip
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('source_discovery',
        Path(__file__).parents[1] / 'tools/discover_public_pair_sources.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sample_count_and_replication_phrase_never_establish_cultures():
    payload = {'result': {'uids': ['123'], '123': {'accession': 'GSE123', 'entrytype': 'GSE',
              'n_samples': 6000, 'title': 'Drug combination',
              'summary': 'Independent experiments tested viability; expression used one sample.'}}}
    record = module.summarize(payload)[0]
    assert record['source_sample_count'] == 6000
    assert record['replication_text_signals'] == ['independent experiment']
    assert not record['primary_confirmation_qualified']
    assert record['qualification'] == 'not_assessed_from_summary'
    assert 'summary' not in record


def test_query_and_incomplete_summary_errors_reject():
    with pytest.raises(ValueError, match='rejected'):
        module.search_ids({'esearchresult': {'errorlist': {'fieldnotfound': ['bad']}}})
    with pytest.raises(ValueError, match='identifiers'):
        module.search_ids({'esearchresult': {'querytranslation': 'valid', 'idlist': ['1', '1']}})
    with pytest.raises(ValueError, match='Expected GEO'):
        module.summarize({'result': {'uids': ['1'], '1': {'accession': 'GSM1'}}})


def test_custom_queries_require_bounded_unique_series_scope():
    for queries in ([], ['single drug'], ['x AND gse[ETYP]'] * 2,
                    ['x AND gse[ETYP]'] * 5, [None]):
        with pytest.raises(ValueError, match='queries'):
            module.validate_queries(queries)
    assert module.validate_queries(['old studies AND gse[ETYP]']) == ['old studies AND gse[ETYP]']


def test_family_design_counts_do_not_pool_or_infer_cultures(tmp_path):
    audit_spec = importlib.util.spec_from_file_location('family_audit',
        Path(__file__).parents[1] / 'tools/audit_geo_family_design.py')
    audit_module = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit_module)
    path = tmp_path / 'GSE1.soft.gz'
    path.write_bytes(gzip.compress(b'^SAMPLE = GSM1\n!Sample_characteristics_ch1 = batch: Pool-1\n'
                                  b'^SAMPLE = GSM2\n!Sample_characteristics_ch1 = batch: Pool-2\n'))
    result = audit_module.audit(path)
    assert result['sample_count'] == 2
    assert result['characteristic_value_counts']['batch'] == {'Pool-1': 1, 'Pool-2': 1}
    assert not result['independent_cultures_inferred']
    path.write_bytes(gzip.compress(b'^SAMPLE = GSM1\n^SAMPLE = GSM1\n'))
    with pytest.raises(ValueError, match='duplicated'):
        audit_module.audit(path)


def test_brief_metadata_rejects_html_and_embedded_expression_tables():
    brief_spec = importlib.util.spec_from_file_location('brief_metadata',
        Path(__file__).parents[1] / 'tools/fetch_geo_brief_metadata.py')
    brief = importlib.util.module_from_spec(brief_spec)
    brief_spec.loader.exec_module(brief)
    with pytest.raises(ValueError, match='Expression table'):
        brief.validate_brief(b'^SAMPLE = GSM1\n!sample_table_begin\n', 'SAMPLE')
    with pytest.raises(ValueError, match='Expected GEO'):
        brief.validate_brief(b'<html>Checking your browser</html>', 'SAMPLE')
    assert brief.validate_brief(b'^SAMPLE = GSM1\n!Sample_title = control', 'SAMPLE').startswith('^SAMPLE')


def test_biostudies_sample_audit_deduplicates_assays_without_inventing_cultures():
    spec = importlib.util.spec_from_file_location('biostudies',
        Path(__file__).parents[1] / 'tools/fetch_biostudies_metadata.py')
    biostudies = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(biostudies)
    data = (b'Source Name\tComment[ENA_SAMPLE]\tFactor Value[treatment]\n'
            b's1\tERS1\tcontrol\ns1\tERS1\tcontrol\ns2\tERS2\tpair\n')
    result = biostudies.summarize_sdrf(data)
    assert result['assay_row_count'] == 3
    assert result['unique_source_count'] == 2
    assert result['unique_source_factor_counts'] == {'Factor Value[treatment]': {'control': 1, 'pair': 1}}
    assert not result['independent_cultures_inferred']
    with pytest.raises(ValueError, match='Conflicting'):
        biostudies.summarize_sdrf(data.replace(b's1\tERS1\tcontrol\ns2', b's1\tERS1\tpair\ns2'))
    with pytest.raises(ValueError, match='Unexpected'):
        biostudies.validate_study(b'{"accno":"E-MTAB-1","section":{}}', 'E-MTAB-2')
    with pytest.raises(ValueError, match='rectangular'):
        biostudies.validate_sdrf(b'<html>error</html>')
