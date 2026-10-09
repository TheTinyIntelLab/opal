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
