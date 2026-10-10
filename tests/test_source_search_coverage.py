import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from summarize_source_search import summarize


def inventory(tmp_path, name, identities):
    path = tmp_path / name
    rows = [{'accession': a, 'source_summary_sha256': h} for a, h in identities]
    path.write_text(json.dumps({'role': 'bounded_metadata_inventory_not_qualification',
        'candidate_count': len(rows), 'candidates': rows, 'queries': [],
        'expression_or_weights_downloaded': False}))
    return path


def test_overlap_and_changed_summaries_preserve_identity(tmp_path):
    a = inventory(tmp_path, 'a.json', [('GSE2', 'first'), ('GSE10', 'same')])
    b = inventory(tmp_path, 'b.json', [('GSE2', 'changed'), ('GSE3', 'new')])
    report = summarize([a, b])
    assert report['unique_summary_records'] == 3
    assert report['repeated_summary_records'] == 1
    assert report['changed_summary_identities'] == 1
    assert report['searches'][1]['new_summary_identities'] == 1
    assert [r['accession'] for r in report['series']] == ['GSE2', 'GSE3', 'GSE10']
    assert report['series'][0]['summary_sha256_values'] == ['first', 'changed']
    assert report['qualification_performed'] is False
    assert all(r['qualification'] == 'not_assessed_by_inventory_merge' for r in report['series'])


def test_duplicate_series_rejected(tmp_path):
    path = inventory(tmp_path, 'duplicate.json', [('GSE2', 'a'), ('GSE2', 'b')])
    with pytest.raises(ValueError, match='duplicated'):
        summarize([path])


def test_incorrect_count_or_exposure_rejected(tmp_path):
    path = inventory(tmp_path, 'bad.json', [('GSE2', 'a')])
    data = json.loads(path.read_text())
    data['candidate_count'] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='count'):
        summarize([path])
    data['candidate_count'] = 1
    data['expression_or_weights_downloaded'] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='unopened'):
        summarize([path])
