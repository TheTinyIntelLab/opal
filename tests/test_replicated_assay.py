from pathlib import Path
import json
from tools.rehearse_replicated_assay import rehearse
from opal.replicated.contracts import sha


def test_count_cli_rehearsal_preserves_custody_and_actual_reference_lineage(tmp_path):
    root=Path(__file__).parents[1]
    report=rehearse(tmp_path/'assay',root/'configs/experiments/replicated_fixed_state_v1.json')
    assert report['development_pairs']==88 and report['confirmation_pairs']==24
    assert report['query_reference_rows']==40
    assert report['query_and_custodian_reference_transforms_agree']
    assert report['restoration_pass'] and not report['scientific_confirmation_claimed']
    assert [r['command'] for r in report['commands']]==['prepare','query-references','prepare-sealed','diagnose','freeze','evaluate']
    access=json.loads((tmp_path/'assay/freeze/access.json').read_text())
    assert not set(access['query_feature_rows'])&set(access['query_reference_rows'])
    manifest=json.loads((tmp_path/'assay/query.preparation.json').read_text())
    assert manifest['query_identities_sha256']==sha(tmp_path/'assay/query-identities.json')
    assert set(manifest['matched_reference_rows'])==set(access['query_feature_rows'])
    assert set(access['query_reference_rows'])<=set(manifest['source_row_ids'])
