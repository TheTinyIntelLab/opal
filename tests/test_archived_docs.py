import hashlib
import json
import zipfile

import pytest

from tools.restore_archived_docs import ROOT, restore


def test_retired_scientific_documents_restore_exactly_without_private_plans(tmp_path):
    manifest = json.loads((ROOT / 'docs/archive/retired-documents.json').read_text())
    report = restore(tmp_path)
    public = [r for r in manifest['files'] if r['disposition'] == 'public_history_archive']
    assert report['restored'] == len(public)
    for record in public:
        assert hashlib.sha256((tmp_path / record['path']).read_bytes()).hexdigest() == record['sha256']
    for record in manifest['files']:
        if record['disposition'] == 'private_backup':
            assert not (tmp_path / record['path']).exists()


def test_restore_refuses_overwrite_before_writing_any_documents(tmp_path):
    manifest = json.loads((ROOT / 'docs/archive/retired-documents.json').read_text())
    last = [r for r in manifest['files'] if r['disposition'] == 'public_history_archive'][-1]
    existing = tmp_path / last['path']
    existing.parent.mkdir(parents=True)
    existing.write_text('keep this file')
    with pytest.raises(FileExistsError):
        restore(tmp_path)
    assert existing.read_text() == 'keep this file'
    assert len([p for p in tmp_path.rglob('*') if p.is_file()]) == 1


def test_private_planning_files_are_absent_from_public_archive():
    manifest = json.loads((ROOT / 'docs/archive/retired-documents.json').read_text())
    with zipfile.ZipFile(ROOT / 'docs/archive/development-records.zip') as archive:
        assert archive.testzip() is None
        assert all(r['path'] not in archive.namelist() for r in manifest['files']
                   if r['disposition'] == 'private_backup')

def test_research_history_preserves_every_original_file_hash():
    manifest = json.loads((ROOT / 'docs/archive/research-records.json').read_text())
    path = ROOT / 'docs/archive/research-records.zip'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['archive_sha256']
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {r['path'] for r in manifest['files']}
        for record in manifest['files']:
            assert hashlib.sha256(archive.read(record['path'])).hexdigest() == record['sha256']
            if record['path'].endswith('.npz'):
                assert record['kind'] == 'synthetic_engineering_state'
                assert record['path'].startswith('results/research-continuation/replication-v1/')


def test_archive_audit_only_accepts_exact_registered_synthetic_states():
    from tools.check_v1_release import inspect_member
    manifest = json.loads((ROOT / 'docs/archive/research-records.json').read_text())
    record = next(r for r in manifest['files'] if r['kind'] == 'synthetic_engineering_state')
    with zipfile.ZipFile(ROOT / 'docs/archive/research-records.zip') as archive:
        payload = archive.read(record['path'])
    whitelist = {record['path']: record['sha256']}
    inspect_member(record['path'], payload, whitelist)
    with pytest.raises(ValueError, match='Private/model/data member'):
        inspect_member(record['path'], payload+b'changed', whitelist)
    with pytest.raises(ValueError, match='Private/model/data member'):
        inspect_member('opal/checkpoint.npz', payload, {'opal/checkpoint.npz': record['sha256']})
