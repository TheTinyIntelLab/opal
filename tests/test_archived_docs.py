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
