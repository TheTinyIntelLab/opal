"""Portable custody check for repository-owned completion freeze dependencies."""
import hashlib
import json
from pathlib import Path


def test_repository_owned_completion_freeze_bytes_survive_checkout():
    root=Path(__file__).resolve().parents[1]
    directory=root/'results/research-wave5-completion'
    freeze_path=directory/'freeze.json'
    freeze=json.loads(freeze_path.read_text(encoding='utf8'))
    authorization=json.loads((directory/'execution-authorized.json').read_text(encoding='utf8'))
    assert hashlib.sha256(freeze_path.read_bytes()).hexdigest()==authorization['freeze_sha256']
    checked=0
    marker='/code/thetinyintellab/opal/'
    for original,expected in freeze['sha256'].items():
        normalized=original.replace('\\','/')
        if marker not in normalized:
            # External saved-state dependencies are checked by the no-fit tools,
            # and are deliberately unavailable in a clean hosted checkout.
            continue
        relative=normalized.split(marker,1)[1]
        path=(root/relative).resolve()
        assert path.is_relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
        checked+=1
    assert checked==35
