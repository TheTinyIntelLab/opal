"""Custody gates for the finite Wave Five completion experiments."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-wave5-completion'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gates():
    freeze = json.loads((OUT/'freeze.json').read_text(encoding='utf8'))
    auth = json.loads((OUT/'execution-authorized.json').read_text(encoding='utf8'))
    if not auth['authorized'] or auth['freeze_sha256'] != digest(OUT/'freeze.json'):
        raise ValueError('Invalid completion authorization')
    for path, expected in freeze['sha256'].items():
        if digest(path) != expected:
            raise ValueError('Frozen input drift: '+path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve violated')
    return digest(OUT/'freeze.json')


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
