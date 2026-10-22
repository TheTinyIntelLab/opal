"""Collect existing delivery evidence without fitting or scientific scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def artifact(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve validation receipts')
    freeze = read(ROOT/'results/research-wave4/freeze.json')
    custody = read(ROOT/'results/research-wave4/custody-before.json')
    protected_drift = [name for name, expected in custody['files'].items()
                       if digest(ROOT/name) != expected]
    freeze_drift = [name for name, expected in freeze['files'].items()
                    if digest(ROOT/name) != expected]
    commands = read(args.delivery/'commands.json')
    suite = ET.parse(args.delivery/'tests.xml').getroot().find('testsuite').attrib
    receipts = {}
    for name in ('historical.json', 'block.json', 'wave2.json', 'wave3.json',
                 'assay/verification.json', 'series-v1/verification.json',
                 'series-v2/verification.json'):
        path = args.delivery/name
        if path.exists():
            receipts[name] = dict(artifact=artifact(path), receipt=read(path))
    doc_folder = args.delivery/('docs-final' if (args.delivery/'docs-final').exists() else 'docs')
    docs = read(doc_folder/'build_manifest.json')
    new_files = subprocess.check_output(['git', 'diff', '--name-only', '45795c6'],
                                        cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'],
                                        cwd=ROOT, text=True).splitlines()
    candidates = sorted(set(new_files+untracked))
    import re
    secret_pattern = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    secret_files = []
    restricted_files = []
    for name in candidates:
        path = ROOT/name
        if not path.is_file():
            continue
        if secret_pattern.search(path.read_bytes()):
            secret_files.append(name)
        if path.suffix.lower() in {'.npz', '.npy', '.pt', '.pth', '.zip', '.xlsx', '.tsv'}:
            restricted_files.append(name)
    result = dict(schema='opal-wave4-validation-v1',
        scientific_successor=False, new_source_response_access=False,
        new_source_response_header_access=False, biological_fits=0, new_scientific_scores=0,
        full_suite=suite, commands=commands,
        builds=[artifact(path) for path in sorted(args.delivery.glob('dist*/*')) if path.is_file()],
        installed=receipts, dependency_install='package-only --no-deps; existing host closure reused',
        followups={name: dict(artifact=artifact(ROOT/'results/research-wave4'/name),
                              receipt=read(ROOT/'results/research-wave4'/name))
                   for name in ('numerical-tests-v3.json', 'numerical-installed-comparison.json')},
        documentation=dict(manifest=artifact(doc_folder/'build_manifest.json'),
                           documents=docs['documents'], unavailable_links=docs['unavailable_links']),
        custody=dict(protected_files=len(custody['files']), protected_drift=protected_drift,
                     frozen_files=len(freeze['files']), freeze_drift=freeze_drift),
        hygiene=dict(files_scanned=len(candidates), credential_pattern_files=secret_files,
                     restricted_asset_files=restricted_files),
        ci='Final remote HEAD and hosted run are verified after push and reported with delivery; initial CI is in initial-state.json')
    result['passed'] = (not protected_drift and not freeze_drift and
        all(command['returncode'] == 0 for command in commands) and
        suite['failures'] == '0' and suite['errors'] == '0' and
        docs['unavailable_links'] == 0 and not secret_files and not restricted_files)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('passed', 'full_suite', 'custody', 'hygiene')}))


if __name__ == '__main__':
    main()
