"""Package frozen research assets without refitting or accessing evaluator outcomes.

Run after committing the release files. Models are local, separately licensed
exports; source/evidence paths come from the Git index, never a workspace glob.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def allowed_source(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        return False
    if any(part in {'web', 'vendor', 'external', '__pycache__'} for part in path.parts):
        return False
    if path.name in {'plan.md', 'outreach.md', 'context.md', 'roadmap.md', 'opal_next_master_plan.md', 'build_evidence_demo.py'}:
        return False
    if len(path.parts) == 1:
        return path.suffix in {'.md', '.toml', '.cff', '.in', '.py'} or name in {'LICENSE', '.gitignore', '.gitattributes', 'pytest.ini'} or (path.name.startswith('requirements') and path.suffix == '.txt')
    return path.parts[0] in {'src', 'tools', 'tests', 'configs', 'docs', 'paper', 'images'} and path.suffix in {
        '.py', '.json', '.yaml', '.yml', '.md', '.tex', '.bib', '.sty', '.pdf', '.txt', '.png', '.svg'}


def allowed_evidence(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and '..' not in path.parts and
            path.parts[0] in {'results', 'data', 'paper'} and
            path.suffix in {'.json', '.md', '.tex', '.pdf'} and
            not any(part in {'raw', 'prepared', 'demo-v1', 'local-release-v1',
                             'local-release-v2', 'local-release-v3', 'local-release-v4'}
                    for part in path.parts))


def archive(output, entries):
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(entries.items()):
            z.write(path, name)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC check failed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dist', type=Path, default=ROOT / 'dist')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/releases/v0.1.1.json')
    p.add_argument('--allow-dirty', action='store_true', help='Local validation only; never upload these assets')
    args = p.parse_args()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    if dirty and not args.allow_dirty:
        raise ValueError('Commit release inputs before creating final assets')
    args.output.mkdir(parents=True, exist_ok=False)
    command = ['git', 'ls-files', '-z']
    if args.allow_dirty:
        command += ['--cached', '--others', '--exclude-standard']
    tracked = set(subprocess.check_output(command, cwd=ROOT).decode().split('\0'))
    source = {n: ROOT / n for n in tracked if n and (ROOT / n).is_file() and allowed_source(n)}
    evidence = {n: ROOT / n for n in tracked if n and (ROOT / n).is_file() and allowed_evidence(n)}
    for name in ['LICENSE', 'THIRD_PARTY_NOTICES.md', 'docs/evidence.md']:
        evidence[name] = ROOT / name
    config = json.loads(args.config.read_text())
    prefix = 'opal-' + config['version']
    models = {n: ROOT / n for n in ['LICENSE', 'THIRD_PARTY_NOTICES.md', 'docs/models.md']}
    for label, directory in config['models'].items():
        directory = ROOT / directory
        manifest = json.loads((directory / 'manifest.json').read_text())
        files = {'manifest.json', *manifest['files']}
        for name in files:
            if Path(name).name != name or Path(name).suffix not in {'.npz', '.json'}:
                raise ValueError('Unexpected model export file')
            path = directory / name
            if name != 'manifest.json' and digest(path) != manifest['files'][name]:
                raise ValueError('Model export hash mismatch')
            models[f'models/{label}/{name}'] = path
    assets = args.output
    archive(assets / (prefix + '-research-source.zip'), source)
    archive(assets / (prefix + '-evidence.zip'), evidence)
    archive(assets / (prefix + '-reference-models.zip'), models)
    for name in [f"opal_research-{config['version']}-py3-none-any.whl", f"opal_research-{config['version']}.tar.gz"]:
        shutil.copyfile(args.dist / name, assets / name)
    shutil.copyfile(ROOT / 'paper/main.pdf', assets / (prefix + '-manuscript.pdf'))
    arxiv_source = ROOT / 'releases/arxiv' / (prefix + '-arxiv-source.zip')
    if not arxiv_source.is_file():
        raise FileNotFoundError('Build the standalone arXiv source bundle before packaging')
    shutil.copyfile(arxiv_source, assets / arxiv_source.name)
    record = {
        'version': config['version'], 'tag': config['tag'], 'scope': config['scope'],
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'dirty_inputs': bool(dirty), 'raw_datasets_included': False,
        'private_plan_included': False, 'model_data_license': config['model_data_license'],
        'excluded_weights': ['Wessels', 'D-SPIN', 'native CPA/GEARS', 'external priors', 'learned state'],
        'source_files': {n: digest(f) for n, f in sorted(source.items())},
        'evidence_files': {n: digest(f) for n, f in sorted(evidence.items())},
        'model_files': {n: digest(f) for n, f in sorted(models.items())},
        'assets': {f.name: {'sha256': digest(f), 'bytes': f.stat().st_size}
                   for f in sorted(assets.iterdir())},
    }
    (assets / (prefix + '-manifest.json')).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    sums = ''.join(f'{digest(f)}  {f.name}\n' for f in sorted(assets.iterdir()))
    (assets / 'SHA256SUMS.txt').write_text(sums, encoding='utf-8')
    print(json.dumps({'output': str(assets), 'files': len(list(assets.iterdir())),
                      'bytes': sum(f.stat().st_size for f in assets.iterdir()),
                      'dirty_inputs': bool(dirty)}, indent=2))


if __name__ == '__main__':
    main()
