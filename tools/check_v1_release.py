"""Check v1 claims, public entry-point links and code-only artifact contents."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DOCS = (
    'README.md', 'RELEASE_NOTES.md', 'RELEASE_CHECKLIST.md', 'docs/README.md',
    'docs/installation.md', 'docs/usage.md', 'docs/research.md', 'docs/reproduction.md',
    'docs/evidence.md', 'docs/models.md', 'docs/distribution.md', 'docs/archive/README.md',
    'docs/frontier/README.md',
    'results/README.md',
)
FORBIDDEN_SUFFIXES = {'.npz', '.npy', '.pt', '.pth', '.h5ad', '.h5', '.parquet', '.xlsx'}
PRIVATE_NAMES = {'plan.md', 'outreach.md', 'outreach_register.json', 'context.md', 'roadmap.md',
                 'release_cleanup_plan.md', 'release_deletions.json', 'opal_repository_audit.md'}


def inspect_member(name, payload, synthetic_hashes=None, history_hashes=None):
    path = Path(name)
    synthetic = (name.startswith('results/research-continuation/replication-v1/')
                 and path.suffix == '.npz' and synthetic_hashes is not None
                 and synthetic_hashes.get(name) == hashlib.sha256(payload).hexdigest())
    if (path.suffix.lower() in FORBIDDEN_SUFFIXES and not synthetic) or path.name.casefold() in PRIVATE_NAMES:
        raise ValueError('Private/model/data member: ' + name)
    historical_metadata = (path.suffix == '.json' and history_hashes is not None
                           and history_hashes.get(name) == hashlib.sha256(payload).hexdigest())
    if {'raw', 'prepared', 'custodian', 'sealed'} & set(path.parts) and not historical_metadata:
        raise ValueError('Private data directory: ' + name)
    # A harmless-looking outer ZIP must not conceal model/reference arrays.
    if path.suffix.lower() in {'.zip', '.whl'}:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                if not member.is_dir():
                    inspect_member(member.filename, archive.read(member), synthetic_hashes, history_hashes)


def check(artifacts=None):
    import markdown
    claims = json.loads((ROOT / 'paper/evidence/v1_release_claims.json').read_text())
    source = ROOT / claims['source']
    if hashlib.sha256(source.read_bytes()).hexdigest() != claims['source_sha256']:
        raise ValueError('Scientific source changed')
    benchmark = json.loads(source.read_text())
    if claims['program'] != benchmark['program'] or benchmark['program']['passes']:
        raise ValueError('Failed promotion gate was changed')
    table = (ROOT / 'paper/evidence/v1_growth_results.tex').read_text()
    for row in claims['tasks']:
        original = benchmark['tasks'][row['task']]
        simple = original['models']['selected_simple']['common_support']['mean']['mse']
        completion = original['models']['selected_policy']['common_support']['mean']['mse']
        if (row['simple_mse'], row['completion_mse'], row['pairs']) != (simple, completion, original['pairs']):
            raise ValueError('Claim projection mismatch')
        if f'{simple:.9g}' not in table or f'{completion:.9g}' not in table:
            raise ValueError('Manuscript table mismatch')
        for name in ('README.md', 'RELEASE_NOTES.md', 'docs/research.md'):
            if f'{100*(1-completion/simple):.2f}%' not in (ROOT / name).read_text(encoding='utf-8'):
                raise ValueError('Missing benchmark gain in ' + name)
    links = []
    for name in ACTIVE_DOCS:
        text = (ROOT / name).read_text(encoding='utf-8')
        rendered = markdown.markdown(text, extensions=['extra', 'toc'])
        for target in re.findall(r'(?:href|src)="([^"]+)"', rendered):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            destination = (ROOT / name).parent / unquote(parsed.path) if parsed.path else ROOT / name
            if not destination.exists():
                raise ValueError(f'Broken active link {name}: {target}')
            if parsed.fragment and destination.suffix == '.md':
                document = markdown.markdown(destination.read_text(encoding='utf-8'), extensions=['extra', 'toc'])
                if f'id="{unquote(parsed.fragment)}"' not in document:
                    raise ValueError(f'Broken active anchor {name}: {target}')
            links.append(dict(source=name, target=target))
    inspected = []
    history = json.loads((ROOT / 'docs/archive/research-records.json').read_text())
    synthetic_hashes = {r['path']: r['sha256'] for r in history['files']
                        if r['kind'] == 'synthetic_engineering_state'}
    history_hashes = {r['path']: r['sha256'] for r in history['files']}
    if artifacts:
        for path in sorted(Path(artifacts).iterdir()):
            if path.suffix in {'.zip', '.whl'}:
                with zipfile.ZipFile(path) as archive:
                    if archive.testzip() is not None:
                        raise ValueError('Archive CRC failure')
                    for member in archive.infolist():
                        if not member.is_dir():
                            inspect_member(member.filename, archive.read(member), synthetic_hashes, history_hashes)
            elif path.name.endswith('.tar.gz'):
                with tarfile.open(path) as archive:
                    for member in archive.getmembers():
                        if member.isfile():
                            inspect_member(member.name, archive.extractfile(member).read(), synthetic_hashes, history_hashes)
            else:
                continue
            inspected.append(dict(name=path.name, bytes=path.stat().st_size,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(status='passed', scientific_tasks=len(claims['tasks']), promotion_gate='failed_preserved',
                active_documents=len(ACTIVE_DOCS), local_links_checked=len(links),
                artifacts=inspected, nested_archives_checked=True,
                scope='active v1 docs; historical files retain dated paths and documented local prerequisites')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.artifacts)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
