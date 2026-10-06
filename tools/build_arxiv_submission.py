"""Build a standalone arXiv TeX bundle and compile an extracted copy locally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tectonic')
    args = parser.parse_args()
    paper = ROOT / 'paper'
    config = json.loads((ROOT / 'configs/releases/v0.1.1.json').read_text())
    needed = {'main.tex', 'arxiv.sty', 'references.bib', 'main.bbl'}
    for name in ['main.tex', *[p.relative_to(paper).as_posix() for p in (paper / 'evidence').glob('*.tex')]]:
        text = (paper / name).read_text()
        for target in re.findall(r'\\input\{([^}]+)\}|\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
            dependency = next(t for t in target if t)
            if not Path(dependency).suffix:
                dependency += '.tex'
            needed.add(dependency)
    for path in (paper / 'evidence').iterdir():
        if path.suffix in {'.tex', '.pdf'}:
            needed.add(path.relative_to(paper).as_posix())
    for name in needed:
        path = (paper / name).resolve()
        if not path.is_relative_to(paper.resolve()) or not path.is_file():
            raise ValueError('Missing or unsafe TeX dependency: ' + name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Choose a new bundle name; previous artifacts are preserved')
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(needed):
            archive.write(paper / name, name)
        archive.write(ROOT / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")
    outside = Path(tempfile.mkdtemp(prefix='opal-arxiv-source-'))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert 'main.pdf' not in archive.namelist()
        archive.extractall(outside)
    binary = args.tectonic or shutil.which('tectonic') or str(ROOT / 'tools/vendor/tectonic.exe')
    env = dict(os.environ)
    env['TECTONIC_CACHE_DIR'] = str(ROOT / 'tools/vendor/tectonic-cache')
    compiled = subprocess.run([binary, '--untrusted', '--keep-logs', 'main.tex'], cwd=outside,
                              env=env, capture_output=True, text=True)
    if compiled.returncode:
        raise RuntimeError(compiled.stdout + compiled.stderr)
    sources = {name: hashlib.sha256((paper / name).read_bytes()).hexdigest() for name in sorted(needed)}
    sources['THIRD_PARTY_NOTICES.md'] = hashlib.sha256((ROOT / 'THIRD_PARTY_NOTICES.md').read_bytes()).hexdigest()
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', (paper / 'main.tex').read_text(), re.S)[1].strip()
    abstract = abstract.replace(r'\%', '%')
    report = {'version': config['version'], 'status': 'local_preparation_not_submitted',
              'files': sources, 'bundle_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'extracted_source_compiled': True, 'compiler': binary,
              'arxiv_server_compiler_tested': False, 'abstract': abstract,
              'title': 'Opal: an observed-single benchmark for combinatorial perturbation prediction',
              'authors': ['Arjun Vijay Prakash'], 'temporary_compile_directory': str(outside)}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'source_bundle': str(args.output), 'files': len(sources),
                      'extracted_source_compiled': True, 'status': report['status']}, indent=2))


if __name__ == '__main__':
    main()
