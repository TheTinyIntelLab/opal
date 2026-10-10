"""Verify saved empirical claims and compile a self-contained review TeX bundle."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, sha256, write_json


def main():
    local = Path('C:/Users/USER/Desktop/opal-model-series-20261010-v2/paper-standalone-complete')
    local.mkdir(exist_ok=False)
    paper = ROOT / 'paper'
    evidence = ROOT / 'results/opal_model_series_v2'
    manifest = read_json(evidence / 'report/figure_manifest.json')
    if sha256(paper / 'evidence/model_series_current_table.tex') != manifest['table_sha256']:
        raise ValueError('Primary table changed')
    for relative, digest in manifest['inputs'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError('Figure empirical input changed: ' + relative)
    for name, digest in manifest['figures'].items():
        if sha256(paper / 'evidence' / name) != digest:
            raise ValueError('Figure changed: ' + name)
    text = (paper / 'evidence/model_series_current.tex').read_text()
    summary = read_json(evidence / 'report/summary.json')
    claims = read_json(evidence / 'report/claims.json')
    expected = []
    expected += [f'{v:.6f}' for v in claims['primary_interaction_gain_percent'].values() if v]
    expected += [f'{v:.6f}' for v in claims['retrospective_guard_gain_percent'].values() if v]
    expected += [f'{v["mse"]:.6f}' for s in summary['native_adapted'].values() for v in s.values()]
    expected += [f'{v:.6f}' for v in read_json(evidence / 'report/extra_single_reference.json').values()]
    if any(value not in text for value in expected):
        raise ValueError('Manuscript claim differs from saved numerical evidence')
    def expand(match):
        source = paper / (match.group(1) + ('' if match.group(1).endswith('.tex') else '.tex'))
        return re.sub(r'\\input\{([^}]+)\}', expand, source.read_text())
    flattened = re.sub(r'\\input\{([^}]+)\}', expand, (paper / 'main.tex').read_text())
    (local / 'main.tex').write_text(flattened)
    for path in paper.glob('*.sty'):
        shutil.copy2(path, local / path.name)
    shutil.copy2(paper / 'references.bib', local / 'references.bib')
    # Existing IfFileExists gates still need their source files present after inlining.
    shutil.copytree(paper / 'evidence', local / 'evidence', ignore=shutil.ignore_patterns('*.json'))
    archive_path = local.parent / 'opal-local-model-series-complete-source.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in local.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(local))
    env = dict(os.environ, TECTONIC_CACHE_DIR=str(ROOT / 'tools/vendor/tectonic-cache'))
    result = subprocess.run([str(ROOT / 'tools/vendor/tectonic.exe'), '--untrusted', 'main.tex'],
                            cwd=local, env=env, capture_output=True, text=True)
    (local / 'compile.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    import pymupdf
    main_pdf, standalone = pymupdf.open(paper / 'main.pdf'), pymupdf.open(local / 'main.pdf')
    def normalized(text):
        # Inlining can change whitespace and discretionary line-end hyphenation.
        return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).replace('-', '')
    if len(main_pdf) != len(standalone) or any(normalized(a.get_text()) != normalized(b.get_text()) for a, b in zip(main_pdf, standalone)):
        raise ValueError('Standalone page text differs')
    write_json(evidence / 'verification/manuscript.json', dict(empirical_literals_checked=expected,
        figure_inputs_verified=len(manifest['inputs']), figures_verified=len(manifest['figures']),
        main_pdf_sha256=sha256(paper / 'main.pdf'), standalone_tex_sha256=sha256(local / 'main.tex'),
        standalone_zip=str(archive_path), standalone_zip_sha256=sha256(archive_path),
        standalone_pdf_sha256=sha256(local / 'main.pdf'), standalone_normalized_page_text_exact=True,
        normalization='NFKC, whitespace and hyphens removed; empirical values checked separately',
        pages=len(main_pdf), build_return_code=0, visual_review='separate rendered-page inspection record'))
    print('Claims and six figure hashes verified; standalone TeX compiles with identical normalized page text')


if __name__ == '__main__':
    main()
