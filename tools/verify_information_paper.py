"""Check numerical literals, compile standalone TeX, render changed pages."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unicodedata

import pymupdf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import read_json,sha256,write_json
from prepare_information import OUT,LOCAL,resource_check


def run(label='paper-review'):
    resource_check(8000000)
    paper=ROOT/'paper';local=LOCAL/label;local.mkdir(exist_ok=False)
    summary=read_json(OUT/'development/summary.json')
    text=(paper/'evidence/information_development.tex').read_text()
    expected=[f"{r['mse'][a]:.8f}" for r in summary['aggregates'] if not r['precision'] for a in ['Base','mean','capacity_mean','variance','shuffle']]
    expected+=['0.231163','0.451677','0.466825'][:2]
    if any(v not in text for v in expected):raise ValueError('Numerical addendum differs from saved results')
    # Preserve earlier six figures and their evidence inputs exactly.
    figures=read_json(ROOT/'results/opal_model_series_v2/report/figure_manifest.json')
    for p,h in figures['inputs'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Historical figure input changed')
    for p,h in figures['figures'].items():
        if sha256(paper/'evidence'/p)!=h:raise ValueError('Historical figure changed')
    def expand(match):
        path=paper/(match.group(1)+('' if match.group(1).endswith('.tex') else '.tex'))
        return re.sub(r'\\input\{([^}]+)\}',expand,path.read_text())
    flattened=re.sub(r'\\input\{([^}]+)\}',expand,(paper/'main.tex').read_text())
    (local/'main.tex').write_text(flattened)
    for p in paper.glob('*.sty'):shutil.copy2(p,local/p.name)
    shutil.copy2(paper/'references.bib',local/'references.bib')
    shutil.copytree(paper/'evidence',local/'evidence',ignore=shutil.ignore_patterns('*.json'))
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(ROOT/'tools/vendor/tectonic-cache'))
    result=subprocess.run([str(ROOT/'tools/vendor/tectonic.exe'),'--untrusted','main.tex'],cwd=local,env=env,capture_output=True,text=True,timeout=1800)
    (local/'build.stdout.txt').write_text(result.stdout);(local/'build.stderr.txt').write_text(result.stderr)
    if result.returncode:raise RuntimeError(result.stderr)
    current=pymupdf.open(paper/'main.pdf');standalone=pymupdf.open(local/'main.pdf')
    normalized=lambda t:re.sub(r'\s+','',unicodedata.normalize('NFKC',t)).replace('-','')
    if len(current)!=len(standalone) or any(normalized(a.get_text())!=normalized(b.get_text()) for a,b in zip(current,standalone)):raise ValueError('Standalone text mismatch')
    addendum=[i for i,p in enumerate(current) if 'Measured-single variance information' in p.get_text() or '0.08256839' in p.get_text()]
    first=min(addendum)
    rendered=[]
    for i in range(first,len(current)):
        destination=local/f'page-{i+1}.png';current[i].get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).save(destination)
        rendered.append(str(destination))
    write_json(OUT/'verification/manuscript.json',{'pdf_sha256':sha256(paper/'main.pdf'),'standalone_pdf_sha256':sha256(local/'main.pdf'),
        'standalone_tex_sha256':sha256(local/'main.tex'),'normalized_page_text_parity':True,'pages':len(current),
        'empirical_literals_checked':expected,'historical_figures_verified':len(figures['figures']),
        'changed_pages_rendered':rendered,'visual_review':'pending inspection of rendered pages','old_pdf_preservation':'paper_preservation.json'})
    print(len(current),'pages; numerical and standalone checks pass; rendered pages',list(range(first+1,len(current)+1)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',default='paper-review')
    run(parser.parse_args().label)
