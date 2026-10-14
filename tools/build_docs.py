"""Build internal Markdown documentation for local review, without experiments.

Outputs are private/local build artifacts, never publication approval. Existing
directories are refused. Requires the existing Markdown package; installs nothing.
"""
import argparse
import html
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import sha256, write_json


def build(output):
    import markdown
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted(set(ROOT.glob("*.md")) | set((ROOT/"docs").rglob("*.md")))
    rendered, links = {}, []
    for source in sources:
        name = source.relative_to(ROOT).as_posix()
        text = source.read_text(encoding="utf-8-sig")
        body = markdown.markdown(text, extensions=["extra", "toc"])
        for destination in re.findall(r'href="([^"#]+)(?:#[^"]*)?"', body):
            destination = html.unescape(destination)
            if re.match(r"^[a-z]+://", destination) or destination.startswith("mailto:"):
                continue
            candidate = (source.parent/destination).resolve()
            links.append(dict(source=name, target=destination, present=candidate.exists(),
                classification="local_reference" if candidate.exists() else "unavailable_local_or_historical_reference"))
        # Rendered Markdown stays within this build. Other source/evidence
        # links open their original local files, without copying large banks.
        def rewrite_link(match):
            destination = html.unescape(match[1])
            if destination.startswith(("#", "mailto:")) or re.match(r"^[a-z]+://", destination):
                return match[0]
            raw, _, fragment = destination.partition("#")
            candidate = (source.parent/unquote(raw)).resolve()
            suffix = "#"+fragment if fragment else ""
            if candidate in sources:
                import os
                rendered_target = output/candidate.relative_to(ROOT).with_suffix(".html")
                location = Path(os.path.relpath(rendered_target, (output/source.relative_to(ROOT)).parent)).as_posix()
                return 'href="'+html.escape(location+suffix, quote=True)+'"'
            if candidate.is_file():
                return 'href="'+html.escape(candidate.as_uri()+suffix, quote=True)+'"'
            return match[0]
        body = re.sub(r'href="([^"]+)"', rewrite_link, body)
        target = output/Path(name).with_suffix(".html")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>'+html.escape(name)+'</title><style>'
            'body{max-width:1000px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#202020}'
            'pre{overflow:auto;background:#f5f5f5;padding:16px}table{border-collapse:collapse}'
            'td,th{border:1px solid #ddd;padding:8px}img{max-width:100%}</style><body>'+body+'</body></html>', encoding="utf-8")
        rendered[name] = dict(source_sha256=sha256(source), output=str(target.relative_to(output)), bytes=target.stat().st_size)
    from shutil import copy2
    # Reference files remain local, retaining source-relative destinations.
    assets = sorted(set((ROOT/"docs").rglob("*.json")) | set((ROOT/"images").rglob("*")) |
        {p for p in ROOT.glob("*") if p.is_file() and p.suffix in (".toml", ".txt", ".cff")})
    for source in assets:
        if not source.is_file():
            continue
        target = output/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, target)
    entries = ''.join('<li><a href="'+html.escape(v["output"])+'">'+html.escape(k)+'</a></li>' for k, v in rendered.items())
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Opal local documentation</title>'
        '<h1>Opal local documentation</h1><p>Internal review build. No publication or scientific execution is authorized.</p><ul>'+entries+'</ul>',encoding="utf-8")
    receipt = dict(status="built", scope="all root/docs Markdown; internal local review only",
        documents=len(rendered), sources=rendered, local_links=links,
        unavailable_links=sum(not r["present"] for r in links), numerical_work=False,
        publication_authorized=False, dependencies_installed=False)
    write_json(output/"build_manifest.json", receipt)
    print(json.dumps(dict(status="built", documents=len(rendered), unavailable_links=receipt["unavailable_links"], output=str(output))))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build(args.output)


if __name__ == "__main__":
    main()
