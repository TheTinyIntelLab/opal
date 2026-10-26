"""Single registered static operation-line projection; no upstream execution."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tmp/colo1-annotation-provenance-v1/Rscript_final.R'
REG = ROOT / 'docs/frontier_execution/processing_projection_v9.md'
EXPECTED = 'a5ca73f871273ba1b370fdea4e3969e9badf58698a085cfc486dd5fa08e08127'
OP = re.compile(r'read\.table|readRDS|filter|rowSums|colSums|sweep|log2|scale|mean|median|is\.na|na\.omit|count|normaliz')
OUTPUT = re.compile(r'^\s*(?:#|\[\d+\]|\d+\s+\S|A tibble|>)')
RENDER = re.compile(r'\b(?:print|plot|ggplot|ggsave|write\.\w+)\s*\(')
LITERAL = re.compile(r'(?i)\b(?:lfc|fitness|score|gamma|phenotype|count\w*)\s*(?:=|<-)\s*[-+]?\d|\bc\s*\(\s*[-+]?\d')


def project(text):
    rows = [(i, line) for i, line in enumerate(text.splitlines(), 1)
            if OP.search(line) and not OUTPUT.search(line) and not RENDER.search(line)
            and not LITERAL.search(line)]
    rendered = ''.join(f'{i}: {line}\n' for i, line in rows)
    if len(rows) > 128 or len(rendered.encode()) > 16384:
        raise ValueError('Frozen line/byte projection cap')
    return rows, rendered


def main():
    out = ROOT / 'results/frontier-execution-20261026/processing-projection-v9.json'
    local = ROOT / 'tmp/colo1-frontier-processing-v9'
    if out.exists() or local.exists():
        raise FileExistsError('Projection receipt/quarantine already exists')
    raw = SOURCE.read_bytes()
    if len(raw) != 87060 or hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Pinned cached source identity')
    rows, rendered = project(raw.decode('utf-8'))
    local.mkdir()
    (local / 'operations.txt').write_text(rendered, encoding='utf-8')
    receipt = dict(registration_sha256=hashlib.sha256(REG.read_bytes()).hexdigest(),
                   code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_sha256=EXPECTED, lines=len(rows), projection_bytes=len(rendered.encode()),
                   projection_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                   line_numbers=[i for i, _ in rows], network_requests=0,
                   response_tables_headers_values_opened=False, upstream_executed=False,
                   source_qualified=False, code_exported=False)
    out.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(lines=len(rows), bytes=len(rendered.encode()), source_qualified=False)))


if __name__ == '__main__':
    main()
