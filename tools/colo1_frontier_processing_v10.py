"""Narrow the existing v9 static view; never re-read the upstream source."""
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.colo1_frontier_processing_v9 import OP

ROOT = Path(__file__).resolve().parents[1]
QUOTED = re.compile(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*' ''', re.VERBOSE)
RENDER = re.compile(r'\b(?:print|plot|barplot|text|legend|pdf|png|jpeg|svg|ggplot|ggsave|points|lines|title|axis|dev\.off|write\.\w+)\s*\(')


def retain(line):
    code = QUOTED.sub('', line)
    return bool(OP.search(code)) and not RENDER.search(code)


def main():
    src = ROOT / 'tmp/colo1-frontier-processing-v9/operations.txt'
    file_bytes = src.read_bytes()
    raw = file_bytes.replace(b'\r\n', b'\n')
    if hashlib.sha256(raw).hexdigest() != '10415afb638d4d689d68a80dc7be7dd848661f5b5eb15fb01a06792789f58281':
        raise ValueError('Registered v9 quarantine hash')
    lines = raw.decode().splitlines(keepends=True)
    kept = [line for line in lines if retain(line.split(': ', 1)[1])]
    if not 0 < len(kept) < len(lines):
        raise ValueError('Correction must strictly narrow the existing view')
    out = ROOT / 'tmp/colo1-frontier-processing-v10'
    out.mkdir(exist_ok=False)
    selected = ''.join(kept).encode()
    (out / 'operations.txt').write_bytes(selected)
    receipt = dict(source='v9 quarantine only', input_sha256=hashlib.sha256(raw).hexdigest(),
                   input_file_sha256=hashlib.sha256(file_bytes).hexdigest(),
                   input_line_endings='v9 receipt binds logical LF projection; Windows quarantine uses CRLF',
                   projection_sha256=hashlib.sha256(selected).hexdigest(),
                   lines=len(kept), removed_lines=len(lines)-len(kept),
                   code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   registration_sha256=hashlib.sha256((ROOT / 'docs/frontier_execution/processing_projection_v10.md').read_bytes()).hexdigest(),
                   new_source_lines=0, response_reads=0, network_requests=0, upstream_executed=False)
    dest = ROOT / 'results/frontier-execution-20261026/processing-projection-v10.json'
    with dest.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(lines=len(kept), removed_lines=len(lines)-len(kept), new_source_lines=0)))


if __name__ == '__main__':
    main()
