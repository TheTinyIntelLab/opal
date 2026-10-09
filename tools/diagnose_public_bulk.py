"""Reproduce the scoped LNCaP processed-bulk diagnostic with literal source joins."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.replicated.bulk_diagnostic import processed_bulk_diagnostic
from opal.replicated.contracts import sha, write_json


def metadata_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        text = stream.read(30_000_001)
    if len(text) > 30_000_000:
        raise ValueError('Metadata exceeds fixed read budget')
    rows = []
    for block in text.split('^SAMPLE = ')[1:]:
        lines = block.splitlines()
        row = {'accession': lines[0].strip()}
        for line in lines:
            if line.startswith('!Sample_title = '):
                row['title'] = line.split(' = ', 1)[1]
            elif line.startswith('!Sample_characteristics_ch1 = '):
                key, value = line.split(' = ', 1)[1].split(': ', 1)
                row[key] = value
            elif line.startswith('!Sample_treatment_protocol_ch1 = '):
                row['treatment_protocol'] = line.split(' = ', 1)[1]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--access', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    access = json.loads(args.access.read_text())
    scope = json.loads(args.scope.read_text())
    if access['matrix_sha256'] != sha(args.matrix) or access['scope_sha256'] != sha(args.scope):
        raise ValueError('Expression access/scope hash mismatch')
    if scope['metadata_sha256'] != sha(args.metadata):
        raise ValueError('Metadata hash mismatch')
    # Existing scope authorized a finite fixed descriptive analysis before access.
    args.output.mkdir(parents=True, exist_ok=False)
    genes, values, size = [], [], 0
    with gzip.open(args.matrix, 'rt', encoding='utf-8', newline='') as stream:
        columns = next(csv.reader([stream.readline()]))[1:]
        for line in stream:
            size += len(line.encode('utf-8'))
            if size > scope['uncompressed_budget_bytes']:
                raise ValueError('Uncompressed matrix exceeds fixed budget')
            fields = next(csv.reader([line]))
            if len(fields) != len(columns) + 1:
                raise ValueError('Malformed processed matrix row')
            genes.append(fields[0])
            values.append(np.asarray(fields[1:], float))
    report = processed_bulk_diagnostic(metadata_rows(args.metadata), columns, np.asarray(values).T, genes)
    report['input_provenance'] = access
    report['metadata_sha256'] = scope['metadata_sha256']
    report['code_sha256'] = {str(path.relative_to(ROOT)): sha(path) for path in
                             [Path(__file__), ROOT / 'src/replicated/bulk_diagnostic.py']}
    write_json(args.output / 'report.json', report)
    print(json.dumps({'genes': report['gene_count'], 'time_strata': [
        {'hour': s['nominal_hour'], 'mse': s['equal_repeat_mean_mse']} for s in report['strata']],
        'primary_claim_pass': False}))


if __name__ == '__main__':
    main()
