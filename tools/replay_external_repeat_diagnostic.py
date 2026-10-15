"""Independent CSV-reader/centered-sum replay of the small repeat diagnostic."""
import argparse
import csv
import gzip
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from opal.artifacts import read_json, write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('New replay receipt required')
    start = time.process_time()
    registration = read_json(args.run/'registration.json')
    report = read_json(args.run/'report.json')
    for name, key in [('tools/run_external_repeat_diagnostic.py', 'code_sha256'),
                      ('src/external_repeat.py', 'helper_sha256'),
                      ('configs/experiments/opal_external_repeat_diagnostic_v1.json', 'config_sha256')]:
        if sha256(ROOT/name) != registration[key]:
            raise ValueError('Diagnostic code/config custody mismatch')
    path = args.input/'GSE95153_Villanueva_processed.txt.gz'
    if sha256(path) != registration['matrix_sha256']:
        raise ValueError('Count source custody mismatch')
    with gzip.open(path, 'rt') as f:
        rows = csv.reader(f, delimiter='\t')
        header = next(rows)
        genes, data = [], []
        for row in rows:
            genes.append(row[0])
            data.append([float(v) for v in row[1:]])
    columns = header[1:]
    if len(genes) != report['gene_count'] or set(columns) != set(report['column_to_sample']):
        raise ValueError('Independent source axis mismatch')
    x = np.asarray(data, dtype=np.float64).T
    z = np.log2(1+1_000_000*x/x.sum(axis=1)[:, None])
    discrepancies = []
    for role, row in report['conditions'].items():
        a = z[[columns.index(f'{role}.{i}') for i in (1, 2, 3)]]
        centered = a-a.sum(axis=0)/3
        value = float(np.sum(centered*centered)/(2*len(genes)))
        discrepancies.append(abs(value-row['mean_gene_sample_variance']))
        if not np.isclose(value, row['mean_gene_sample_variance'], rtol=1e-12, atol=1e-14):
            raise ValueError('Independent variance mismatch')
        for pair in row['within_condition_pairwise_mse']:
            i, j = pair['repeats']
            value = float(np.sum((a[i-1]-a[j-1])**2)/len(genes))
            discrepancies.append(abs(value-pair['mse']))
            if not np.isclose(value, pair['mse'], rtol=1e-12, atol=1e-14):
                raise ValueError('Independent pairwise difference mismatch')
    write_json(args.output, dict(status='independent_repeat_summary_replay_passed',
        variance_summaries=5, pairwise_summaries=15, maximum_absolute_difference=max(discrepancies),
        report_sha256=sha256(args.run/'report.json'), matrix_sha256=sha256(path),
        replay_tool_sha256=sha256(__file__), cpu_seconds=time.process_time()-start,
        biological_confirmation=False, independent_cultures_verified=False))
    print('independent summary replay passed; maximum absolute difference', max(discrepancies))


if __name__ == '__main__':
    main()
