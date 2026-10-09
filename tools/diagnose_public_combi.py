"""Freeze scope, fetch bounded public aggregate reads, and report processing agreement."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.replicated.contracts import sha, write_json
from opal.replicated.public_diagnostic import processing_diagnostic

URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE174nnn/GSE174695/suppl/GSE174695_large_scale_matrix.csv.gz'


def load_matrix(path, source_ids, maximum_uncompressed_bytes=600_000_000):
    """Read genes by barcodes and restore metadata order, never positional joins."""
    vectors, genes, size = [], [], 0
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        header = next(csv.reader([stream.readline()]))
        columns = header[1:]
        if len(set(columns)) != len(columns) or set(columns) != set(source_ids):
            raise ValueError('Matrix source columns must match metadata IDs exactly')
        lookup = {name: i for i, name in enumerate(columns)}
        order = [lookup[name] for name in source_ids]
        for line in stream:
            size += len(line.encode('utf-8'))
            if size > maximum_uncompressed_bytes:
                raise ValueError('Uncompressed read matrix exceeds fixed byte budget')
            fields = next(csv.reader([line]))
            if len(fields) != len(header):
                raise ValueError('Malformed matrix row')
            genes.append(fields[0])
            vectors.append(np.asarray(fields[1:], dtype=np.float64)[order])
    if not vectors:
        raise ValueError('Empty public read matrix')
    return np.asarray(vectors).T, genes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fetch', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.metadata, 'rt', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    protocol = {'scope_fixed_utc': datetime.now(timezone.utc).isoformat(),
                'source_id': 'GSE174695', 'matrix_url': URL,
                'metadata_sha256': sha(args.metadata), 'runner_sha256': sha(Path(__file__)),
                'diagnostic_code_sha256': sha(ROOT / 'src/replicated/public_diagnostic.py'),
                'matrix_already_present': args.matrix.exists(),
                'role': 'exploratory_nominal_identity_processing_diagnostic',
                'normalization': 'library total to 10000, log2(1 + normalized reads); mean controls/singles before log',
                'pair_variants': 'mean logged effects within each run and canonical pair',
                'exclusions': 'zero-library rows; missing positive-library controls/singles/pairs; complete panel across runs',
                'gene_universe': 'all source-provided genes; upstream filtering is not independent held-pair selection',
                'metrics': ['per-run zero/additive/shared/repeated-pair MSE', 'pairwise-run residual cosine summaries'],
                'unqualified': ['physical dose units', 'vehicle compatibility', 'independent culture initiation'],
                'no_architecture_search': True, 'no_biological_intervals': True,
                'no_scientific_advancement': True, 'public_registration': False}
    write_json(args.output / 'scope_before_expression_access.json', protocol)
    if args.fetch and not args.matrix.exists():
        args.matrix.parent.mkdir(parents=True, exist_ok=True)
        partial = args.matrix.with_suffix(args.matrix.suffix + '.partial')
        with urlopen(Request(URL, headers={'User-Agent': 'Opal public processing diagnostic'}), timeout=30) as src, partial.open('xb') as dest:
            total = 0
            while block := src.read(65536):
                total += len(block)
                if total > 80_000_000:
                    raise ValueError('Compressed read matrix exceeds 80MB download budget')
                dest.write(block)
        partial.replace(args.matrix)
    access = {'expression_access_started_utc': datetime.now(timezone.utc).isoformat(),
              'matrix_sha256': sha(args.matrix), 'matrix_bytes': args.matrix.stat().st_size,
              'scope_sha256': sha(args.output / 'scope_before_expression_access.json')}
    write_json(args.output / 'expression_access.json', access)
    counts, genes = load_matrix(args.matrix, [r[''] for r in rows])
    report = processing_diagnostic(rows, counts, genes)
    report['input_provenance'] = access
    report['metadata_sha256'] = protocol['metadata_sha256']
    report['gene_order_sha256'] = __import__('hashlib').sha256(json.dumps(genes, separators=(',', ':')).encode()).hexdigest()
    write_json(args.output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ['role', 'source_rows', 'gene_count', 'complete_nominal_pair_count', 'primary_claim_pass']}))


if __name__ == '__main__':
    main()
