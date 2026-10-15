"""Reproduce bounded public source-metadata counts, never infer cultures."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import sha256, write_json, read_json
from opal.external_repeat import soft_samples, metadata_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.process_time()
    receipts = read_json(args.input/'acquisition.json') + read_json(args.input/'additional_acquisition.json')
    sources = {}
    for record in receipts:
        if record['status'] != 'metadata_acquired':
            raise ValueError('Incomplete acquisition: '+record['accession'])
        path = args.input/(record['accession']+'_family.soft.gz')
        if sha256(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('Source metadata custody mismatch')
        sources[record['accession']] = dict(receipt=record, **metadata_summary(soft_samples(path)))
    write_json(args.output/'metadata_counts.json', dict(sources=sources,
        helper_sha256=sha256(ROOT/'src/external_repeat.py'), tool_sha256=sha256(__file__),
        expression_opened_by_this_command=False, independent_biological_qualification=False,
        compressed_metadata_bytes=sum(r['bytes'] for r in receipts)))
    write_json(args.output/'resource_ledger.json', dict(cpu_seconds=time.process_time()-start,
        network_requests=0, fitting=False, inputs=len(sources),
        maximum_expanded_bytes_per_source=20_000_000, retained_bytes=sum(p.stat().st_size for p in args.output.iterdir())))
    print('metadata sources', len(sources), 'samples', {k:v['sample_count'] for k,v in sources.items()})


if __name__ == '__main__':
    main()
