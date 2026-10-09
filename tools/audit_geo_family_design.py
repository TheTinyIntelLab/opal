"""Summarize local GEO SOFT metadata; never read supplementary expression."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def audit(path):
    raw = path.read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError('Compressed metadata exceeds budget')
    with gzip.open(path, 'rb') as stream:
        decoded = stream.read(20_000_001)
    if len(decoded) > 20_000_000:
        raise ValueError('Decompressed metadata exceeds budget')
    samples, current = [], None
    for line in decoded.decode('utf-8').splitlines():
        if line.startswith('^SAMPLE = '):
            current = {'accession': line.split(' = ', 1)[1], 'characteristics': {}}
            samples.append(current)
        elif current is not None and line.startswith('!Sample_characteristics_ch1 = '):
            value = line.split(' = ', 1)[1]
            if ': ' in value:
                key, value = value.split(': ', 1)
                current['characteristics'].setdefault(key.lower(), []).append(value)
    ids = [s['accession'] for s in samples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Missing or duplicated sample identities')
    values = defaultdict(Counter)
    for sample in samples:
        for key, entries in sample['characteristics'].items():
            values[key].update(entries)
    return {'source_file': path.name, 'source_sha256': hashlib.sha256(raw).hexdigest(),
            'compressed_bytes': len(raw), 'sample_count': len(samples),
            'characteristic_value_counts': {k: dict(sorted(v.items())) for k, v in sorted(values.items())},
            'expression_accessed': False, 'independent_cultures_inferred': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [audit(path) for path in sorted(args.snapshots.glob('GSE*.soft.gz'))]
    if not records:
        raise ValueError('No SOFT metadata found')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'role': 'sample_metadata_counts_not_culture_qualification', 'sources': records}, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'audited_sources': len(records), 'expression_accessed': False}))


if __name__ == '__main__':
    main()
