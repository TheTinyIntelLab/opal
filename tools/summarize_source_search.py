"""Merge bounded GEO discovery inventories without inferring assay qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def summarize(paths):
    sources, accessions, searches = [], {}, []
    for path in paths:
        raw = path.read_bytes()
        inventory = json.loads(raw)
        if inventory.get('role') != 'bounded_metadata_inventory_not_qualification':
            raise ValueError('Expected a metadata discovery inventory')
        rows = inventory['candidates']
        ids = [r['accession'] for r in rows]
        if len(set(ids)) != len(ids) or any(not re.fullmatch(r'GSE\d+', a) for a in ids):
            raise ValueError('Invalid or duplicated series identities')
        if inventory['candidate_count'] != len(rows):
            raise ValueError('Candidate count does not match identities')
        if inventory.get('expression_or_weights_downloaded') is not False:
            raise ValueError('Expected expression-unopened metadata inventory')
        before = set(accessions)
        source_id = len(sources)
        sources.append({'path': path.as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()})
        searches.append({'source_index': source_id, 'records': len(rows),
                         'new_summary_identities': len(set(ids) - before),
                         'overlapping_summary_identities': len(set(ids) & before),
                         'queries': inventory['queries']})
        for row in rows:
            entry = accessions.setdefault(row['accession'], {'accession': row['accession'],
                'source_indices': [], 'summary_sha256_values': [],
                'qualification': 'not_assessed_by_inventory_merge'})
            entry['source_indices'].append(source_id)
            if row['source_summary_sha256'] not in entry['summary_sha256_values']:
                entry['summary_sha256_values'].append(row['source_summary_sha256'])
    return {'role': 'summary_identity_coverage_only', 'sources': sources, 'searches': searches,
            'unique_summary_records': len(accessions),
            'repeated_summary_records': sum(len(s['source_indices']) - 1 for s in accessions.values()),
            'changed_summary_identities': sum(len(s['summary_sha256_values']) > 1 for s in accessions.values()),
            'qualification_performed': False,
            'boundary': 'Series identity overlap is not biological replication, canonical intervention identity, or deep source review coverage.',
            'series': [accessions[a] for a in sorted(accessions, key=lambda a: int(a[3:]))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventories', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(set(p.resolve() for p in args.inventories)) != len(args.inventories):
        raise ValueError('Duplicate inventory paths')
    report = summarize(args.inventories)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('unique_summary_records', 'repeated_summary_records',
                                          'changed_summary_identities', 'qualification_performed')}))


if __name__ == '__main__':
    main()
