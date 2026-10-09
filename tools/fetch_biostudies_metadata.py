"""Retrieve bounded BioStudies study JSON only, without following data files."""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import io
from pathlib import Path
import re
from urllib.request import urlopen


def validate_study(data, accession):
    if len(data) > 2_000_000:
        raise ValueError('Study metadata exceeds budget')
    payload = json.loads(data)
    if payload.get('accno') != accession or not isinstance(payload.get('section'), dict):
        raise ValueError('Unexpected study identity or missing section')
    return payload


def validate_sdrf(data):
    if len(data) > 2_000_000:
        raise ValueError('Sample metadata exceeds budget')
    rows = list(csv.reader(io.StringIO(data.decode('utf-8')), delimiter='\t'))
    if len(rows) < 2 or rows[0][0] != 'Source Name' or any(len(r) != len(rows[0]) for r in rows[1:]):
        raise ValueError('Expected rectangular SDRF sample metadata')
    return rows


def summarize_sdrf(data):
    rows = validate_sdrf(data)
    header = rows[0]
    factors = [i for i, name in enumerate(header) if name.startswith('Factor Value[')]
    identities = [i for i, name in enumerate(header) if name in
                  ('Comment[ENA_SAMPLE]', 'Extract Name', 'Characteristics[cell line]')]
    if not factors or len(set(header[i] for i in factors)) != len(factors):
        raise ValueError('Missing or duplicate experimental factors')
    sources = defaultdict(list)
    for row in rows[1:]:
        if not row[0].strip():
            raise ValueError('Missing source identity')
        sources[row[0]].append(row)
    for source_rows in sources.values():
        for i in identities + factors:
            if len({r[i] for r in source_rows}) != 1:
                raise ValueError('Conflicting metadata within a source')
    return {'source_sha256': hashlib.sha256(data).hexdigest(), 'assay_row_count': len(rows) - 1,
            'unique_source_count': len(sources),
            'assay_rows_per_source_histogram': dict(sorted(Counter(len(v) for v in sources.values()).items())),
            'unique_source_factor_counts': {header[i]: dict(sorted(Counter(v[0][i] for v in sources.values()).items()))
                                             for i in factors},
            'independent_cultures_inferred': False, 'matched_initiation_blocks_inferred': False,
            'expression_accessed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accessions', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--sdrf', action='store_true', help='Fetch sample relationship metadata instead of study JSON')
    args = parser.parse_args()
    if (not 1 <= len(args.accessions) <= 12 or len(set(args.accessions)) != len(args.accessions)
            or any(not re.fullmatch(r'E-MTAB-\d+', a) for a in args.accessions)):
        raise ValueError('Require one through twelve unique ArrayExpress studies')
    args.output.mkdir(parents=True, exist_ok=False)
    args.snapshots.mkdir(parents=True, exist_ok=False)
    scope = {'fixed_utc': datetime.now(timezone.utc).isoformat(), 'accessions': args.accessions,
             'role': 'sdrf_metadata_only' if args.sdrf else 'study_metadata_only_no_data_file_following',
             'request_budget_bytes': 2_000_000,
             'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'published_narratives_may_be_encountered': True}
    (args.output / 'scope.json').write_text(json.dumps(scope, indent=2) + '\n', encoding='utf-8')
    ledger = []
    for accession in args.accessions:
        name = accession + ('.sdrf.txt' if args.sdrf else '.json')
        url = (f'https://www.ebi.ac.uk/biostudies/files/{accession}/{name}' if args.sdrf
               else f'https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}')
        record = {'accession': accession, 'url': url, 'expression_accessed': False}
        try:
            with urlopen(url, timeout=30) as stream:
                data = stream.read(2_000_001)
            if args.sdrf:
                validate_sdrf(data)
            else:
                validate_study(data, accession)
            (args.snapshots / name).write_bytes(data)
            record.update({'status': 'retrieved', 'snapshot': name, 'bytes': len(data),
                           'sha256': hashlib.sha256(data).hexdigest()})
        except Exception as error:
            record.update({'status': 'retrieval_failed', 'error_type': type(error).__name__})
        ledger.append(record)
    (args.output / 'retrieval.json').write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'successful_requests': sum(r['status'] == 'retrieved' for r in ledger),
                      'failed_requests': sum(r['status'] != 'retrieved' for r in ledger)}))


if __name__ == '__main__':
    main()
