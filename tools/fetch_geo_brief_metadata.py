"""Fetch bounded GEO brief metadata without family expression tables."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import urlopen


def validate_brief(data, expected_kind):
    if len(data) > 2_000_000:
        raise ValueError('Brief metadata response exceeds budget')
    text = data.decode('utf-8')
    if any(marker in text.lower() for marker in ('!sample_table_begin', '!series_matrix_table_begin')):
        raise ValueError('Expression table in metadata response')
    if not text.startswith('^' + expected_kind + ' = '):
        raise ValueError('Expected GEO brief text, possibly blocked or HTML response')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accessions', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    args = parser.parse_args()
    if (not 1 <= len(args.accessions) <= 12 or len(set(args.accessions)) != len(args.accessions)
            or any(not re.fullmatch(r'GSE\d+', a) for a in args.accessions)):
        raise ValueError('Require one through twelve unique GEO series')
    args.output.mkdir(parents=True, exist_ok=False)
    args.snapshots.mkdir(parents=True, exist_ok=False)
    (args.output / 'scope.json').write_text(json.dumps({'fixed_utc': datetime.now(timezone.utc).isoformat(),
        'accessions': args.accessions, 'role': 'brief_metadata_only', 'request_budget_bytes': 2_000_000,
        'maximum_requests': 2 * len(args.accessions), 'published_narratives_may_be_encountered': True,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    ledger = []
    for accession in args.accessions:
        for target, kind in (('self', 'SERIES'), ('gsm', 'SAMPLE')):
            url = f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}&targ={target}&form=text&view=brief'
            record = {'accession': accession, 'target': target, 'url': url}
            try:
                with urlopen(url, timeout=30) as stream:
                    data = stream.read(2_000_001)
                text = validate_brief(data, kind)
                name = f'{accession}-{target}.soft'
                (args.snapshots / name).write_bytes(data)
                record.update({'snapshot': name, 'bytes': len(data),
                               'sha256': hashlib.sha256(data).hexdigest(),
                               'record_count': sum(line.startswith('^' + kind + ' = ') for line in text.splitlines()),
                               'status': 'retrieved', 'expression_table_present': False})
            except Exception as error:
                record.update({'status': 'retrieval_failed', 'error_type': type(error).__name__})
            ledger.append(record)
            time.sleep(.5)
    (args.output / 'retrieval.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps({'successful_requests': sum(r['status'] == 'retrieved' for r in ledger),
                      'failed_requests': sum(r['status'] != 'retrieved' for r in ledger)}))


if __name__ == '__main__':
    main()
