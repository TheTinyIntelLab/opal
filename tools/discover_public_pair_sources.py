"""Bounded GEO metadata discovery, without downloading expression or weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

QUERIES = [
    '(combinatorial[Title] OR "drug combination"[Title] OR "double knockout"[Title] OR "double mutant"[Title]) AND gse[ETYP]',
    '("genetic interaction" OR "combination treatment" OR "co-treatment") AND ("RNA-seq" OR "transcriptome") AND gse[ETYP]',
]
PAIR_TERMS = ('drug combination', 'combinatorial perturb', 'double knockout', 'double mutant',
              'genetic interaction', 'dual perturbation', 'co-treatment', 'cotreatment', 'combination treatment')
REPEAT_TERMS = ('biological replic', 'independent cultures', 'independently cultured', 'independent experiment')


def validate_queries(queries):
    if (not isinstance(queries, list) or not 1 <= len(queries) <= 4
            or any(not isinstance(q, str) or not q.strip() or len(q) > 4000
                   or 'gse[etyp]' not in q.lower() for q in queries)
            or len(set(queries)) != len(queries)):
        raise ValueError('Require one through four unique bounded GEO series queries')
    return queries


def search_ids(payload):
    result = payload.get('esearchresult', {})
    if result.get('errorlist') or payload.get('error') or not result.get('querytranslation'):
        raise ValueError('Search query rejected or untranslated')
    identifiers = result.get('idlist', [])
    if len(identifiers) != len(set(identifiers)) or any(not str(i).isdigit() for i in identifiers):
        raise ValueError('Invalid search identifiers')
    return identifiers


def summarize(payload):
    """Index sample counts and textual leads, never infer biological units."""
    result = payload.get('result', {})
    if payload.get('error') or 'uids' not in result:
        raise ValueError('Missing metadata summary result')
    records = []
    for uid in result['uids']:
        row = result.get(str(uid), {})
        accession = row.get('accession', '')
        if row.get('error') or not re.fullmatch(r'GSE\d+', accession) or row.get('entrytype') != 'GSE':
            raise ValueError('Expected GEO series metadata')
        text = (row.get('title', '') + ' ' + row.get('summary', '')).lower()
        pairs = [term for term in PAIR_TERMS if term in text]
        repeats = [term for term in REPEAT_TERMS if term in text]
        samples = int(row.get('n_samples', 0))
        if samples < 0:
            raise ValueError('Negative source sample count')
        records.append({'accession': accession, 'uid': str(uid), 'title': row.get('title', ''),
                        'organism': row.get('taxon', ''), 'assay': row.get('gdstype', ''),
                        'source_sample_count': samples, 'source_publication_date': row.get('pdat'),
                        'pubmed_ids': row.get('pubmedids', []),
                        'primary_url': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=' + accession,
                        'pair_text_signals': pairs, 'replication_text_signals': repeats,
                        'priority_score': 3 * len(pairs) + 2 * len(repeats),
                        'source_summary_sha256': hashlib.sha256(row.get('summary', '').encode()).hexdigest(),
                        'primary_confirmation_qualified': False,
                        'qualification': 'not_assessed_from_summary', 'expression_accessed': False})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--snapshots', type=Path, required=True)
    p.add_argument('--limit-per-query', type=int, default=150)
    p.add_argument('--query-file', type=Path, help='JSON list of one through four explicit GEO series queries')
    args = p.parse_args()
    if not 1 <= args.limit_per_query <= 150:
        raise ValueError('Fixed query budget is 1 through 150 records')
    selected_queries = validate_queries(json.loads(args.query_file.read_text(encoding='utf-8'))
                                        if args.query_file else QUERIES)
    args.output.mkdir(parents=True, exist_ok=False)
    args.snapshots.mkdir(parents=True, exist_ok=False)
    scope = {'fixed_utc': datetime.now(timezone.utc).isoformat(), 'queries': selected_queries,
             'query_file_sha256': hashlib.sha256(args.query_file.read_bytes()).hexdigest() if args.query_file else None,
             'limit_per_query': args.limit_per_query, 'sort': 'NCBI default, no unsupported sort parameter',
             'role': 'metadata_only_discovery', 'published_narratives_may_be_encountered': True,
             'expression_or_weights_downloaded': False, 'maximum_total_bytes': 80_000_000,
             'per_response_bytes': 12_000_000, 'minimum_request_spacing_seconds': .5,
             'priority_rule': '3 per matched pair-text phrase plus 2 per replication-text phrase; ties by sample count then accession',
             'qualification_boundary': 'Text signals and sample counts never establish independent cultures, matching references or assay fit',
             'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / 'scope.json').write_text(json.dumps(scope, indent=2) + '\n')
    ledger, used, last_request = [], 0, 0.

    def fetch(endpoint, parameters, name):
        nonlocal used, last_request
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/' + endpoint + '?' + urlencode(parameters)
        for attempt in range(3):
            time.sleep(max(0, .5 - (time.monotonic() - last_request)))
            last_request = time.monotonic()
            try:
                with urlopen(Request(url, headers={'User-Agent': 'Opal bounded metadata discovery'}), timeout=30) as stream:
                    data = stream.read(12_000_001)
                used += len(data)
                if len(data) > 12_000_000 or used > 80_000_000:
                    raise ValueError('Metadata response exceeds fixed byte budget')
                (args.snapshots / name).write_bytes(data)
                ledger.append({'url': url, 'snapshot': name, 'bytes': len(data),
                               'sha256': hashlib.sha256(data).hexdigest(), 'attempt': attempt + 1})
                return json.loads(data)
            except (HTTPError, URLError, TimeoutError) as error:
                if attempt == 2:
                    raise RuntimeError(f'Metadata request failed: {endpoint}') from error
                time.sleep(2 ** (attempt + 1))

    queries, identifiers = [], set()
    for i, query in enumerate(selected_queries):
        payload = fetch('esearch.fcgi', {'db': 'gds', 'term': query, 'retmode': 'json',
                        'retmax': args.limit_per_query, 'tool': 'opal-source-audit'}, f'search-{i}.json')
        ids = search_ids(payload)
        identifiers.update(ids)
        result = payload['esearchresult']
        queries.append({'query': query, 'total_index_matches': int(result['count']),
                        'retrieved_ids': len(ids), 'translation': result['querytranslation'],
                        'warnings': result.get('warninglist', {})})
    records = []
    ordered = sorted(identifiers, key=int)
    for start in range(0, len(ordered), 20):
        payload = fetch('esummary.fcgi', {'db': 'gds', 'id': ','.join(ordered[start:start + 20]),
                        'retmode': 'json', 'tool': 'opal-source-audit'}, f'summary-{start // 20}.json')
        batch = summarize(payload)
        if {r['uid'] for r in batch} != set(ordered[start:start + 20]):
            raise ValueError('Summary identity coverage mismatch')
        records.extend(batch)
    records.sort(key=lambda r: (-r['priority_score'], -r['source_sample_count'], r['accession']))
    inventory = {'role': 'bounded_metadata_inventory_not_qualification', 'queries': queries,
                 'candidate_count': len(records), 'primary_qualified_count': 0,
                 'primary_qualification_performed': False, 'expression_or_weights_downloaded': False,
                 'candidates': records, 'retrievals': ledger, 'total_retrieved_bytes': used}
    (args.output / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({'metadata_records': len(records), 'retrieval_bytes': used,
                      'top_accessions': [r['accession'] for r in records[:20]],
                      'primary_qualification_performed': False}))


if __name__ == '__main__':
    main()
