"""Hash-check cached COLO1 projections and summarize allowlisted ENA aliases."""
import collections
import hashlib
import json
import pathlib
import re
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / 'results/frontier-execution-20261026'
OUT = RECEIPTS / 'Grok_source_audit_v1.json'
ALIAS_RULES = (
    ('d14_index_coverage', re.compile(r'^EXP_d14-[123]-(?:100x|500x)_S\d+_L001_R[12]_001$')),
    ('d14_r_label', re.compile(r'^EXP_d14-R[123]_S\d+_L001_R[12]_001$')),
    ('d3_coverage', re.compile(r'^EXP_d3-(?:100x|500x)_S\d+_L001_R[12]_001$')),
    ('d3_plain', re.compile(r'^EXP_d3_S\d+_L001_R[12]_001$')),
    ('lib_gi', re.compile(r'^EXP_lib-GI_S\d+_L001_R[12]_001$')),
    ('lib_pilot', re.compile(r'^EXP_lib-PILOT_S\d+_L001_R[12]_001$')),
    ('lib_plain', re.compile(r'^EXP_lib_S\d+_L001_R[12]_001$')),
    ('sanger_sc', re.compile(r'^SC_EXP_35467_[12]#\d+$')),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_aliases(aliases):
    counts = {name: 0 for name, _rule in ALIAS_RULES}
    unmatched = []
    for alias in aliases:
        matched = False
        for name, rule in ALIAS_RULES:
            if rule.match(alias):
                counts[name] += 1
                matched = True
                break
        if not matched:
            unmatched.append(alias)
    return counts, unmatched


def sample_experiment_counts(rows):
    experiments = collections.defaultdict(set)
    for row in rows:
        experiments[row['sample_accession']].add(row['experiment_accession'])
    histogram = collections.Counter(len(value) for value in experiments.values())
    return {
        'samples': len(experiments),
        'experiments_per_sample': dict(sorted(histogram.items())),
    }


def load(name):
    return json.loads((RECEIPTS / name).read_text(encoding='utf-8'))


def main():
    if OUT.exists():
        raise RuntimeError('immutable audit exists')
    checks = []

    def check(label, ok, **extra):
        checks.append({'label': label, 'ok': bool(ok), **extra})

    v2 = load('A_source_requests_v2.json')
    for record in v2['requests']:
        path = record.get('path')
        if not path:
            continue
        cached = ROOT / 'tmp/colo1_frontier_v2' / path.replace('/', '__')
        check('v2 ' + path, cached.is_file() and sha256(cached) == record['sha256'], bytes=record.get('bytes'))
    v3 = load('A_source_requests_v3.json')
    article = ROOT / 'tmp/colo1_frontier_v3/article.xml'
    projection = ROOT / 'tmp/colo1_frontier_v3/methods_projection.json'
    check('v3 article bytes', article.is_file() and sha256(article) == v3['sha256'] and article.stat().st_size == v3['bytes'])
    check('v3 methods projection', projection.is_file() and sha256(projection) == v3['projection_sha256'])
    methods = json.loads(projection.read_text(encoding='utf-8'))
    method_text = methods['sections'][0]['text']
    readme = (ROOT / 'tmp/colo1_frontier_v2/README.md').read_text(encoding='utf-8')
    input_readme = (ROOT / 'tmp/colo1_frontier_v2/Analysis__ENCORE__input__readme.txt').read_text(encoding='utf-8')
    described = 40 * 24 + 404 * 12 + (40 * 404 * 4)
    documentary = {
        'readme_states_100136_vectors': '100136' in readme,
        'readme_states_40_anchors_and_404_library_genes': '40 anchor genes' in readme and '404 library genes' in readme,
        'readme_category_products': {'anchor_singletons': 960, 'library_singletons': 4848, 'library_combinations': 64640},
        'readme_described_category_sum': described,
        'readme_stated_total_minus_described_sum': 100136 - described,
        'methods_state_100136_vectors_and_444_library_genes': '100,136' in method_text and '444 library genes' in method_text,
        'methods_name_pycroquet_1_6_0_and_day_14': 'pyCROQUET v1.6.0' in method_text and '14 days' in method_text,
        'input_readme_names_annotation_object': 'IDs_colo1.rds' in input_readme,
        'library_gene_count_conflict': '404 versus 444',
    }
    v6 = load('A_source_requests_v6.json')
    for index, record in enumerate(v6['requests']):
        cached = ROOT / 'tmp/colo1_frontier_v6' / f'item_{index}.raw'
        check(f'v6 item {index}', cached.is_file() and sha256(cached) == record['sha256'], outcome=record['outcome'])
    catalog = v6['requests'][0]['catalog']
    v7 = load('A_source_projection_v7.json')
    check('v7 source hash', v7['source_sha256'] == v6['requests'][1]['sha256'])
    rows = v7['metadata_rows']
    alias_counts, unmatched = group_aliases([row['experiment_alias'] for row in rows])
    valued_attributes = 0
    sample_titles = []
    v8 = load('A_source_requests_v8.json')
    for record in v8['requests']:
        cached = ROOT / 'tmp/colo1_frontier_v8' / f"{record['accession']}.xml"
        check('v8 ' + record['accession'], cached.is_file() and sha256(cached) == record['sha256'], outcome=record['outcome'])
        for attribute in record['metadata'].get('sample_attributes', []):
            valued_attributes += int('value' in attribute)
        for title in record['metadata'].get('TITLE', []):
            if record['accession'].startswith('SAMEA'):
                sample_titles.append({'accession': record['accession'], 'title': title.get('text')})
    audit = {
        'attribution': 'Cursor Grok 4.7 continuation; not an A completion',
        'checks': checks,
        'checks_failed': sum(not item['ok'] for item in checks),
        'documentary_strings': documentary,
        'figshare_catalog': {
            'doi': catalog.get('doi'),
            'title': catalog.get('title'),
            'license_name': catalog.get('license', {}).get('name'),
            'files': [
                {'id': item.get('id'), 'name': item.get('name'), 'size': item.get('size'), 'computed_md5': item.get('computed_md5')}
                for item in catalog.get('files', [])
            ],
            'archive_body_downloaded_by_A': False,
        },
        'ena_allowlisted_summary': {
            'rows': len(rows),
            'study_accessions': sorted({row['study_accession'] for row in rows}),
            'sample_accession_count': len({row['sample_accession'] for row in rows}),
            'experiment_accession_count': len({row['experiment_accession'] for row in rows}),
            'run_accession_count': len({row['run_accession'] for row in rows}),
            'experiments_per_sample': sample_experiment_counts(rows),
            'library_layout': dict(collections.Counter(row['library_layout'] for row in rows)),
            'instrument_model': dict(collections.Counter(row['instrument_model'] for row in rows)),
            'library_name': dict(collections.Counter(row['library_name'] or '<blank>' for row in rows)),
            'alias_groups': alias_counts,
            'unmatched_alias_count': len(unmatched),
            'allowlisted_sample_attribute_values': valued_attributes,
            'sample_titles': sample_titles,
        },
        'article_xml_parsed_by_this_audit': False,
        'response_tables_opened': False,
        'archive_directory_request': 'not_started',
        'free_bytes': shutil.disk_usage(ROOT).free,
    }
    OUT.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'checks_failed': audit['checks_failed'], 'rows': len(rows), 'unmatched': len(unmatched)}))


if __name__ == '__main__':
    main()
