"""Run the scoped GSE78083 single-pair donor diagnostic and exact replay."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from opal.replicated.donor_diagnostic import (matched_donors, parse_brief_samples,
                                            read_series_matrix, single_pair_donor_diagnostic)

MATRIX_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE78nnn/GSE78083/matrix/GSE78083_series_matrix.txt.gz'
EXCLUDED_DONORS = ('AA_185', 'EA_252')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = args.metadata.read_text(encoding='utf-8')
    all_rows = parse_brief_samples(metadata)
    rows = [row for row in all_rows if row['title'].rsplit('_', 1)[0] not in EXCLUDED_DONORS]
    design = matched_donors(rows, [row['accession'] for row in rows])
    if len(all_rows) != 80 or len(design) != 18 or len(rows) != 72:
        raise ValueError('Expected 18 unambiguous donors after two metadata-only exclusions')
    processing = '!Sample_data_processing = quantile normalized and variance transformed using LUMI package in R'
    if sum(line.strip() == processing for line in metadata.splitlines()) != 80:
        raise ValueError('Source scale statement missing or inconsistent')
    import opal.replicated.donor_diagnostic as implementation
    scope = {'fixed_utc': datetime.now(timezone.utc).isoformat(), 'source': 'GSE78083',
             'metadata_sha256': sha(args.metadata), 'implementation_sha256': sha(Path(implementation.__file__)),
             'runner_sha256': sha(Path(__file__)), 'role': 'exposed_retrospective_single_pair_donor_diagnostic',
             'published_narratives_already_encountered': True, 'expression_previously_accessed_by_this_runner': False,
             'expected_donors': 18, 'expected_samples': 72, 'deposited_samples': 80,
             'excluded_donors_before_expression_access': list(EXCLUDED_DONORS),
             'exclusion_reason': 'GSM2066485 and GSM2066487 are titled EL but characterized as combined VL treatment; exclude complete donors without outcome-based label repair',
             'primary_methods': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4856085/',
             'methods_review': {'vitamin_d_nM': 100, 'lps_ng_per_mL': 10, 'ethanol_vehicle_percent': 1,
                                'hours': 24, 'within_donor_wells_pooled_per_arm': 3,
                                'biological_unit': 'donor, not individual pooled well'},
             'endpoint': 'All deposited finite unique probe IDs, no gene filtering or remapping',
             'processing': 'Retain deposited LUMI normalized intensity, no counts or log-CPM conversion',
             'comparisons': ['zero', 'matched-donor additive', 'leave-one-donor-out same-pair residual transfer'],
             'diagnostics': ['equal-donor MSE', 'per-donor losses', 'pairwise residual cosine median'],
             'same_pair_transfer_boundary': 'Other donors reveal this same pair, forbidden under original held-pair design',
             'no_model_menu_search': True, 'no_biological_gate_or_fresh_confirmation': True,
             'no_confidence_intervals': True, 'maximum_download_bytes': 40_000_000,
             'matrix_url': MATRIX_URL, 'matrix_already_local': args.matrix.exists()}
    dump(args.output / 'scope_before_expression_access.json', scope)
    if args.download:
        if args.matrix.exists():
            raise ValueError('Refuse to overwrite source matrix')
        args.matrix.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(MATRIX_URL, timeout=90) as stream:
            data = stream.read(40_000_001)
        if len(data) > 40_000_000:
            raise ValueError('Source download exceeds budget')
        with args.matrix.open('xb') as stream:
            stream.write(data)
    digest = sha(args.matrix)
    dump(args.output / 'expression_access.json', {'url': MATRIX_URL, 'sha256': digest,
         'bytes': args.matrix.stat().st_size, 'access_utc': datetime.now(timezone.utc).isoformat(),
         'expression_now_exposed': True})
    columns, probes, values = read_series_matrix(args.matrix)
    if len(set(columns)) != 80 or set(columns) != {row['accession'] for row in all_rows}:
        raise ValueError('Deposited matrix must match all 80 metadata sample identities')
    selected = [i for i, column in enumerate(columns) if column in {row['accession'] for row in rows}]
    columns, values = [columns[i] for i in selected], values[selected]
    report = single_pair_donor_diagnostic(rows, columns, probes, values)
    report.update({'source_matrix_sha256': digest, 'metadata_sha256': sha(args.metadata),
                   'probe_order_sha256': hashlib.sha256(json.dumps(probes).encode()).hexdigest()})
    dump(args.output / 'report.json', report)
    # Replay from independently parsed inputs, including reversed metadata order.
    replay_columns, replay_probes, replay_values = read_series_matrix(args.matrix)
    replay_columns, replay_values = [replay_columns[i] for i in selected], replay_values[selected]
    replay = single_pair_donor_diagnostic(list(reversed(rows)), replay_columns, replay_probes, replay_values)
    expected = {key: report[key] for key in replay}
    if expected != replay:
        raise ValueError('Exact diagnostic replay mismatch')
    dump(args.output / 'verification.json', {'all_diagnostic_fields_replayed_exactly': True,
         'independent_input_parse': True, 'source_hash_verified': sha(args.matrix) == digest,
         'donors': report['donors'], 'source_probe_count': report['source_probe_count'],
         'original_protocol_gate_pass': False})
    print(json.dumps({'donors': report['donors'], 'probes': report['source_probe_count'],
                      'mse': report['equal_donor_mean_mse'], 'gate_pass': False}))


if __name__ == '__main__':
    main()
