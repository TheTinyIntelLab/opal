"""Run scoped GSE56188 diagnostics without assuming matched culture blocks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from opal.replicated.donor_diagnostic import read_series_matrix
from opal.replicated.unmatched_diagnostic import lncap_array_design, unmatched_replicate_diagnostic

URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE56nnn/GSE56188/matrix/GSE56188_series_matrix.txt.gz'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--series-metadata', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    rows = lncap_array_design(args.metadata.read_text(encoding='utf-8'))
    processing = "Data were analysed using Partek Genomics Suite. Briefly, .cel files were normalised using RMA background correction, Partek's own GC content correction, and mean probe summarization."
    if args.metadata.read_text(encoding='utf-8').count('!Sample_data_processing = ' + processing) != 42:
        raise ValueError('Expected deposited continuous-assay processing statement for every sample')
    series = args.series_metadata.read_text(encoding='utf-8')
    if ('^SERIES = GSE56188' not in series or '5 uM bicalutamide' not in series
            or '1 uM vorinostat (SAHA)' not in series or '40 nM 17-AAG' not in series
            or 'for 6 hours' not in series):
        raise ValueError('Expected source dose/time metadata')
    args.output.mkdir(parents=True, exist_ok=False)
    import opal.replicated.unmatched_diagnostic as implementation
    scope = {'fixed_utc': datetime.now(timezone.utc).isoformat(), 'source': 'GSE56188',
             'role': 'descriptive_processed_assay_diagnostic', 'matrix_url': URL,
             'metadata_sha256': sha(args.metadata), 'series_metadata_sha256': sha(args.series_metadata),
             'implementation_sha256': sha(Path(implementation.__file__)), 'runner_sha256': sha(Path(__file__)),
             'source_parser_sha256': sha(Path(__import__('opal.replicated.donor_diagnostic', fromlist=['x']).__file__)),
             'published_narratives_encountered': True,
             'partial_expression_values_encountered_in_indexed_GSM_search_results': True,
             'full_matrix_already_local': args.matrix.exists(), 'expected_samples': 42,
             'reported_biological_repeats_per_arm': 6, 'canonical_pairs': 3,
             'scope_fixed_before_full_matrix_download': not args.matrix.exists(),
             'endpoint': 'All deposited finite unique probes; no source-significant gene filtering',
             'scale': 'Deposited Partek RMA/GC-corrected mean probe signal, no normalization refit or count inversion',
             'references': 'Arm means; equal repeat numbering is never treated as matched experimental blocks',
             'comparisons': ['zero', 'additive', 'same-pair-other-samples', 'whole-held-pair other-pair mean offset'],
             'diagnostics': ['equal-pair MSE', 'per-sample losses', 'within-pair residual cosine'],
             'source_qualification_limits': ['initiation-block map unknown', 'vehicle fraction unknown',
                                              'three pairs provide a small graph', 'deposited normalization uses study outcomes'],
             'no_confidence_intervals_or_biological_gate': True, 'no_model_menu_search': True,
             'maximum_download_bytes': 20_000_000}
    dump(args.output / 'scope_before_full_matrix_access.json', scope)
    if args.download:
        if args.matrix.exists():
            raise ValueError('Refuse to overwrite source matrix')
        args.matrix.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=60) as stream:
            data = stream.read(20_000_001)
        if len(data) > 20_000_000:
            raise ValueError('Matrix exceeds download budget')
        with args.matrix.open('xb') as stream:
            stream.write(data)
    digest = sha(args.matrix)
    dump(args.output / 'expression_access.json', {'url': URL, 'sha256': digest,
         'bytes': args.matrix.stat().st_size, 'access_utc': datetime.now(timezone.utc).isoformat(),
         'expression_now_exposed': True, 'earlier_indexed_expression_snippet_exposure': True})
    columns, probes, values = read_series_matrix(args.matrix)
    report = unmatched_replicate_diagnostic(rows, columns, probes, values)
    report.update({'source_matrix_sha256': digest, 'metadata_sha256': sha(args.metadata)})
    dump(args.output / 'report.json', report)
    columns2, probes2, values2 = read_series_matrix(args.matrix)
    replay = unmatched_replicate_diagnostic(rows[::-1], columns2[::-1], probes2, values2[::-1])
    if any(report[k] != value for k, value in replay.items()):
        raise ValueError('Exact replay mismatch')
    dump(args.output / 'verification.json', {'all_diagnostic_fields_replayed_exactly': True,
         'independent_matrix_parse': True, 'reversed_columns_and_metadata': True,
         'source_hash_verified': sha(args.matrix) == digest, 'original_protocol_gate_pass': False})
    print(json.dumps({'probes': len(probes), 'mse': report['equal_pair_mean_mse'], 'gate_pass': False}))


if __name__ == '__main__':
    main()
