"""Project safe SPIDR library metadata; never parse guide-activity outcomes.

The public library CSV includes historical activity annotations. Only the
allowlisted identity/sequence columns enter memory as field values. Other field
characters are scanned for CSV structure and discarded, never converted,
retained, scored or exported. No response-count table is accepted by this tool.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


ARCHIVE_HASH = 'c5bb66b0f3bbd1d1677ce862b5188a6a9201c47cf68e27af56e4a9a62d52a9a6'
MEMBER = 'Supplementary Table 2/Supplementary Table 2.csv'
SAFE_NAMES = ('sgRNA name', 'gene', 'perfect match sgRNA', 'sequence', 'transcript', 'mismatched')
SAFE_COLUMNS = tuple(range(1, 7)) + tuple(range(13, 19))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def project_line(line, selected, width):
    """Read one non-multiline CSV record, discarding unselected field values.

    This deliberately rejects multiline records instead of broadening its
    source-specific contract. Quoted commas and escaped quotes are supported.
    """
    text = line.rstrip('\r\n')
    keep = set(selected)
    output = {}
    column = 0
    value = []
    quoted = False
    start = True
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            if quoted and index + 1 < len(text) and text[index + 1] == '"':
                if column in keep:
                    value.append('"')
                index += 1
            elif quoted:
                quoted = False
            elif start:
                quoted = True
            else:
                raise ValueError('Unexpected quote inside unquoted field')
        elif char == ',' and not quoted:
            if column in keep:
                output[column] = ''.join(value)
            value = []
            column += 1
            start = True
            index += 1
            continue
        elif column in keep:
            value.append(char)
        start = False
        index += 1
    if quoted:
        raise ValueError('Multiline or unterminated quoted field is not allowed')
    if column in keep:
        output[column] = ''.join(value)
    if column + 1 != width or set(output) != keep:
        raise ValueError('Unexpected source record width')
    return output


def qualify(archive_path, original_root):
    archive_path = Path(archive_path)
    original_root = Path(original_root)
    if sha256(archive_path) != ARCHIVE_HASH:
        raise ValueError('Unregistered library archive bytes')
    original = original_root / 'prepared/profiles.npz'
    with np.load(original, allow_pickle=False) as data:
        genes = data['genes'].astype(str)
        supported = set(genes[np.all(data['single_counts'] >= 20, axis=(0, 1))])
        original_pairs = {tuple(sorted((genes[a], genes[b]))) for a, b in data['pairs']}
    if len(supported) != 427:
        raise ValueError('Original metadata support changed')
    old_guides = {}
    guide_path = original_root / 'GSE116198_GImap_sgRNA_sequences_barcodes.txt.gz'
    with gzip.open(guide_path, 'rt', encoding='utf-8-sig', newline='') as stream:
        for row in csv.reader(stream, delimiter='\t'):
            if len(row) != 9:
                raise ValueError('Unexpected original guide schema')
            old_guides[row[1]] = (row[1].split('_')[0], row[5].upper())
    guides, pairs, rows = {}, set(), 0
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != [MEMBER]:
            raise ValueError('Archive contains unexpected members')
        with archive.open(MEMBER) as binary:
            with io.TextIOWrapper(binary, encoding='utf-8-sig', newline='') as stream:
                group_header = next(csv.reader([stream.readline()]))
                header = next(csv.reader([stream.readline()]))
                if len(header) != 25 or tuple(header[i] for i in range(1, 7)) != SAFE_NAMES:
                    raise ValueError('Unexpected first-arm metadata header')
                if tuple(header[i] for i in range(13, 19)) != SAFE_NAMES:
                    raise ValueError('Unexpected second-arm metadata header')
                if group_header[1:13] != ['gene 1'] * 12 or group_header[13:] != ['gene 2'] * 12:
                    raise ValueError('Unexpected arm-group header')
                for line in stream:
                    projected = project_line(line, SAFE_COLUMNS, 25)
                    row_genes = []
                    for start in (1, 13):
                        identity = tuple(projected[start + j] for j in range(6))
                        name, gene, perfect, sequence, transcript, mismatched = identity
                        if name in guides and guides[name] != identity:
                            raise ValueError('Inconsistent guide identity metadata')
                        guides[name] = identity
                        row_genes.append(gene)
                    pairs.add(tuple(sorted(row_genes)))
                    rows += 1
    source_genes = {value[1] for value in guides.values()}
    overlap = source_genes & supported
    supported_pairs = {pair for pair in pairs if pair[0] != pair[1] and set(pair) <= supported}
    matches = []
    old_sequence_to_genes = {}
    for gene, sequence in old_guides.values():
        old_sequence_to_genes.setdefault(sequence, set()).add(gene)
    for identity in guides.values():
        name, gene, perfect, sequence, transcript, mismatched = identity
        if gene in supported and gene in old_sequence_to_genes.get(sequence.upper(), set()):
            matches.append({'name': name, 'gene': gene, 'mismatched': mismatched})
    match_genes = {item['gene'] for item in matches}
    sequence_matched_pairs = {pair for pair in supported_pairs if set(pair) <= match_genes}
    return {
        'schema_version': 1,
        'status': 'metadata_only_candidate_not_prediction_or_endpoint_qualification',
        'source': {'doi': '10.1038/s41586-025-08815-4', 'bioproject': 'PRJNA988447',
                   'url': 'https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-08815-4/MediaObjects/' + archive_path.name,
                   'path': str(archive_path), 'bytes': archive_path.stat().st_size, 'sha256': ARCHIVE_HASH,
                   'archive_member': MEMBER, 'member_bytes': 190604826,
                   'article_license': 'CC BY 4.0 unless material-specific exception',
                   'license_evidence': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12018271/fullTextXML',
                   'raw_sra_license': 'not separately verified'},
        'projection': {'allowed_columns': list(SAFE_NAMES), 'allowed_zero_based_indices': list(SAFE_COLUMNS),
                       'disallowed_guide_activity_headers': ['predicted relative activity (mean)', 'mean relative gamma'],
                       'outcome_values_converted_retained_or_scored': False,
                       'response_count_tables_downloaded_or_opened': False,
                       'archive_member_crc_validated': True},
        'counts': {'library_rows': rows, 'unique_guides': len(guides), 'unique_gene_labels_including_controls': len(source_genes),
                   'unique_unordered_label_pairs_including_controls_and_same_gene': len(pairs),
                   'original_supported_genes': len(supported), 'shared_supported_genes': len(overlap),
                   'distinct_supported_gene_pairs': len(supported_pairs),
                   'pairs_absent_original_eligible_bank': len(supported_pairs - original_pairs),
                   'pairs_present_original_eligible_bank': len(supported_pairs & original_pairs),
                   'supported_exact_sequence_matched_guides': len(matches),
                   'supported_genes_with_at_least_one_exact_sequence_match': len(match_genes),
                   'supported_pairs_with_both_genes_having_at_least_one_exact_sequence_match': len(sequence_matched_pairs)},
        'shared_supported_genes': sorted(overlap),
        'sequence_matched_supported_genes': sorted(match_genes),
        'original_metadata': {'profiles_sha256': sha256(original), 'guide_metadata_sha256': sha256(guide_path)},
        'limitations': [
            'Library membership is not observed T0/T14 coverage or eligible matched-single support.',
            'A gene with one matched sequence does not mean all constructs for that gene are reagent matched.',
            'Exact sequences do not match knockdown level, growth window, oxygen, cell context or gamma scale.',
            'Source RPE-1 TP53 KO cultures were split after transduction, not independent infection initiations.',
            'Keep response table unopened until task-specific LFC protocol, prediction freeze and coverage rules are registered.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous qualification')
    result = qualify(args.archive, args.original_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result['counts'], indent=2))
