"""Prepare frozen external-screen reference growth; never fit or score predictions."""
from array import array
from contextlib import contextmanager
import argparse
import csv
import gzip
import io
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import sha256, write_json

# Source identities and preprocessing were fixed before external model scoring.
COHORTS = {
    'environment1': ('environment/GSE312636_GI_expt1_counts.txt.gz',
        '105353899a4b91bc5446bfacc20ee91a13ff6bf17f8d53805fc8bb543f2a1403',
        ('Rep1_T0', 'Rep2_T0', 'Rep1_UT', 'Rep2_UT'), (9.632926829, 9.559756098)),
    'environment2': ('environment/GSE312636_GI_expt2_counts.txt.gz',
        'f9f5c85fc81e366709fd5d42f1c3c85388f00ab19624e14ebede53f72e09080b',
        ('t0_r1', 't0_r2', 'dmso_r1', 'dmso_r2'), (5.761137265, 6.324829553)),
    'parp2022': ('parp/screen2022_raw_counts.zip',
        'e72067bdc280ea8ed41c5a629273068eadc939db5194e6590b395526340e052e',
        ('T0.R1', 'T0.R2', 'DMSO.R1', 'DMSO.R2'), (7.19, 7.73)),
    'parp2023': ('parp/screen2023_raw_counts.zip',
        '1b3ecdc668e4bf693987dc00e9eb93ae8e5f1d5e5f78f694fbfe7096519dbb96',
        ('T0.R1', 'T0.R2', 'DMSO.R1', 'DMSO.R2'), (8.04, 7.98)),
}


@contextmanager
def count_stream(path):
    """Read without extracting archives or retaining large text tables."""
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith('_raw_counts.tsv')]
            if len(names) != 1:
                raise ValueError('Expected one raw count TSV in ZIP')
            with archive.open(names[0]) as binary:
                with io.TextIOWrapper(binary, encoding='utf-8-sig') as stream:
                    yield stream
    else:
        with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as stream:
            yield stream


def read_counts(path, columns, kind):
    """Return compact integer guide identities and reference-only count arrays."""
    names, genes, index = [], [], {}
    positions = [array('I'), array('I')]
    values = [array('d') for _ in columns]

    def guide_id(name, gene):
        if not name or not gene:
            raise ValueError('Missing guide or gene identity')
        if name not in index:
            index[name] = len(names)
            names.append(name)
            genes.append(gene)
        elif genes[index[name]] != gene:
            raise ValueError('Inconsistent guide-to-gene annotation')
        return index[name]

    with count_stream(Path(path)) as stream:
        first_line = stream.readline()
        delimiter = '\t' if '\t' in first_line else ','
        headers = next(csv.reader([first_line], delimiter=delimiter))
        required = list(columns) + (['FirstPosition', 'FirstGene', 'SecondPosition', 'SecondGene']
                                    if kind == 'parp' else ['double'])
        if len(set(headers)) != len(headers) or not set(required).issubset(headers):
            raise ValueError('Unexpected count schema')
        cols = [headers.index(c) for c in columns]
        if kind == 'parp':
            ids = [headers.index(c) for c in ['FirstPosition', 'FirstGene', 'SecondPosition', 'SecondGene']]
        else:
            double = headers.index('double')
        for row in csv.reader(stream, delimiter=delimiter):
            if len(row) != len(headers):
                raise ValueError('Malformed count row')
            if kind == 'parp':
                a, ga, b, gb = [row[i] for i in ids]
            else:
                pair = row[double].split(':')
                if len(pair) != 2:
                    raise ValueError('Malformed environmental guide pair')
                a, b = pair
                ga, gb = a.split('_')[0], b.split('_')[0]
            positions[0].append(guide_id(a, ga))
            positions[1].append(guide_id(b, gb))
            for out, col in zip(values, cols):
                value = float(row[col]) if row[col].strip() else float('nan')
                if not np.isnan(value) and (not np.isfinite(value) or value < 0 or value != int(value)):
                    raise ValueError('Counts must be finite nonnegative integers')
                out.append(value)
    first, second = [np.asarray(v, dtype=np.int64) for v in positions]
    counts = np.column_stack(values)
    if len(first) == 0 or len(np.unique(first * len(names) + second)) != len(first):
        raise ValueError('Empty screen or duplicate ordered guide constructs')
    return np.asarray(names), np.asarray(genes), first, second, counts


def normalize_reference(guide_genes, first, second, counts, doublings):
    """Apply prespecified T0-only QC, then native reference gamma normalization."""
    guide_genes = np.asarray(guide_genes)
    first, second = np.asarray(first), np.asarray(second)
    counts = np.asarray(counts, dtype=float)
    doublings = np.asarray(doublings, dtype=float)
    if counts.shape != (len(first), 4) or len(second) != len(first):
        raise ValueError('Expected T0 R1/R2 and reference R1/R2 counts')
    if np.any(np.isinf(counts)) or np.any(counts < 0):
        raise ValueError('Invalid count matrix')
    if doublings.shape != (2,) or not np.all(np.isfinite(doublings)) or np.any(doublings <= 0):
        raise ValueError('Expected two positive finite population doublings')
    if any(np.any(pos < 0) or np.any(pos >= len(guide_genes)) for pos in [first, second]):
        raise ValueError('Guide index outside schema')
    bad = np.zeros(len(guide_genes), dtype=bool)
    # Median across partners separately for each T0 replicate and orientation,
    # then arithmetic mean across replicates, exactly as the PARPi implementation.
    for positions in [first, second]:
        order = np.argsort(positions, kind='stable')
        boundaries = np.r_[0, np.flatnonzero(np.diff(positions[order])) + 1, len(order)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            selected = order[lo:hi]
            medians = np.nanmedian(counts[selected, :2], axis=0)
            bad[positions[selected[0]]] |= not np.isfinite(medians).all() or medians.mean() <= 35
    keep = ~bad[first] & ~bad[second] & np.all(counts[:, :2] > 50, axis=1)
    keep &= np.isfinite(counts).all(axis=1)
    if not np.any(keep):
        raise ValueError('No constructs survive T0-only coverage filters')
    a, b = first[keep], second[keep]
    nt = guide_genes == 'non-targeting'
    controls = nt[a] & nt[b]
    if not np.any(controls):
        raise ValueError('No surviving double non-targeting controls')
    pseudo = counts[keep] + 10.
    fractions = pseudo / pseudo.sum(axis=0)
    enrichment = np.log2(fractions[:, 2:] / fractions[:, :2])
    centers = np.median(enrichment[controls], axis=0)
    gamma = (enrichment - centers) / doublings
    return a, b, gamma, {
        'input_constructs': len(first), 'retained_constructs': int(keep.sum()),
        'missing_required_count_constructs': int((~np.isfinite(counts).all(axis=1)).sum()),
        'excluded_guides': int(bad.sum()), 'double_nt_controls': int(controls.sum()),
        'nt_log2_centers': centers.tolist(), 'doublings': doublings.tolist(),
    }


def aggregate_genes(guide_genes, first, second, gamma):
    """Keep biological replicates separate; guides/orientations are subsamples."""
    guide_genes = np.asarray(guide_genes)
    genes, gene_codes = np.unique(guide_genes, return_inverse=True)
    a, b = gene_codes[first], gene_codes[second]
    nt = genes == 'non-targeting'
    pair_rows = ~nt[a] & ~nt[b] & (a != b)
    left, right = np.minimum(a[pair_rows], b[pair_rows]), np.maximum(a[pair_rows], b[pair_rows])
    codes, inverse = np.unique(left * len(genes) + right, return_inverse=True)
    observed = np.bincount(inverse, minlength=len(codes))
    sums = np.column_stack([np.bincount(inverse, weights=gamma[pair_rows, r], minlength=len(codes)) for r in range(2)])
    eligible = observed >= 2
    pair_names = np.column_stack([genes[codes[eligible] // len(genes)], genes[codes[eligible] % len(genes)]])
    singles_rows = nt[a] ^ nt[b]
    single_codes = np.where(nt[a[singles_rows]], b[singles_rows], a[singles_rows])
    single_counts = np.bincount(single_codes, minlength=len(genes))
    single_sums = np.column_stack([np.bincount(single_codes, weights=gamma[singles_rows, r], minlength=len(genes)) for r in range(2)])
    single_keep = (single_counts >= 2) & ~nt
    arrays = {
        'genes': genes[~nt], 'pair_names': pair_names,
        'gamma': sums[eligible] / observed[eligible, None],
        'counts': np.repeat(observed[eligible, None], 2, axis=1),
        'single_genes': genes[single_keep],
        'singles': single_sums[single_keep] / single_counts[single_keep, None],
        'single_counts': np.repeat(single_counts[single_keep, None], 2, axis=1),
    }
    if not len(pair_names) or not np.all(np.isfinite(arrays['gamma'])):
        raise ValueError('No finite replicated gene-pair endpoints')
    return arrays


def prepare(source_root, output):
    source_root, output = Path(source_root), Path(output)
    if output.exists():
        raise FileExistsError('Preserve prior external preparations')
    # Verify all sources before creating any outputs.
    for relative, expected, _, _ in COHORTS.values():
        if sha256(source_root / relative) != expected:
            raise ValueError(f'Unexpected source bytes: {relative}')
    started = time.perf_counter()
    output.mkdir(parents=True)
    receipts = {}
    for name, (relative, expected, columns, doublings) in COHORTS.items():
        _, genes, first, second, counts = read_counts(source_root / relative, columns,
                                                     'parp' if name.startswith('parp') else 'environment')
        first, second, gamma, metadata = normalize_reference(genes, first, second, counts, doublings)
        arrays = aggregate_genes(genes, first, second, gamma)
        np.savez_compressed(output / f'{name}.npz', **arrays)
        receipts[name] = dict(metadata, source_path=relative, source_sha256=expected,
            output_sha256=sha256(output / f'{name}.npz'), genes=len(arrays['genes']),
            pairs=len(arrays['pair_names']), singles=len(arrays['single_genes']),
            coverage_filter='guide mean replicate T0 median >35 in both orientations; construct both T0 >50',
            native_filter=name.startswith('parp'))
        print(f'{name}: {len(arrays["pair_names"])} pairs prepared', flush=True)
        del first, second, gamma, counts, arrays
    receipt = {
        'cohorts': receipts, 'preparer_sha256': sha256(__file__),
        'seconds': time.perf_counter() - started,
        'target': 'mean ordered-guide reference gamma within each independent infection replicate',
        'normalization': 'pseudocount10 after T0 QC; all surviving construct fractions; NT+NT median centered log2 reference/T0 per population doubling',
        'environment_filter_departure': 'Official environmental pipeline uses endpoint coverage; prespecified T0-only filter avoids conditioning inclusion on external outcomes.',
        'pair_eligibility': 'different target genes; >=2 ordered guide observations in each of two biological replicates; no outcome-magnitude or correlation filtering',
        'replication': 'two independent infections per cohort; guide pairs, orientations and technical sequencing repeats are not biological replicates',
        'scoring_performed': False,
        'sources': {'parp': 'https://github.com/simpsondl/parpi-manuscript/tree/a66c13fecbaab0cf026a380343b7d6a9f860afb3',
                    'environment': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE312636'},
    }
    write_json(output / 'preparation.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_root, args.output)
