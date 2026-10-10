"""Bounded full-hash source access and chunked label-conditional preparation.

Does not load expression-derived annotations, embeddings or upstream weights.
Gene-level labels average deposited guide mixtures, not matched-guide effects.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.request import Request, urlopen

import h5py
import numpy as np
import psutil
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import fingerprint, read_json, sha256, write_json

URL = 'https://pert-spectra.s3.amazonaws.com/data/inhouse.h5ad'
SIZE = 3142132144
ETAG = '"5f6e072df69ee8383171c46e945a7fde-375"'


def column(node):
    if isinstance(node, h5py.Group):
        values, codes = column(node['categories']), node['codes'][:]
        return np.asarray([values[c] if c >= 0 else 'nan' for c in codes])
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in node[:]])


def label(raw):
    components = str(raw).split('|')
    if len(components) > 2 or any(not c or c == 'nan' or c.startswith('INTERGENIC') for c in components):
        return None, 'ambiguous_or_intergenic'
    components = [c for c in components if not re.fullmatch(r'CTRL00\w*', c)]
    if len(components) != len(set(components)):
        return None, 'self_pair'
    return tuple(sorted(components)), None


def block(matrix, start, stop, genes):
    if isinstance(matrix, h5py.Dataset):
        return sparse.csr_matrix(matrix[start:stop])
    encoding = matrix.attrs.get('encoding-type', '')
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding != 'csr_matrix':
        raise ValueError('Only backed dense or CSR expression is supported; no full densification')
    pointers = matrix['indptr'][start:stop + 1]
    left, right = int(pointers[0]), int(pointers[-1])
    return sparse.csr_matrix((matrix['data'][left:right], matrix['indices'][left:right], pointers - left),
                             shape=(stop - start, genes))


def csc_columns(matrix, start, stop, rows):
    pointers = matrix['indptr'][start:stop + 1]
    left, right = int(pointers[0]), int(pointers[-1])
    return sparse.csc_matrix((matrix['data'][left:right], matrix['indices'][left:right], pointers - left),
                             shape=(rows, stop - start))


def aggregate_csc(matrix, indices, treatments, labels, gene_count, use_counts):
    """Two bounded column passes, sparse label membership, dense aggregates only."""
    n = len(labels)
    libraries = np.zeros(n)
    for start in range(0, gene_count, 128):
        x = csc_columns(matrix, start, min(start + 128, gene_count), n)
        if not np.isfinite(x.data).all() or (use_counts and (np.any(x.data < 0) or not np.allclose(x.data, np.rint(x.data), atol=1e-6, rtol=0))):
            raise ValueError('Invalid deposited count scale; counts layer not verified raw counts')
        libraries += np.asarray(x.sum(1)).ravel()
    keys = sorted({(str(indices[i]), str(treatments[i]), p) for i, p in enumerate(labels) if p is not None})
    lookup = {key: i for i, key in enumerate(keys)}
    row, col, weight, counts = [], [], [], Counter()
    for i, assigned in enumerate(labels):
        if assigned is None or (use_counts and libraries[i] <= 0):
            continue
        key = (str(indices[i]), str(treatments[i]), assigned)
        row.append(lookup[key]); col.append(i)
        weight.append(10000 / libraries[i] if use_counts else 1.)
        counts[key] += 1
    membership = sparse.csr_matrix((weight, (row, col)), shape=(len(keys), n))
    sums = np.zeros((len(keys), gene_count))
    peak = psutil.Process().memory_info().rss
    for start in range(0, gene_count, 128):
        stop = min(start + 128, gene_count)
        x = csc_columns(matrix, start, stop, n)
        sums[:, start:stop] = (membership @ x).toarray()
        peak = max(peak, psutil.Process().memory_info().rss)
        if peak > 4000000000:
            raise ValueError('CSC aggregate resource cap')
    return {key: sums[i] / counts[key] for i, key in enumerate(keys) if counts[key] > 0}, counts, peak


def prepare(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    args.matrix.parent.mkdir(parents=True, exist_ok=True)
    scope = {'utc': datetime.now(timezone.utc).isoformat(), 'url': URL, 'object_bytes': SIZE,
             'etag': ETAG, 'etag_is_file_hash': False, 'code_sha256': sha256(Path(__file__)),
             'menu_sha256': sha256(ROOT / 'configs/experiments/opal_model_series_v1.json'),
             'resource_amendment': 'Earlier refusal preserved; free disk increased during local work. Recheck permits one bounded transfer.',
             'free_disk_bytes': shutil.disk_usage(args.matrix.parent).free,
             'memory_available_bytes': psutil.virtual_memory().available,
             'prior_exposure': ['observation metadata', 'adjacent uninterpreted bytes', 'published narratives', 'author notebook saved outputs'],
             'allowed_inputs': ['controls', 'observed singles', 'literal target labels', 'SampleIndex', 'Treatment'],
             'forbidden_inputs': ['archetypes', 'clusters', 'embeddings', 'StringDB', 'pretrained weights', 'query pair outcomes'],
             'control': 'one or two CTRL00* calls only; intergenic calls excluded',
             'single': 'one non-control target with optional CTRL00*; deposited guide mixtures averaged',
             'pair': 'two distinct non-control, non-intergenic targets; canonical unordered identity held globally',
             'QC': 'at least ten cells per control/single/pair stratum; finite values; positive libraries for verified counts',
             'scale': 'Use deposited counts layer only if all consumed values finite nonnegative integers; library-total 10000, aggregate mean normalized counts then log2(1+x). Otherwise retain deposited X arithmetic without inversion, explicitly upstream-dependent.',
             'endpoint': '512 highest-variance unique observed-single profiles across four source strata per TNF context, lexical gene ties; no pair responses',
             'population': 'complete reference-supported canonical pairs across all four SampleIndex strata within each TNF context',
             'seeds': [20261009, 20261010, 11, 29, 47], 'chunk_rows': 1024,
             'limits': {'seconds': 7200, 'rss_bytes': 4000000000, 'transfer_bytes': SIZE, 'disk_reserve': 1000000000},
             'independent_cultures_claimed': False, 'matched_guide_causal_claim': False,
             'paper_65648_cells_reconciled': False, 'minimum_pairs_for_model_fit': 12}
    scope['adapter_amendment'] = 'Full-file structure revealed CSC counts/X. Earlier row adapter refused before expression values. Use 128-gene column chunks and sparse group membership; only aggregates are dense. QC, scale, endpoint and model menu unchanged before first expression decoding.'
    write_json(output / 'scope_before_expression.json', scope)
    if not args.matrix.exists() and scope['free_disk_bytes'] < SIZE + 1000000000:
        write_json(output / 'refusal.json', {'reason': 'Insufficient full-object plus reserve disk', 'scope_hash': sha256(output / 'scope_before_expression.json')})
        return
    started = time.monotonic()
    if not args.matrix.exists():
        partial = args.matrix.with_suffix('.partial')
        width, size = 8 * 1024 * 1024, 0
        def fetch(start):
            stop = min(start + width, SIZE) - 1
            request = Request(URL, headers={'User-Agent': 'Opal bounded source extension', 'Range': f'bytes={start}-{stop}'})
            with urlopen(request, timeout=30) as source:
                if (source.status != 206 or source.headers.get('ETag') != ETAG
                        or source.headers.get('Content-Range') != f'bytes {start}-{stop}/{SIZE}'):
                    raise ValueError('Range object identity/size mismatch')
                data = source.read(stop - start + 1)
                if len(data) != stop - start + 1:
                    raise ValueError('Incomplete range')
            return start, data
        # Four bounded 8 MB ranges at once; never retain the full object in RAM.
        with partial.open('xb') as target, ThreadPoolExecutor(max_workers=4) as pool:
            starts = list(range(0, SIZE, width))
            for batch in range(0, len(starts), 4):
                for start, data in pool.map(fetch, starts[batch:batch + 4]):
                    target.seek(start)
                    target.write(data)
                    size += len(data)
                target.flush()
                print(f'Transferred {size}/{SIZE} bytes', flush=True)
                if size > SIZE or time.monotonic() - started > 1800:
                    raise ValueError('Bounded transfer exceeded')
        if size != SIZE:
            raise ValueError('Incomplete assembled full file')
        partial.replace(args.matrix)
    if args.matrix.stat().st_size != SIZE:
        raise ValueError('File size mismatch')
    digest = sha256(args.matrix)
    write_json(output / 'access.json', {'full_file_sha256': digest, 'bytes': SIZE,
               'scope_sha256': sha256(output / 'scope_before_expression.json'),
               'full_file_hash_verified_before_expression_decode': True, 'local_path': str(args.matrix)})
    with h5py.File(args.matrix, 'r') as file:
        names = column(file['var'][file['var'].attrs.get('_index', '_index')]).tolist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate source gene IDs need a new contract')
        indices, treatments = column(file['obs/SampleIndex']), column(file['obs/Treatment'])
        targets, guides = column(file['obs/target_gene_name']), column(file['obs/feature_call'])
        if len(targets) != 73015:
            raise ValueError('Deposited metadata count changed')
        labels, exclusions = [], Counter()
        for target, guide in zip(targets, guides):
            assigned, reason = label(target)
            # Preserve molecular assignment ambiguity rather than assuming guide equivalence.
            if assigned is not None and len(str(guide).split('|')) != len(str(target).split('|')):
                assigned, reason = None, 'guide_target_assignment_length_mismatch'
            labels.append(assigned)
            if reason:
                exclusions[reason] += 1
        layers = list(file.get('layers', {}).keys())
        matrix_key = 'layers/counts' if 'counts' in layers else 'X'
        matrix = file[matrix_key]
        use_counts = matrix_key == 'layers/counts'
        encoding = matrix.attrs.get('encoding-type', '')
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        counts, mixtures = Counter(), defaultdict(Counter)
        refs, pair_rows = defaultdict(list), defaultdict(list)
        for i, assigned in enumerate(labels):
            if assigned is None:
                continue
            key = (str(indices[i]), str(treatments[i]), assigned)
            counts[key] += 1
            mixtures[key][str(guides[i])] += 1
            (refs if len(assigned) < 2 else pair_rows)[key].append(i)
        eligible_refs = {key for key in refs if counts[key] >= 10}
        sums = {key: np.zeros(len(names)) for key in eligible_refs}
        consumed_counts = Counter()
        peak = psutil.Process().memory_info().rss
        group_means = None
        if encoding == 'csc_matrix':
            group_means, consumed_counts, peak = aggregate_csc(matrix, indices, treatments, labels, len(names), use_counts)
            for key in sums:
                if key in group_means:
                    sums[key] = group_means[key] * consumed_counts[key]
        for start in range(0, len(targets), 1024) if group_means is None else []:
            stop = min(start + 1024, len(targets))
            x = block(matrix, start, stop, len(names))
            if not np.isfinite(x.data).all() or (use_counts and (np.any(x.data < 0) or not np.allclose(x.data, np.rint(x.data), atol=1e-6, rtol=0))):
                raise ValueError('Invalid deposited expression/count scale')
            if use_counts:
                libraries = np.asarray(x.sum(1)).ravel()
                x = sparse.diags(np.divide(10000., libraries, out=np.zeros_like(libraries, dtype=float), where=libraries > 0)) @ x
            for offset in range(stop - start):
                i = start + offset
                key = (str(indices[i]), str(treatments[i]), labels[i])
                if key in sums and (not use_counts or libraries[offset] > 0):
                    sums[key] += x.getrow(offset).toarray().ravel()
                    consumed_counts[key] += 1
            peak = max(peak, psutil.Process().memory_info().rss)
            if peak > 4000000000 or time.monotonic() - started > 7200:
                raise ValueError('Preparation resource cap')
        means = {key: sums[key] / consumed_counts[key] for key in sums if consumed_counts[key] >= 10}
        if use_counts:
            means = {key: np.log2(1 + value) for key, value in means.items()}
        contexts = sorted(set(treatments))
        prepared = {}
        support = []
        for context in contexts:
            strata = sorted(set(indices[treatments == context]))
            if len(strata) != 4 or any((s, context, ()) not in means for s in strata):
                write_json(output / 'refusal.json', {'reason': 'Missing ten-cell CTRL-only reference in all four context strata', 'context': str(context)})
                return
            single_keys = [k for k in means if k[1] == context and len(k[2]) == 1]
            profiles = np.asarray([means[k] - means[(k[0], context, ())] for k in single_keys])
            variance = profiles.var(0)
            panel = sorted(range(len(names)), key=lambda i: (-variance[i], names[i]))[:512]
            available = {}
            for stratum in strata:
                available[stratum] = {p for (s, t, p), rows in pair_rows.items() if s == stratum and t == context and counts[(s, t, p)] >= 10
                                      and all((s, t, (c,)) in means for c in p)}
            population = sorted(set.intersection(*(available[s] for s in strata)))
            if len(population) < 36:
                write_json(output / 'refusal.json', {'reason': 'Fewer than 36 complete canonical pairs after fixed reference/cell rules',
                           'context': str(context), 'pairs': len(population)})
                return
            psums = {(s, context, p): np.zeros(len(panel)) for s in strata for p in population}
            pcounts = Counter()
            if group_means is not None:
                for key in psums:
                    if key in group_means:
                        pcounts[key] = consumed_counts[key]
                        psums[key] = group_means[key][panel] * pcounts[key]
            for start in range(0, len(targets), 1024) if group_means is None else []:
                stop = min(start + 1024, len(targets))
                x = block(matrix, start, stop, len(names))
                if use_counts:
                    libraries = np.asarray(x.sum(1)).ravel()
                    x = sparse.diags(np.divide(10000., libraries, out=np.zeros_like(libraries, dtype=float), where=libraries > 0)) @ x
                x = x[:, panel].tocsr()
                for offset in range(stop - start):
                    i = start + offset
                    key = (str(indices[i]), str(treatments[i]), labels[i])
                    if key in psums and (not use_counts or libraries[offset] > 0):
                        psums[key] += x.getrow(offset).toarray().ravel()
                        pcounts[key] += 1
            bank = {'singles': [], 'responses': [], 'pair_ids': [], 'culture_ids': []}
            for s in strata:
                control = means[(s, context, ())][panel]
                for pair in population:
                    key = (s, context, pair)
                    if pcounts[key] < 10:
                        raise ValueError('Pair support changed after positive-library QC; no implicit population repair')
                    effect = psums[key] / pcounts[key]
                    if use_counts:
                        effect = np.log2(1 + effect)
                    bank['responses'].append(effect - control)
                    bank['singles'].append([means[(s, context, (c,))][panel] - control for c in pair])
                    bank['pair_ids'].append(json.dumps(pair, separators=(',', ':')))
                    bank['culture_ids'].append(s)
            bank = {k: np.asarray(v) for k, v in bank.items()}
            destination = args.local / ('TNFA-positive' if context == 'TNFA+' else 'TNFA-negative')
            destination.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(destination / 'canonical_bank.npz', **bank, genes=np.asarray([names[i] for i in panel]))
            prepared[str(context)] = {'local': str(destination), 'pairs': len(population), 'strata': strata,
                    'genes': [names[i] for i in panel], 'bank_sha256': sha256(destination / 'canonical_bank.npz'),
                    'units': 'log2_library_10000_change' if use_counts else 'deposited_log_change',
                    'source': 'PertSpectra-inhouse/' + str(context), 'matrix_key': matrix_key}
            for key in sorted(set(refs) | set(pair_rows)):
                if key[1] == context:
                    support.append({'sample_index': key[0], 'context': key[1], 'targets': key[2], 'cells': counts[key],
                                    'guide_mixtures': dict(mixtures[key]), 'reference_supported': key in means,
                                    'included_pair': len(key[2]) == 2 and key[2] in population})
        write_json(output / 'prepared.json', {'contexts': prepared, 'full_file_sha256': digest, 'layers': layers,
                   'matrix_key': matrix_key, 'counts_verified': use_counts, 'source_genes': len(names),
                   'deposited_rows': len(targets), 'paper_filtered_cells': 65648,
                   'paper_count_reconciliation': 'Different author filtering, pooled condition>=50 and HVG/noise filtering not reproduced; this endpoint and per-stratum QC are intentionally separate',
                   'exclusions': dict(exclusions), 'wall_seconds': time.monotonic() - started,
                   'peak_sampled_rss_bytes': peak, 'upstream_processing_dependence': True,
                   'independent_culture_confirmation': False})
        write_json(output / 'support_and_exclusions.json', support)
        print(json.dumps({'contexts': {k: v['pairs'] for k, v in prepared.items()}, 'matrix_key': matrix_key, 'sha256': digest}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=ROOT / 'data/raw/pertspectra-model-series/inhouse.h5ad')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/opal_model_series_v1/pertspectra/access-v4')
    parser.add_argument('--local', type=Path, default=Path('E:/opal-model-series-20261010/pertspectra-banks'))
    args = parser.parse_args()
    try:
        prepare(args)
    except Exception as error:
        write_json(args.output / 'failure.json', {'type': type(error).__name__, 'reason': str(error), 'expression_access_may_have_started': (args.output / 'access.json').exists()})
        raise
