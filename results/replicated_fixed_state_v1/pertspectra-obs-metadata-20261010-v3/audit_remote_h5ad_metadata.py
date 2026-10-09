"""Inspect selected observation metadata using bounded, identity-checked HTTP ranges."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

URL = 'https://pert-spectra.s3.amazonaws.com/data/inhouse.h5ad'
ALLOWED = {'condition', 'treatment', 'replicate', 'rep', 'biological_replicate', 'batch',
           'sample', 'sample_id', 'donor', 'culture', 'culture_id', 'plate',
           'sampleindex', 'target_gene_name', 'feature_call'}


def summarize_strata(sample_ids, treatments, labels):
    if not len(sample_ids) == len(treatments) == len(labels):
        raise ValueError('Observation metadata lengths disagree')
    groups = {}
    for sample, treatment, label in zip(sample_ids, treatments, labels):
        groups.setdefault((sample, treatment), []).append(label)
    records = []
    for (sample, treatment), group in sorted(groups.items()):
        assignments = {tuple(sorted(label.split('|'))) for label in group}
        singles = {a[0] for a in assignments if len(a) == 1 and a[0] and not a[0].startswith('INTERGENIC')}
        pairs = {a for a in assignments if len(a) == 2 and len(set(a)) == 2
                 and all(c and not c.startswith('INTERGENIC') for c in a)}
        records.append({'sample_index_label': sample, 'treatment_label': treatment,
                        'rows': len(group), 'unique_literal_assignments': len(assignments),
                        'literal_assignment_arity_counts': dict(sorted(Counter(len(a) for a in assignments).items())),
                        'literal_single_target_count_excluding_intergenic': len(singles),
                        'literal_distinct_two_target_count_excluding_intergenic': len(pairs),
                        'literal_pairs_with_both_single_labels': sum(set(pair) <= singles for pair in pairs),
                        'independent_culture_identity_verified': False})
    return records


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, etag, snapshots, opener=urlopen, budget=32_000_000):
        self.url, self.size, self.etag, self.snapshots = url, size, etag, snapshots
        self.opener, self.budget = opener, budget
        self.position, self.used = 0, 0
        self.cache, self.ledger = {}, []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset + (0 if whence == 0 else self.position if whence == 1 else self.size if whence == 2 else -self.size * 2)
        if whence not in (0, 1, 2) or not 0 <= target <= self.size:
            raise ValueError('Seek outside source object')
        self.position = target
        return target

    def read(self, length=-1):
        length = self.size - self.position if length < 0 else min(length, self.size - self.position)
        if length > self.budget:
            raise ValueError('Read exceeds fixed metadata budget')
        start, end = self.position, self.position + length
        chunks = []
        while self.position < end:
            block = self.position // 65536 * 65536
            last = min(block + 65536, self.size) - 1
            if block not in self.cache:
                size = last - block + 1
                if self.used + size > self.budget:
                    raise ValueError('HTTP ranges exceed fixed metadata budget')
                request = Request(self.url, headers={'Range': f'bytes={block}-{last}', 'If-Match': self.etag})
                with self.opener(request, timeout=30) as response:
                    if (response.status != 206 or response.headers.get('Content-Range') != f'bytes {block}-{last}/{self.size}'
                            or response.headers.get('ETag') != self.etag):
                        raise ValueError('Source ignored range or changed object identity')
                    data = response.read(size + 1)
                if len(data) != size:
                    raise ValueError('Truncated or oversized range response')
                self.used += size
                self.cache[block] = data
                name = f'bytes-{block}-{last}.bin'
                (self.snapshots / name).write_bytes(data)
                self.ledger.append({'start': block, 'end': last, 'bytes': size, 'snapshot': name,
                                    'sha256': hashlib.sha256(data).hexdigest()})
            stop = min(end, last + 1)
            chunks.append(self.cache[block][self.position - block:stop - block])
            self.position = stop
        assert self.position == start + length
        return b''.join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.snapshots.mkdir(parents=True, exist_ok=False)
    dump = lambda name, data: (args.output / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    dump('scope.json', {'fixed_utc': datetime.now(timezone.utc).isoformat(), 'url': URL,
         'role': 'selected_obs_metadata_only', 'allowed_columns': sorted(ALLOWED),
         'expression_values_decoded': False, 'model_weights_accessed': False,
         'maximum_range_bytes': 32_000_000, 'cache_block_bytes': 65536,
         'adjacent_expression_bytes_may_arrive_uninterpreted_in_cached_blocks': True,
         'unknown_columns_and_expression_datasets_not_read': True,
         'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    with urlopen(Request(URL, method='HEAD'), timeout=30) as response:
        size, etag = int(response.headers['Content-Length']), response.headers['ETag']
    reader = RangeReader(URL, size, etag, args.snapshots)
    try:
        import h5py
        with h5py.File(reader, 'r') as file:
            obs = file['obs']
            names = sorted(obs.keys())
            counts, selected_values = {}, {}
            for name in names:
                if name.lower() not in ALLOWED:
                    continue
                node = obs[name]
                if isinstance(node, h5py.Group):
                    category_node = node['categories']
                    categories = (category_node.asstr()[:] if h5py.check_string_dtype(category_node.dtype)
                                  else category_node[:])
                    codes = node['codes'][:]
                    if any(code < 0 or code >= len(categories) for code in codes):
                        raise ValueError('Missing or invalid categorical metadata')
                    values = [str(categories[code]) for code in codes]
                elif h5py.check_string_dtype(node.dtype):
                    values = node.asstr()[:].tolist()
                else:
                    values = [str(value) for value in node[:]]
                if len(values) > 200_000 or len(set(values)) > 2000:
                    raise ValueError('Metadata cardinality exceeds declared report budget')
                counts[name] = dict(sorted(Counter(values).items()))
                selected_values[name] = values
            strata = (summarize_strata(selected_values['SampleIndex'], selected_values['Treatment'],
                                      selected_values['target_gene_name'])
                      if {'SampleIndex', 'Treatment', 'target_gene_name'} <= set(selected_values) else [])
            dump('report.json', {'url': URL, 'object_bytes': size, 'etag': etag,
                 'observation_column_names': names, 'selected_column_value_counts': counts,
                 'sample_treatment_strata': strata,
                 'expression_values_decoded': False, 'independent_cultures_inferred': False,
                 'primary_confirmation_qualified': False})
    finally:
        dump('retrieval.json', {'object_bytes': size, 'etag': etag, 'range_bytes': reader.used,
                               'ranges': reader.ledger, 'expression_values_decoded': False})
    print(json.dumps({'range_bytes': reader.used, 'metadata_columns': list(counts), 'expression_values_decoded': False}))


if __name__ == '__main__':
    main()
