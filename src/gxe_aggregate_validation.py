"""Validate recovered aggregate axes independently of raw reconstruction."""
import numpy as np
from pathlib import Path
import zipfile
from .artifacts import read_json, sha256


def validate_aggregates(sums, support, totals, mapping):
    if mapping.ndim != 1 or mapping.dtype.kind not in 'iu' or totals.shape != mapping.shape:
        raise ValueError('Original cell-axis membership required')
    if sums.ndim != 3 or support.shape != sums.shape[:2] or sums.shape[1] != 2:
        raise ValueError('Aggregate partition axes mismatch')
    if support.dtype.kind not in 'iu' or np.any(support < 0) or not np.isfinite(sums).all() or np.any(sums < 0):
        raise ValueError('Finite nonnegative sums and integer support required')
    if np.any(mapping < -1) or np.any(mapping >= len(sums)):
        raise ValueError('Cell-to-condition index outside declared axes')
    if not np.isfinite(totals).all() or np.any(totals < 0) or np.any(totals[mapping >= 0] <= 0):
        raise ValueError('Invalid full-library denominators')
    counted = np.zeros_like(support)
    valid = mapping >= 0
    np.add.at(counted, (mapping[valid], np.flatnonzero(valid) % 2), 1)
    if not np.array_equal(counted, support):
        raise ValueError('Retained membership/support mismatch')


def load_recovery(folder):
    """Require every primary asset to be hash-bound before opening safe arrays."""
    folder = Path(folder).resolve()
    manifest = read_json(folder/'manifest.json')
    records = manifest.get('files', [])
    names = [r.get('path') for r in records]
    required = {'aggregates.npz', 'axes.json', 'metrics.json', 'registration.json', 'decision.json'}
    if not required.issubset(names) or len(names) != len(set(names)):
        raise ValueError('Recovery manifest missing or duplicating a required asset')
    for record in records:
        path = (folder/record['path']).resolve()
        if not path.is_relative_to(folder) or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Recovered asset hash/path/size mismatch')
    with zipfile.ZipFile(folder/'aggregates.npz') as archive:
        if sum(r.file_size for r in archive.infolist()) > 10_000_000:
            raise ValueError('Aggregate decompression exceeds retained envelope')
    with np.load(folder/'aggregates.npz', allow_pickle=False) as archive:
        expected = {'normalized_count_sums', 'partition_support', 'full_library_totals', 'cell_condition_index'}
        if set(archive.files) != expected:
            raise ValueError('Unknown aggregate array schema')
        arrays = {k: archive[k].copy() for k in archive.files}
    validate_aggregates(*(arrays[k] for k in ('normalized_count_sums', 'partition_support', 'full_library_totals', 'cell_condition_index')))
    axes = read_json(folder/'axes.json')
    if len(axes['group_keys']) != len(arrays['normalized_count_sums']) or len(set(axes['panel_genes'])) != len(axes['panel_genes']) or len(axes['panel_genes']) != arrays['normalized_count_sums'].shape[-1]:
        raise ValueError('Recovered gene/condition axis mismatch')
    return arrays, axes
