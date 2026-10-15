"""Recover fixed-panel GxE aggregates from retained inputs under hard audit limits."""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import psutil
from opal.artifacts import read_json, write_json, sha256
from opal.gxe_measurement import assignment, rt_context, key
from opal.gxe_reconstruction import coordinates, reconstruct_metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = ROOT/'configs/experiments/opal_gxe_reconstruction_v1.json'
    config = read_json(config_path)
    args.output.mkdir(parents=True, exist_ok=False)
    start, wall, peak = time.process_time(), time.monotonic(), 0
    status = 'reconstruction_failed'
    def check():
        nonlocal peak
        process = psutil.Process()
        peak = max(peak, process.memory_info().rss)
        if process.children(recursive=True) or time.process_time()-start > config['cpu_seconds'] or peak > config['ram_bytes']:
            raise RuntimeError('Bounded reconstruction CPU/RAM/process stop')
        if shutil.disk_usage(args.output).free < config['reserve_bytes']:
            raise RuntimeError('Writable reserve stop')
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > config['retained_bytes']-100000:
            raise RuntimeError('Output/failure reserve stop')
    try:
        raw, previous = ROOT/config['input'], ROOT/config['historical_run']
        registered = read_json(previous/'registration.json')
        for name, field in [('tools/run_gxe_measurement.py', 'code_sha256'),
                            ('src/gxe_measurement.py', 'helper_sha256'),
                            ('configs/experiments/opal_gxe_measurement_v1.json', 'config_sha256')]:
            if sha256(ROOT/name) != registered[field]:
                raise ValueError('Historical transformation identity mismatch')
        for record in registered['source_receipts']['files']:
            if sha256(raw/record['path']) != record['sha256']:
                raise ValueError('Raw source checksum mismatch')
            check()
        write_json(args.output/'registration.json', dict(config=config, config_sha256=sha256(config_path),
            code_sha256=sha256(__file__), helper_sha256=sha256(ROOT/'src/gxe_reconstruction.py'),
            historical_registration_sha256=sha256(previous/'registration.json'),
            frozen_panel_sha256=sha256(previous/'panel.json'), source_receipts=registered['source_receipts'],
            acceptance_created=False, reviewer_attestation_created=False, scientific_models_fit=0))
        with gzip.open(raw/'cells.txt.gz', 'rt') as stream:
            cells = [line.split('\t')[0].strip() for line in stream]
        with gzip.open(raw/'genes.txt.gz', 'rt') as stream:
            genes = [line.split('\t')[0].strip() for line in stream]
        if len(set(cells)) != len(cells) or len(set(genes)) != len(genes):
            raise ValueError('Duplicate source axes')
        contexts = []
        for cell in cells:
            try:
                contexts.append(rt_context(cell))
            except ValueError:
                contexts.append('')
        lookup = {c: i for i, c in enumerate(cells)}
        with gzip.open(raw/'sample_sheet.txt.gz', 'rt') as stream:
            labels = sorted({line.split('\t')[0].strip() for line in stream if line.strip()})
        records = {label: assignment(label) for label in labels}
        fixed = {label: r for label, r in records.items() if
            (r['drug'] == 'vehicle' and r['trametinib_dose'] in (0., 1.)) or
            (r['drug'] != 'vehicle' and r['drug_dose'] == 1. and r['trametinib_dose'] in (0., 1.))}
        historic = registered['config']
        keys = sorted({key(c, r) for c in historic['contexts'] for r in fixed.values()})
        group_map = {k: i for i, k in enumerate(keys)}
        calls = defaultdict(lambda: defaultdict(int))
        for chunk in pd.read_csv(raw/'hashes.txt.gz', sep='\t', header=None, chunksize=100000):
            if not (chunk[3] == 1).all():
                raise ValueError('Unverified hash family')
            for cell, label, count in chunk.iloc[:, [1, 2, 4]].itertuples(index=False, name=None):
                if cell in lookup:
                    calls[cell][str(label)] += int(count)
            check()
        mapping = np.full(len(cells), -1, np.int32)
        excluded = defaultdict(int)
        for cell, i in lookup.items():
            if not contexts[i]:
                excluded['outside_author_rt_allocation'] += 1
                continue
            ranked = sorted(calls.get(cell, {}).items(), key=lambda r: (-r[1], r[0]))
            if not ranked or ranked[0][1] < historic['minimum_hash_umis']:
                excluded['hash_support'] += 1
                continue
            if len(ranked) > 1 and ranked[0][1] < historic['minimum_hash_ratio']*ranked[1][1]:
                excluded['ambiguous_hash'] += 1
                continue
            label = ranked[0][0]
            if label not in fixed:
                excluded['other_dose_or_unknown_assignment'] += 1
                continue
            mapping[i] = group_map[key(contexts[i], fixed[label])]
        del calls, lookup, cells, contexts
        halves = np.arange(len(mapping)) % 2
        support = np.zeros((len(keys), 2), np.int64)
        valid = mapping >= 0
        np.add.at(support, (mapping[valid], halves[valid]), 1)
        qualification = read_json(previous/'qualification.json')
        if support.tolist() != qualification['partition_support'] or [list(k) for k in keys] != qualification['group_keys'] or dict(excluded) != qualification['excluded_cells']:
            raise ValueError('Historical population/support mismatch')
        panel = read_json(previous/'panel.json')['genes']
        positions = {g: i for i, g in enumerate(genes)}
        reverse = np.full(len(genes), -1, np.int16)
        reverse[[positions[g] for g in panel]] = np.arange(len(panel))
        total = np.zeros(len(mapping), np.float64)
        selected = []
        entries = selected_entries = 0
        for row, col, value in coordinates(raw/'counts.mtx.gz', (len(genes), len(mapping)), check):
            np.add.at(total, col, value)
            keep = (mapping[col] >= 0) & (reverse[row] >= 0)
            selected.append((reverse[row[keep]].copy(), col[keep].astype(np.int32), value[keep].astype(np.float64)))
            entries += len(row)
            selected_entries += int(keep.sum())
            check()
        if np.any(total[valid] <= 0):
            raise ValueError('Zero retained-cell full-library denominator')
        sums = np.zeros((len(keys), 2, len(panel)))
        for gene, cell, value in selected:
            np.add.at(sums, (mapping[cell], halves[cell], gene), value*10000/total[cell])
            check()
        del selected
        drugs = sorted({r['drug'] for r in fixed.values() if r['drug'] != 'vehicle' and r['trametinib_dose'] == 1.})
        metrics = reconstruct_metrics(sums, support, keys, historic['contexts'], drugs, historic['minimum_cells_per_partition'])
        expected = read_json(previous/'metrics.json')
        exact = metrics == expected
        if not exact:
            raise ValueError('Historical endpoint replay is not exact')
        logical_bytes = sum(a.nbytes for a in (sums, support, total, mapping))
        if logical_bytes+100000+sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > config['retained_bytes']:
            raise RuntimeError('Uncompressed aggregate custody exceeds output envelope')
        np.savez(args.output/'aggregates.npz', normalized_count_sums=sums, partition_support=support,
                 full_library_totals=total, cell_condition_index=mapping)
        write_json(args.output/'axes.json', dict(group_keys=[list(k) for k in keys], panel_genes=panel,
            cell_axis='original cells.txt.gz row order, all rows including exclusions',
            halves='original cell index modulo two', contexts=historic['contexts'], drugs=drugs,
            dose_uM=historic['dose_uM'], duration_hours=historic['duration_hours'],
            excluded_cells=dict(excluded), frozen_panel_reselected=False, coordinate_entries=entries,
            selected_coordinate_entries=selected_entries, aggregate_logical_bytes=logical_bytes))
        write_json(args.output/'metrics.json', metrics)
        write_json(args.output/'decision.json', dict(status='fixed_panel_aggregate_reconstruction_exact',
            all_66_rows_exact=exact, complete_panel_selection_replayed=False, independent_confirmation=False,
            scientific_models_fit=0, source_reacquisition_required=False))
        check()
        write_json(args.output/'manifest.json', dict(files=[dict(path=p.name, bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(args.output.iterdir()) if p.is_file()], safe_array_format='NPZ without pickle',
            original_inputs_not_copied=True))
        status = 'fixed_panel_aggregate_reconstruction_exact'
        print(status, 'entries', entries, 'selected', selected_entries, 'logical bytes', logical_bytes, flush=True)
    except Exception as error:
        write_json(args.output/'failure.json', dict(status='failed_bounded_reconstruction', reason=str(error), partial_assets_retained=True))
        raise
    finally:
        write_json(args.output/'resource_ledger.json', dict(status=status, cpu_seconds=time.process_time()-start,
            wall_seconds=time.monotonic()-wall, peak_sampled_rss_bytes=peak, continuous_peak_measured=False,
            cpu_limit_seconds=config['cpu_seconds'], ram_limit_bytes=config['ram_bytes'],
            retained_bytes=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),
            retained_limit_bytes=config['retained_bytes'], threads=1, downloads=0, fitting=False, acceptance_created=False))


if __name__ == '__main__':
    main()
