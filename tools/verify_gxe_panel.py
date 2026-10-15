"""Reproduce the historical reference-only panel, charging the recovery CPU budget."""
import argparse
from collections import defaultdict
import gzip
import math
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from opal.artifacts import read_json, write_json, sha256
from opal.gxe_measurement import reference_variance
from opal.gxe_streaming import coordinates_c, RuntimeGuard


def panel_variance(sums, support, keys, contexts, block_genes=2048):
    variance = np.zeros(sums.shape[-1])
    groups = {tuple(k): i for i, k in enumerate(keys)}
    reference = np.array([k[3] == 'vehicle' or k[5] == 0 for k in keys])
    for start in range(0, len(variance), block_genes):
        stop = min(start+block_genes, len(variance))
        logged = np.log2(1+sums[:, :, start:stop]/np.maximum(support[:, :, None], 1))
        for context in contexts:
            effects = defaultdict(list)
            for i, k in enumerate(keys):
                if k[0] != context or not reference[i] or (k[3] == 'vehicle' and k[5] == 0) or min(support[i]) < 10:
                    continue
                control = groups.get((context, k[1], k[2], 'vehicle', 0., 0.))
                if control is None or min(support[control]) < 10:
                    continue
                identity = ('Trametinib', k[5]) if k[3] == 'vehicle' else (k[3], k[4])
                effects[identity].append((logged[i]-logged[control]).mean(0))
            variance[start:stop] += reference_variance(effects)/len(contexts)
    return variance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    prior = read_json(args.recovery/'resource_ledger.json')
    if prior['status'] != 'fixed_panel_aggregate_reconstruction_exact':
        raise ValueError('A completed exact recovery is required')
    limits = read_json(ROOT/'configs/experiments/opal_gxe_reconstruction_v2.json')
    limits['cpu_seconds'] = min(240, math.floor(300-prior['cpu_seconds'])-2)
    if limits['cpu_seconds'] <= 5:
        raise ValueError('Combined recovery/panel CPU allowance exhausted')
    args.output.mkdir(parents=True, exist_ok=False)
    guard, status = RuntimeGuard(args.output, limits), 'panel_verification_failed'
    try:
        for record in read_json(args.recovery/'manifest.json')['files']:
            path = (args.recovery/record['path']).resolve()
            if not path.is_relative_to(args.recovery.resolve()) or sha256(path) != record['sha256']:
                raise ValueError('Recovered input custody mismatch')
        registration = read_json(args.recovery/'registration.json')
        raw = ROOT/'data/raw/gxe-measurement-20261014'
        for record in registration['source_receipts']['files']:
            if sha256(raw/record['path']) != record['sha256']:
                raise ValueError('Raw identity mismatch')
            guard.check()
        with gzip.open(raw/'genes.txt.gz', 'rt') as f:
            genes = [line.split('\t')[0].strip() for line in f]
        axes = read_json(args.recovery/'axes.json')
        keys = [tuple(k) for k in axes['group_keys']]
        with np.load(args.recovery/'aggregates.npz', allow_pickle=False) as bank:
            total, mapping, support = (bank[n].copy() for n in ('full_library_totals', 'cell_condition_index', 'partition_support'))
        halves = np.arange(len(mapping)) % 2
        reference = np.array([k[3] == 'vehicle' or k[5] == 0 for k in keys])
        sums = np.zeros((len(keys), 2, len(genes)))
        entries = 0
        guard.check(phase='reference_count_stream')
        write_json(args.output/'registration.json', dict(code_sha256=sha256(__file__),
            streaming_sha256=sha256(ROOT/'src/gxe_streaming.py'),
            reference_rule_sha256=sha256(ROOT/'src/gxe_measurement.py'),
            recovery_manifest_sha256=sha256(args.recovery/'manifest.json'),
            prior_cpu_seconds=prior['cpu_seconds'], combined_cpu_limit_seconds=300,
            operation_limits=limits, source_receipts=registration['source_receipts'], fitting=False))
        for row, col, value in coordinates_c(raw/'counts.mtx.gz', (len(genes), len(mapping)), guard.check):
            active = mapping[col] >= 0
            active[active] &= reference[mapping[col[active]]]
            rc, cc = row[active], col[active]
            np.add.at(sums, (mapping[cc], halves[cc], rc), value[active].astype(float)*10000/total[cc])
            entries += len(row)
            guard.check(coordinate_entries=entries)
        guard.check(phase='reference_only_variance')
        variance = panel_variance(sums, support, keys, axes['contexts'])
        columns = sorted(range(len(genes)), key=lambda i: (-variance[i], genes[i]))[:512]
        selected = [genes[i] for i in columns]
        if selected != axes['panel_genes']:
            raise ValueError('Reference-only panel selection does not match historical panel')
        np.savez(args.output/'reference_variance.npz', variance=variance)
        write_json(args.output/'decision.json', dict(status='reference_only_panel_reproduced', exact_gene_order=True,
            selected_genes=selected, original_gene_count=len(genes), coordinate_entries=entries,
            pair_outcomes_used_for_variance=False, technical_reference_scope=True,
            independent_confirmation=False, panel_selection_replayed=True,
            variance_sha256=sha256(args.output/'reference_variance.npz')))
        guard.check(phase='panel_verified')
        status = 'reference_only_panel_reproduced'
        print(status, len(selected), 'genes', flush=True)
    except Exception as error:
        write_json(args.output/'failure.json', dict(status=status, reason=str(error)))
        raise
    finally:
        receipt = guard.receipt(status)
        receipt['combined_recovery_panel_cpu_seconds'] = prior['cpu_seconds']+receipt['cpu_seconds']
        write_json(args.output/'resource_ledger.json', receipt)


if __name__ == '__main__':
    main()
