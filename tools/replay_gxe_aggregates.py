"""Verify recovered aggregate custody and replay all historical GxE row endpoints."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from opal.artifacts import read_json, write_json, sha256
from opal.gxe_reconstruction import reconstruct_metrics
from opal.gxe_aggregate_validation import validate_aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    folder = args.input.resolve()
    for record in read_json(folder/'manifest.json')['files']:
        path = (folder/record['path']).resolve()
        if not path.is_relative_to(folder) or sha256(path) != record['sha256']:
            raise ValueError('Recovered asset hash/path mismatch')
    axes = read_json(folder/'axes.json')
    with np.load(folder/'aggregates.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    validate_aggregates(arrays['normalized_count_sums'], arrays['partition_support'],
                        arrays['full_library_totals'], arrays['cell_condition_index'])
    actual = reconstruct_metrics(arrays['normalized_count_sums'], arrays['partition_support'],
                                 axes['group_keys'], axes['contexts'], axes['drugs'])
    if actual != read_json(folder/'metrics.json') or actual != read_json(ROOT/'results/opal_gxe_measurement_v1/technical-a4-20261014/metrics.json'):
        raise ValueError('Historical measurement replay mismatch')
    write_json(args.output, dict(status='aggregate_replay_exact', declared_rows=actual['declared_population'],
        supported_rows=actual['supported_population'], all_rows_exact=True, safe_npz=True,
        aggregate_sha256=sha256(folder/'aggregates.npz'), panel_selection_replayed=False,
        independent_confirmation=False, fitting=False))
    print('all', actual['declared_population'], 'measurement rows replay exactly')


if __name__ == '__main__':
    main()
