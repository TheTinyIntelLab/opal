"""Execute the frozen small public-count diagnostic; no network or fitting."""
import argparse
import gzip
import io
import os
from pathlib import Path
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
from opal.external_repeat import soft_samples, repeat_diagnostic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start, wall = time.process_time(), time.monotonic()
    process = psutil.Process()
    cfg_path = ROOT/'configs/experiments/opal_external_repeat_diagnostic_v1.json'
    cfg = read_json(cfg_path)
    status = 'failed'
    try:
        metadata = args.input/'GSE95153_family.soft.gz'
        matrix = args.input/'GSE95153_Villanueva_processed.txt.gz'
        acquisition = read_json(args.input/'diagnostic_acquisition.json')
        if (sha256(metadata) != cfg['metadata_sha256'] or sha256(matrix) != acquisition['sha256']
                or sha256(cfg_path) != acquisition['config_sha256']):
            raise ValueError('Source/config acquisition identity mismatch')
        if matrix.stat().st_size > cfg['compressed_input_limit_bytes']:
            raise ValueError('Compressed source cap')
        with gzip.open(matrix, 'rb') as f:
            data = f.read(cfg['expanded_input_limit_bytes']+1)
        if len(data) > cfg['expanded_input_limit_bytes']:
            raise ValueError('Expanded source cap')
        header = data.splitlines()[0].decode().split('\t')
        if len(header) != len(set(header)) or header[0] != 'Gene':
            raise ValueError('Duplicate or unknown source header')
        frame = pd.read_csv(io.BytesIO(data), sep='\t', index_col=0)
        samples = soft_samples(metadata)
        write_json(args.output/'registration.json', dict(config=cfg, config_sha256=sha256(cfg_path),
            code_sha256=sha256(__file__), helper_sha256=sha256(ROOT/'src/external_repeat.py'),
            metadata_sha256=sha256(metadata), matrix_sha256=sha256(matrix),
            source_scope=read_json(args.input/'diagnostic_scope_before_acquisition.json'),
            predictive_evaluation=False, authorization='current user authorizes bounded public-data analysis'))
        report = repeat_diagnostic(list(frame.columns), list(frame.index), frame.to_numpy(dtype=np.float64).T, samples)
        memory = process.memory_info()
        peak = max(memory.rss, getattr(memory, 'peak_wset', 0))
        if time.process_time()-start > cfg['cpu_limit_seconds'] or time.monotonic()-wall > cfg['wall_limit_seconds'] or peak > cfg['ram_limit_bytes']:
            raise RuntimeError('Bounded diagnostic resource limit')
        write_json(args.output/'report.json', report)
        if sum(q.stat().st_size for q in args.output.iterdir()) > cfg['retained_output_limit_bytes']:
            raise RuntimeError('Retained diagnostic limit')
        status = report['status']
        print(status, report['gene_count'], 'genes; no predictive or biological confirmation claim')
    except Exception as error:
        write_json(args.output/'failure.json', dict(status='failed', reason=str(error)))
        raise
    finally:
        memory = process.memory_info()
        write_json(args.output/'resource_ledger.json', dict(status=status,
            cpu_seconds=time.process_time()-start, wall_seconds=time.monotonic()-wall,
            peak_sampled_rss_bytes=memory.rss, os_peak_working_set_bytes=getattr(memory, 'peak_wset', None),
            continuous_peak_available=hasattr(memory, 'peak_wset'), limits=cfg,
            fitting=False, installs=0, threads=1, workers=1, network_requests=0))


if __name__ == '__main__':
    main()
