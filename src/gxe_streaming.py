"""C-parser recovery with measured, reserved, progress-reporting resource limits."""
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import psutil
from .artifacts import write_json


def coordinates_c(path, shape, check, chunk_lines=250000):
    if chunk_lines <= 0 or len(shape) != 2 or min(shape) <= 0:
        raise ValueError('Positive chunk and annotation dimensions required')
    with pd.read_csv(path, sep='\t', header=None, dtype=np.int64,
                     chunksize=chunk_lines, engine='c') as stream:
        for chunk in stream:
            block = chunk.to_numpy()
            if block.ndim != 2 or block.shape[1] != 3:
                raise ValueError('Exactly three integer coordinate columns required')
            row, col, value = block[:, 0]-1, block[:, 1]-1, block[:, 2]
            if np.any(row < 0) or np.any(row >= shape[0]) or np.any(col < 0) or np.any(col >= shape[1]) or np.any(value < 0):
                raise ValueError('Coordinate bounds/nonnegative count violation')
            check()
            yield row, col, value


def validate_limits(config):
    caps = dict(cpu_seconds=300, ram_bytes=1_000_000_000, retained_bytes=10_000_000,
                wall_seconds=600)
    for key, cap in caps.items():
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= cap:
            raise ValueError('Bounded recovery limit outside authorized envelope: '+key)
    if config.get('reserve_bytes', 0) < 1073741824 or config.get('threads') != 1 or config.get('workers') != 1:
        raise ValueError('One worker/thread and writable reserve required')
    if any(config.get(k) is not False for k in ('downloads', 'installs', 'fitting')):
        raise ValueError('Downloads, installation and model fitting forbidden')
    return config


class RuntimeGuard:
    def __init__(self, output, config):
        self.output, self.config = Path(output), validate_limits(config)
        self.start, self.wall = time.process_time(), time.monotonic()
        self.peak_sampled = self.peak_os = 0
        self.os_peak_available = False
        self.last_report = -10.
        self.progress = dict(phase='input_identity', coordinate_entries=0, selected_coordinate_entries=0)
        self.process = psutil.Process()
        self.finalization_reserve = min(5., config['cpu_seconds']/10)

    def check(self, **progress):
        self.progress.update(progress)
        cpu, wall = time.process_time()-self.start, time.monotonic()-self.wall
        memory = self.process.memory_info()
        self.peak_sampled = max(self.peak_sampled, memory.rss)
        os_peak = getattr(memory, 'peak_wset', 0)
        self.os_peak_available |= bool(os_peak)
        self.peak_os = max(self.peak_os, os_peak)
        size = sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())
        if wall-self.last_report >= 5:
            write_json(self.output/'progress.json', dict(self.progress, cpu_seconds=cpu, wall_seconds=wall,
                peak_sampled_rss_bytes=self.peak_sampled, partial_state_resumable=False))
            print(json.dumps(dict(self.progress, cpu_seconds=round(cpu, 3))), flush=True)
            self.last_report = wall
        if cpu >= self.config['cpu_seconds']-self.finalization_reserve or wall >= self.config['wall_seconds']-5:
            raise RuntimeError('Recovery CPU/wall stop with finalization reserve')
        if max(self.peak_sampled, self.peak_os) > self.config['ram_bytes'] or self.process.children(recursive=True):
            raise RuntimeError('Recovery memory/process stop')
        if size > self.config['retained_bytes']-100000 or shutil.disk_usage(self.output).free < self.config['reserve_bytes']:
            raise RuntimeError('Recovery output/failure/volume reserve stop')

    def receipt(self, status):
        return dict(status=status, cpu_seconds=time.process_time()-self.start,
            wall_seconds=time.monotonic()-self.wall, peak_sampled_rss_bytes=self.peak_sampled,
            os_peak_working_set_bytes=self.peak_os if self.os_peak_available else None,
            os_process_peak_available=self.os_peak_available, continuous_peak_measured=self.os_peak_available,
            cpu_limit_seconds=self.config['cpu_seconds'], wall_limit_seconds=self.config['wall_seconds'],
            finalization_reserve_seconds=self.finalization_reserve, ram_limit_bytes=self.config['ram_bytes'],
            retained_bytes=sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()),
            retained_limit_bytes=self.config['retained_bytes'], progress=self.progress,
            threads=1, workers=1, downloads=0, installs=0, fitting=False, acceptance_created=False)
