"""Launch bounded local work with append-only process-tree receipts.

CPU is the last observed cumulative CPU of each PID, counted once. Polling can
miss short-lived children; receipts disclose that limitation rather than infer it.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import tempfile

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/opal_information_v1'


def write_receipt(path, receipt):
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(receipt, output, indent=2, allow_nan=False)
            output.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def retained(roots):
    roots = {Path(p).resolve() for p in roots}
    roots = [p for p in roots if not any(p != q and p.is_relative_to(q) for q in roots)]
    return {str(p): sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in roots}


def run(command, identity, timeout=1800, cwd=None):
    if Path(command[0]).is_file():
        command = [str(Path(command[0]).resolve())] + command[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = OUT / 'receipts'
    receipts.mkdir(exist_ok=True)
    path = receipts / (identity + '.json')
    if path.exists():
        raise FileExistsError(path)
    stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt = {'id': identity, 'command': command, 'start_utc': stamp(),
               'status': 'launched', 'poll_seconds': .02, 'hardware': platform.platform(),
               'cpu_count': psutil.cpu_count(), 'host_ram_bytes': psutil.virtual_memory().total,
               'python_monitor': platform.python_version(), 'historical_cpu': 'unknown partial lower bound',
               'limitations': 'Sampled RSS and last-observed per-PID CPU; short-lived children and final exit increments may be missed. No claim of exact CPU accounting.',
               'threads': 1, 'gpu_seconds': None}
    write_receipt(path, receipt)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    started = time.monotonic()
    cpus, peak, seen = {}, 0, set()
    # Files avoid pipe deadlocks while preserving failures and full logs.
    with (receipts / (identity + '.stdout.txt')).open('w') as stdout, (receipts / (identity + '.stderr.txt')).open('w') as stderr:
        try:
            process = subprocess.Popen(command, cwd=cwd or ROOT, env=env, stdout=stdout, stderr=stderr)
        except OSError as error:
            receipt.update(status='launch_failed', end_utc=stamp(), error=str(error),
                           wall_seconds=time.monotonic()-started, cpu_seconds_observed=0,
                           peak_tree_rss_bytes=0, exit_code=None)
            write_receipt(path, receipt)
            raise
        receipt['pid'] = process.pid
        write_receipt(path, receipt)
        timed_out = False
        while process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                tree = [parent] + parent.children(recursive=True)
            except psutil.Error:
                tree = []
            rss = 0
            for child in tree:
                try:
                    key = f'{child.pid}:{child.create_time()}'
                    times = child.cpu_times()
                    cpus[key] = times.user + times.system
                    seen.add(child.pid)
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, rss)
            if time.monotonic() - started > timeout:
                timed_out = True
                for child in reversed(tree):
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
                break
            time.sleep(.02)
        code = process.wait()
    receipt.update(end_utc=stamp(), wall_seconds=time.monotonic()-started,
                   cpu_seconds_observed=sum(cpus.values()), peak_tree_rss_bytes=peak,
                   per_process_cpu_seconds=cpus, pids=sorted(seen), exit_code=code,
                   status='timeout' if timed_out else ('complete' if code == 0 else 'failed'))
    write_receipt(path, receipt)
    print(json.dumps({k: receipt[k] for k in ('id', 'exit_code', 'wall_seconds', 'cpu_seconds_observed', 'peak_tree_rss_bytes')}))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--cwd')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    raise SystemExit(run(command, args.id, args.timeout, args.cwd))
