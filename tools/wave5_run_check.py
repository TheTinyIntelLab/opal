"""Bound, monitored delivery/experiment command; preserve every attempted invocation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--cwd', type=Path, default=Path.cwd())
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--peak-bytes', type=int, default=2*1024**3)
    parser.add_argument('--pythonpath', type=Path)
    parser.add_argument('--bind', action='append', default=[])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('command required')
    receipt = args.receipt.resolve()
    start_path = Path(str(receipt)+'.started.json')
    log_path = Path(str(receipt)+'.log')
    if any(p.exists() for p in (receipt, start_path, log_path)):
        raise FileExistsError('Preserve previous command')
    root = Path(__file__).resolve().parents[1]
    bound = [Path(__file__), root/'results/research-wave5/freeze.json', *map(Path,args.bind)]
    hashes = {str(p.resolve()):digest(p) for p in bound}
    record = dict(command=command,cwd=str(args.cwd.resolve()),
                  pythonpath=str(args.pythonpath.resolve()) if args.pythonpath else None,
                  started_utc=datetime.now(timezone.utc).isoformat(), bound_sha256=hashes,
                  caps=dict(timeout_seconds=args.timeout,peak_bytes=args.peak_bytes),
                  sampling_seconds=.05, timeout_enforcement='sampled external watchdog; kills process tree',
                  cpu_measurement='sampled sum across live process tree; may miss process-exit tail')
    write(start_path, record)
    env=dict(os.environ)
    if args.pythonpath:
        env['PYTHONPATH']=str(args.pythonpath.resolve())
    started=time.perf_counter()
    peak=0
    cpu=0.
    stopped=None
    with log_path.open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=args.cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        proc=psutil.Process(child.pid)
        while child.poll() is None:
            members=[proc]
            try:
                members+=proc.children(recursive=True)
            except psutil.NoSuchProcess:
                pass
            rss=tree_cpu=0
            for member in members:
                try:
                    info=member.memory_info()
                    rss+=max(info.rss,getattr(info,'peak_wset',0))
                    ct=member.cpu_times()
                    tree_cpu+=ct.user+ct.system
                except (psutil.NoSuchProcess,psutil.AccessDenied):
                    pass
            peak=max(peak,rss)
            cpu=max(cpu,tree_cpu)
            if time.perf_counter()-started>args.timeout:
                stopped='wall cap'
            elif peak>args.peak_bytes:
                stopped='memory cap'
            if stopped:
                for member in reversed(members):
                    try:member.kill()
                    except psutil.NoSuchProcess:pass
                break
            time.sleep(.05)
        rc=child.wait()
    drift=[name for name,h in hashes.items() if digest(name)!=h]
    record.update(returncode=rc,stopped=stopped,wall_seconds=time.perf_counter()-started,
                  sampled_cpu_seconds=cpu,sampled_tree_peak_bytes=peak,
                  log=str(log_path),log_sha256=digest(log_path),log_bytes=log_path.stat().st_size,
                  bound_drift=drift,passed=rc==0 and stopped is None and not drift)
    write(receipt,record)
    print(json.dumps({k:record[k] for k in ('passed','returncode','stopped','wall_seconds','sampled_cpu_seconds','sampled_tree_peak_bytes','bound_drift')}))
    if not record['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
