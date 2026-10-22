"""Serialize delivery checks and preserve every command receipt; no scientific fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    python = str(Path(sys.executable).resolve())
    installed = out / 'installed'
    records = []

    def run(name, arguments, *, package=False, timeout=900):
        command = [python, *map(str, arguments)]
        env = dict(os.environ)
        if package:
            env['PYTHONPATH'] = str(installed)
        started = time.time()
        log = out / (name + '.log')
        record = dict(name=name, command=command, cwd=str(ROOT),
                      installed_pythonpath=package, started_unix=started)
        try:
            with log.open('w', encoding='utf-8') as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=timeout)
            record['returncode'] = result.returncode
        except Exception as error:
            record['error'] = repr(error)
            record['returncode'] = None
        record.update(wall_seconds=time.time()-started, log=str(log),
                      log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        records.append(record)
        (out / 'commands.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
        print(name, record['returncode'], flush=True)
        return record['returncode'] == 0

    run('tests', ['-m', 'pytest', '-q', '--basetemp', out/'tests', '--junitxml', out/'tests.xml'])
    built = run('build', ['-m', 'build', '--outdir', out/'dist'])
    wheel = out/'dist/opal_research-0.1.3.dev0-py3-none-any.whl'
    if built and run('install', ['-m', 'pip', 'install', '--no-deps', '--target', installed, wheel]):
        run('historical', [ROOT/'tools/verify_reset_delivery.py', '--installed', installed, '--output', out/'historical.json'])
        run('block', [ROOT/'tools/verify_block_delivery.py', '--installed', installed,
                     '--block-run', 'C:/Users/USER/Desktop/opal-research-next-20261020/block-v2',
                     '--native-run', 'E:/opal-research-next-20261021/native-block-v6', '--output', out/'block.json'])
        run('wave2', [ROOT/'tools/verify_wave2_delivery.py', '--installed', installed,
                     '--q1b', 'E:/opal-wave2-20261021/q1b-v2', '--partners', 'E:/opal-wave2-20261021/q1b-partners-v2',
                     '--flexible', 'E:/opal-wave2-20261021/q2b-v1', '--budget', 'C:/Users/USER/Desktop/opal-wave2-20261021/q3b-v3',
                     '--output', out/'wave2.json'])
        run('wave3', [ROOT/'tools/verify_wave3_delivery.py', '--installed', installed,
                     '--prior', 'E:/opal-wave3-20261021/nuisance-v1', '--stopped', 'E:/opal-wave3-20261021/flexible-nuisance-v2',
                     '--output', out/'wave3.json'])
        run('assay', [ROOT/'tools/rehearse_replicated_assay.py', '--output', out/'assay'], package=True)
        for version in (1, 2):
            run(f'series-v{version}', [ROOT/'tools/check_installed_series.py', '--require-installed', '--synthetic',
                '--config', ROOT/f'configs/experiments/opal_model_series_v{version}.json', '--output', out/f'series-v{version}'], package=True)
    run('docs', [ROOT/'tools/build_docs.py', '--output', out/'docs'])
    if any(record['returncode'] != 0 for record in records):
        raise SystemExit('Delivery failures preserved in commands.json')


if __name__ == '__main__':
    main()
