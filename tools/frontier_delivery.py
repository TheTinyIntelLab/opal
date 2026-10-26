"""Measured engineering-delivery commands. No biological fit and no shell.

Run only when root is not using the numerical lease. Commands are fixed argv
lists. Historical replay paths are the ones recorded in
tools/validate_wave4_delivery.py; missing custody is a ledger blocker.
Failed commands stay in the ledger. Nothing is deleted or overwritten.
Package installation uses pip --no-deps into an empty directory outside this
checkout and reuses the host dependency closure. No clean virtualenv is created.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')

# Copied from tools/validate_wave4_delivery.py. These are not searched or guessed.
BLOCK_RUN = Path('C:/Users/USER/Desktop/opal-research-next-20261020/block-v2')
NATIVE_RUN = Path('E:/opal-research-next-20261021/native-block-v6')
Q1B_RUN = Path('E:/opal-wave2-20261021/q1b-v2')
PARTNERS_RUN = Path('E:/opal-wave2-20261021/q1b-partners-v2')
FLEXIBLE_RUN = Path('E:/opal-wave2-20261021/q2b-v1')
BUDGET_RUN = Path('C:/Users/USER/Desktop/opal-wave2-20261021/q3b-v3')
WAVE3_PRIOR = Path('E:/opal-wave3-20261021/nuisance-v1')
WAVE3_STOPPED = Path('E:/opal-wave3-20261021/flexible-nuisance-v2')
RESET_FILES = (
    'results/fitness-external-20261020/protocol.json',
    'results/research-reset-diagnostics/protocol.json',
    'results/research-reset-reference-swap/protocol.json',
    'results/research-reset-support/v1/protocol.json',
    'results/research-reset-development/protocol.json',
    'results/research-reset-development/results.json',
    'results/research-reset-development/predictions.npz',
    'results/research-reset-development/selection.json',
    'results/research-reset-development/prefit_tests.json',
    'results/research-reset-development/run_started.json',
    'results/research-reset-development/assignment.npz',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outside_repository(path):
    resolved = Path(path).resolve()
    root = ROOT.resolve()
    return resolved != root and root not in resolved.parents


def missing_files(base, names):
    return [str(base / name) for name in names if not (base / name).is_file()]


def reset_blockers():
    missing = [str(ROOT / relative) for relative in RESET_FILES if not (ROOT / relative).is_file()]
    if missing:
        return missing
    try:
        development = json.loads((ROOT / 'results/research-reset-development/protocol.json').read_text(encoding='utf-8'))
        data_root = Path(development['data_root'])
        required = [data_root / 'profiles.npz', data_root.parent / 'prepared/profiles.npz',
                    data_root.parent / 'fitness-v1/assignment.npz', data_root.parent / 'fitness-v1/predictions.npz']
        missing.extend(str(path) for path in required if not path.is_file())
        if not (data_root.parent / 'fitness-reference-model').exists():
            missing.append(str(data_root.parent / 'fitness-reference-model'))
        external = json.loads((ROOT / 'results/fitness-external-20261020/protocol.json').read_text(encoding='utf-8'))
        matches = [Path(key) for key in external['files'] if str(key).endswith('environment1_predictions.npz')]
        if len(matches) != 1:
            missing.append('fitness-external protocol has no unique environment1_predictions.npz path')
        else:
            for cohort in ('environment1', 'environment2', 'parp2022', 'parp2023'):
                path = matches[0].parent / f'{cohort}_predictions.npz'
                if not path.is_file():
                    missing.append(str(path))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        missing.append(f'{type(error).__name__}: {error}')
    return missing


def classify_install(path):
    if not outside_repository(path):
        return 'inside-repository'
    if path.exists() and not path.is_dir():
        return 'not-a-directory'
    if not path.exists():
        return 'missing'
    if any(path.iterdir()):
        if (path / 'opal' / '__init__.py').is_file():
            return 'reuse'
        return 'occupied'
    return 'empty'


def persist(output, record):
    directory = output / 'receipts'
    directory.mkdir(exist_ok=True)
    with (directory / f"{record['name']}.json").open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')


def blank(name, argv):
    return dict(name=name, argv=[str(part) for part in argv], cwd=str(ROOT), returncode=None,
                biological_fit=False, source_acquisition=False, deletion=False,
                started_utc=datetime.now(timezone.utc).isoformat())


def block(output, name, reason, argv=()):
    record = blank(name, argv)
    record.update(status='blocked', blocker=reason, wall_seconds=0)
    persist(output, record)
    print(name, 'blocked', flush=True)
    return record


def environment(installed=None):
    env = dict(os.environ)
    for key in THREAD_KEYS:
        env[key] = '1'
    if installed is None:
        env.pop('PYTHONPATH', None)
    else:
        env['PYTHONPATH'] = str(installed)
    return env


def execute(output, name, argv, *, installed=None, timeout=900):
    log = output / 'logs' / f'{name}.log'
    log.parent.mkdir(exist_ok=True)
    record = blank(name, argv)
    record['timeout_seconds'] = timeout
    started = time.perf_counter()
    try:
        with log.open('x', encoding='utf-8') as stream:
            result = subprocess.run(record['argv'], cwd=ROOT, env=environment(installed),
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, shell=False)
        record['returncode'] = result.returncode
        record['status'] = 'passed' if result.returncode == 0 else 'failed'
    except subprocess.TimeoutExpired:
        record.update(status='failed', error='timeout')
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
    record['wall_seconds'] = time.perf_counter() - started
    if log.exists():
        data = log.read_bytes()
        record.update(log=str(log), log_sha256=hashlib.sha256(data).hexdigest(), log_bytes=len(data))
    print(name, record['status'], record['returncode'], flush=True)
    return record


def artifact_hashes(directory, pattern):
    if not directory.is_dir():
        return {}
    return {path.name: digest(path) for path in sorted(directory.glob(pattern)) if path.is_file()}


def prepare_install(output, requested):
    if requested is None:
        created = Path(tempfile.mkdtemp(prefix='opal-frontier-installed-'))
        if not outside_repository(created):
            return None, f'temp install resolved inside the checkout: {created}'
        return created, None
    kind = classify_install(requested)
    if kind == 'inside-repository':
        return None, f'installed root is inside the checkout: {requested}'
    if kind == 'not-a-directory':
        return None, f'installed root is not a directory: {requested}'
    if kind == 'occupied':
        return None, f'installed root is non-empty and has no opal package; refusing overwrite: {requested}'
    if kind == 'missing':
        if not outside_repository(requested):
            return None, f'installed root is inside the checkout: {requested}'
        requested.mkdir(parents=True, exist_ok=False)
    return requested, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--installed-root', type=Path)
    parser.add_argument('--tournament-run', type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    python = str(Path(sys.executable).resolve())
    records = []
    wheel = args.wheel.resolve() if args.wheel else None
    if wheel is not None:
        if not wheel.is_file():
            records.append(block(output, 'build', f'supplied wheel is missing: {wheel}'))
        else:
            record = blank('build', ())
            record.update(status='skipped', wall_seconds=0, reason='caller supplied --wheel; build was not repeated',
                          wheel=str(wheel), wheel_sha256=digest(wheel))
            persist(output, record)
            records.append(record)
    else:
        built = execute(output, 'build', [python, '-m', 'build', '--no-isolation', '--outdir', output / 'dist'])
        built['wheels'] = artifact_hashes(output / 'dist', '*.whl')
        built['sdists'] = artifact_hashes(output / 'dist', '*.tar.gz')
        wheels = list((output / 'dist').glob('*.whl')) if (output / 'dist').is_dir() else []
        if built['status'] == 'passed' and len(wheels) == 1:
            wheel = wheels[0]
            built['wheel'] = str(wheel)
        elif built['status'] == 'passed':
            built['status'] = 'failed'
            built['error'] = f'expected one wheel, found {len(wheels)}'
        persist(output, built)
        records.append(built)
    installed, install_blocker = prepare_install(output, None if args.installed_root is None else args.installed_root.resolve())
    if install_blocker:
        records.append(block(output, 'install', install_blocker))
        installed = None
    elif classify_install(installed) == 'reuse':
        record = blank('install', ())
        record.update(status='skipped', wall_seconds=0, reason='existing outside install reused; pip was not run',
                      installed_root=str(installed))
        persist(output, record)
        records.append(record)
    elif wheel is None or not Path(wheel).is_file():
        records.append(block(output, 'install', 'no wheel is available'))
        installed = None
    else:
        installed_record = execute(output, 'install', [python, '-m', 'pip', 'install', '--no-deps', '--target', installed, wheel])
        installed_record.update(installed_root=str(installed), wheel=str(wheel),
                                dependency_install='host closure reused; no clean virtualenv',
                                outside_repository=outside_repository(installed))
        persist(output, installed_record)
        if installed_record['status'] != 'passed':
            installed = None
        records.append(installed_record)

    def needs_install(name, command):
        if installed is None:
            records.append(block(output, name, 'installed package is unavailable', command))
            return False
        return True

    reset = [python, str(ROOT / 'tools/verify_reset_delivery.py'), '--installed', installed or '', '--output', output / 'historical.json']
    reset_missing = reset_blockers()
    if reset_missing:
        records.append(block(output, 'historical-reset-original-external', 'missing custody: ' + '; '.join(reset_missing), reset))
    elif needs_install('historical-reset-original-external', reset):
        record = execute(output, 'historical-reset-original-external', reset, installed=installed)
        persist(output, record)
        records.append(record)

    block_missing = missing_files(BLOCK_RUN, ('protocol.json', 'results.json', 'selection.json', 'assignments.npz'))
    native_missing = missing_files(NATIVE_RUN, ('protocol.json', 'selection.json', 'results.json'))
    block_command = [python, str(ROOT / 'tools/verify_block_delivery.py'), '--installed', installed or '',
                     '--block-run', BLOCK_RUN, '--output', output / 'block.json']
    if block_missing:
        records.append(block(output, 'block', 'missing custody: ' + '; '.join(block_missing), block_command))
    elif needs_install('block', block_command):
        if not native_missing:
            block_command = [python, str(ROOT / 'tools/verify_block_delivery.py'), '--installed', installed,
                             '--block-run', BLOCK_RUN, '--native-run', NATIVE_RUN, '--output', output / 'block.json']
        record = execute(output, 'block', block_command, installed=installed)
        persist(output, record)
        records.append(record)
    if native_missing:
        records.append(block(output, 'native', 'missing custody: ' + '; '.join(native_missing),
                             ['--native-run', NATIVE_RUN]))

    wave2_missing = (missing_files(Q1B_RUN, ('registration.json', 'replay.json', 'prerequisites.json'))
                     + missing_files(PARTNERS_RUN, ('replay_receipt.json',))
                     + missing_files(FLEXIBLE_RUN, ('protocol.json',)))
    budget_missing = missing_files(BUDGET_RUN, ('protocol.json',))
    wave2 = [python, str(ROOT / 'tools/verify_wave2_delivery.py'), '--installed', installed or '',
             '--q1b', Q1B_RUN, '--partners', PARTNERS_RUN, '--flexible', FLEXIBLE_RUN,
             '--output', output / 'wave2.json']
    if not budget_missing:
        wave2 = [python, str(ROOT / 'tools/verify_wave2_delivery.py'), '--installed', installed or '',
                 '--q1b', Q1B_RUN, '--partners', PARTNERS_RUN, '--flexible', FLEXIBLE_RUN,
                 '--budget', BUDGET_RUN, '--output', output / 'wave2.json']
    if wave2_missing:
        records.append(block(output, 'reagent-flexible-budget', 'missing custody: ' + '; '.join(wave2_missing), wave2))
    elif needs_install('reagent-flexible-budget', wave2):
        record = execute(output, 'reagent-flexible-budget', wave2, installed=installed)
        persist(output, record)
        records.append(record)
    if budget_missing:
        records.append(block(output, 'budget', 'missing custody: ' + '; '.join(budget_missing), ['--budget', BUDGET_RUN]))

    wave3_missing = (missing_files(WAVE3_PRIOR, ('registration.json', 'selection.json', 'results.json', 'replay.json', 'features.npz'))
                     + missing_files(WAVE3_STOPPED, ('registration.json', 'fit_failure.json')))
    wave3 = [python, str(ROOT / 'tools/verify_wave3_delivery.py'), '--installed', installed or '',
             '--prior', WAVE3_PRIOR, '--stopped', WAVE3_STOPPED, '--output', output / 'wave3.json']
    if wave3_missing:
        records.append(block(output, 'wave3', 'missing custody: ' + '; '.join(wave3_missing), wave3))
    elif needs_install('wave3', wave3):
        record = execute(output, 'wave3', wave3, installed=installed)
        persist(output, record)
        records.append(record)

    assay = [python, str(ROOT / 'tools/rehearse_replicated_assay.py'), '--output', output / 'assay']
    if needs_install('count-assay', assay):
        # The assay script inserts the checkout at sys.path[0], so this checks the synthetic CLI
        # with the installed tree also present. It does not by itself prove an installed import.
        record = execute(output, 'count-assay', assay, installed=installed)
        record['installed_import'] = 'checkout inserted by rehearse_replicated_assay.py'
        persist(output, record)
        records.append(record)
    for version in (1, 2):
        command = [python, str(ROOT / 'tools/check_installed_series.py'), '--require-installed', '--synthetic',
                   '--config', ROOT / f'configs/experiments/opal_model_series_v{version}.json',
                   '--output', output / f'series-v{version}']
        name = f'series-v{version}'
        if needs_install(name, command):
            record = execute(output, name, command, installed=installed)
            persist(output, record)
            records.append(record)

    replay = [python, str(ROOT / 'tools/frontier_tournament.py'), 'replay', '--output', args.tournament_run or '',
              '--installed-root', installed or '']
    if args.tournament_run is None:
        records.append(block(output, 'tournament-replay', 'root has not supplied --tournament-run', replay))
    else:
        run = args.tournament_run.resolve()
        replay[replay.index('--output') + 1] = str(run)
        absent = [name for name in ('protocol.json', 'code-binding.json', 'all-predictions-sealed.json', 'results.json')
                  if not (run / name).is_file()]
        markers = [name for name in ('replay-start.json', 'replay.json') if (run / name).exists()]
        if absent:
            records.append(block(output, 'tournament-replay', 'missing run files: ' + ', '.join(absent), replay))
        elif markers:
            records.append(block(output, 'tournament-replay', 'refusing duplicate replay marker: ' + ', '.join(markers), replay))
        elif needs_install('tournament-replay', replay):
            record = execute(output, 'tournament-replay', replay, installed=installed)
            persist(output, record)
            records.append(record)
    docs = execute(output, 'docs', [python, str(ROOT / 'tools/build_docs.py'), '--output', output / 'docs'])
    persist(output, docs)
    records.append(docs)
    ledger = dict(schema='opal-frontier-delivery-v1', biological_fits=0, source_acquisition=0, deletion=False,
                  clean_venv=False, dependency_policy='pip --no-deps reuses the host closure; hosted CI owns clean installs',
                  installed_root=None if installed is None else str(installed),
                  numerical_thread_limit=1, commands=records,
                  utc=datetime.now(timezone.utc).isoformat())
    with (output / 'ledger.json').open('x', encoding='utf-8') as stream:
        json.dump(ledger, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), failed=sum(record['status'] == 'failed' for record in records),
                          blocked=sum(record['status'] == 'blocked' for record in records))), flush=True)
    return 2 if any(record['status'] == 'failed' for record in records) else 0


if __name__ == '__main__':
    raise SystemExit(main())
