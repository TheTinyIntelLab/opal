"""Verify wheel/sdist privacy and installed inference/synthetic routes outside checkout."""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
import numpy as np
from opal.artifacts import read_json, write_json, load_npz, sha256, environment
from opal import __version__

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.release_policy import public_name


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data-root', type=Path, required=True); p.add_argument('--cycle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dose-exports', type=Path)
    p.add_argument('--dist-directory', type=Path, default=ROOT / 'dist')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    outside = Path(tempfile.mkdtemp(prefix='opal-stage1-wheel-')); installed = outside / 'installed'
    wheel = args.dist_directory / f'opal_research-{__version__}-py3-none-any.whl'; sdist = args.dist_directory / f'opal_research-{__version__}.tar.gz'
    with zipfile.ZipFile(wheel) as z:
        assert z.testzip() is None; wheel_names = z.namelist()
    with tarfile.open(sdist) as t:
        source_names = t.getnames()
    for name in wheel_names + source_names:
        path = Path(name)
        if not public_name(name.rstrip('/')):
            raise ValueError('Private or unsafe archive path: ' + name)
        if set(path.parts) & {'raw', 'prepared', 'vendor', 'external', 'results'}:
            raise ValueError('Dataset/resource leaked into package: ' + name)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(installed), str(wheel)], check=True, capture_output=True)
    checks = []
    for name in ['norman-log2', 'wessels-log2']:
        run = args.cycle / 'run-v1' / name / 'random'; fold = read_json(run / 'results.json')['folds'][-1]
        with np.load(args.data_root / 'prepared' / name / 'profiles.npz', allow_pickle=False) as f:
            identities = f['targets'][f['pairs'][fold['query']]]
        saved = load_npz(run / f"fold-{fold['fold']}-predictions.npz")
        for family in ['shared', 'pair', 'original_network', 'original_sum_network']:
            checkpoint = (args.cycle / 'exports-v2' / name / family).resolve()
            expected = outside / f'{name}-{family}-expected.npz'
            np.savez_compressed(expected, identities=identities, prediction=saved['neural_' + family])
            checks.append(dict(name=name, family=family, checkpoint=str(checkpoint), expected=str(expected)))
    (outside / 'checks.json').write_text(json.dumps(checks), encoding='utf-8')
    dose_checks = []
    if args.dose_exports:
        for name in ['100', '1000']:
            expected = load_npz(args.dose_exports / f'dose-{name}-expected.npz')
            np.savez_compressed(outside / f'dose-{name}-expected.npz', **expected)
            dose_checks.append(dict(path=str((args.dose_exports / f'dose-{name}').resolve()),
                                   expected=f'dose-{name}-expected.npz', dose=float(name)))
    (outside / 'dose-checks.json').write_text(json.dumps(dose_checks), encoding='utf-8')
    script = outside / 'verify.py'
    script.write_text('''import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import opal,numpy as np
assert Path(opal.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert opal.__version__==EXPECTED_VERSION
records=[]
for check in json.loads(Path('checks.json').read_text()):
    with np.load(check['expected'],allow_pickle=False) as f:
        identities,expected=f['identities'],f['prediction']
    model=opal.load_model(check['checkpoint'])
    predicted=np.stack([model.predict(*pair)['effect'] for pair in identities])
    error=float(np.max(np.abs(predicted-expected)))
    assert error < 1e-10
    np.testing.assert_array_equal(model.predict(*identities[0])['effect'],model.predict(*identities[0][::-1])['effect'])
    records.append(dict(name=check['name'],family=check['family'],pairs=len(expected),maximum_absolute_error=error))
dose_records=[]
for check in json.loads(Path('dose-checks.json').read_text()):
    with np.load(check['expected'],allow_pickle=False) as f:
        states,compounds,expected=f['states'],f['compounds'],f['prediction']
    model=opal.load_model(check['path'])
    predicted=np.stack([model.predict(s,c,check['dose'],endpoint_hours=24)['effect'] for s,c in zip(states,compounds)])
    error=float(np.max(np.abs(predicted-expected)))
    assert error < 1e-10
    dose_records.append(dict(dose_nM=check['dose'],profiles=len(expected),maximum_absolute_error=error))
print(json.dumps(dict(module=opal.__file__,models=records,dose_models=dose_records,version=opal.__version__)))
'''.replace('EXPECTED_VERSION', repr(__version__)), encoding='utf-8')
    env = dict(os.environ); env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(script), str(installed)], cwd=outside, env=env, capture_output=True, text=True)
    (args.output / 'inference.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Outside-checkout installed inference failed; log preserved')
    report = json.loads(result.stdout)
    invocation = f"import sys;sys.path.insert(0,{str(installed)!r});from opal.cli import main;main()"
    quickstart = subprocess.run([sys.executable, '-c', invocation, 'reproduce', '--config',
        str(ROOT / 'configs/experiments/release_quickstart.json'), '--output', str(outside / 'synthetic')],
        cwd=outside, env=env, capture_output=True, text=True)
    (args.output / 'synthetic.log').write_text(quickstart.stdout + quickstart.stderr, encoding='utf-8')
    if quickstart.returncode:
        raise RuntimeError('Installed synthetic CLI failed; log preserved')
    report.update(dict(wheel_sha256=sha256(wheel), sdist_sha256=sha256(sdist),
        private_files_excluded=True, raw_and_external_resources_excluded=True, outside_checkout=True,
        dependencies_reused=True, fresh_dependency_installation=False, synthetic_cli='passed',
        environment=environment(), temporary_validation_directory=str(outside)))
    write_json(args.output / 'verification.json', report); print(report)


if __name__ == '__main__':
    main()
