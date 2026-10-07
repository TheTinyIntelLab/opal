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

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data-root', type=Path, required=True); p.add_argument('--cycle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    outside = Path(tempfile.mkdtemp(prefix='opal-stage1-wheel-')); installed = outside / 'installed'
    wheel = ROOT / 'dist/opal_research-0.1.2.dev1-py3-none-any.whl'; sdist = ROOT / 'dist/opal_research-0.1.2.dev1.tar.gz'
    with zipfile.ZipFile(wheel) as z:
        assert z.testzip() is None; wheel_names = z.namelist()
    with tarfile.open(sdist) as t:
        source_names = t.getnames()
    for name in wheel_names + source_names:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.name in {'plan.md', 'context.md', 'roadmap.md', 'outreach.md'}:
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
    script = outside / 'verify.py'
    script.write_text('''import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import opal,numpy as np
assert Path(opal.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert opal.__version__=='0.1.2.dev1'
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
print(json.dumps(dict(module=opal.__file__,models=records,version=opal.__version__)))
''', encoding='utf-8')
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
