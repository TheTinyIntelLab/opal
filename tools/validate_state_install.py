"""Validate a wheel and NumPy exports outside the checkout in a clean venv."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import venv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import load_npz, read_json, sha256, write_json
from opal.state.inference import StatePredictor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    # A source snapshot uses an allowlist, so private files cannot enter the archive.
    import zipfile
    snapshot = args.output / 'source.zip'
    source_files = sorted((ROOT / 'src/opal').rglob('*.py')) + sorted((ROOT / 'tools').glob('*state*.py'))
    with zipfile.ZipFile(snapshot, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in source_files:
            archive.write(file, file.relative_to(ROOT).as_posix())
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                    '--wheel-dir', str(args.output.resolve()), str(ROOT)], check=True)
    wheel = next(args.output.glob('*.whl'))
    clean = args.output / 'clean-install'; venv.EnvBuilder(with_pip=True).create(clean)
    python = clean / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    subprocess.run([str(python), '-m', 'pip', 'install', 'numpy>=2.1,<3', '--no-deps', str(wheel.resolve())], check=True)
    query_dir = args.output / 'queries'; query_dir.mkdir()
    for i, path in enumerate(args.checkpoints):
        manifest = read_json(path / 'manifest.json')
        fitted = load_npz(path / (manifest['members'][0] if 'members' in manifest else '') / 'fitted.npz')
        predictor = StatePredictor(path)
        targets = [t for j, t in enumerate(manifest['targets']) if fitted['target_support'][j] > 0]
        names = targets[:2] if manifest['residual'] is not None else targets[:1]
        # A previously unregistered compatible reference tests the new-context route.
        baseline = fitted['source_controls'].mean(0) + .001
        kwargs = dict(baseline=baseline, genes=manifest['genes'], assay=manifest['assay'], scale=manifest['scale'])
        expected = predictor.predict(*names, **kwargs)
        np.savez_compressed(query_dir / f'{i}.npz', baseline=baseline, genes=np.asarray(manifest['genes']),
                            assay=np.asarray(manifest['assay']), scale=np.asarray(manifest['scale']),
                            targets=np.asarray(names), effect=expected['effect'], absolute=expected['absolute_response'])
        shutil.copytree(path, args.output / 'exports' / str(i))
    import tempfile
    outside = Path(tempfile.mkdtemp(prefix='opal-state-install-'))
    script = outside / 'validate.py'
    script.write_text('''import sys,json
from pathlib import Path
import numpy as np
import opal
from opal.state import StatePredictor
root=Path(sys.argv[1]); results=[]
for query in sorted((root/'queries').glob('*.npz')):
    with np.load(query,allow_pickle=False) as f: q={k:f[k] for k in f.files}
    predictor=StatePredictor(root/'exports'/query.stem)
    assert isinstance(opal.load_model(root/'exports'/query.stem),StatePredictor)
    r=predictor.predict(*q['targets'].tolist(),baseline=q['baseline'],genes=q['genes'],assay=q['assay'].item(),scale=q['scale'].item(),context='unregistered_baseline')
    error=float(np.max(np.abs(r['effect']-q['effect'])))
    assert error < 1e-10
    np.testing.assert_allclose(r['absolute_response'],q['absolute'],atol=1e-10)
    results.append({'checkpoint':query.stem,'maximum_absolute_error':error,'route':r['route']})
assert 'torch' not in sys.modules
print(json.dumps({'predictions':results,'torch_imported':False,'module':opal.__file__,'outside_source_checkout':True}))
''', encoding='utf-8')
    # Isolated Python ignores PYTHONPATH and user site packages.
    result = subprocess.run([str(python.resolve()), '-I', str(script), str(args.output.resolve())], cwd=outside,
                            capture_output=True, text=True, check=True)
    import json
    validation = json.loads(result.stdout)
    cli_checks = []
    for i, _ in enumerate(args.checkpoints):
        q = load_npz(query_dir / f'{i}.npz')
        baseline = outside / f'baseline-{i}.npz'
        np.savez_compressed(baseline, **{k: q[k] for k in ['baseline', 'genes', 'assay', 'scale']})
        destination = outside / f'prediction-{i}.npz'
        command = [str(python.resolve()), '-I', '-c', 'from opal.cli import main; main()',
            'state-predict', '--checkpoint', str((args.output / 'exports' / str(i)).resolve()),
            '--baseline', str(baseline), '--output', str(destination)]
        command += ['--pair', *q['targets'].tolist()] if len(q['targets']) == 2 else ['--single', q['targets'][0]]
        subprocess.run(command, cwd=outside, capture_output=True, text=True, check=True)
        error = float(np.max(np.abs(load_npz(destination)['effect'] - q['effect'])))
        if error > 1e-10:
            raise ValueError('Installed CLI prediction differs from the Python route')
        cli_checks.append({'checkpoint': i, 'maximum_absolute_error': error})
    validation['cli_checks'] = cli_checks
    validation.update({'wheel_sha256': sha256(wheel), 'source_snapshot_sha256': sha256(snapshot),
                       'system_site_packages': False, 'source_files': {str(p.relative_to(ROOT)): sha256(p) for p in source_files}})
    write_json(args.output / 'installed_validation.json', validation)
    print(result.stdout)


if __name__ == '__main__':
    main()
