"""Verify assets and installed-wheel inference without refitting or scoring.

Install into a fresh target directory outside the checkout. Dependencies are
reused from this host, explicitly not a clean dependency installation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import stat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.release_policy import public_name


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    assets = args.assets.resolve()
    manifest = json.loads(next(assets.glob('opal-*-manifest.json')).read_text())
    version = manifest['version']
    prefix = 'opal-' + version
    for line in (assets / 'SHA256SUMS.txt').read_text().splitlines():
        checksum, name = line.split('  ', 1)
        assert Path(name).name == name and sha(assets / name) == checksum, name
    for path in assets.iterdir():
        if path.suffix in {'.zip', '.whl'}:
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None
                assert not any(stat.S_ISLNK(i.external_attr >> 16) for i in archive.infolist()), 'Archive symlink'
                names = archive.namelist()
        elif path.name.endswith('.tar.gz'):
            with tarfile.open(path) as archive:
                assert all(m.isfile() or m.isdir() for m in archive.getmembers()), 'Unsafe tar member type'
                names = archive.getnames()
        else:
            continue
        for name in names:
            assert public_name(name.rstrip('/')), name
    archives = [(prefix+'-research-source.zip','source_files'),(prefix+'-evidence.zip','evidence_files')]
    if manifest['model_files']:
        archives.append((prefix+'-reference-models.zip','model_files'))
    for archive_name,field in archives:
        with zipfile.ZipFile(assets/archive_name) as archive:
            assert set(archive.namelist())==set(manifest[field]), 'Archive membership differs from manifest'
            for name,digest in manifest[field].items():
                assert hashlib.sha256(archive.read(name)).hexdigest()==digest, 'Per-file hash mismatch'
    outside = Path(tempfile.mkdtemp(prefix='opal-v1-verify-'))
    installed = outside / 'installed'
    wheel = assets / f'opal_research-{version}-py3-none-any.whl'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(installed), str(wheel)],
                   check=True, capture_output=True, text=True)
    if not manifest['model_files']:
        # Source-only releases must work without reaching for private weights.
        env = dict(os.environ)
        env['PYTHONPATH'] = str(installed)
        checked = subprocess.run([sys.executable, '-c',
            'import json,opal;print(json.dumps({"version":opal.__version__,"module":opal.__file__}))'],
            cwd=outside, env=env, capture_output=True, text=True, check=True)
        imported = json.loads(checked.stdout)
        assert imported['version'] == version
        assert Path(imported['module']).resolve().is_relative_to(installed.resolve())
        subprocess.run([sys.executable, '-c', 'from opal.cli import main;main()', '--help'],
                       cwd=outside, env=env, capture_output=True, text=True, check=True)
        with zipfile.ZipFile(assets / (prefix + '-research-source.zip')) as archive:
            archive.extractall(outside / 'source')
        subprocess.run([sys.executable, str(outside / 'source/tools/report_trained_growth_benchmark.py'),
                       '--source', str(ROOT / 'results/frontier-execution-20261026/growth-v3/results.json'),
                       '--output', str(outside / 'benchmark.json')],
                       cwd=outside, env=env, capture_output=True, text=True, check=True)
        report = dict(status='passed', asset_hashes_verified=True, source_only=True,
                      installed=imported, cli_help='passed', aggregate_replay='passed',
                      model_weights_included=False, dependencies_reused=True,
                      temporary_validation_directory=str(outside))
        (assets.parent / (assets.name + '-validation.json')).write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        return
    with zipfile.ZipFile(assets / (prefix + '-reference-models.zip')) as archive:
        archive.extractall(outside)
    with zipfile.ZipFile(assets / (prefix + '-research-source.zip')) as archive:
        archive.extractall(outside / 'source')
    import numpy as np
    nm = json.loads((outside / 'models/norman-kernel/manifest.json').read_text())
    with np.load(ROOT / 'results/improvements/residual-kernel-v1/norman-log2/unscored_query_predictions.npz', allow_pickle=False) as f:
        norman_expected = f['predictions']
    with np.load(ROOT / 'data/prepared/emeraldbay-pilot-v1/queries.npz', allow_pickle=False) as f:
        query = {k: f[k] for k in f.files}
    with np.load(ROOT / 'results/next/emeraldbay-pilot-v1/predictions.npz', allow_pickle=False) as f:
        emerald_expected = f['selected']
        np.testing.assert_array_equal(f['identities'], query['identities'])
    np.savez_compressed(outside / 'checks.npz', norman_expected=norman_expected,
                        norman_pairs=np.asarray(nm['benchmark_pairs']),
                        emerald_expected=emerald_expected, pairs=query['pairs'], contexts=query['contexts'])
    script = outside / 'verify.py'
    script.write_text('''import json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'installed'))
import numpy as np,opal
assert Path(opal.__file__).resolve().is_relative_to(root/'installed')
from opal import load_model
with np.load(root/'checks.npz',allow_pickle=False) as f:q={k:f[k] for k in f.files}
results=[]
for label in ['norman','emerald']:
    model=load_model(root/'models'/('norman-kernel' if label=='norman' else 'emeraldbay-kernel'))
    if label=='norman':
        predictions=np.stack([model.predict(a,b)['effect'] for a,b in q['norman_pairs']])
        a,b=q['norman_pairs'][0];left=model.predict(a,b)['effect'];right=model.predict(b,a)['effect']
    else:
        bank=model.bank.arrays
        predictions=np.stack([model.predict(*bank['targets'][p],context=str(bank['contexts'][c]))['effect'] for p,c in zip(q['pairs'],q['contexts'])])
        a,b=bank['targets'][q['pairs'][0]];c=str(bank['contexts'][q['contexts'][0]])
        left=model.predict(a,b,context=c)['effect'];right=model.predict(b,a,context=c)['effect']
    np.testing.assert_allclose(predictions,q[label+'_expected'],rtol=1e-7,atol=1e-8)
    np.testing.assert_allclose(left,right,atol=1e-10)
    genes=model.reference['genes'] if label=='norman' else model.bank.arrays['genes']
    kwargs={} if label=='norman' else {'context':c}
    for bad_kwargs in [{'context':'__unsupported__'},dict(kwargs,genes=genes[::-1])]:
        try:model.predict(a,b,**bad_kwargs)
        except ValueError:pass
        else:raise AssertionError('Unsupported context/gene order was accepted')
    try:model.predict('__unknown__',b,**kwargs)
    except ValueError:pass
    else:raise AssertionError('Missing single was accepted')
    result=model.predict(a,b,**kwargs)
    assert result['genes'].tolist()==model.manifest['gene_order']
    assert result.get('units',result.get('scale'))
    results.append({'model':label,'saved_predictions_replayed':len(predictions),'maximum_absolute_error':float(np.max(np.abs(predictions-q[label+'_expected']))),'gene_order_scale_support_guards':'passed'})
print(json.dumps({'models':results,'installed_module':opal.__file__,'dependencies_reused':True,'outside_checkout':True,'version':opal.__version__}))
''', encoding='utf-8')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    checked = subprocess.run([sys.executable, str(script), str(outside)], cwd=outside, env=env,
                             capture_output=True, text=True, check=True)
    report = json.loads(checked.stdout)
    assert report['version'] == version, 'Installed wheel version mismatch'
    quickstart = subprocess.run([sys.executable, '-c',
        f"import sys;sys.path.insert(0,{str(installed)!r});from opal.cli import main;main()",
        'reproduce', '--config', str(outside / 'source/configs/experiments/release_quickstart.json'),
        '--output', str(outside / 'quickstart')], cwd=outside, env=env, capture_output=True, text=True)
    if quickstart.returncode:
        raise RuntimeError(quickstart.stdout + quickstart.stderr)
    report.update({'asset_hashes_verified': True, 'private_files_excluded': True,
                   'synthetic_quickstart': 'passed', 'wheel_sha256': sha(wheel),
                   'python': sys.version, 'temporary_validation_directory': str(outside)})
    (assets.parent / (assets.name + '-validation.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
