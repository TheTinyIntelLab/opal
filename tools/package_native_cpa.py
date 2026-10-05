"""Export allowed references and native weights, verify without fitting labels."""
import argparse
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, load_npz, write_json, sha256
from opal.contextual.contracts import load_dataset
from tools.study_native_cpa import run_native


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT / 'results/next/native-cpa-aggregate-v1')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    protocol = read_json(args.run / 'protocol.json')
    result = read_json(args.run / 'results.json')
    bank, _, query, _ = load_dataset(ROOT / protocol['config']['dataset'])
    bank.save(args.output / 'checkpoint')
    state = load_npz(args.run / 'native_state.npz')
    np.savez_compressed(args.output / 'checkpoint/native_state.npz', **state)
    # CPA's categorical dictionaries require the same allowed reference and
    # intervention identities, but no training response values are necessary.
    fitting = {'pairs': np.empty((0, 2), dtype=int), 'contexts': np.empty(0, dtype=int),
               'responses': np.empty((0, len(bank.arrays['genes'])))}
    pred, _, _ = run_native(bank, fitting, None, query, result['config'],
                            protocol['config']['seed'], restored_state=state)
    original = load_npz(args.run / 'predictions.npz')['prediction']
    error = float(np.max(np.abs(pred - original)))
    np.testing.assert_allclose(pred, original, atol=1e-6, rtol=1e-6)
    with zipfile.ZipFile(args.output / 'provenance.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in (ROOT / 'tools/external/CPA').rglob('*'):
            if path.is_file() and (path.suffix == '.py' or path.name.startswith('LICENSE') or path.suffix == '.toml'):
                archive.write(path, path.relative_to(ROOT).as_posix())
        for path in [ROOT / 'tools/study_native_cpa.py', Path(__file__), ROOT / 'configs/environments/native-cpa-win310.txt']:
            archive.write(path, path.relative_to(ROOT).as_posix())
    files = ['references.json', 'references.npz', 'native_state.npz']
    write_json(args.output / 'checkpoint/native_manifest.json', {
        'schema': 'opal-native-cpa-aggregate-v1', 'config': result['config'], 'seed': protocol['config']['seed'],
        'files': {name: sha256(args.output / 'checkpoint' / name) for name in files},
        'combination_outcomes_included': False, 'native_dependencies_required': True,
        'source_run_sha256': sha256(args.run / 'results.json'), 'provenance_sha256': sha256(args.output / 'provenance.zip'),
        'roundtrip_maximum_absolute_error': error, 'reliability': 'unvalidated',
        'scope': 'Gaussian population-mean adaptation; not the canonical cell-level CPA benchmark'})
    print('Native CPA no-label restore maximum error:', error)


if __name__ == '__main__':
    main()
