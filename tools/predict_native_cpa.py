"""Predict from a native CPA export without any combination outcome bundle."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, load_npz, write_json, sha256
from opal.contextual.contracts import ReferenceBank
from tools.study_native_cpa import run_native


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--context', required=True)
    p.add_argument('--pair', nargs=2, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Prediction output already exists')
    manifest = read_json(args.checkpoint / 'native_manifest.json')
    expected = {'references.json', 'references.npz', 'native_state.npz'}
    if manifest.get('schema') != 'opal-native-cpa-aggregate-v1' or set(manifest['files']) != expected:
        raise ValueError('Unsupported native CPA manifest')
    for name, checksum in manifest['files'].items():
        if sha256(args.checkpoint / name) != checksum:
            raise ValueError('Native CPA checkpoint fingerprint mismatch')
    bank = ReferenceBank.load(args.checkpoint)
    targets = {str(t): i for i, t in enumerate(bank.arrays['targets'])}
    contexts = {str(c): i for i, c in enumerate(bank.arrays['contexts'])}
    if any(t not in targets for t in args.pair) or args.context not in contexts:
        raise ValueError('Unknown exact-dose intervention or context')
    pair, context = bank.validate_rows(np.array([[targets[t] for t in args.pair]]), np.array([contexts[args.context]]))
    query = {'pairs': pair, 'contexts': context}
    empty = {'pairs': np.empty((0, 2), dtype=int), 'contexts': np.empty(0, dtype=int),
             'responses': np.empty((0, len(bank.arrays['genes'])))}
    pred, _, _ = run_native(bank, empty, None, query, manifest['config'], manifest['seed'],
                             restored_state=load_npz(args.checkpoint / 'native_state.npz'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, effect=pred[0], genes=bank.arrays['genes'])
    write_json(args.output.with_suffix('.json'), {'method': 'native_CPA_aggregate_input',
        'context': args.context, 'pair': args.pair, 'reliability': 'unvalidated',
        'training_outcomes_accessed': False, 'checkpoint_manifest_sha256': sha256(args.checkpoint / 'native_manifest.json')})


if __name__ == '__main__':
    main()
