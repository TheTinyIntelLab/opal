"""Replay registered missing-block artifacts through an installed Opal wheel."""
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--installed', type=Path, required=True)
parser.add_argument('--block-run', type=Path, required=True)
parser.add_argument('--native-run', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.installed.resolve()))

import numpy as np
import opal
import torch
from threadpoolctl import threadpool_limits
from opal.artifacts import load_npz, read_json, sha256, write_json
from opal.block_completion import ScalarCompletion, ScalarReference
from opal.scalar_screen import TASKS, load_screen_roster

if not Path(opal.__file__).resolve().is_relative_to(args.installed.resolve()):
    raise RuntimeError('Expected installed-wheel imports')
if args.output.exists():
    raise FileExistsError('Preserve replay receipts')


def verify_files(files, root=None):
    for name, expected in files.items():
        path = Path(name) if root is None else root / name
        if sha256(path) != expected:
            raise ValueError(f'Artifact drift: {path}')


torch.set_num_threads(1)
with threadpool_limits(limits=1):
    run = args.block_run
    protocol = read_json(run / 'protocol.json')
    verify_files(protocol['files'])
    results, selection = read_json(run / 'results.json'), read_json(run / 'selection.json')
    if results['protocol_sha256'] != sha256(run / 'protocol.json') or results['selection_sha256'] != sha256(run / 'selection.json'):
        raise ValueError('Q2 lifecycle drift')
    screens = load_screen_roster(Path(protocol['original']), Path(protocol['prepared']))
    assignments = load_npz(run / 'assignments.npz')
    checks = {}
    for task in TASKS:
        screen, pieces = screens[task], []
        for record in (r for r in selection['fits'] if r['task'] == task):
            path = run / record['path']
            verify_files({str(path): record['sha256']})
            r = read_json(path)
            verify_files(r['files'], path.parent)
            indices = np.flatnonzero(assignments[f"{task}__{r['fold']}"] == 2)
            refs = [ScalarReference.load(path.parent / name) for name in r['reference_checkpoints']]
            base = refs[r['chosen_reference']].predict(screen['pairs'][indices])
            arrays = dict(test_indices=indices, baseline=base)
            for model in r['models']:
                key = 'candidate' if model['name'] == 'aligned' else model['name']
                arrays[key] = base + ScalarCompletion.load(path.parent / model['checkpoint']).predict(screen['pairs'][indices])
            arrays.update({label.replace('/', '_'): model.predict(screen['pairs'][indices])
                           for label, model in zip(r['reference_labels'], refs)})
            pieces.append(arrays)
        if len(pieces) != 5:
            raise ValueError('All five blocks required')
        stored = load_npz(run / (task + '_predictions.npz'))
        verify_files({str(run / (task + '_predictions.npz')): results['prediction_hashes'][task]})
        if set(stored) != set(pieces[0]) or any(not np.array_equal(stored[k], np.concatenate([p[k] for p in pieces])) for k in stored):
            raise RuntimeError('Installed Q2 replay mismatch')
        checks[task] = dict(pairs=len(stored['test_indices']), arrays=len(stored), exact=True)
    native_checks = {}
    if args.native_run:
        from opal.naiad_reference import NativeNAIAD
        nr = args.native_run
        np_protocol, ns = read_json(nr / 'protocol.json'), read_json(nr / 'selection.json')
        nresults = read_json(nr / 'results.json')
        if nresults['selection_sha256'] != sha256(nr / 'selection.json') or nresults['protocol_sha256'] != sha256(nr / 'protocol.json'):
            raise ValueError('Native lifecycle drift')
        for task in TASKS:
            pieces = []
            for record in sorted((r for r in ns['fits'] if r['task'] == task), key=lambda r: r['fold']):
                path = nr / record['path']
                verify_files({str(path): record['sha256']})
                r = read_json(path)
                verify_files(r['files'], nr)
                indices = np.flatnonzero(assignments[f"{task}__{r['fold']}"] == 2)
                model = NativeNAIAD.load(np_protocol['source'], nr / r['checkpoint'])
                pieces.append((indices, model.predict(screens[task]['pairs'][indices])))
            if len(pieces) != 5:
                raise ValueError('All five native blocks required')
            stored = load_npz(nr / (task + '_predictions.npz'))
            verify_files({str(nr / (task + '_predictions.npz')): nresults['prediction_hashes'][task]})
            if not np.array_equal(stored['test_indices'], np.concatenate([p[0] for p in pieces])) or not np.array_equal(stored['native'], np.concatenate([p[1] for p in pieces])):
                raise RuntimeError('Installed native replay mismatch')
            native_checks[task] = dict(pairs=len(stored['native']), exact=True)
    write_json(args.output, dict(installed_module=str(Path(opal.__file__).resolve()),
        block=checks, native=native_checks, status='exact installed-wheel numerical replay'))
