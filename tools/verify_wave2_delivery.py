"""Replay wave-two APIs from a package-only installed wheel and local assets."""
import argparse
import importlib.util
from pathlib import Path
import sys
import time
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--installed', type=Path, required=True)
parser.add_argument('--q1b', type=Path, required=True)
parser.add_argument('--partners', type=Path, required=True)
parser.add_argument('--flexible', type=Path, required=True)
parser.add_argument('--budget', type=Path)
parser.add_argument('--output', type=Path, required=True)
a = parser.parse_args()
sys.path.insert(0, str(a.installed.resolve()))

import numpy as np
import opal
import torch
import psutil
from threadpoolctl import threadpool_limits
from opal.artifacts import load_npz, read_json, sha256, write_json
from opal.block_completion import ScalarReference, ScalarCompletion
from opal.flexible_singles import FlexibleSingles
from opal.naiad_reference import NativeNAIAD
from opal.processed_gamma import library, rows, collect_singles, aggregate
from opal.scalar_screen import TASKS, load_screen_roster

if not Path(opal.__file__).resolve().is_relative_to(a.installed.resolve()):
    raise RuntimeError('Expected imports from installed wheel')
if a.output.exists():
    raise FileExistsError('Preserve replay evidence')
started, cpu_started, peak = time.monotonic(), time.process_time(), 0


def guard():
    global peak
    info = psutil.Process().memory_info()
    peak = max(peak, getattr(info, 'peak_wset', info.rss))
    if max(time.monotonic()-started, time.process_time()-cpu_started) > 600 or peak > 2*1024**3:
        raise RuntimeError('Installed replay time/memory cap reached')
    if shutil.disk_usage(a.output.parent).free < 1024**3:
        raise RuntimeError('Installed replay disk reserve reached')


def exact(left, right):
    if left.keys() != right.keys() or any(not np.array_equal(left[k], right[k], equal_nan=left[k].dtype.kind == 'f') for k in left):
        raise ValueError('Installed replay array mismatch')


def check(files):
    for path, digest in files.items():
        if sha256(path) != digest:
            raise ValueError('Installed replay custody mismatch: '+str(path))


torch.set_num_threads(1)
with threadpool_limits(limits=1):
    receipt = dict(installed_module=str(Path(opal.__file__).resolve()),
                   dependencies='Host closure reused; not a fresh dependency installation')
    q = read_json(a.q1b/'registration.json')
    qr = read_json(a.q1b/'replay.json')
    assert qr['passed'] and read_json(a.q1b/'prerequisites.json')['passed']
    check(q['inputs'])
    check({str(a.q1b/'results.json'): qr['results_sha256'], str(a.q1b/'prepared.npz'): qr['arrays_sha256']})
    package = Path(opal.__file__).parent
    assert sha256(package/'processed_gamma.py') == q['code']['src/processed_gamma.py']
    source, guide_path, bank_path = map(Path, q['paths'])
    guides, bank = library(guide_path), load_npz(bank_path)
    singles, counts, _ = collect_singles(rows(source, guides, guard), len(guides), q['config']['minimum_controls'])
    arrays, _ = aggregate(rows(source, guides, guard), bank['genes'], bank['pairs'], singles, q['config']['minimum_constructs'])
    saved = load_npz(a.q1b/'prepared.npz')
    exact(arrays, {k: saved[k] for k in arrays})
    np.testing.assert_array_equal(singles, saved['guide_single'])
    np.testing.assert_array_equal(counts, saved['guide_single_counts'])
    receipt['q1b'] = dict(exact=True, arrays=len(arrays)+2, results_sha256=qr['results_sha256'])
    # This tool imports the already installed opal modules before the sidecar.
    path = Path(__file__).with_name('audit_gamma_control_partners.py')
    spec = importlib.util.spec_from_file_location('installed_partner_replay', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pr = read_json(a.partners/'replay_receipt.json')
    assert pr['passed']
    check({str(a.partners/'partners.npz'): pr['arrays_sha256'], str(a.partners/'results.json'): pr['results_sha256']})
    partner_arrays = module.collect(rows(source, guides, guard), guides)
    exact(partner_arrays, load_npz(a.partners/'partners.npz'))
    receipt['partners'] = dict(exact=True, results_sha256=pr['results_sha256'])
    p = read_json(a.flexible/'protocol.json')
    bp = read_json(Path(p['block_run'])/'protocol.json')
    check(bp['files'])
    assert sha256(package/'flexible_singles.py') == p['code']['src/flexible_singles.py']
    screens = load_screen_roster(Path(bp['original']), Path(bp['prepared']))
    assignments = load_npz(Path(p['block_run'])/'assignments.npz')
    selection = read_json(a.flexible/'selection.json')
    results = read_json(a.flexible/'results.json')
    receipt['flexible'] = {}
    for task in TASKS:
        chunks = []
        for rr in sorted((r for r in selection['fits'] if r['task'] == task), key=lambda r: r['fold']):
            guard()
            path = a.flexible/rr['path']
            check({str(path): rr['sha256']})
            r = read_json(path)
            check({str(path.parent/name): digest for name, digest in r['files'].items()})
            ids = np.flatnonzero(assignments[f'{task}__{rr["fold"]}'] == 2)
            predictions = {f: FlexibleSingles.load(path.parent/f'{f}-{r["chosen"][f]}.npz').predict(screens[task]['pairs'][ids]) for f in ('phenotype', 'nodes')}
            chunks.append(dict(test_indices=ids, **predictions, selected=predictions[r['winner']]))
        arrays = {k: np.concatenate([v[k] for v in chunks]) for k in chunks[0]}
        file = a.flexible/(task+'_predictions.npz')
        check({str(file): results['prediction_hashes'][task]})
        exact(arrays, load_npz(file))
        receipt['flexible'][task] = dict(pairs=len(arrays['test_indices']), exact=True)
    if a.budget:
        from opal.label_budget import BUDGETS, supported_queries
        p = read_json(a.budget/'protocol.json')
        assert sha256(package/'label_budget.py') == p['code']['src/label_budget.py']
        check(p['files'])
        check(p['bank_files'])
        assignments = load_npz(a.budget/'assignments.npz')
        selection, results = read_json(a.budget/'selection.json'), read_json(a.budget/'results.json')
        receipt['budget'] = {}
        for total, _, _ in BUDGETS:
            for task in TASKS:
                chunks = []
                screen = screens[task]
                for rr in sorted((r for r in selection['records'] if r['budget'] == total and r['task'] == task), key=lambda r: r['fold']):
                    guard()
                    path = a.budget/rr['path']
                    check({str(path): rr['sha256']})
                    r = read_json(path)
                    check({str(path.parent/name): digest for name, digest in r['files'].items()})
                    ids = supported_queries(screen['pairs'], assignments[f'{total}__{task}__{rr["fold"]}'], len(screen['genes']))['test']
                    pairs = screen['pairs'][ids]
                    refs = [ScalarReference.load(path.parent/name) for name in r['reference_files']]
                    base = refs[r['chosen_reference']].predict(pairs)
                    values = dict(indices=ids, simple=base)
                    for name in ('aligned', 'shuffled-20261023', 'shuffled-20261024', 'shuffled-20261025'):
                        values['rank8' if name == 'aligned' else name] = base+ScalarCompletion.load(path.parent/(name+'.npz')).predict(pairs)
                    values['native'] = NativeNAIAD.load(p['source'], path.parent/'native').predict(pairs)
                    values['candidate'] = values[r['chosen_family']]
                    for label, ref in zip(r['reference_labels'], refs):
                        values['ref_'+label.replace('/', '_')] = ref.predict(pairs)
                    chunks.append(values)
                arrays = {k: np.concatenate([v[k] for v in chunks]) for k in chunks[0]}
                filename = f'{total}__{task}_predictions.npz'
                check({str(a.budget/filename): results['prediction_hashes'][filename]})
                exact(arrays, load_npz(a.budget/filename))
                receipt['budget'][f'{total}__{task}'] = dict(exact=True, pairs=len(arrays['indices']))
    receipt['status'] = 'all installed APIs replay exactly'
    guard()
    receipt['resources'] = dict(wall_seconds=time.monotonic()-started,
                               cpu_seconds=time.process_time()-cpu_started, peak_bytes=peak)
    write_json(a.output, receipt)
    print(receipt['status'])
