"""Isolated unchanged historical budget prediction replay; no fitting/scoring."""
import argparse
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed', type=Path, required=True)
    parser.add_argument('--budget', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    installed = args.installed.resolve()
    root = Path(__file__).resolve().parents[1]
    if installed == root or installed.is_relative_to(root):
        raise ValueError('An external installed package is required')
    if args.output.exists():
        raise FileExistsError('Preserve replay evidence')
    sys.path.insert(0, str(installed))
    import numpy as np
    import opal
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    from opal.artifacts import load_npz, read_json, sha256, write_json
    from opal.block_completion import ScalarReference, ScalarCompletion
    from opal.naiad_reference import NativeNAIAD
    from opal.label_budget import BUDGETS, supported_queries
    from opal.scalar_screen import TASKS, load_screen_roster

    package = Path(opal.__file__).resolve().parent
    if not package.is_relative_to(installed):
        raise ImportError('Replay imported checkout runtime')
    started, cpu, peak = time.monotonic(), time.process_time(), 0

    def guard():
        nonlocal peak
        info = psutil.Process().memory_info()
        peak = max(peak, getattr(info, 'peak_wset', info.rss))
        if max(time.monotonic()-started, time.process_time()-cpu) > 600 or peak > 2*1024**3:
            raise RuntimeError('Historical replay time/memory cap')
        if shutil.disk_usage(args.output.parent).free < 1024**3:
            raise RuntimeError('Historical replay disk reserve')

    def check(files):
        for name, expected in files.items():
            guard()
            if sha256(Path(name)) != expected:
                raise ValueError('Historical custody mismatch: '+str(name))

    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        protocol = read_json(args.budget/'protocol.json')
        for name in ('artifacts.py', 'block_completion.py', 'label_budget.py', 'naiad_reference.py', 'scalar_screen.py'):
            if sha256(package/name) != protocol['code']['src/'+name]:
                raise ValueError('Historical installed source hash mismatch: '+name)
        check(protocol['files'])
        check(protocol['bank_files'])
        screens = load_screen_roster(Path(protocol['original']), Path(protocol['prepared']))
        assignments = load_npz(args.budget/'assignments.npz')
        selection = read_json(args.budget/'selection.json')
        results = read_json(args.budget/'results.json')
        receipt, checkpoints, folds = {}, 0, 0
        for total, _, _ in BUDGETS:
            for task in TASKS:
                chunks = []
                screen = screens[task]
                records = sorted((r for r in selection['records'] if r['budget'] == total and r['task'] == task), key=lambda r: r['fold'])
                if [r['fold'] for r in records] != list(range(5)):
                    raise ValueError('Incomplete historical fold roster')
                for rr in records:
                    guard()
                    path = args.budget/rr['path']
                    check({str(path): rr['sha256']})
                    record = read_json(path)
                    check({str(path.parent/name): digest for name, digest in record['files'].items()})
                    ids = supported_queries(screen['pairs'], assignments[f'{total}__{task}__{rr["fold"]}'], len(screen['genes']))['test']
                    pairs = screen['pairs'][ids]
                    refs = [ScalarReference.load(path.parent/name) for name in record['reference_files']]
                    base = refs[record['chosen_reference']].predict(pairs)
                    values = dict(indices=ids, simple=base)
                    for name in ('aligned', 'shuffled-20261023', 'shuffled-20261024', 'shuffled-20261025'):
                        values['rank8' if name == 'aligned' else name] = base+ScalarCompletion.load(path.parent/(name+'.npz')).predict(pairs)
                    values['native'] = NativeNAIAD.load(protocol['source'], path.parent/'native').predict(pairs)
                    values['candidate'] = values[record['chosen_family']]
                    for label, ref in zip(record['reference_labels'], refs):
                        values['ref_'+label.replace('/', '_')] = ref.predict(pairs)
                    chunks.append(values)
                    checkpoints += len(refs)+5
                    folds += 1
                arrays = {k: np.concatenate([v[k] for v in chunks]) for k in chunks[0]}
                filename = f'{total}__{task}_predictions.npz'
                check({str(args.budget/filename): results['prediction_hashes'][filename]})
                stored = load_npz(args.budget/filename)
                if arrays.keys() != stored.keys() or any(not np.array_equal(arrays[k], stored[k], equal_nan=arrays[k].dtype.kind == 'f') for k in arrays):
                    raise ValueError('Historical exact prediction mismatch: '+filename)
                receipt[f'{total}__{task}'] = dict(exact=True, pairs=len(arrays['indices']))
        guard()
        write_json(args.output, dict(passed=True, installed_module=str(package), refits=0,
            new_scores=0, folds=folds, checkpoints=checkpoints, tasks=receipt,
            protocol_sha256=sha256(args.budget/'protocol.json'),
            results_sha256=sha256(args.budget/'results.json'),
            selected_budget_unchanged=results['selected_budget'],
            wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
            peak_bytes=peak, host_dependencies_reused=True))
        print(dict(passed=True, folds=folds, checkpoints=checkpoints, refits=0))


if __name__ == '__main__':
    main()
