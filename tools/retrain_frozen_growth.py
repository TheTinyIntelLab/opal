"""Reproduce frozen biological training without another test score or selection rule."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np

# The existing adapter sets thread controls before importing its numerical stack.
from tools import frontier_growth_v3 as growth


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    protocol = growth.verify_binding(source)
    if protocol['id'] != 'opal-frontier-growth-v3':
        raise ValueError('Requires the original growth-v3 source')
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Preserve the original and use a fresh separate output')
    output.mkdir(parents=True, exist_ok=False)
    registration = Path(__file__).resolve().parents[1]/'docs/model_delivery/training_reproduction.md'
    write(output/'started.json', dict(source=str(source), protocol_sha256=sha(source/'protocol.json'),
        registration_sha256=sha(registration), runner_sha256=sha(__file__),
        fit_seed=growth.FIT_SEED, test_targets_opened=False, new_scores=0))
    input_seal = growth.read(source/'inputs-sealed.json')['fit_args']
    # Seals are checked with the original implementation, without opening targets.
    growth._verify_prediction_seal(source, growth.REGISTERED_PROCEDURES)
    t0, c0 = time.perf_counter(), time.process_time()
    records = []

    def check():
        if time.perf_counter()-t0 > 900 or time.process_time()-c0 > 900:
            raise RuntimeError('Training reproduction time cap')
        if shutil.disk_usage(output).free < 1024**3:
            raise RuntimeError('Training reproduction disk reserve')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 80*1024**2:
            raise RuntimeError('Training reproduction artifact cap')

    try:
        for task in protocol['tasks']:
            check()
            original = source/'procedures'/task/'fold-0'
            fit_path = original/'fit-args.npz'
            if sha(fit_path) != input_seal[fit_path.relative_to(source).as_posix()]:
                raise ValueError('Original fit-input drift')
            destination = output/task/'fold-0'
            destination.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(fit_path, destination/'fit-args.npz')
            growth.fit_procedure(destination, check=check)
            old, new = growth.read(original/'selection.json'), growth.read(destination/'selection.json')
            if old != new:
                # Container hashes can differ despite identical array states.
                def decisions(choice):
                    return {k:v for k,v in choice.items() if k != 'models'}
                if decisions(old) != decisions(new):
                    raise ValueError('Training selection or validation record mismatch')
            if len(old['models']) != 28 or len(new['models']) != 28:
                raise ValueError('Training roster mismatch')
            with np.load(original/'predictions.npz', allow_pickle=False) as a, np.load(destination/'predictions.npz', allow_pickle=False) as b:
                if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                    raise ValueError('Training prediction reproduction mismatch')
            for before, after in zip(old['models'], new['models']):
                if any(before[k] != after[k] for k in ('name','family','checkpoint','prediction_key')):
                    raise ValueError('Model roster alignment mismatch')
                left = growth.load_checkpoint(original/'checkpoints'/before['checkpoint'], before['checkpoint_sha256'])
                right = growth.load_checkpoint(destination/'checkpoints'/after['checkpoint'], after['checkpoint_sha256'])
                for x,y in ((left,right),(left.base,right.base)):
                    if (x is None) != (y is None): raise ValueError('Residual layer mismatch')
                    if x is not None and (set(x.state) != set(y.state) or any(not np.array_equal(x.state[k], y.state[k]) for k in x.state)):
                        raise ValueError('Model-state reproduction mismatch')
            records.append(dict(task=task, fold=0, models=28, state_equal=True,
                predictions_equal=True, selected_policy=new['selected_policy'],
                selection_sha256=sha(destination/'selection.json')))
            check()
            print(json.dumps(records[-1]), flush=True)
        write(output/'reproduction.json', dict(passed=True, new_biological_training_procedures=6,
            models=168, selected_models=6, new_scores=0, test_targets_opened=False,
            independent_confirmation=False, wall_seconds=time.perf_counter()-t0,
            cpu_seconds=time.process_time()-c0, tasks=records,
            artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())))
    except Exception as error:
        write(output/'failure.json', dict(error=type(error).__name__, message=str(error),
            completed_tasks=records, wall_seconds=time.perf_counter()-t0))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source_run, args.output)


if __name__ == '__main__':
    main()
