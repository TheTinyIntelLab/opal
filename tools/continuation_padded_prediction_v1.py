"""Frozen no-fit padding matrix. Outputs are numerical receipts, never scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import wave5_numerical_matrix as original

OUT = ROOT / 'results/research-continuation'
FOLDER = OUT / 'padded-prediction-v1'


def write(path, value):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def gates():
    freeze_path = OUT / 'padded-prediction-freeze-v1.json'
    binding_path = OUT / 'padded-prediction-code-binding-v1.json'
    freeze, binding = original.read(freeze_path), original.read(binding_path)
    if original.digest(freeze_path) != binding['freeze_sha256']:
        raise ValueError('Padding freeze drift')
    for path, expected in {**freeze['repo_sha256'], **freeze['external_sha256'], **binding['code_sha256']}.items():
        if original.digest(ROOT / path) != expected:
            raise ValueError('Padding input drift: ' + path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    return freeze


def padded_stage(model, pairs, tile, joint, trailing=False, repeat_filler=False):
    import numpy as np
    # Trace wrapper intentionally computes the original stage too. Timings include
    # this diagnostic overhead and cannot be advertised as an optimized predictor.
    raw = original.stage(model, pairs)
    n, k = raw['kernel'].shape
    if not 0 < n <= tile:
        raise ValueError('Invalid tile occupancy')
    slot = tile - n if trailing else 0
    padded = np.zeros((tile, k), dtype=np.float64)
    if repeat_filler:
        padded[:] = raw['kernel'][0]
    padded[slot:slot+n] = raw['kernel']
    white = (padded @ np.ascontiguousarray(model.whitener))[slot:slot+n]
    features = np.concatenate((raw['features'][:, :-k], white), axis=1)
    standardized = (features-model.center)/model.scale
    if joint:
        padded_features = np.zeros((tile, standardized.shape[1]), dtype=np.float64)
        if repeat_filler:
            padded_features[:] = standardized[0]
        padded_features[slot:slot+n] = standardized
        dot = (padded_features @ model.coefficient)[slot:slot+n]
    else:
        dot = standardized @ model.coefficient
    raw.update(whitened=white, features=features, standardized=standardized,
               dot=dot, prediction=(raw['additive']+model.offset)+dot)
    if not all(np.isfinite(v).all() for v in raw.values()):
        raise ValueError('Nonfinite padded arithmetic')
    return raw


def predict(model, pairs, config, sizes=None, trailing=False, repeat_filler=False):
    import numpy as np
    result = np.empty(len(pairs))
    sizes = sizes or (len(pairs),)
    position = calls = 0
    while position < len(pairs):
        end = min(len(pairs), position + sizes[calls % len(sizes)])
        for start in range(position, end, config['tile']):
            chunk = pairs[start:min(start+config['tile'], end)]
            result[start:start+len(chunk)] = padded_stage(model, chunk, config['tile'], config['joint'], trailing, repeat_filler)['prediction']
        position = end
        calls += 1
    return result


def anchored_stages(model, pairs, anchors, config):
    import numpy as np
    result = {}
    for start in range(0, len(pairs), config['tile']):
        chosen = np.flatnonzero((anchors >= start) & (anchors < start+config['tile']))
        if not len(chosen):
            continue
        current = padded_stage(model, pairs[start:start+config['tile']], config['tile'], config['joint'])
        if not result:
            result = {k: np.empty((len(anchors),)+v.shape[1:]) for k, v in current.items()}
        for key, value in current.items():
            result[key][chosen] = value[anchors[chosen]-start]
    return result


def worker(index):
    config = gates()['configs'][index]
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(config['threads'])
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_info, threadpool_limits
    from opal.flexible_singles import FlexibleSingles
    def forbidden(*args, **kwargs):
        raise RuntimeError('No fit/prepare/solve/lstsq in padding experiment')
    FlexibleSingles.fit = FlexibleSingles.prepare = FlexibleSingles.solve = forbidden
    np.linalg.solve = np.linalg.lstsq = forbidden
    begun, cpu = time.perf_counter(), time.process_time()
    with threadpool_limits(1):
        model, bank, ids = original.load_allowed({})
        query = bank['pairs'][ids]
        histories = original.history(model, bank, ids)
        reference = model.predict(query)
        anchors = np.unique(np.r_[np.linspace(0, len(query)-1, 129, dtype=int), np.arange(len(query)-4, len(query))])
        legacy_stages = original.global_trace(model, query, anchors)
        if not np.array_equal(legacy_stages['prediction'], reference[anchors]):
            raise ValueError('Reference trace drift')
    with threadpool_limits(config['threads']), np.errstate(over='raise', invalid='raise', divide='raise'):
        baseline = predict(model, query, config)
        traced = anchored_stages(model, query, anchors, config)
        stages = {k: original.difference(traced[k], legacy_stages[k], ids[anchors]) for k in original.STAGES}
        first = next((k for k in original.STAGES if not stages[k]['exact']), None)
        controls = {}
        def record(name, pairs, expected, target, sizes=None, restore=None, **kwargs):
            output = predict(model, pairs, config, sizes, **kwargs)
            if restore is not None:
                output = output[restore]
            controls[name] = dict(self_difference=original.difference(output, expected),
                                  legacy_difference=original.difference(output, target),
                                  prediction_sha256=original.array_hash(output), queries=len(pairs))
        for size in (63, 257, 2047, 2049, len(query)):
            record('caller-'+str(size), query, baseline, reference, sizes=(size,))
        record('singletons-anchors', query[anchors], baseline[anchors], reference[anchors], sizes=(1,))
        record('irregular-anchors', query[anchors], baseline[anchors], reference[anchors], sizes=(1, 3, 17, 2, 65))
        reverse = np.arange(len(query)-1, -1, -1)
        permutation = np.random.default_rng(510523).permutation(len(query))
        record('reverse', query[reverse], baseline, reference, restore=reverse)
        record('permutation', query[permutation], baseline, reference, restore=np.argsort(permutation))
        record('orientation', query[:, ::-1], baseline, reference)
        record('fortran-query', np.asfortranarray(query), baseline, reference)
        record('strided-query', np.repeat(query, 2, axis=0)[::2], baseline, reference)
        record('trailing-slots-anchors', query[anchors], baseline[anchors], reference[anchors], trailing=True)
        record('repeat-fillers-anchors', query[anchors], baseline[anchors], reference[anchors], repeat_filler=True)
        occurrences = np.repeat([0, len(query)//2, len(query)-1], 65)
        record('repeated-occurrences', query[occurrences], baseline[occurrences], reference[occurrences])
        pools = threadpool_info()
    gates()
    result = dict(schema='opal-padded-prediction-v1', config=config, executor='root; no independent review',
                  history_reproduction=histories, reference_threads=1,
                  validation_queries=len(query), query_sha256=original.array_hash(query),
                  reference_sha256=original.array_hash(reference), prediction_sha256=original.array_hash(baseline),
                  compatibility=original.difference(baseline, reference, ids),
                  trace_anchor_positions=anchors.tolist(), first_divergent_stage_on_anchors=first,
                  traced_stage_differences=stages, controls=controls,
                  all_self_controls_atol=all(r['self_difference']['within_atol'] for r in controls.values()),
                  all_self_controls_exact=all(r['self_difference']['exact'] for r in controls.values()),
                  all_legacy_controls_atol=all(r['legacy_difference']['within_atol'] for r in controls.values()),
                  absolute_tolerance=1e-12, relative_tolerance=0, fits=0, scores=0,
                  outcome_arrays_loaded=False, allowed_bank_fields=['genes', 'pairs', 'singles'],
                  environment=dict(python=sys.version, numpy=np.__version__, threadpools=pools),
                  resources=dict(wall_seconds=time.perf_counter()-begun, cpu_seconds=time.process_time()-cpu,
                                 lifetime_peak_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', None)),
                  timing_caveat='includes legacy-stage diagnostic overhead, not an optimized predictor benchmark')
    write(FOLDER / f'cell-{index:02}.json', result)
    print(json.dumps(dict(index=index, config=config, compatibility=result['compatibility']['max_abs'],
                          max_self=max(c['self_difference']['max_abs'] for c in controls.values()),
                          all_self_atol=result['all_self_controls_atol'], all_legacy_atol=result['all_legacy_controls_atol'])))


def controller():
    freeze = gates()
    FOLDER.mkdir(exist_ok=False)
    begun = time.perf_counter()
    commands = []
    for index in range(len(freeze['configs'])):
        if time.perf_counter()-begun > 1800:
            raise RuntimeError('Padding controller wall ceiling')
        command = [sys.executable, '-B', str(ROOT/'tools/wave5_run_check.py'), '--receipt', str(FOLDER/f'command-{index:02}.json'),
                   '--timeout', '300', '--peak-bytes', str(2*1024**3), '--bind', str(Path(__file__).resolve()),
                   '--', sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(index)]
        completed = subprocess.run(command, cwd=ROOT)
        commands.append(dict(index=index, returncode=completed.returncode))
        if completed.returncode:
            break
    gates()
    write(FOLDER/'controller.json', dict(commands=commands, wall_seconds=time.perf_counter()-begun,
                                       registered_jobs=len(freeze['configs']), fits=0, scores=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    controller() if args.worker is None else worker(args.worker)
