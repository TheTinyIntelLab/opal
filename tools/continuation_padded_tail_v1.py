"""Finite contraction-shape falsification on frozen validation kernels only."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import continuation_padded_prediction_v1 as padding
from tools import wave5_numerical_matrix as original

OUT = ROOT / 'results/research-continuation'


def run():
    output = OUT / 'padded-tail-results-v1.json'
    cases_path = OUT / 'padded-tail-cases-v1.json.gz'
    if output.exists() or cases_path.exists():
        raise FileExistsError('Tail diagnostic output exists')
    freeze_path = OUT / 'padded-tail-freeze-v1.json'
    binding_path = OUT / 'padded-tail-code-binding-v1.json'
    freeze, binding = original.read(freeze_path), original.read(binding_path)
    if original.digest(freeze_path) != binding['freeze_sha256']:
        raise ValueError('Tail freeze drift')
    checks = {**freeze['repo_sha256'], **freeze['external_sha256'], **binding['code_sha256']}
    def check():
        padding.gates()
        for path, expected in checks.items():
            if original.digest(ROOT / path) != expected:
                raise ValueError('Tail input drift: ' + path)
    check()
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits
    from opal.flexible_singles import FlexibleSingles
    def forbidden(*args, **kwargs):
        raise RuntimeError('No fit/prepare/solve/lstsq in tail diagnostic')
    FlexibleSingles.fit = FlexibleSingles.prepare = FlexibleSingles.solve = forbidden
    np.linalg.solve = np.linalg.lstsq = forbidden
    begun, cpu = time.perf_counter(), time.process_time()
    with threadpool_limits(1):
        model, bank, ids = original.load_allowed({})
        query = bank['pairs'][ids]
        positions = np.array([0, len(query)//2, len(query)-1])
        reference = original.global_trace(model, query, positions)
        kernels, targets = reference['kernel'], reference['whitened']
        oracles = np.array([[math.fsum(np.multiply(row[:, None], model.whitener)[:, column])
                             for column in range(model.whitener.shape[1])] for row in kernels])
    whiteners = {'C': np.ascontiguousarray(model.whitener), 'F': np.asfortranarray(model.whitener),
                 'strided': np.ascontiguousarray(model.whitener)}
    cases, pools = [], []
    for threads in freeze['threads']:
        with threadpool_limits(threads):
            pools.append(dict(requested=threads, pools=threadpool_info()))
            for n in freeze['sizes']:
                for position in sorted({0, n//2, n-1}):
                    for filler in freeze['fillers']:
                        for layout in freeze['layouts']:
                            rows = []
                            for q, kernel in enumerate(kernels):
                                left = np.zeros((n, len(kernel)))
                                if filler == 'repeat':
                                    left[:] = kernel
                                left[position] = kernel
                                if layout == 'F':
                                    left = np.asfortranarray(left)
                                elif layout == 'strided':
                                    left = np.repeat(left, 2, axis=1)[:, ::2]
                                value = (left @ whiteners[layout])[position]
                                if not np.isfinite(value).all():
                                    raise ValueError('Nonfinite contraction')
                                delta = np.abs(value-targets[q])
                                relative = np.divide(delta, np.abs(targets[q]), out=np.zeros_like(delta), where=targets[q] != 0)
                                rows.append(dict(validation_position=int(positions[q]), bitwise_equal=bool(np.array_equal(value, targets[q])),
                                                 max_abs=float(delta.max()), max_relative_nonzero_reference=float(relative.max()),
                                                 nonzero_at_zero_reference=int(np.count_nonzero((targets[q] == 0) & (delta != 0))),
                                                 max_abs_to_rounded_product_fsum=float(np.max(np.abs(value-oracles[q]))),
                                                 whitening_sha256=original.array_hash(value),
                                                 first_divergent_operation='kernel @ whitener' if np.any(delta) else None))
                            cases.append(dict(threads=threads, rows=n, position=position, filler=filler, layout=layout, observations=rows))
    check()
    raw = (json.dumps(cases, separators=(',', ':'), allow_nan=False)+'\n').encode()
    with cases_path.open('xb') as stream:
        stream.write(gzip.compress(raw, mtime=0))
    result = dict(schema='opal-padded-tail-diagnostic-v1', executor='root; no independent execution',
                  freeze_sha256=original.digest(freeze_path), code_binding_sha256=original.digest(binding_path),
                  configurations=len(cases), whitening_observations=sum(len(c['observations']) for c in cases),
                  cases_gzip_sha256=original.digest(cases_path), cases_uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                  case_path=str(cases_path.relative_to(ROOT)), validation_positions=positions.tolist(),
                  reference_whitening_sha256=[original.array_hash(row) for row in targets],
                  reference_to_rounded_product_fsum_max_abs=[float(np.max(np.abs(v-o))) for v, o in zip(targets, oracles)],
                  oracle_limit='math.fsum of float64-rounded products, not exact unrounded products',
                  maximum_whitening_delta=max(r['max_abs'] for c in cases for r in c['observations']),
                  bitwise_equal_observations=sum(r['bitwise_equal'] for c in cases for r in c['observations']),
                  threadpools=pools, wall_seconds=time.perf_counter()-begun, cpu_seconds=time.process_time()-cpu,
                  outcome_arrays_loaded=False, fits=0, scores=0, production_implementation_changed=False)
    padding.write(output, result)
    print(json.dumps({k: result[k] for k in ('configurations', 'whitening_observations', 'maximum_whitening_delta', 'bitwise_equal_observations')}))


if __name__ == '__main__':
    run()
