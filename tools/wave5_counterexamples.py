"""Root execution of C's frozen constructions after worker provider exhaustion."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    import psutil
    output = ROOT/'results/research-wave5/C-designed-root-execution.json'
    freeze = ROOT/'results/research-wave5/freeze.json'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    names = [Path(__file__), ROOT/'docs/research_reset/wave5/C_initial.md', freeze,
             ROOT/'docs/research_reset/wave5/C_slot.json']
    hashes = {str(p): sha(p) for p in names}
    if output.exists():
        raise FileExistsError('Preserve C experiment')
    auth = json.loads((ROOT/'docs/research_reset/wave5/C_slot.json').read_text())
    if not auth['authorized'] or auth['freeze_sha256'] != sha(freeze):
        raise ValueError('C slot mismatch')
    started = time.perf_counter()
    cpu = time.process_time()
    terms = np.array([2.**54, 1., -2.**54])
    delta, scale = 2.**-40, 2.**-20
    epsilon = 2.**-26
    x = np.array([[1, 1], [1, 1+epsilon], [1, 1-epsilon], [1, 1]])
    exact = [[sum(Fraction(float(row[i]))*Fraction(float(row[j])) for row in x)
              for j in range(2)] for i in range(2)]
    controls = {}
    for dtype in (np.float64, np.float32):
        for construction in ('C1', 'C2'):
            rows = np.tile(terms.astype(dtype), (512, 1)) if construction == 'C1' else np.full((512, 1), delta, dtype=dtype)
            def evaluate(array):
                if construction == 'C2':
                    return array[:, 0]/dtype(scale)
                return np.add.accumulate(array, axis=1, dtype=dtype)[:, -1]
            base = evaluate(rows)
            probes = {}
            for size in (1, 3, 7, 64, 512):
                probes[f'partition-{size}'] = np.concatenate([evaluate(rows[s:s+size]) for s in range(0, len(rows), size)])
            perm = np.random.default_rng(41).permutation(512)
            probes.update(reverse=evaluate(rows[::-1])[::-1], permutation=evaluate(rows[perm])[np.argsort(perm)],
                          F=evaluate(np.asfortranarray(rows)), strided=evaluate(np.repeat(rows, 2, axis=1)[:, ::2]))
            controls[f'{np.dtype(dtype)}-{construction}'] = {name:dict(exact=bool(np.array_equal(value, base)),
                max_abs=float(abs(value-base).max())) for name, value in probes.items()}
    tol = Fraction(1, 10**12)
    right = Fraction(3, 10**12)
    info = psutil.Process().memory_info()
    result = dict(schema='opal-wave5-C-designed-root-execution-v1', executor='root',
        independent_C_execution=False, independent_C_final_review=False,
        reason='A/B provider usage limit; C supplied independent proposal, critique and static tooling review before interruption',
        C1=dict(exact_sum=str(sum(map(Fraction, terms))), ascending=float(np.add.accumulate(terms)[-1]),
                cancellation_first=float((terms[0]+terms[2])+terms[1]), compensated=math.fsum(terms),
                longdouble_sum=str(np.sum(terms.astype(np.longdouble))),
                longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant)),
        C2=dict(raw_delta=delta, scale=scale, prediction_delta=delta/scale, amplification_factor=1/scale),
        C3=dict(computed_gram=(x.T@x).tolist(), exact_gram=[[str(v) for v in row] for row in exact],
                erased_increment=str(exact[1][1]-Fraction(float((x.T@x)[1, 1])))),
        C4=controls, C5=dict(exact_first_interval=[str(-tol), str(tol)],
                exact_second_interval=[str(right-tol), str(right+tol)], disjoint=tol < right-tol,
                implication='No common value matches both; this does not preclude one invariant value matching one target'),
        hashes=hashes, resources=dict(wall_seconds=time.perf_counter()-started,
            cpu_seconds=time.process_time()-cpu, lifetime_peak_bytes=max(info.rss, getattr(info, 'peak_wset', 0))),
        fit=False, score=False, biological_inputs=False)
    if any(sha(Path(name)) != h for name, h in hashes.items()):
        raise ValueError('Code drift')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
