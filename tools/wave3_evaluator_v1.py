"""Hash-bound independent synthetic evaluator audit; no source-data access."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update({k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
CODE = ['tools/wave3_evaluator_v1.py', 'tests/test_wave3_evaluator_v1.py',
        'docs/research_reset/wave3_evaluator_protocol_v1.md', 'src/block_completion.py',
        'src/scalar_screen.py', 'src/models/fitness.py', 'src/flexible_singles.py',
        'src/processed_gamma.py', 'src/splits/__init__.py', 'src/artifacts.py',
        'tools/study_block_completion.py']
SEED = 310021


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    from opal.artifacts import write_json
    write_json(path, value)


def fixture():
    import numpy as np
    from opal.scalar_screen import pair_assignments
    rng = np.random.default_rng(SEED)
    pairs = np.array([(a, b) for a in range(30) for b in range(a+1, 30)])
    singles = rng.normal(size=30)*.2
    u = rng.normal(size=(30, 2))
    interaction = (u[pairs[:, 0]]*u[pairs[:, 1]]*np.array([1., -1.])).sum(1)
    screen = dict(genes=np.array([f'G{i:03}' for i in range(30)]), pairs=pairs,
                  singles=singles, accession='wave3-synthetic-v1')
    return dict(**screen, split=pair_assignments(screen), interaction=interaction,
                additive=singles[pairs].sum(1))


def load_study():
    spec = importlib.util.spec_from_file_location('wave3_actual_block_study', ROOT/'tools/study_block_completion.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(output):
    import numpy as np
    from opal.artifacts import environment
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'inputs.npz', **fixture())
    write(output/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        code={name: digest(ROOT/name) for name in CODE},
        inputs={'inputs.npz': digest(output/'inputs.npz')}, seed=SEED,
        caps=dict(wall_seconds=1800, rss_bytes=1024**3, output_bytes=100*1024**2),
        uncertainty=dict(trials=128, draws=512, mechanisms=['iid', 'gene', 'cohort', 'global']),
        environment=environment(), biological_inputs=False))


def verify(output):
    p = json.loads((output/'protocol.json').read_text())
    if p['code'] != {name: digest(ROOT/name) for name in CODE}:
        raise ValueError('Frozen code drift')
    if p['inputs'] != {'inputs.npz': digest(output/'inputs.npz')}:
        raise ValueError('Frozen input drift')
    return p


def wilson(k, n):
    z = 1.959963984540054
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    half = z*((k/n*(1-k/n)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
    return [center-half, center+half]


def coverage(output):
    import numpy as np
    from opal.scalar_screen import paired_gene_uncertainty
    d = fixture()
    pairs, m = d['pairs'], len(d['pairs'])
    records, summaries = [], {}
    for mechanism in ('iid', 'gene', 'cohort', 'global'):
        start = time.perf_counter()
        for trial in range(128):
            rng = np.random.default_rng(SEED+trial)
            noise = rng.normal(size=m)
            if mechanism == 'iid':
                delta = noise
            elif mechanism == 'gene':
                delta = rng.normal(size=30)[pairs].sum(1)/np.sqrt(2)+.1*noise
            elif mechanism == 'cohort':
                delta = rng.normal(size=4)[np.arange(m) % 4]+.1*noise
            else:
                delta = rng.normal()+.1*noise
            target = -delta/2
            actual = (.5-target)**2-(-.5-target)**2
            np.testing.assert_allclose(actual, delta, atol=1e-14, rtol=1e-14)
            r = paired_gene_uncertainty(actual, pairs, draws=512, seed=SEED+trial)
            prng = np.random.default_rng(SEED+trial)
            pi = np.quantile(actual[prng.integers(m, size=(512, m))].mean(1), [.025, .975])
            lo, hi = r['interval']
            records.append(dict(mechanism=mechanism, trial=trial, mean=float(actual.mean()),
                gene_interval=r['interval'], pair_interval=pi.tolist(),
                gene_covers_zero=bool(lo <= 0 <= hi), pair_covers_zero=bool(pi[0] <= 0 <= pi[1]),
                empty_draws=r['discarded_empty_draws']))
        rows = [r for r in records if r['mechanism'] == mechanism]
        summaries[mechanism] = dict(trials=128, wall_seconds=time.perf_counter()-start,
            mean_estimator_sd=float(np.std([r['mean'] for r in rows], ddof=1)),
            **{kind: dict(covered=sum(r[kind+'_covers_zero'] for r in rows),
                coverage=sum(r[kind+'_covers_zero'] for r in rows)/128,
                wilson_95=wilson(sum(r[kind+'_covers_zero'] for r in rows), 128),
                mean_width=float(np.mean([r[kind+'_interval'][1]-r[kind+'_interval'][0] for r in rows])))
               for kind in ('gene', 'pair')},
            empty_draws=sum(r['empty_draws'] for r in rows))
        print('coverage', mechanism, summaries[mechanism]['gene']['coverage'], flush=True)
    write(output/'coverage_trials.json', records)
    write(output/'coverage.json', summaries)


def recovery(output):
    import numpy as np
    from opal.block_completion import fit_reference_grid, ScalarCompletion
    from opal.scalar_screen import residual_permutations, metrics
    d = fixture()
    p, split = d['pairs'], d['split']
    train, val, test = [np.flatnonzero(split == k) for k in range(3)]
    result = {}
    for name, y in [('additive_null', d['additive']), ('signed_low_rank', d['additive']+d['interaction'])]:
        refs = fit_reference_grid(d['singles'], p[train], y[train])
        losses = [metrics(y[val], r.predict(p[val]))['mse'] for r in refs]
        chosen = int(np.argmin(losses))
        base = refs[chosen]
        residual = y[train]-base.predict(p[train])
        screen = dict(d, counts=np.full((len(p), 2), 8))
        controls = residual_permutations(screen, split)
        models = []
        for label, perm in [('aligned', None)]+[(str(c['seed']), c['local_permutation']) for c in controls]:
            model = ScalarCompletion(rank=2, penalty=.0001, steps=400, seed=SEED).fit(
                p[train], residual if perm is None else residual[perm], 30,
                max(float(y[train].std()), 1e-6), (p[val], y[val]-base.predict(p[val])))
            models.append((label, model))
        # Persist selection before accessing test losses.
        write(output/(name+'_selection.json'), dict(chosen_reference=chosen, validation_mse=losses,
            model_steps={k: m.best_step for k, m in models}))
        base_pred = base.predict(p[test])
        result[name] = dict(pairs=len(test), reference_kind=base.kind, reference_alpha=base.alpha,
            reference_mse=metrics(y[test], base_pred)['mse'],
            additive_mse=metrics(y[test], d['additive'][test])['mse'],
            candidate_and_control_mse={k: metrics(y[test], base_pred+m.predict(p[test]))['mse'] for k, m in models},
            controls=[{k: c[k] for k in ('seed', 'moved_fraction', 'permutable_fraction', 'sufficient')} for c in controls])
        result[name]['ratio_to_unmodeled_interaction'] = (result[name]['candidate_and_control_mse']['aligned']/
            float(np.mean(d['interaction'][test]**2))) if name == 'signed_low_rank' else None
    result['technical_confounding'] = dict(observationally_identical_to='signed_low_rank',
        biological_interaction=0, interpretation='same fitted predictions; recovery is not causal identification')
    write(output/'recovery.json', result)


def worker(output):
    import numpy as np
    import pytest
    import torch
    from threadpoolctl import threadpool_limits
    verify(output)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        rc = pytest.main(['-q', 'tests/test_wave3_evaluator_v1.py', '--tb=short', '-p', 'no:cacheprovider'])
        write(output/'tests.json', dict(returncode=int(rc), wall_seconds=time.perf_counter()-start))
        if rc:
            raise RuntimeError('Prerequisite failure; preserve v1, register successor')
        recovery(output)
        coverage(output)
        write(output/'complete.json', dict(protocol_sha256=digest(output/'protocol.json'),
            artifact_hashes={p.name: digest(p) for p in output.glob('*.json') if p.name not in ('started.json', 'resources.json')}))


def run(output):
    import psutil
    protocol = verify(output)
    marker = output/'started.json'
    with marker.open('x') as f:
        json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat()), f)
    started, peak, cpu, stopped = time.perf_counter(), 0, 0., None
    with (output/'worker.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--output', str(output.resolve())],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        proc = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                info = proc.memory_info()
                peak = max(peak, info.rss, getattr(info, 'peak_wset', 0))
                ct = proc.cpu_times()
                cpu = max(cpu, ct.user+ct.system)
            except psutil.NoSuchProcess:
                break
            size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if peak > protocol['caps']['rss_bytes']:
                stopped = 'RSS cap'
            elif time.perf_counter()-started > protocol['caps']['wall_seconds']:
                stopped = 'wall cap'
            elif size > protocol['caps']['output_bytes']:
                stopped = 'output cap'
            if stopped:
                child.kill()
                break
            time.sleep(.05)
        rc = child.wait()
    write(output/'resources.json', dict(wall_seconds=time.perf_counter()-started,
        cpu_seconds_sampled=cpu, peak_rss_bytes=peak, sampled_period_seconds=.05,
        output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        returncode=rc, stopped=stopped, controller_peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)))
    if rc or stopped:
        write(output/'failure.json', dict(returncode=rc, reason=stopped or 'worker failure'))
        raise RuntimeError('Scientific job failed; see preserved worker.log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'run', 'worker'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'freeze': freeze, 'run': run, 'worker': worker}[args.action](args.output)
