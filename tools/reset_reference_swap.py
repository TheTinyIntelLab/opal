"""No-fit, unit-weight replacement of exposed-cohort main-effect references."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import environment, load_npz, read_json, sha256, write_json
from tools.reset_diagnostics import COHORTS, conditional_interval, correlation, metrics, panel, summary, verify

REGISTERED = ['tools/reset_reference_swap.py', 'tests/test_reset_reference_swap.py',
              'docs/research_reset/reference_swap_protocol.md', 'tools/reset_diagnostics.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def signature():
    return {name: sha256(ROOT/name) for name in REGISTERED}


def compare(names, reference, correction, target):
    reference, correction, target = [np.asarray(x) for x in (reference, correction, target)]
    if correction.shape != reference.shape or target.shape != reference.shape:
        raise ValueError('Aligned correction, reference and target required')
    base, candidate = metrics(reference, target), metrics(reference+correction, target)
    delta = (reference+correction-target)**2-(reference-target)**2
    residual = target-reference
    return dict(reference=base, reference_plus_old_correction=candidate,
        relative_mse_gain=1-candidate['mse']/base['mse'] if base['mse'] else None,
        paired_difference=conditional_interval(names, delta, draws=2000),
        old_correction_local_residual_centered_pearson=correlation(correction, residual),
        old_correction=summary(correction), local_residual=summary(residual),
        correction_rms=float(np.sqrt(np.mean(correction**2))),
        local_residual_rms=float(np.sqrt(np.mean(residual**2))))


def analyze(panels):
    cohorts = {}
    for name in COHORTS:
        p = panels[name]
        old_correction = p['original_completion']-p['original_main_effect']
        target = p['target'].mean(1)
        results = dict(pairs=len(p['names']), genes=len(np.unique(p['names'])),
            strict_novel_total=p['strict_novel_total'], missing_single_pairs=p['missing_single_pairs'],
            old_correction_definition='historical completion prediction minus historical main-effect prediction',
            primary={}, cross_infection={})
        for label in ['minimum', 'additive']:
            results['primary'][label] = compare(p['names'], p[label].mean(1), old_correction, target)
            results['cross_infection'][label] = {
                'infection1_singles_to_infection2': compare(p['names'], p[label][:, 0], old_correction, p['target'][:, 1]),
                'infection2_singles_to_infection1': compare(p['names'], p[label][:, 1], old_correction, p['target'][:, 0])}
        cohorts[name] = results
    return dict(label='exploratory extra-information successor diagnostic; no fitting, external validation or promotion',
        correction_weight=1, cohorts=cohorts,
        uncertainty='paired gene-resampling sensitivity conditional on observed infections and genes; no formal population coverage guarantee')


def freeze(output, diagnostic_protocol):
    if output.exists():
        raise FileExistsError('Preserve previous registrations')
    historical = read_json(ROOT/'results/fitness-external-20261020/protocol.json')
    diagnostic = read_json(diagnostic_protocol)
    files = dict(historical['files'])
    files[str(diagnostic_protocol.resolve())] = sha256(diagnostic_protocol)
    for name, digest in signature().items():
        files[str((ROOT/name).resolve())] = digest
    verify(files)
    output.mkdir(parents=True)
    write_json(output/'protocol.json', dict(created_utc=now(), code=signature(), files=files,
        prepared=diagnostic['prepared'], frozen=diagnostic['frozen'], environment=environment(),
        registration='after all historical outcomes and earlier reset diagnostics; before reference-swap scores',
        threads=1, wall_seconds=600, peak_rss_bytes=2*1024**3, artifact_bytes=16*1024**2,
        reference_weight=1, correction_weight=1, fitting=False))


def prerequisite(output):
    p = read_json(output/'protocol.json')
    verify(p['files'])
    if (output/'prefit_tests.json').exists() or (output/'run_started.json').exists():
        raise FileExistsError('Preserve prerequisite ordering receipts')
    started = now()
    command = [sys.executable, '-m', 'pytest', 'tests/test_reset_reference_swap.py', '-q']
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    write_json(output/'prefit_tests.json', dict(started_utc=started, completed_utc=now(),
        command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        protocol_sha256=sha256(output/'protocol.json'), code=signature()))
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)


def execute(output, replay=False):
    p, tests = read_json(output/'protocol.json'), read_json(output/'prefit_tests.json')
    verify(p['files'])
    if p['code'] != signature() or tests['code'] != signature() or tests['returncode'] != 0:
        raise ValueError('Frozen code requires passing prerequisite tests')
    if tests['protocol_sha256'] != sha256(output/'protocol.json') or tests['started_utc'] < p['created_utc']:
        raise ValueError('Prerequisites must follow registration')
    if not replay and (output/'run_started.json').exists():
        raise FileExistsError('Preserve completed and partial diagnostic attempts')
    if not replay:
        write_json(output/'run_started.json', dict(started_utc=now(),
            protocol_sha256=sha256(output/'protocol.json'), prefit_tests_sha256=sha256(output/'prefit_tests.json')))
    start = time.perf_counter()
    panels = {name: panel(load_npz(Path(p['prepared'])/f'{name}.npz'),
        load_npz(Path(p['frozen'])/f'{name}_predictions.npz')) for name in COHORTS}
    result = analyze(panels)
    elapsed = time.perf_counter()-start
    mem = psutil.Process().memory_info()
    peak = getattr(mem, 'peak_wset', mem.rss)
    if elapsed > p['wall_seconds'] or peak > p['peak_rss_bytes']:
        raise RuntimeError('Post-run resource check exceeds registered budget')
    if replay:
        if result != read_json(output/'results.json'):
            raise RuntimeError('Exact aggregate metric replay differs')
        write_json(output/'replay.json', dict(completed_utc=now(), status='exact scientific JSON replay',
            results_sha256=sha256(output/'results.json'), protocol_sha256=sha256(output/'protocol.json')))
    else:
        write_json(output/'results.json', result)
        write_json(output/'resources.json', dict(completed_utc=now(), wall_seconds=elapsed,
            peak_rss_bytes=peak, protocol_sha256=sha256(output/'protocol.json'),
            results_sha256=sha256(output/'results.json')))
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > p['artifact_bytes']:
        raise RuntimeError('Aggregate artifact budget exceeded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'verify', 'run', 'replay'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-protocol', type=Path, default=ROOT/'results/research-reset-diagnostics/protocol.json')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == 'freeze':
            freeze(args.output, args.diagnostic_protocol)
        elif args.action == 'verify':
            prerequisite(args.output)
        else:
            execute(args.output, replay=args.action == 'replay')
