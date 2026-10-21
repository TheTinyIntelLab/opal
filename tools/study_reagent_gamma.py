"""Frozen no-fit exact-guide/arm gamma diagnosis on exposed Horlbeck sources."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import sha256, read_json, write_json, environment, load_npz
from opal.processed_gamma import library, rows, collect_singles, aggregate, CONTEXTS
from opal.scalar_screen import block_assignments, metrics, paired_gene_uncertainty

CODE = ('src/processed_gamma.py', 'src/scalar_screen.py', 'src/artifacts.py',
        'tools/study_reagent_gamma.py', 'tests/test_processed_gamma.py',
        'tests/test_reagent_gamma_study.py', 'docs/research_reset/reagent_gamma_protocol.md')
HASHES = ('ba9772f3b0e929bcf5b23bebf17d4caab86c7f2ccd50dbd62fbc47e40c2aac21',
          '84a785108a5bf830ce2ee569e14757fe5c31738ba6d88c95206f38a0b1bbc150',
          '7a0a6a4c7f79e79983071deb59b18d52493f4333133dfa89c100d6720b12cb94')
CONFIG = dict(minimum_controls=2, minimum_constructs=2, tolerance=1e-12,
              bootstrap_draws=2000, seed=20261022, wall_seconds=1800,
              rss_bytes=1024**3, artifact_bytes=50*1024**2, reserve_bytes=1024**3)


class Guard:
    def __init__(self, output, consumed=None):
        self.output, self.consumed = output, consumed or {}
        self.wall, self.cpu, self.peak = time.monotonic(), time.process_time(), 0

    def check(self):
        info = psutil.Process().memory_info()
        self.peak = max(self.peak, getattr(info, 'peak_wset', info.rss))
        if self.receipt()['wall_seconds'] > CONFIG['wall_seconds'] or self.receipt()['cpu_seconds'] > CONFIG['wall_seconds']:
            raise RuntimeError('Registered time cap reached')
        if self.peak > CONFIG['rss_bytes']:
            raise RuntimeError('Registered memory cap reached')
        if shutil.disk_usage(self.output).free < CONFIG['reserve_bytes']:
            raise RuntimeError('Registered disk reserve reached')
        if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()) > CONFIG['artifact_bytes']:
            raise RuntimeError('Registered output cap reached')

    def receipt(self):
        return dict(wall_seconds=time.monotonic()-self.wall+self.consumed.get('wall_seconds', 0),
            cpu_seconds=time.process_time()-self.cpu+self.consumed.get('cpu_seconds', 0),
            peak_rss_bytes=max(self.peak, self.consumed.get('peak_rss_bytes', 0)))


def signature():
    return {name: sha256(ROOT/name) for name in CODE}


def verify(output):
    registration = read_json(output/'registration.json')
    if registration['code'] != signature() or registration['config'] != CONFIG:
        raise ValueError('Frozen recipe/code differs')
    if registration['dependencies'] != environment():
        raise ValueError('Frozen environment differs')
    for path, digest in registration['inputs'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen source differs')
    return registration


def compute(registration, check):
    source, guide_path, bank_path = map(Path, registration['paths'])
    guides = library(guide_path)
    single, single_counts, single_tally = collect_singles(rows(source, guides, check), len(guides), CONFIG['minimum_controls'])
    bank = load_npz(bank_path)
    arrays, double_tally = aggregate(rows(source, guides, check), bank['genes'], bank['pairs'], single, CONFIG['minimum_constructs'])
    arrays.update(genes=bank['genes'], pairs=bank['pairs'], guide_single=single,
                  guide_single_counts=single_counts,
                  guide_names=np.asarray(list(guides)), guide_sequences=np.asarray([g['sequence'] for g in guides.values()]))
    results = dict(exposure='Previously exposed original sources, exploratory no-fit diagnosis',
        single_tally=single_tally, double_tally=double_tally,
        annotations=dict(transcript='unavailable', intentional_mismatch='unavailable'), contexts={})
    commutation = True
    guide_equivalence = True
    for c, context in enumerate(CONTEXTS):
        check()
        data, counts = arrays['means'][c], arrays['counts'][c]
        qualified = (counts >= CONFIG['minimum_constructs']).all(0)
        screen = dict(genes=bank['genes'], pairs=bank['pairs'], accession='GSE116198')
        splits = block_assignments(screen)
        test = np.logical_or.reduce([split == 2 for split in splits])
        difference = np.abs(data[..., 4]-data[..., 1]-data[..., 2])
        finite = difference[np.isfinite(difference)]
        sum_error = float(finite.max()) if len(finite) else None
        commutation &= sum_error is not None and sum_error <= CONFIG['tolerance']
        target_difference = np.abs(data[..., 0]-bank['replicates'][c])
        finite_target = target_difference[np.isfinite(target_difference)]
        full_target_difference = float(finite_target.max()) if len(finite_target) else None
        complete = bool(qualified.all() and np.array_equal(counts, arrays['raw_counts'][c]) and
                        full_target_difference is not None and full_target_difference <= CONFIG['tolerance'])
        guide_equivalence &= complete
        record = dict(original_pairs=len(qualified), qualified_pairs=int(qualified.sum()),
            excluded_pairs=int((~qualified).sum()),
            original_block_test_pairs=int(test.sum()), qualified_block_test_pairs=int((qualified & test).sum()),
            unsupported_construct_measurements=int((arrays['raw_counts'][c]-counts).sum()),
            full_panel_target_max_abs_difference=full_target_difference,
            guide_branch_identical_population_and_target=complete,
            sum_commutation_max_abs_error=sum_error,
            control_support_histogram={str(k): int(v) for k, v in zip(*np.unique(single_counts[c], return_counts=True))},
            qualified_construct_count_histogram={str(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))},
            target_difference_comparable_cells=int(len(finite_target)),
            infection_qualified_pairs=[int((counts[r] >= CONFIG['minimum_constructs']).sum()) for r in range(2)],
            panels={})
        for panel_name, select in (('all_qualified', qualified), ('qualified_block_test', qualified & test)):
            if not select.any():
                record['panels'][panel_name] = dict(available=False)
                continue
            target = data[:, select, 0].mean(0)
            endpoint_a, endpoint_b = data[:, select, 1], data[:, select, 2]
            historical = bank['singles'][c][:, bank['pairs'][select]].transpose(1, 0, 2)
            # Explicit indexing gives [pair,infection,endpoint], avoiding mixed-axis ambiguity.
            old_sum = historical.sum(-1).mean(1)
            old_min = historical.min(-1).mean(1)
            predictions = dict(zero=np.zeros(len(target)), construct_sum=data[:, select, 4].mean(0),
                mean_first_sum=(endpoint_a+endpoint_b).mean(0),
                construct_min=data[:, select, 3].mean(0),
                mean_first_min=np.minimum(endpoint_a, endpoint_b).mean(0),
                historical_sum=old_sum, historical_min=old_min)
            infection_predictions = [dict(zero=np.zeros(len(target)), construct_sum=data[r, select, 4],
                mean_first_sum=endpoint_a[r]+endpoint_b[r], construct_min=data[r, select, 3],
                mean_first_min=np.minimum(endpoint_a[r], endpoint_b[r]),
                historical_sum=historical[:, r].sum(-1), historical_min=historical[:, r].min(-1)) for r in range(2)]
            orientation_values = arrays['orientation_means'][c][:, select]
            orientation_difference = orientation_values[..., 0]-orientation_values[..., 1]
            orientation_finite = orientation_difference[np.isfinite(orientation_difference)]
            report = dict(available=True, metrics={name: metrics(target, pred) for name, pred in predictions.items()},
                infection_metrics=[{name: metrics(data[r, select, 0], pred) for name, pred in infection_predictions[r].items()} for r in range(2)],
                orientation_comparable_infection_pair_cells=int(len(orientation_finite)),
                orientation_target_gap_mse=float(np.mean(orientation_finite**2)) if len(orientation_finite) else None,
                orientation_target_signed_gap=float(orientation_finite.mean()) if len(orientation_finite) else None,
                matched_target_vs_historical_mse=float(np.mean((target-bank['responses'][c, select])**2)),
                min_aggregation_prediction_mse=float(np.mean((predictions['construct_min']-predictions['mean_first_min'])**2)),
                both_orientations_pairs=int((arrays['orientation_counts'][c, :, select] > 0).all(axis=(1, 2)).sum()))
            if panel_name == 'qualified_block_test':
                delta = (predictions['construct_min']-target)**2-(old_min-target)**2
                report['matched_min_minus_historical_min'] = paired_gene_uncertainty(delta,
                    bank['pairs'][select], len(bank['genes']), CONFIG['bootstrap_draws'], CONFIG['seed'])
            record['panels'][panel_name] = report
        results['contexts'][context] = record
    results['integrity_gate'] = bool(commutation)
    results['guide_aware_historical_comparison_gate'] = bool(guide_equivalence and commutation)
    results['q2b_original_scalar_prerequisites'] = bool(commutation)
    results['interpretation'] = 'Integrity/replay qualifies original scalar Q2b; missing guide support or target change stops only guide-aware historical comparison/refit. Shared controls/guides are not biological replication.'
    return arrays, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'verify', 'run', 'replay'))
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output
    if args.command == 'freeze':
        if out.exists():
            raise FileExistsError('Preserve registrations')
        paths = [args.source_root/'GSE116198_sgRNA_pair_phenotypes.txt.gz',
                 args.source_root/'GSE116198_GImap_sgRNA_sequences_barcodes.txt.gz',
                 args.source_root/'prepared/profiles.npz']
        inputs = {str(path.resolve()): sha256(path) for path in paths}
        if tuple(inputs.values()) != HASHES:
            raise ValueError('Historical source hashes differ')
        out.mkdir(parents=True)
        write_json(out/'registration.json', dict(version='q1b-gamma-v1', paths=list(inputs), inputs=inputs,
            config=CONFIG, code=signature(), dependencies=environment(), frozen_at=datetime.now(timezone.utc).isoformat()))
        return
    registration = verify(out)
    if args.command == 'verify':
        if (out/'prerequisites.json').exists():
            raise FileExistsError('Preserve prerequisite receipts')
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_processed_gamma.py', 'tests/test_reagent_gamma_study.py',
            '--junitxml', str(out/'prerequisites.xml')], cwd=ROOT, capture_output=True, text=True, timeout=180)
        (out/'prerequisites.log').write_text(result.stdout+result.stderr, encoding='utf-8')
        write_json(out/'prerequisites.json', dict(passed=result.returncode == 0,
            registration_sha256=sha256(out/'registration.json'), junit_sha256=sha256(out/'prerequisites.xml'), log_sha256=sha256(out/'prerequisites.log')))
        if result.returncode:
            raise RuntimeError('Mandatory prerequisites failed')
        return
    prerequisites = read_json(out/'prerequisites.json')
    if (not prerequisites['passed'] or prerequisites['registration_sha256'] != sha256(out/'registration.json')
        or prerequisites['junit_sha256'] != sha256(out/'prerequisites.xml')
        or prerequisites['log_sha256'] != sha256(out/'prerequisites.log')):
        raise ValueError('Passing frozen prerequisites required')
    replay = args.command == 'replay'
    if any((out/name).exists() for name in (('replay_started.json', 'replay_failure.json', 'replay.json') if replay else
                                           ('run_started.json', 'failure.json', 'results.json', 'prepared.npz'))):
        raise FileExistsError('Preserve scientific attempts')
    consumed = read_json(out/'resources.json') if replay else None
    guard = Guard(out, consumed)
    write_json(out/('replay_started.json' if replay else 'run_started.json'), dict(started=datetime.now(timezone.utc).isoformat()))
    try:
        with threadpool_limits(1):
            arrays, results = compute(registration, guard.check)
        if not results['integrity_gate']:
            raise RuntimeError('Sum-commutation scientific integrity gate failed')
        if replay:
            saved = load_npz(out/'prepared.npz')
            for key in arrays:
                if not np.array_equal(arrays[key], saved[key], equal_nan=arrays[key].dtype.kind == 'f'):
                    raise ValueError('Numerical replay differs: '+key)
            if results != read_json(out/'results.json'):
                raise ValueError('Result replay differs')
            guard.check()
            write_json(out/'replay.json', dict(passed=True, results_sha256=sha256(out/'results.json'),
                arrays_sha256=sha256(out/'prepared.npz'), resources=guard.receipt()))
        else:
            np.savez_compressed(out/'prepared.npz', **arrays)
            write_json(out/'results.json', results)
            guard.check()
            write_json(out/'resources.json', guard.receipt())
    except Exception as error:
        write_json(out/('replay_failure.json' if replay else 'failure.json'), dict(error=str(error), resources=guard.receipt()))
        raise


if __name__ == '__main__':
    main()
