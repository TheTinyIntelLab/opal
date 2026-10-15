"""Audit retained crossed-single predictions and GxE receipts; never fit or download."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from opal.artifacts import read_json, write_json, sha256
from opal.context_contrast.artifacts import load_state, load_npz
from opal.contextual.contracts import ReferenceBank
from opal.crossed_single import predict, grouped_folds, menu, FAMILIES
from opal.evidence_audit import checked_losses, conditional_bounds, measurement_summary
from opal.gxe_aggregate_validation import load_recovery
from opal.gxe_reconstruction import reconstruct_metrics


def audit(root, check):
    root = Path(root).resolve()
    result = dict(schema='opal-retained-evidence-audit-v1', fitting=False, downloads=0,
                  acceptance_created=False, biological_confirmation=False)
    hashes = {}

    def verify(path, expected):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Inventory escapes repository')
        if path not in hashes:
            check()
            h = hashlib.sha256()
            with path.open('rb') as stream:
                while block := stream.read(1024*1024):
                    h.update(block)
                    check()
            hashes[path] = h.hexdigest()
        if hashes[path] != expected:
            raise ValueError('Custody mismatch: '+str(path.relative_to(root)))

    single = root/'results/opal_crossed_single_v1/development-20261014'
    gxe = root/'results/opal_gxe_measurement_v1'
    for folder in (single, gxe):
        inventory = read_json(folder/'local_asset_inventory.json')
        for record in inventory['files']:
            path = folder/record['path']
            if path.stat().st_size != record['bytes']:
                raise ValueError('Asset size mismatch')
            verify(path, record['sha256'])
    config_path = root/'configs/experiments/opal_crossed_single_v1.json'
    config = read_json(config_path)
    registration = read_json(single/'registration.json')
    verify(config_path, registration['config_sha256'])
    for path, digest in registration['code_sha256'].items():
        verify(root/path, digest)
    verify(root/'data/raw/state/SrivatsanTrapnell2020_sciplex3.h5ad', config['source_sha256'])
    verify(root/'data/raw/identity-reference-20261007/GSM4150378_sciPlex3_pData.txt.gz', config['primary_metadata_sha256'])
    seal = read_json(single/'prediction_seal.json')
    verify(single/'attempts.json', seal['attempts_sha256'])
    attempts = read_json(single/'attempts.json')
    if len(attempts) != config['logical_invocations'] or seal['logical_invocations'] != len(attempts):
        raise ValueError('Invocation population mismatch')
    bank = ReferenceBank.load(single/'prepared')
    labels = np.array([bank.metadata['context_records'][str(c)]['biological_state'] for c in bank.arrays['contexts']])
    effects = []
    for context in config['contexts']:
        rows = np.flatnonzero(labels == context)
        effects.append(np.stack([bank.arrays['singles'][rows[bank.arrays['available'][rows, i]], i].mean(0) for i in range(96)]))
    effects = np.asarray(effects)
    assignment = grouped_folds(96, config['split_seed'])
    errors = {c: {f: np.empty(96) for f in (*FAMILIES, 'selector')} for c in config['contexts']}
    seen = {c: np.zeros(96, bool) for c in config['contexts']}
    fallbacks = {c: {f: 0 for f in FAMILIES} for c in config['contexts']}
    inner_replays = final_replays = 0
    if len(seal['records']) != 9:
        raise ValueError('Missing outer direction/fold')
    for record in seal['records']:
        check()
        context, held = record['target'], record['outer']
        folder = single/record['path']
        fit = load_npz(folder, record['fit'])
        query = load_npz(folder, record['query'])
        evaluator = load_npz(folder, record['evaluator'])
        predictions = load_npz(folder, record['predictions'])
        target = config['contexts'].index(context)
        x = np.transpose(effects[[j for j in range(3) if j != target]], (1, 0, 2))
        train, ids = np.flatnonzero(assignment != held), np.flatnonzero(assignment == held)
        for a, b in ((fit['identities'], train), (query['identities'], ids), (evaluator['identities'], ids),
                     (fit['x'], x[train]), (fit['y'], effects[target, train]),
                     (query['x'], x[ids]), (evaluator['y'], effects[target, ids])):
            if not np.array_equal(a, b):
                raise ValueError('Prepared-bank, fold or role mismatch')
        if set(predictions) != set(errors[context]):
            raise ValueError('Missing comparison arm')
        own = [a for a in attempts if a['target'] == context and a['outer'] == held]
        inner_folds = grouped_folds(len(train), config['split_seed'])
        specs = menu(config)
        scores = []
        for index, spec in enumerate(specs):
            losses = []
            for inner in range(3):
                match = [a for a in own if a['stage'] == f'inner-{inner}-spec-{index}']
                if len(match) != 1:
                    raise ValueError('Missing/duplicate nested attempt')
                attempt = match[0]
                tr, val = np.flatnonzero(inner_folds != inner), np.flatnonzero(inner_folds == inner)
                if attempt['train'] != tr.tolist() or attempt['validation'] != val.tolist() or attempt['spec'] != spec:
                    raise ValueError('Nested selection split mismatch')
                if attempt['state_path'] is None:
                    if not attempt['error'] or attempt['mse'] is not None:
                        raise ValueError('Unsupported fit accounting mismatch')
                    continue
                state = load_state(single/attempt['state_path'])
                value = float(np.mean((predict(state, fit['x'][val])-fit['y'][val])**2))
                if not np.isclose(value, attempt['mse'], rtol=1e-12, atol=1e-14):
                    raise ValueError('Nested selection loss mismatch')
                losses.append(value)
                inner_replays += 1
            scores.append(dict(spec=spec, supported=len(losses) == 3, mse=float(np.mean(losses)) if len(losses) == 3 else None))
        choices = {}
        for family in FAMILIES:
            valid = [s for s in scores if s['supported'] and s['spec']['family'] == family]
            choices[family] = min(valid, key=lambda s: (s['mse'], s['spec']['alpha'])) if valid else None
        refs = [v for f, v in choices.items() if v and f not in config['selector_excludes']]
        locked = min(refs, key=lambda s: (s['mse'], s['spec']['family'], s['spec']['alpha']))['spec']['family']
        if locked != record['locked_reference']:
            raise ValueError('Reference selector mismatch')
        for family, choice in choices.items():
            stored = record['choices'][family]
            if choice is None:
                if stored is not None:
                    raise ValueError('Unsupported choice mismatch')
            elif stored is None or stored['spec'] != choice['spec'] or not np.isclose(stored['mse'], choice['mse'], rtol=1e-12, atol=1e-14):
                raise ValueError('Stored selected score mismatch')
        for family in FAMILIES:
            final = [a for a in own if a['stage'] == 'final-'+family]
            if len(final) != 1:
                raise ValueError('Final attempt missing')
            if final[0]['state_path'] is None:
                if family not in record['unsupported'] or not np.array_equal(predictions[family], predictions['selector']):
                    raise ValueError('Fallback population mismatch')
                fallbacks[context][family] += len(ids)
                continue
            if final[0]['spec'] != choices[family]['spec']:
                raise ValueError('Final penalty selection mismatch')
            state = load_state(single/final[0]['state_path'])
            if not np.array_equal(predict(state, query['x']), predictions[family]):
                raise ValueError('Exact final replay failed')
            final_replays += 1
        if not np.array_equal(predictions[locked], predictions['selector']):
            raise ValueError('Locked reference prediction mismatch')
        losses = checked_losses(predictions, evaluator['y'], ids, 96, seen[context])
        for family, loss in losses.items():
            errors[context][family][ids] = loss
    if not all(v.all() for v in seen.values()):
        raise ValueError('Incomplete prediction population')
    draws = np.random.default_rng(config['sensitivity_seed']).integers(0, 96, (config['sensitivity_draws'], 96))
    saved = read_json(single/'metrics.json')['targets']
    rebuilt = {}
    for context, loss in errors.items():
        means = {f: float(v.mean()) for f, v in loss.items()}
        if fallbacks[context] != saved[context]['fallback_counts']:
            raise ValueError('Reported fallback count mismatch')
        bounds = conditional_bounds(loss['pca_concat'], loss['selector'], draws)
        gain = 1-means['pca_concat']/means['selector']
        for f, value in means.items():
            if value != saved[context]['mse'][f]:
                raise ValueError('Reported primary loss mismatch')
        if bounds.tolist() != saved[context]['conditional_paired_loss_bounds'] or gain != saved[context]['candidate_relative_gain']:
            raise ValueError('Reported sensitivity mismatch')
        passes = bool(gain >= .05 and bounds[0] > 0 and not any(saved[context]['fallback_counts'].values()) and
                      all(means['pca_concat'] < means[f] for f in ('pca_mean', 'pca_difference', 'kernel', 'shuffled_concat')))
        if passes != saved[context]['spending_filter_passed']:
            raise ValueError('Spending gate mismatch')
        rebuilt[context] = dict(mse=means, gain=gain, conditional_bounds=bounds.tolist(), spending_filter_passed=passes)
    decision = read_json(single/'decision.json')
    if decision['spending_filter_passed'] != all(v['spending_filter_passed'] for v in rebuilt.values()) or not decision['complete_population']:
        raise ValueError('Global decision mismatch')
    result['crossed_single'] = dict(inner_replays=inner_replays, final_exact_replays=final_replays,
        attempts=len(attempts), population=288, results=rebuilt, preprocessing_redecoded=False,
        biological_units_verified=False, seal_scope='logical code ordering and retained identities; no independent timestamp attestation')
    registered = read_json(gxe/'technical-a4-20261014/registration.json')
    for p, key in [('configs/experiments/opal_gxe_measurement_v1.json', 'config_sha256'),
                   ('src/gxe_measurement.py', 'helper_sha256'), ('tools/run_gxe_measurement.py', 'code_sha256')]:
        verify(root/p, registered[key])
    for record in registered['source_receipts']['files']:
        verify(root/'data/raw/gxe-measurement-20261014'/record['path'], record['sha256'])
    summary = measurement_summary(read_json(gxe/'technical-a4-20261014/metrics.json'))
    if summary != read_json(gxe/'technical-a4-20261014/summary.json')['contexts']:
        raise ValueError('Measurement summary mismatch')
    result['gxe'] = dict(summary=summary, raw_hashes_verified=5, aggregates_retained=False,
        measurement_errors_recomputed=False, status='custody and reported-row summaries verified; raw transformation replay incomplete')
    recovery = root/'results/opal_gxe_reconstruction_v2/recovery-20261015'
    if recovery.exists():
        arrays, axes = load_recovery(recovery)
        actual = reconstruct_metrics(arrays['normalized_count_sums'], arrays['partition_support'],
                                     axes['group_keys'], axes['contexts'], axes['drugs'])
        if actual != read_json(gxe/'technical-a4-20261014/metrics.json'):
            raise ValueError('Recovered GxE endpoint mismatch')
        result['gxe'].update(aggregates_retained=True, measurement_errors_recomputed=True,
            recovery_manifest_sha256=sha256(recovery/'manifest.json'), status='fixed-panel source recovery and exact all-row endpoint replay complete')
        panel = root/'results/opal_gxe_reconstruction_v2/panel-verification-20261015'
        if panel.exists():
            panel_record = read_json(panel/'decision.json')
            panel_registration = read_json(panel/'registration.json')
            verify(root/'tools/verify_gxe_panel.py', panel_registration['code_sha256'])
            verify(panel/'reference_variance.npz', panel_record['variance_sha256'])
            if panel_record['selected_genes'] != axes['panel_genes'] or panel_record['panel_selection_replayed'] is not True:
                raise ValueError('Reproduced reference-panel mismatch')
            result['gxe'].update(panel_selection_replayed=True, panel_verification_sha256=sha256(panel/'decision.json'),
                status='raw-to-fixed-panel recovery, reference-only panel reproduction and exact endpoint replay complete')
    protocol = read_json(root/'configs/experiments/opal_context_contrast_v1.json')
    verify(root/protocol['raw_path'], protocol['raw_sha256'])
    result['verified_files'] = len(hashes)
    result['input_hashes'] = {str(p.relative_to(root)): h for p, h in hashes.items()}
    result['capacity'] = dict(writable_free_bytes=shutil.disk_usage(root).free,
        pair_prediction_bytes=55148544, pair_checkpoint_unprofiled_upper_bytes=47800684800,
        uncompressed_predictions_exceed_25MB=True,
        pair_25MB_retention_feasibility_demonstrated=False,
        complete_pair_run_feasibility_demonstrated=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    start, wall, peak = time.process_time(), time.monotonic(), 0
    status = 'audit_failed'
    def check():
        nonlocal peak
        peak = max(peak, psutil.Process().memory_info().rss)
        if time.process_time()-start > 300 or peak > 1_000_000_000 or shutil.disk_usage(args.output).free < 1073741824:
            raise RuntimeError('Bounded audit CPU/RAM/reserve limit')
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            result = audit(ROOT, check)
        result['audit_tool_sha256'] = sha256(__file__)
        if len(json.dumps(result).encode('utf-8')) > 10_000_000:
            raise RuntimeError('Bounded audit output limit')
        write_json(args.output/'audit.json', result)
        status = 'retained_output_audit_completed'
        print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'crossed_single', 'gxe')}, indent=2))
    except Exception as error:
        write_json(args.output/'failure.json', dict(status='audit_failed', reason=str(error)))
        raise
    finally:
        write_json(args.output/'resource_ledger.json', dict(cpu_seconds=time.process_time()-start,
            status=status, cpu_accounting_complete=True, retained_limit_bytes=10_000_000,
            wall_seconds=time.monotonic()-wall, sampled_rss_bytes=peak, continuous_peak_measured=False,
            cpu_limit_seconds=300, ram_limit_bytes=1_000_000_000, numerical_threads=1,
            downloads=0, fitting=False, acceptance_created=False))


if __name__ == '__main__':
    main()
