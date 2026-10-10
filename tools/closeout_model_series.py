"""Record completed local model-series evidence; never stages or publishes files."""
import ast
import gzip
import json
from pathlib import Path
import tarfile
import zipfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path('E:/opal-model-series-20261010')
OUT = ROOT / 'results/opal_model_series_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    log = LOCAL / 'verification.log'
    content = log.read_bytes()
    encoding = 'utf-16' if content.startswith(b'\xff\xfe') else 'utf-8'
    refit = ast.literal_eval(content.decode(encoding).splitlines()[-1])
    assert refit['exact_selection_history_replayed'] and refit['prediction_max_abs'] == 0
    refit['retained_run_log_sha256'] = digest(log)
    write(OUT / 'combi-seq/refit_verification.json', refit)
    sources = [('combi-seq', LOCAL / 'fits', 'GSE174695'),
               ('pertspectra/TNFA-positive', LOCAL / 'pertspectra-banks/TNFA-positive/evaluation', 'PertSpectra/TNFA-positive'),
               ('pertspectra/TNFA-negative', LOCAL / 'pertspectra-banks/TNFA-negative/evaluation', 'PertSpectra/TNFA-negative')]
    registry = []
    for relative, local, source in sources:
        records = []
        for path in sorted(local.glob('*/receipt.json')):
            receipt = read(path)
            records.append({'fit_id': path.parent.name, 'receipt_sha256': digest(path),
                            'selection_records': receipt['selection_records'], 'unavailable': receipt['unavailable']})
        with gzip.open(OUT / relative / 'all_selection_records.json.gz', 'wt', encoding='utf-8') as handle:
            json.dump(records, handle, sort_keys=True)
        manifest = OUT / relative / ('all_budget_checkpoint_manifest.json' if source == 'GSE174695' else 'checkpoint_manifest.json')
        registry.append({'source_version': 'opal-model-series-v1/' + source + '/20261010',
                         'manifest': str(manifest.relative_to(ROOT)), 'manifest_sha256': digest(manifest),
                         'selection_receipts': len(records), 'default': read(OUT / relative / 'summary.json')['default']})
    write(OUT / 'series_registry.json', {'sources': registry, 'universal_cross_assay_weights': False,
                                      'checkpoint_role': 'frozen outer-evaluation fits', 'all_development_refit': False})
    installed = []
    for name in ['installed-final314', 'installed-final311', 'installed-final-synthetic',
                 'installed-final-pertspectra-positive', 'installed-final-pertspectra-negative']:
        path = LOCAL / name / 'verification.json'
        record = read(path)
        assert record['status'] == 'verified' and record['outside_checkout']
        installed.append(record)
    write(OUT / 'installed_verification.json', {'records': installed, 'final_wheel': str(LOCAL / 'wheel-final'),
                                             'all_source_checks_use_final_wheel': True})
    packages = []
    for path in sorted((LOCAL / 'wheel-final').iterdir()):
        if path.suffix == '.whl':
            with zipfile.ZipFile(path) as archive:
                members = archive.namelist()
            assert 'opal/series.py' in members
        else:
            with tarfile.open(path) as archive:
                members = archive.getnames()
        assert not any(Path(name).name.casefold() == 'plan.md' for name in members)
        assert not any('/data/raw/' in name or name.endswith('.npz') or name.endswith('.h5ad') for name in members)
        packages.append({'path': str(path), 'sha256': digest(path), 'members': len(members)})
    write(OUT / 'local_package_verification.json', {'packages': packages, 'private_plan_and_expression_and_weights_excluded': True,
                                                  'publication': 'not published; existing v0.1.2 assets unchanged'})
    write(OUT / 'test_verification.json', {
        'full_suite': {'python': '3.14', 'passed': 255, 'log_sha256': digest(LOCAL / 'tests-final255.log')},
        'focused_series_suite': {'python': '3.11', 'passed': 21, 'log_sha256': digest(LOCAL / 'tests-focused-final311.log')},
        'python311_full_suite_attempts': 'Retained collection failures in tests-final311*.log: research-only pandas/psutil, then pyarrow, then anndata missing. Full 3.11 suite not claimed.',
        'new_remote_ci': 'Configured Windows/Linux Python 3.11/3.14 matrix; not run because owner prohibits pushing.',
        'prior_ci': 'results/opal_model_series_v1/prior_ci.json',
        'numerical_surface_audit': {'checked_literals': 767, 'grounded': 767, 'unresolved': 0, 'semantic_certification': False},
        'manuscript': 'results/opal_model_series_v1/manuscript_build_verification.json',
        'standalone_source': {'path': str(LOCAL / 'opal-local-model-series-tex.zip'),
                              'sha256': digest(LOCAL / 'opal-local-model-series-tex.zip'), 'extracted_source_compiled': True}})
    write(OUT / 'verification_amendments.json', {'repairs': [
        'Strict unit, gene and canonical acquired/held declarations; retained full refit replay is exact.',
        'CLI module entry point repaired after retained installed verification failure.',
        'Frozen predictions and evaluator require declared held membership and exclude in-sample pairs.',
        'CI explicitly installs psutil required by the new sparse source-preparation tests.'],
        'failures_retained_locally': True, 'numerical_training_changes_after_frozen_run': False})
    status_path = ROOT / 'results/replicated_fixed_state_v1/task_status.json'
    status = read(status_path)
    status['model_series'].update({
        'status': 'complete locally; new remote CI deferred by owner no-push instruction',
        'actual_model_implementations_complete': True, 'actual_model_checkpoints_complete': True,
        'config_path': 'configs/experiments/opal_model_series_v1.json',
        'evidence_path': 'results/opal_model_series_v1/series_registry.json',
        'results_path': 'docs/next_cycle/opal_model_series.md',
        'training_sources': ['GSE174695 Combi-seq', 'PertSpectra TNFA-positive', 'PertSpectra TNFA-negative'],
        'budget_checkpoints': 1296, 'sample_label_transport_checkpoints': 72,
        'local_full_suite': {'passed': 255, 'python': '3.14', 'log': str(LOCAL / 'tests-final255.log')},
        'installed_verification': 'results/opal_model_series_v1/installed_verification.json',
        'backlog_audit': 'results/opal_model_series_v1/original_backlog_audit.json',
        'new_remote_ci': 'Not run: owner explicitly prohibits commit and push.',
        'original_biological_gates_passed': False,
        'externally_deferred': [
            'Qualified independent cultures, matched-dose/reference provenance and empirical biological precision/power.',
            'Actual external registration, independent custody, collaboration/contact and laboratory execution.',
            'Source-term authorization for weight publication; publication, visibility changes, Zenodo/arXiv remain unauthorized.',
            'Unseen singles, state/dose extrapolation, calibrated uncertainty and prospective acquisition validation.',
            'New remote CI needs owner-authorized commit/push; prior successful CI remains tied to its original HEAD.']})
    write(status_path, status)
    plan_path = ROOT / 'PLAN.md'
    plan = plan_path.read_text(encoding='utf-8')
    replacements = {
        'Scope documented; configuration not started': 'Complete; frozen configuration and scopes verified',
        'Metadata reviewed; expression/model work not started': 'Complete; both TNF contexts trained and scored',
        'Audit required': 'Complete software audit; named external dependencies deferred'}
    lines = []
    for line in plan.splitlines():
        if line.startswith('| MS'):
            for old, new in replacements.items():
                line = line.replace(old, new)
            line = line.replace('Not started', 'Complete locally; remote CI deferred under no-push instruction' if line.startswith('| MS7 ') else 'Complete with saved evidence')
        lines.append(line)
    plan = '\n'.join(lines) + '\n'
    plan = plan.replace('MS1 is complete with this configuration; MS2-MS8 are in progress.',
                        'MS1-MS8 computational work is complete with source-specific evidence; new remote CI is deferred under the no-push instruction.')
    plan = plan.replace('Documentation of this amendment is complete; model implementation is not yet complete.',
                        'Model implementations, actual checkpoints, budget runs, installed inference, reports and executable repairs are complete locally.')
    marker = '### Completed local execution, 10 October 2026'
    if marker not in plan:
        plan = plan.replace('### Active implementation backlog', marker + '\n\n'
            'All three variants are trained on Combi-seq and both PertSpectra TNF contexts. '
            'The 891 declared budget/control conditions retain 1,296 replayed model checkpoints, plus 72 source sample-label transport checkpoints. '
            'Full local tests: 255 passed; installed wheel train/predict/evaluate passed on Windows Python 3.11 and 3.14. '
            'Combi-seq Base MSE 3.872700 remains the dependable default; nonlinear gains are only about 0.03%. '
            'Both nonlinear variants lose to Base in PertSpectra, and no-change wins on TNF-negative. '
            'The original adverse benchmark, biological gates and approved release are preserved. '
            'All original backlog items are audited in results/opal_model_series_v1/original_backlog_audit.json. '
            'External dependencies are explicitly retained in task_status.json. No commit, push or publication was performed.\n\n'
            '### Active implementation backlog')
    plan_path.write_text(plan, encoding='utf-8')
    print('Local evidence closeout recorded')


if __name__ == '__main__':
    main()
