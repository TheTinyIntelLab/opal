"""Record final local verification, bounded refusals and physical resource accounting."""
import json
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import zipfile

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, sha256, write_json


def main():
    local = Path('C:/Users/USER/Desktop/opal-model-series-20261010-v2')
    evidence = ROOT / 'results/opal_model_series_v2'
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    original = read_json(Path(settings['artifact_roots'][0]) / 'preservation.json')
    preservation = []
    for relative, record in original['files'].items():
        if relative.startswith(('results/opal_model_series_v1/', 'results/architecture/', 'releases/')):
            path = ROOT / relative
            if not path.exists() or sha256(path) != record['sha256']:
                raise ValueError('Preserved original evidence changed: ' + relative)
            preservation.append(relative)
    sizes = {p: sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file()) for p in settings['artifact_roots']}
    sizes[str(evidence)] = sum(f.stat().st_size for f in evidence.rglob('*') if f.is_file())
    total = sum(sizes.values())
    free = {letter: shutil.disk_usage(letter + ':/').free for letter in ['C', 'D', 'E']}
    if total > settings['new_artifact_bytes_cap'] or free['C'] < 2**30:
        raise ValueError('Final resource constraint failed')
    receipts = read_json(evidence / 'report/retained_receipt_costs.json')
    cost_groups = {'deduplicated_acquisition_receipts': receipts['cpu_seconds']}
    for name in ['retrospective', 'retrospective-ablations', 'sensitivities', 'sensitivity-references',
                 'native-completed', 'native-extra-single-controls', 'normalization-diagnostics', 'deployment', 'deployment-final']:
        records = [read_json(p) for p in (evidence / name).rglob('*.json')]
        cost_groups[name] = sum(r.get('cpu_seconds', 0) for r in records if isinstance(r, dict))
    # Full retrospective fits reused by budget receipts are already counted separately.
    # Every retained duplicate with a distinct receipt remains counted as actual work.
    write_json(evidence / 'verification/final_resources.json', dict(artifact_bytes=total, physical_roots=sizes,
        artifact_cap_bytes=settings['new_artifact_bytes_cap'], free_bytes=free, reserve_bytes=2**30,
        readonly_volumes=['D', 'E'], cpu_seconds_by_group=cost_groups,
        retained_measured_cpu_core_hours=sum(cost_groups.values()) / 3600, cpu_ceiling_core_hours=24,
        missing_costs='Some failed/import probes, audits, installed subprocesses and document builds lack cumulative CPU counters; not assigned zero. Full historical ceiling certification cannot be reconstructed.',
        primary_receipt_peak_rss_bytes=max(r.get('process_peak_rss_bytes') or 0 for r in receipts['receipts']),
        native_training_ram=None, native_ram_reason='Only replay RAM was captured; it is not training RAM',
        hardware=dict(platform=platform.platform(), processor=platform.processor(), logical_cpus=psutil.cpu_count()),
        original_files_preserved=len(preservation), original_paths=preservation, no_automatic_deletions=True))
    packages = []
    for path in (local / 'wheel-final-review').iterdir():
        if path.suffix == '.whl':
            with zipfile.ZipFile(path) as archive:
                members = archive.namelist()
        else:
            with tarfile.open(path) as archive:
                members = archive.getnames()
        for name in members:
            if Path(name).name.lower() == 'plan.md' or name.endswith(('.npz', '.npy', '.h5ad')) or '/data/raw/' in name or '/data/prepared/' in name or '/results/' in name:
                raise ValueError('Private/data/weight package member: ' + name)
        packages.append(dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size, members=len(members)))
    installed = [read_json(local / f'installed-reviewed{version}/real-example/verification.json') for version in ['311', '314']]
    deployment = [read_json(local / f'installed-complete{version}/deployment/verification.json') for version in ['311', '314']]
    deployment.append(read_json(local / 'installed-complete314/deployment-full-gene/verification.json'))
    if any(r['status'] != 'verified' for r in installed + deployment):
        raise ValueError('Installed verification incomplete')
    write_json(evidence / 'verification/package-final.json', dict(version='0.1.3.dev0', packages=packages,
        installed=installed, deployment_inference=deployment,
        full_suite=dict(python='3.14', passed=268, seconds=38.85),
        focused_suite=dict(python='3.11', passed=34, seconds=9.01),
        unchanged_source_code='wheel includes final deployment API, safe shared arrays and internal dispatch',
        private_raw_weight_exclusions_passed=True, old_release_assets_unchanged=True,
        remote_ci='pending; Windows/Linux Python 3.11/3.14 configured, no push authorized'))
    write_json(evidence / 'task_status.json', dict(status='complete with bounded refusals and negative scientific criteria',
        primary_conditions=858, retrospective_budget_conditions=312, retrospective_ablation_folds=12,
        sensitivities=51, completed_sensitivity_reference_folds=51, native_adaptation_folds=72,
        extra_single_reference_folds=24, final_deployment_checkpoints=21,
        primary_promotion_passed=False, native_competitiveness_passed=False, full_efficiency_criterion_passed=False,
        artifact_cap_bytes=3000000000, reserve_bytes=2**30,
        external_dependencies=['Canonical PertSpectra graph and matched preprocessing', 'Supported public TxPert dependency closure and unavailable proprietary graphs',
            'Working upstream PRESAGE PCA/vector/pair aggregation contract', 'Source/publication authorization before weight redistribution',
            'Qualified independent cultures, matched reference/dose provenance, biological precision/power and laboratory validation'],
        nonrecoverable_measurements=['Native training RAM', 'Complete cumulative CPU for failed and verification subprocesses'],
        commits=[], remote_ci='pending; no push authorized', publication='none'))
    print('Final resources, package, preservation and negative/refusal status saved')


if __name__ == '__main__':
    main()
