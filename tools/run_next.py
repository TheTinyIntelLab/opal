"""Build the next research cycle without reopening historical evaluator labels."""
from pathlib import Path
import argparse
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from opal.artifacts import environment, read_json, sha256, source_hash, write_json
from opal.contextual.benchmark import run_benchmark
from opal.contextual.synthetic import prepare_fixture
from tools.next_audit import archive_evidence, audit_ceiling, data_gate, native_probe
from tools.study_kernel_attribution import study
from tools.study_acquisition import study as acquisition_study


def run(output, skip_attribution=False):
    from threadpoolctl import threadpool_limits
    output = Path(output)
    if output.exists():
        raise FileExistsError("Cycle outputs are immutable; choose a new --output directory")
    output.mkdir(parents=True)
    started = time.perf_counter()
    config_path = ROOT / "configs/experiments/kernel-attribution-v1.json"
    code_paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "main.py", ROOT / "pyproject.toml"]
    code_paths += [ROOT / "tools" / name for name in ["run_next.py", "next_audit.py", "study_kernel_attribution.py",
                    "study_acquisition.py", "improve_residual_model.py", "report_ceiling.py", "prepare_dspin.py",
                    "evaluate_confirmation.py", "study_main_effects.py", "study_native_cpa.py", "study_native_gears.py",
                    "probe_native_pertadapt.py", "package_native_cpa.py", "predict_native_cpa.py",
                    "download_dspin.py", "report_next.py"]]
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in code_paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    write_json(output / "protocol.json", {"version": 1, "source_hash": source_hash(), "environment": environment(),
               "source_files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code_paths},
               "source_snapshot_sha256": sha256(output / "source_snapshot.zip"),
               "attribution_config_sha256": sha256(config_path), "historical_evaluator_access": False,
               "acquisition_config_sha256": sha256(ROOT / "configs/experiments/acquisition-v1.json"),
               "master_plan_is_local_only": True, "skip_attribution": skip_attribution,
               "scope": "v1 audit, development attribution and synthetic contextual engineering; biological context transfer gated"})
    status = {"status": "running", "phases": {}}
    write_json(output / "status.json", status)
    try:
        for label, action in [("evidence_archive", lambda: archive_evidence(ROOT, output / "archive")),
                              ("ceiling_audit", lambda: audit_ceiling(ROOT)),
                              ("data_gate", lambda: data_gate(ROOT)),
                              ("native_methods", lambda: native_probe(ROOT))]:
            print("Starting", label, flush=True)
            result = action()
            write_json(output / f"{label}.json", result)
            status["phases"][label] = "complete"
            write_json(output / "status.json", status)
        config = read_json(config_path)
        if not skip_attribution:
            with threadpool_limits(limits=1):
                summary = {name: study(ROOT / "data/prepared" / name, output / "attribution" / name, config) for name in config["datasets"]}
            write_json(output / "attribution.json", {name: {kind: {k: v for k, v in record.items() if k != "records"}
                                                        for kind, record in value.items() if isinstance(record, dict)} for name, value in summary.items()})
            status["phases"]["kernel_attribution"] = "complete"
        else:
            status["phases"]["kernel_attribution"] = "explicitly_skipped"
        acquisition_config = read_json(ROOT / "configs/experiments/acquisition-v1.json")
        with threadpool_limits(limits=1):
            acquisition = {name: acquisition_study(ROOT / "data/prepared" / name, output / "acquisition" / name, acquisition_config)
                           for name in config["datasets"]}
        write_json(output / "acquisition.json", {name: {k: v for k, v in value.items() if k != "records"} for name, value in acquisition.items()})
        status["phases"]["retrospective_acquisition"] = "complete"
        print("Starting synthetic contextual benchmark", flush=True)
        prepare_fixture(output / "synthetic_dataset")
        with threadpool_limits(limits=1):
            result = run_benchmark(output / "synthetic_dataset", output / "synthetic_benchmark")
        write_json(output / "synthetic_engineering.json", result)
        status["phases"]["contextual_engineering"] = "complete"
        # Reuse recorded exploratory evidence without reopening its now-exposed
        # held-context outcomes. A new confirmation requires a new protocol.
        pilot = ROOT / 'results/next/emeraldbay-pilot-v1'
        biological = False
        if (pilot / 'results.json').exists():
            registered = read_json(pilot / 'run_protocol.json')
            expected = sha256(ROOT / 'data/prepared/emeraldbay-pilot-v1/protocol.json')
            if registered['dataset_protocol_hash'] != expected:
                raise ValueError('Recorded biological pilot no longer matches its dataset')
            record = read_json(pilot / 'results.json')
            if record['status'] != 'complete' or record['synthetic']:
                raise ValueError('Invalid recorded biological pilot')
            write_json(output / 'biological_pilot.json', {'results': record, 'source': str(pilot),
                       'results_sha256': sha256(pilot / 'results.json'), 'dataset_protocol_sha256': expected,
                       'previously_evaluated': True, 'evaluator_outcomes_reopened': False,
                       'independent_confirmation': False})
            status['phases']['recorded_biological_pilot'] = 'reused_existing_evidence'
            biological = True
        status.update({"status": "engineering_and_development_complete_biological_transfer_blocked",
                       "seconds": time.perf_counter() - started, "historical_evaluator_access": False,
                       "biological_context_transfer_established": False,
                       "remaining": ["qualified crossed-context combination data", "native information-aligned comparators",
                                     "independent biological confirmation", "validated uncertainty/abstention"]})
        if biological:
            status['status'] = 'engineering_and_exploratory_chemical_pilot_complete_confirmation_pending'
            status['exploratory_chemical_context_transfer_evaluated'] = True
            status['remaining'] = ['crossed-context genetic combination corpus', 'native information-aligned comparators',
                                   'independent biological confirmation', 'validated uncertainty/abstention']
        evidence_paths = {
            'independent_procedure_test': 'dspin-confirmation-v1/confirmation_review.json',
            'native_chemical_comparison': 'native-cpa-aggregate-v1/results.json',
            'native_genetic_norman': 'native-gears-norman-v1/results.json',
            'native_genetic_wessels': 'native-gears-wessels-v1/results.json',
            'estimated_single_pilot': 'missing-single-pilot-v3/results.json',
            'pertadapt_feasibility': 'pertadapt-native-probe-v1/results.json'}
        recorded = {}
        for label, relative in evidence_paths.items():
            path = ROOT / 'results/next' / relative
            if path.exists():
                recorded[label] = {'results': read_json(path), 'source': str(path),
                    'sha256': sha256(path), 'evaluator_outcomes_reopened': False}
                status['phases'][label] = 'reused_existing_evidence'
        if recorded:
            write_json(output / 'recorded_extensions.json', recorded)
            status['status'] = 'research_software_and_registered_extensions_complete_scientific_gates_unmet'
            status['remaining'] = ['Replicated crossed-context genetic corpus',
                'Canonical cell-level comparators including a compatible PertAdapt run',
                'Positive independent confirmation against the strongest matched reference',
                'Validated reliability under context shift']
            status['independent_procedure_evaluated'] = 'independent_procedure_test' in recorded
            status['missing_single_experimental_route'] = 'estimated_single_pilot' in recorded
        write_json(output / "status.json", status)
        return status
    except Exception as error:
        status.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        write_json(output / "status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/next/cycle-v1")
    parser.add_argument("--skip-attribution", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.output, args.skip_attribution))


if __name__ == "__main__":
    main()
