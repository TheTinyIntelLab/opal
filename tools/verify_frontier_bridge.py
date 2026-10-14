"""Run the five frozen engineering cases and strict replay, one process at a time."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from opal.frontier_bridge.contracts import ContractError, load_config, read_json, sha256
from opal.frontier_bridge.preparation import CASES
from opal.frontier_bridge.resources import Budget, inventory
from opal.frontier_bridge.workflow import code_hashes

ROOT = Path(__file__).resolve().parents[1]

# The wrapper measures every child's complete Python process CPU, including
# failed commands. Abrupt OS termination can still prevent this final receipt.
CHILD = """
import json, pathlib, runpy, sys, time
receipt, kind, target, *args = sys.argv[1:]
sys.argv = [target, *args]
try:
    if kind == 'module':
        runpy.run_module(target, run_name='__main__', alter_sys=True)
    else:
        runpy.run_path(target, run_name='__main__')
finally:
    pathlib.Path(receipt).write_text(json.dumps({'cpu_seconds': time.process_time(),
        'scope': 'complete child Python process; native single-thread kernels included'}) + '\\n', encoding='utf-8')
"""


def protected_snapshot(cfg):
    paths = [ROOT / cfg[name]["path"] for name in
             ("prepared_source", "source_access", "source_registry", "source_support")]
    paths += sorted((ROOT / "src").glob("series*.py"))
    paths += sorted((ROOT / "src/replicated").glob("*.py"))
    paths += [ROOT / "configs/experiments/replicated_fixed_state_v1.json"]
    return {str(path.resolve()): {"available": path.is_file(),
            "sha256": sha256(path) if path.is_file() else None} for path in paths}


def assert_frozen(expected):
    if code_hashes(ROOT) != expected:
        raise ContractError("Verification inference code changed during the suite")


def terminate_tree(process):
    """Stop the numerical child and its interpreter-launcher descendants."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def execute(output, label, kind, target, arguments, budget, command_records):
    """Monitor cumulative storage while retaining complete command logs."""
    budget.check(65536)
    receipt = output / f"{label}-cpu.json"
    log = output / f"{label}.log"
    started = time.monotonic()
    interrupted = None
    command = [sys.executable, "-c", CHILD, str(receipt), kind, str(target), *map(str, arguments)]
    with log.open("xb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        try:
            while process.poll() is None:
                if (output / "CANCEL").exists():
                    raise InterruptedError("Verification cancellation requested by output/CANCEL")
                budget.check()
                # Receipt-based cumulative CPU is complete for finished
                # children; a currently running child's own Budget enforces
                # its bound. Pytest gets a conservative elapsed-time bound.
                observed = sum(r.get("cpu_seconds") or 0 for r in command_records)
                if (sum(budget.prior_cpu.values()) + observed + time.process_time()
                        + time.monotonic() - started) >= budget.cfg["resources"]["cpu_core_hours"] * 3600:
                    raise ContractError("Conservative sequential verification CPU allowance reached")
                time.sleep(1)
        except BaseException as error:
            interrupted = error
            terminate_tree(process)
    measured = read_json(receipt) if receipt.exists() else {}
    record = dict(label=label, command=[kind, str(target), *map(str, arguments)],
                  exit_code=process.returncode, log=str(log.resolve()),
                  cpu_seconds=measured.get("cpu_seconds"),
                  wall_seconds=time.monotonic() - started,
                  cpu_receipt_available=receipt.exists())
    command_records.append(record)
    if interrupted is not None:
        # Preserve failed-command timing before propagating the guard error.
        (output / "commands.json").write_text(json.dumps(command_records, indent=2) + "\n", encoding="utf-8")
        raise interrupted
    budget.write_json(output / "commands.json", command_records)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/opal_frontier_bridge_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--completed-summary", type=Path, help="Reuse verified completed cases from an interrupted suite with identical frozen code")
    parser.add_argument("--tests", nargs="+", help="Pytest paths; defaults to all frontier bridge test files")
    parser.epilog = "Create OUTPUT/CANCEL to stop the child process tree and retain an incomplete final receipt."
    parser.add_argument("--allow-resource-code-variant", action="store_true", help="Import prior strict receipts when only resources.py changed; numerical code must match")
    parser.add_argument("--reuse-prepared", action="append", default=[], metavar="CASE=MANIFEST", help="Use a retained synthetic prepared manifest for an interrupted case")
    args = parser.parse_args(argv)
    reused = dict(value.split("=", 1) for value in args.reuse_prepared)
    if set(reused) - set(CASES):
        parser.error("Unknown prepared case")
    cfg = load_config(args.config)
    output = args.output.resolve()
    # Existing outputs are immutable, including an incomplete earlier suite.
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(ROOT, cfg, output)
    commands, cases = [], {}
    initial_code = code_hashes(ROOT)
    protected_before = protected_snapshot(cfg)
    imported = None
    if args.completed_summary:
        imported = read_json(args.completed_summary)
        previous_code = imported.get("inference_code_hashes", {})
        differences = {key for key in set(previous_code) | set(initial_code) if previous_code.get(key) != initial_code.get(key)}
        resource_path = str((ROOT / "src/frontier_bridge/resources.py").resolve())
        if (differences and not (args.allow_resource_code_variant and differences == {resource_path})) or imported.get("protected_before") != protected_before:
            raise ContractError("Completed suite numerical code or protected assets changed")
        for case, record in imported.get("cases", {}).items():
            receipt = record.get("replay", {})
            if case not in CASES or not record.get("decision", {}).get("engineering_complete") or receipt.get("status") != "passed" or not receipt.get("exact_frozen_code") or receipt.get("max_absolute_error", 1) > 1e-10:
                raise ContractError("Imported case lacks completed strict replay")
            manifest = Path(record["output"]) / "run_manifest.json"
            if sha256(manifest) != receipt["run_manifest_sha256"] or any(read_json(manifest)["code_hashes"].get(key) != value for key, value in initial_code.items() if not (args.allow_resource_code_variant and key == resource_path)):
                raise ContractError("Imported case manifest or frozen code changed")
            cases[case] = dict(record, imported_from=str(args.completed_summary.resolve()))
    summary = dict(status="running", role="synthetic_engineering_only",
                   independent_biological_confirmation=False, frontier_claim=False,
                   cases=cases, commands=commands, inference_code_hashes=initial_code,
                   allowed_resource_code_variant=args.allow_resource_code_variant,
                   completed_summary_asset={"path": str(args.completed_summary.resolve()), "sha256": sha256(args.completed_summary)} if args.completed_summary else None,
                   protected_before=protected_before, cpu_accounting_complete=False,
                   inherited_inventory_before=inventory(ROOT, cfg),
                   protected_hash_scope="Registered source registry/access/support/prepared metadata, historical src/series*.py and src/replicated/*.py, and replicated fixed-state config; inherited data directories receive live size/availability inventory, not complete content hashing")
    failure = None
    try:
        budget.check()
        budget.write_json(output / "verification_summary.json", summary)
        for case in CASES:
            if case in cases:
                continue
            assert_frozen(initial_code)
            run_output = output / f"synthetic-{case}"
            mode = ["--synthetic", case]
            if case in reused:
                prepared_path = Path(reused[case]).resolve()
                prepared = read_json(prepared_path)
                if prepared.get("role") != "synthetic_engineering" or prepared.get("synthetic_case") != case:
                    raise ContractError("Retained prepared manifest is not the requested synthetic case")
                mode = ["--prepared", prepared_path]
            result = execute(output, f"run-{case}", "path", ROOT / "tools/run_frontier_bridge.py",
                             ["--config", args.config.resolve(), *mode,
                              "--output", run_output], budget, commands)
            case_record = dict(run=result, output=str(run_output),
                               decision=read_json(run_output / "decision.json")
                               if (run_output / "decision.json").exists() else {})
            cases[case] = case_record
            manifest = run_output / "run_manifest.json"
            if result["exit_code"] == 0 and manifest.exists():
                assert_frozen(initial_code)
                replay_output = output / f"replay-{case}"
                replay_command = execute(output, f"replay-{case}", "path", ROOT / "tools/run_frontier_bridge.py",
                    ["--config", args.config.resolve(), "--replay", manifest,
                     "--output", replay_output], budget, commands)
                case_record["replay_command"] = replay_command
                replay_path = replay_output / "replay.json"
                case_record["replay"] = read_json(replay_path) if replay_path.exists() else None
            budget.write_json(output / "verification_summary.json", summary)
        if not args.skip_tests:
            assert_frozen(initial_code)
            temporary = output / "pytest-temp"
            budget.register_temporary_root(temporary)
            tests = args.tests or [str(p.relative_to(ROOT)) for p in sorted((ROOT / "tests").glob("test_frontier*.py"))]
            summary["tests"] = execute(output, "pytest", "module", "pytest",
                ["-q", "-p", "no:cacheprovider", "--basetemp", temporary, *tests], budget, commands)
        assert_frozen(initial_code)
    except (Exception, KeyboardInterrupt) as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        after = protected_snapshot(cfg)
        current_inventory = inventory(ROOT, cfg)
        summary.update(protected_after=after, protected_available_files_unchanged=after == protected_before,
                       current_inference_code_hashes=code_hashes(ROOT),
                       inherited_inventory_after=current_inventory,
                       unavailable_historical_roots=current_inventory["unavailable_physical_roots"],
                       failure=failure, observed_child_cpu_seconds=sum(r.get("cpu_seconds") or 0 for r in commands),
                       cpu_unknown_scope="Earlier unreceipted commands and abruptly terminated children remain unknown; missing historical files cannot be freshly verified")
        complete = (failure is None and len(cases) == len(CASES)
                    and all(c.get("decision", {}).get("engineering_complete", False)
                            and c.get("replay", {}).get("status") == "passed"
                            and c["replay"].get("exact_frozen_code")
                            and c["replay"].get("max_absolute_error", 1) <= 1e-10 for c in cases.values())
                    and (args.skip_tests or summary.get("tests", {}).get("exit_code") == 0)
                    and after == protected_before)
        summary["status"] = "passed" if complete else "incomplete"
        # Final failure receipts use the reserved allowance even if a guard
        # tripped. CPU is aggregated here once: nested child ledgers are not
        # counted as independent top-level commands by future Budget runs.
        ledger = budget.evidence()
        ledger["cpu_seconds"] += summary["observed_child_cpu_seconds"]
        ledger["cpu_scope"] = "orchestrator plus separately receipted sequential children; child ledgers nested under this output are not independently charged"
        ledger["cumulative_observed_cpu_seconds"] = sum(budget.prior_cpu.values()) + ledger["cpu_seconds"]
        for name, value in (("verification_summary.json", summary), ("resource_ledger.json", ledger)):
            (output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(output), "failure": failure}))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
