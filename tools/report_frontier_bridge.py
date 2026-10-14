"""Summarize retained engineering verification without fitting or label access."""
import argparse
from pathlib import Path
from opal.frontier_bridge.contracts import load_config, read_json, sha256, fingerprint
from opal.frontier_bridge.resources import Budget

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = read_json(args.verification)
    args.output.mkdir(parents=True, exist_ok=False)
    budget = Budget(ROOT, load_config(ROOT / "configs/experiments/opal_frontier_bridge_v1.json"), args.output)
    cases = {}
    lines = ["# Frontier engineering closeout", "", "These are synthetic engineering checks; public source qualification and independent measurement remain unavailable.", "", "| Case | Direction | Candidate MSE | Locked selector MSE | Candidate fallback fraction | Numerical gate |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for name, receipt in summary["cases"].items():
        output = Path(receipt["output"])
        metrics = read_json(output / "metrics.json")
        gates = read_json(output / "controls.json")["direction_gates"]
        stability = read_json(output / "rank_and_stability.json")
        fold_sizes = {f["fold"]: len(f["query"]) for f in read_json(output / "folds.json")}
        frozen_config = read_json(output / "protocol.json")
        scientific_config = {k: v for k, v in frozen_config.items() if k != "resources"}
        manifest = read_json(output / "run_manifest.json")
        fallback = {}
        for direction in metrics:
            arms = sorted({r["arm"] for r in stability if r["direction"] == direction})
            fallback[direction] = {}
            for arm in arms:
                rows = [r for r in stability if r["direction"] == direction and r["arm"] == arm]
                total = sum(fold_sizes[r["fold"]] for r in rows)
                failed = sum(fold_sizes[r["fold"]] for r in rows if not r["supported"])
                fallback[direction][arm] = dict(query_pair_stratum_rows=total, fallback_rows=failed, fallback_fraction=failed / total if total else None)
            source_free_rows = [c for c in manifest["checkpoints"] if c["direction"] == direction and c["arm"] == "source_free"]
            total, failed = 0, 0
            for checkpoint in source_free_rows:
                state_path = Path(checkpoint["path"]) / "state.json"
                if sha256(state_path) != checkpoint["state_sha256"]:
                    raise ValueError("Checkpoint state changed before report")
                total += fold_sizes[checkpoint["fold"]]
                if read_json(state_path)["model"]["kind"] != "source_free":
                    failed += fold_sizes[checkpoint["fold"]]
            fallback[direction]["source_free"] = dict(query_pair_stratum_rows=total, fallback_rows=failed, fallback_fraction=failed / total if total else None)
            candidate = fallback[direction]["candidate"]["fallback_fraction"]
            lines.append(f"| {name} | {direction} | {metrics[direction]['candidate']['mse']:.10g} | {metrics[direction]['selector']['mse']:.10g} | {candidate:.4g} | {gates[direction]['passed']} |")
        cases[name] = dict(output=str(output), configuration_hash=manifest["configuration_hash"], scientific_configuration_hash=fingerprint(scientific_config), decision=receipt["decision"], replay=receipt.get("replay"), fallback=fallback,
                           primary_mse={d: {a: m["mse"] for a, m in arms.items()} for d, arms in metrics.items()},
                           direction_gates=gates, metrics_sha256=sha256(output / "metrics.json"))
    science_hashes = {c["scientific_configuration_hash"] for c in cases.values()}
    if len(science_hashes) != 1:
        raise ValueError("Cannot combine different scientific contracts")
    result = dict(scientific_contract_hash=next(iter(science_hashes)), status=summary["status"], engineering_complete=summary["status"] == "passed", cases=cases,
                  qualification=read_json(args.qualification), measurement_readiness=read_json(args.readiness),
                  verification_asset=dict(path=str(args.verification.resolve()), sha256=sha256(args.verification)),
                  total_replayed_checkpoints=sum(c.get("replay", {}).get("checkpoints", 0) for c in cases.values()),
                  protected_available_files_unchanged=summary.get("protected_available_files_unchanged"),
                  development_gate_passed=False, independent_biological_confirmation=False, frontier_claim=False,
                  uncertainty_scope="conditional target-incidence sensitivity; no biological confidence or procedure-refitting uncertainty",
                  next_action="independently initiated cultures with qualified references, empirical covariance/power and reviewed confirmation contract")
    budget.write_json(args.output / "closeout.json", result)
    lines += ["", "Candidate fallback fractions retain the entire registered query population; failed rank or stability checks do not remove pairs. Required unsupported controls block advancement.", "", "Every replay is a separate process using its frozen code and numerical assets, with no evaluator expression reopening. Earlier interrupted and incompatible-code attempts remain separate receipts.", "", "The registered source metadata has 207 common genes, below the frozen 256 requirement, and current prepared banks are unavailable. No public response scoring or biological confirmation occurred.", ""]
    text = "\n".join(lines)
    budget.check(len(text.encode("utf-8")))
    (args.output / "report.md").write_text(text, encoding="utf-8")
    budget.write_json(args.output / "resource_ledger.json", budget.evidence())
    print(result["status"])
    return 0 if result["engineering_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
