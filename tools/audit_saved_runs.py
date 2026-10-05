"""Read-only retrospective integrity audit; never fits or reselects models."""
from pathlib import Path
import sys
import zipfile
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import read_json, write_json, load_npz, sha256, fingerprint
from opal.training.benchmark import dataset_fingerprint


def main():
    pairs = {}
    for config in (ROOT / "configs/experiments").glob("*.json"):
        value = read_json(config)
        if "dataset" in value and "output" in value:
            pairs[ROOT / value["output"]] = ROOT / value["dataset"]
    for root in [ROOT / "results/synthetic", ROOT / "results/robustness"]:
        for file in root.rglob("final_status.json"):
            run = file.parent
            pairs[run] = run.parent / "dataset"
    records = []
    for run, dataset in sorted(pairs.items(), key=lambda p: str(p[0])):
        if not (run / "final_status.json").exists():
            continue
        state = read_json(run / "final_status.json")
        if state["status"] != "complete":
            continue
        selection = read_json(run / "selection.json")
        access = read_json(run / "test_access.json")
        data = load_npz(dataset / "profiles.npz")
        evaluator = load_npz(dataset / "evaluator/profiles.npz")
        predictions = load_npz(run / "predictions.npz")
        results = read_json(run / "test_results.json")
        assert dataset_fingerprint(dataset) == selection["data_fingerprint"], str(run)
        assert sha256(run / "predictions.npz") == access["predictions_sha256"] == state["predictions_sha256"], str(run)
        assert np.array_equal(data["query_indices"], evaluator["pair_indices"]), str(run)
        assert np.array_equal(predictions["pair_indices"], evaluator["pair_indices"]), str(run)
        max_difference = 0.0
        for cid, reported in results["models"].items():
            # Preserve the registered equal-pair reduction and saved dtype.
            # A flat float32 reduction has a different rounding order.
            actual = float(np.mean(np.mean((predictions[cid].mean(0) - evaluator["doubles"]) ** 2, axis=1)))
            max_difference = max(max_difference, abs(actual - reported["mse"]))
            assert abs(actual - reported["mse"]) < 1e-12, (run, cid)
        with zipfile.ZipFile(run / "source_snapshot.zip") as snapshot:
            assert snapshot.testzip() is None
            frozen_hash = fingerprint({str(Path(name.removeprefix("opal/"))): hashlib.sha256(snapshot.read(name)).hexdigest()
                                       for name in snapshot.namelist()})
            assert frozen_hash == selection["source_hash"], str(run)
        records.append({"run": str(run.relative_to(ROOT)), "status": "passed", "models": len(results["models"]),
            "maximum_mse_discrepancy": max_difference, "evaluator_sha256_at_audit": sha256(dataset / "evaluator/profiles.npz"),
            "source_hash_at_fitting": selection["source_hash"],
            "evaluator_frozen_at_development": "evaluator_sha256" in selection})
    write_json(ROOT / "results/integrity_audit.json", {"runs": records, "completed_runs_checked": len(records),
        "scope": "retrospective current-file checks; hashes recorded now do not create a historical evaluator freeze"})
    print(len(records), "completed runs pass read-only integrity checks")


if __name__ == "__main__":
    main()
