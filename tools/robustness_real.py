"""Reuse frozen count-derived references under independently fitted holdouts.

The procedure and seed list stay fixed after the primary evaluation. Repeated
splits are correlated secondary analyses, never new biological replicates.
"""
import argparse
from pathlib import Path
import sys
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import read_json, write_json, load_npz, sha256
from opal.data.synthetic import save_prepared
from opal.splits import partner_splits
from opal.training.studies import save_partner_dataset
from opal.training.benchmark import development, final, dataset_fingerprint

SEEDS = [1729, 2027, 4099, 7919, 104729]


def reconstruct(dataset):
    data, evaluator = load_npz(dataset / "profiles.npz"), load_npz(dataset / "evaluator/profiles.npz")
    n = len(data["pair_indices"]) + len(data["query_indices"])
    pairs = np.zeros((n, 2), int)
    doubles = np.zeros((n, len(data["genes"])), dtype=data["doubles"].dtype)
    pairs[data["pair_indices"]], pairs[data["query_indices"]] = data["pairs"], data["query_pairs"]
    doubles[data["pair_indices"]], doubles[data["query_indices"]] = data["doubles"], evaluator["doubles"]
    arrays = {k: data[k] for k in ["singles", "targets", "genes", "control", "control_sd"] if k in data}
    arrays.update({"pairs": pairs, "doubles": doubles})
    halves = None
    if (dataset / "profile_uncertainty.npz").exists():
        devhalf, testhalf = load_npz(dataset / "profile_uncertainty.npz"), load_npz(dataset / "evaluator/profile_uncertainty.npz")
        halves = {k: devhalf[k] for k in ["singles_0", "singles_1"]}
        for i in range(2):
            z = np.empty_like(doubles)
            z[data["pair_indices"]], z[data["query_indices"]] = devhalf[f"doubles_{i}"], testhalf[f"doubles_{i}"]
            halves[f"doubles_{i}"] = z
    return arrays, halves


def attach_uncertainty(path, halves):
    if halves is None:
        return
    split = read_json(path / "splits.json")
    for dest, indices in [(path, split["development"]), (path / "evaluator", split["test"])]:
        np.savez_compressed(dest / "profile_uncertainty.npz", **{k: v[indices] if k.startswith("doubles") else v for k, v in halves.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["norman", "wessels"])
    p.add_argument("--outer-only", action="store_true")
    p.add_argument("--partner-only", action="store_true")
    p.add_argument("--track", help="Execute one already registered track in a separate worker")
    args = p.parse_args()
    parent = ROOT / f"data/prepared/{args.dataset}-log2"
    root = ROOT / f"results/robustness/{args.dataset}"
    protocol = read_json(ROOT / f"configs/experiments/{args.dataset}-log2.json")["training"]
    write_json(root / "protocol.json", {"parent_data_fingerprint": dataset_fingerprint(parent), "training": protocol,
               "random_outer_seeds": SEEDS, "partner_seed": 2718, "partner_folds": 5,
               "minimum_development": 12, "minimum_test": 4, "inner_minimum_development": 12,
               "scope": "secondary repeated protocol; overlapping tests correlated; primary result preserved"})
    arrays, halves = reconstruct(parent)
    source = read_json(parent / "source_manifest.json")
    source["secondary_parent_fitting_fingerprint"] = dataset_fingerprint(parent)
    tasks = []
    if not args.partner_only:
        tasks += [(f"random_{seed}", seed, None) for seed in SEEDS[1:]]
    if not args.outer_only:
        tasks += [(f"partner_{f['fold']}", None, f) for f in partner_splits(arrays["targets"][arrays["pairs"]], seed=2718)]
    if args.track:
        tasks = [task for task in tasks if task[0] == args.track]
        if len(tasks) != 1:
            raise ValueError("Track must identify exactly one registered holdout")
    progress = []
    for name, seed, fold in tasks:
        dataset, run = root / name / "dataset", root / name / "run"
        if fold is not None and not fold["feasible"]:
            progress.append({"track": name, "status": "infeasible", "fold": fold})
            continue
        if not (dataset / "profiles.npz").exists():
            if fold is None:
                save_prepared(dataset, arrays, source, seed)
            else:
                save_partner_dataset(dataset, arrays, source, fold)
                # Preserve additional allowed control dispersion reference.
                bank = load_npz(dataset / "profiles.npz")
                if "control_sd" in arrays:
                    bank["control_sd"] = arrays["control_sd"]
                    np.savez_compressed(dataset / "profiles.npz", **bank)
            attach_uncertainty(dataset, halves)
        try:
            if not (run / "selection.json").exists():
                development(dataset, run, protocol)
            if not (run / "final_status.json").exists():
                final(dataset, run)
            results = read_json(run / "test_results.json")
            progress.append({"track": name, "status": "complete", "selected": results["models"][results["selected_id"]],
                             "simpler": results["models"][results["simpler_id"]], "comparisons": results["comparisons"],
                             "run": str(run), "test_access_sha256": sha256(run / "test_access.json")})
        except ValueError as e:
            if "feasible inner holdouts" not in str(e):
                raise
            progress.append({"track": name, "status": "inner_cv_infeasible", "reason": str(e)})
        write_json(root / (f"worker_{args.track}.json" if args.track else "progress.json"), progress)
    write_json(root / (f"worker_{args.track}.json" if args.track else "summary.json"), progress)
    print(f"Secondary suite recorded: {root}", flush=True)


if __name__ == "__main__":
    main()
