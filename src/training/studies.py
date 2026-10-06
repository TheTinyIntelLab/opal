"""Prespecified sensitivity experiments with their information access stated."""
from pathlib import Path
import numpy as np
from opal.artifacts import write_json
from opal.splits import partner_splits
from opal.data.synthetic import save_prepared, freeze_evaluator
from opal.training import fit, predict


def sample_predictions(selection, data, fractions=(0.1, 0.25, 0.5, 0.75, 1.0), orders=(103, 211, 307)):
    record = next(r for r in selection["comparisons"] if r["config"]["id"] == selection["selected_id"])
    config = {**record["config"], "epochs": record["refit_epochs"]}
    predictions, records = {}, []
    for seed in orders:
        order = np.random.default_rng(seed).permutation(len(data["pairs"]))
        for fraction in fractions:
            n = max(2, int(np.ceil(fraction * len(order))))
            subset = order[:n]
            member = fit(config, data["singles"], data["pairs"][subset], data["doubles"][subset], seed,
                         single_partitions=data.get("single_partitions"))
            key = f"order{seed}_fraction{fraction}"
            predictions[key] = predict(member, data["query_pairs"])
            support = np.bincount(data["pairs"][subset].ravel(), minlength=len(data["singles"]))
            records.append({"key": key, "subset_order_seed": seed, "fraction": fraction, "pairs": n,
                            "targets_covered": int(sum(support > 0)), "target_degrees": support.tolist(),
                            "subset_pair_indices": data["pair_indices"][subset].tolist(),
                            "interpretation": "Fixed full-development hyperparameters and epoch budget; not subset-retuned sample efficiency"})
    return predictions, records


def save_partner_dataset(path, arrays, manifest, outer_fold):
    path = Path(path)
    save_prepared(path, arrays, manifest)
    dev, test = np.array(outer_fold["development"]), np.array(outer_fold["test"])
    np.savez_compressed(path / "profiles.npz", singles=arrays["singles"], pairs=arrays["pairs"][dev],
                        doubles=arrays["doubles"][dev], pair_indices=dev,
                        query_pairs=arrays["pairs"][test], query_indices=test,
                        control=arrays["control"], targets=arrays["targets"], genes=arrays["genes"])
    np.savez_compressed(path / "evaluator/profiles.npz", doubles=arrays["doubles"][test], pair_indices=test)
    freeze_evaluator(path, test)
    inner = partner_splits(arrays["targets"][arrays["pairs"][dev]], seed=31415, minimum_development=12)
    write_json(path / "cv_splits.json", inner)
    write_json(path / "splits.json", {"development": dev.tolist(), "test": test.tolist(), "fold": {str(i): 0 for i in dev},
                                     "outer_partner_holdout": outer_fold, "folds": 5, "inner_cv": "analogous partner holdout"})
    write_json(path / "fit_access_log.json", {"development_pair_indices": dev.tolist(), "test_double_expression_consumed": False,
               "resources": ["controls", "all observed singles"], "held_targets": outer_fold["held_targets"]})


def run_secondary_suite(root, arrays, protocol, seeds=(1729, 2027, 4099, 7919, 104729), partner_seed=2718):
    from opal.training.benchmark import development, final
    root = Path(root)
    write_json(root / "secondary_protocol.json", {"random_outer_seeds": list(seeds), "partner_seed": partner_seed,
               "partner_folds": 5, "minimum_development": 12, "minimum_test": 4,
               "scope": "synthetic low-rank robustness; overlapping outer splits are correlated"})
    rows = []
    for seed in seeds[1:]:
        dataset, run = root / f"random_{seed}/dataset", root / f"random_{seed}/run"
        save_prepared(dataset, arrays, {"synthetic": True, "scenario": "low_rank", "context": "synthetic"}, seed)
        development(dataset, run, protocol)
        result = final(dataset, run)
        rows.append({"track": "random_pair", "seed": seed, "selected": result["models"][result["selected_id"]]})
        write_json(root / "progress.json", rows)
    folds = partner_splits(arrays["targets"][arrays["pairs"]], partner_seed)
    for fold in folds:
        if not fold["feasible"]:
            rows.append({"track": "partner", "fold": fold["fold"], "status": "infeasible", "counts": fold})
            continue
        dataset, run = root / f"partner_{fold['fold']}/dataset", root / f"partner_{fold['fold']}/run"
        save_partner_dataset(dataset, arrays, {"synthetic": True, "scenario": "low_rank", "context": "synthetic"}, fold)
        development(dataset, run, protocol)
        result = final(dataset, run)
        rows.append({"track": "partner", "fold": fold["fold"], "held_targets": fold["held_targets"],
                     "selected": result["models"][result["selected_id"]]})
        write_json(root / "progress.json", rows)
    write_json(root / "summary.json", rows)
    return rows
