"""Conditional technical-cell bootstrap with shared control/single draws.

Fitted weights, predictions, QC, gene eligibility and normalization are fixed.
This measures response-estimation uncertainty, not fitting or biological variance.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import read_json, write_json, load_npz, sha256
from opal.data.prepare import load_raw
from opal.profiles import library_normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["norman", "wessels"])
    parser.add_argument("--draws", type=int, default=200)
    args = parser.parse_args()
    dataset = ROOT / f"data/prepared/{args.dataset}-log2"
    run = ROOT / f"results/runs/{args.dataset}-log2-primary"
    out = run / "measurement_bootstrap"
    if (out / "summary.json").exists():
        raise FileExistsError("Retained measurement-bootstrap evidence exists")
    out.mkdir(exist_ok=True)
    config = read_json(ROOT / f"configs/datasets/{args.dataset}-log2.json")
    data = load_npz(dataset / "profiles.npz")
    selected = read_json(run / "test_results.json")
    predictions = load_npz(run / "predictions.npz")
    fixed_pred = {"selected": predictions[selected["selected_id"]].mean(0),
                  "simpler": predictions[selected["simpler_id"]].mean(0)}
    write_json(out / "protocol.json", {"draws": args.draws, "seed": 87231, "unit": "cells within condition and matched stratum",
        "shared_draws": "one control draw per stratum and one single draw per target reused across all pairs",
        "fixed": "QC, eligible pairs, gene universe, normalizer, fitted weights and method predictions",
        "not_captured": "biological replication, fitting variation, new-target or new-context generalization",
        "registered_after_primary": True, "predictions_sha256": sha256(run / "predictions.npz"),
        "evaluator_sha256": sha256(dataset / "evaluator/profiles.npz")})
    x, genes, labels, strata, cells, digest, source_n = load_raw(config)
    mapping = config["condition_map"]
    totals = np.asarray(x.sum(1)).ravel()
    qc = config["qc"]
    valid = (totals >= qc["minimum_library"]) & (totals <= qc["maximum_library"])
    valid &= np.array([mapping[l]["context"] == config["context"] and not mapping[l].get("exclude", False) for l in labels])
    x, labels, strata = x[valid], labels[valid], strata[valid]
    prep = read_json(dataset / "preprocessing.json")
    keep = np.array(prep["gene_mask"], bool)
    if not np.array_equal(genes[keep], data["genes"]):
        raise ValueError("Bootstrap gene schema changed")
    # Normalize on the full source library before restricting output genes.
    x = library_normalize(x, prep["normalizer"])[:, keep].tocsr()
    pair_groups = read_json(dataset / "qc_report.json")["pair_groups"]
    query_groups = [pair_groups[i] for i in data["query_indices"]]
    single_labels = {t: next(l for l in mapping if mapping[l]["components"] == [t] and np.any(labels == l)) for t in data["targets"]}
    required = set(single_labels.values()) | {l for group in query_groups for l in group}
    controls = np.array([len(mapping[l]["components"]) == 0 for l in labels])
    groups = {(l, s): np.flatnonzero((labels == l) & (strata == s)) for l in sorted(required) for s in sorted(set(strata[labels == l]))}
    control_groups = {s: np.flatnonzero(controls & (strata == s)) for s in sorted(set(strata))}
    if any(len(i) == 0 for i in control_groups.values()):
        raise ValueError("Missing matched control stratum")

    def effects(means, control_means):
        profiles = {}
        for l in required:
            values = [np.log2(means[l, s] + 1) - np.log2(control_means[s] + 1) for ll, s in groups if ll == l]
            profiles[l] = np.mean(values, axis=0)
        single = np.stack([profiles[single_labels[t]] for t in data["targets"]], axis=-2)
        double = np.stack([np.mean([profiles[l] for l in g], axis=0) for g in query_groups], axis=-2)
        return single, double

    point_single, point_double = effects({k: np.asarray(x[i].mean(0)).ravel() for k, i in groups.items()},
                                        {s: np.asarray(x[i].mean(0)).ravel() for s, i in control_groups.items()})
    evaluated = load_npz(dataset / "evaluator/profiles.npz")
    if not np.array_equal(evaluated["pair_indices"], data["query_indices"]):
        raise ValueError("Evaluator order changed")
    if not np.allclose(point_single, data["singles"], atol=2e-6) or not np.allclose(point_double, evaluated["doubles"], atol=2e-6):
        raise ValueError("Raw reconstruction disagrees with frozen profiles")
    rng = np.random.default_rng(87231)
    errors = {name: [] for name in ["additive", "selected", "simpler"]}
    pairs = data["query_pairs"]

    def sample_mean(indices, batch):
        n = len(indices)
        weights = sparse.csr_matrix(rng.multinomial(n, np.full(n, 1 / n), size=batch) / n)
        return (weights @ x[indices]).toarray()

    with threadpool_limits(limits=1):
        for start in range(0, args.draws, 8):
            batch = min(8, args.draws - start)
            control_means = {s: sample_mean(i, batch) for s, i in control_groups.items()}
            means = {k: sample_mean(i, batch) for k, i in groups.items()}
            singles, doubles = effects(means, control_means)
            additive = singles[:, pairs[:, 0]] + singles[:, pairs[:, 1]]
            errors["additive"].extend(np.mean((additive - doubles) ** 2, axis=-1))
            for name, pred in fixed_pred.items():
                errors[name].extend(np.mean((pred[None] - doubles) ** 2, axis=-1))
            print(args.dataset, min(start + batch, args.draws), flush=True)
    errors = {k: np.array(v) for k, v in errors.items()}
    np.savez_compressed(out / "draws.npz", **errors, pair_indices=data["query_indices"])
    summaries = {k: {"mean_mse": float(v.mean()), "technical_profile_percentile_95": np.quantile(v.mean(1), [.025, .975]).tolist()} for k, v in errors.items()}
    gain = errors["simpler"].mean(1) - errors["selected"].mean(1)
    summaries["selected_vs_simpler"] = {"mean_improvement": float(gain.mean()), "technical_profile_percentile_95": np.quantile(gain, [.025, .975]).tolist()}
    write_json(out / "summary.json", {"results": summaries, "draws": args.draws,
        "interpretation": "conditional technical measurement sensitivity with fixed predictions; not a population confidence interval"})


if __name__ == "__main__":
    main()
