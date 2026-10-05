"""Evaluator-only swapped-half emergent panels on the original gene universe.

Outside the learned gene schema, the extension predicts additivity only.
This does not claim an extended trained model or use panel outcomes for fitting.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import read_json, write_json, load_npz
from opal.data.prepare import load_raw
from opal.profiles import library_normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["norman", "wessels"])
    args = parser.parse_args()
    bank = ROOT / f"data/prepared/{args.dataset}-log2"
    run = ROOT / f"results/runs/{args.dataset}-log2-primary"
    out = run / "extended_gene_diagnostic"
    out.mkdir(exist_ok=True)
    config = read_json(ROOT / f"configs/datasets/{args.dataset}-log2.json")
    data = load_npz(bank / "profiles.npz")
    mapping = config["condition_map"]
    x, genes, labels, strata, cells, digest, source_cells = load_raw(config)
    totals = np.asarray(x.sum(1)).ravel()
    qc = config["qc"]
    valid = (totals >= qc["minimum_library"]) & (totals <= qc["maximum_library"])
    valid &= np.array([mapping[l]["context"] == config["context"] and not mapping[l].get("exclude", False) for l in labels])
    x, labels, strata = x[valid], labels[valid], strata[valid]
    prep = read_json(bank / "preprocessing.json")
    x = library_normalize(x, prep["normalizer"]).tocsr()
    controls = np.array([len(mapping[l]["components"]) == 0 for l in labels])
    pair_groups = read_json(bank / "qc_report.json")["pair_groups"]
    query_groups = [pair_groups[i] for i in data["query_indices"]]
    single_labels = {t: next(l for l in mapping if mapping[l]["components"] == [t] and np.any(labels == l)) for t in data["targets"]}
    required = set(single_labels.values()) | {l for g in query_groups for l in g}
    rng = np.random.default_rng(config.get("measurement_seed", 31415))
    partitions = np.zeros(len(labels), int)
    for label in sorted(set(labels)):
        for stratum in sorted(set(strata[labels == label])):
            idx = np.flatnonzero((labels == label) & (strata == stratum))
            partitions[rng.permutation(idx)[::2]] = 1

    def profiles(mask):
        effects = {}
        control = {s: np.asarray(x[mask & controls & (strata == s)].mean(0)).ravel() for s in sorted(set(strata))}
        for label in sorted(required):
            values = []
            for s in sorted(set(strata[labels == label])):
                idx = mask & (labels == label) & (strata == s)
                values.append(np.log2(np.asarray(x[idx].mean(0)).ravel() + 1) - np.log2(control[s] + 1))
            effects[label] = np.mean(values, axis=0)
        singles = np.stack([effects[single_labels[t]] for t in data["targets"]])
        doubles = np.stack([np.mean([effects[l] for l in g], axis=0) for g in query_groups])
        return singles, doubles

    with threadpool_limits(limits=1):
        point_single, point_double = profiles(np.ones(len(labels), bool))
        halves = [profiles(partitions == i) for i in [0, 1]]
    keep = np.array(prep["gene_mask"], bool)
    if not np.array_equal(genes[keep], data["genes"]) or not np.allclose(point_single[:, keep], data["singles"], atol=2e-6):
        raise ValueError("Reference reconstruction changed")
    result = read_json(run / "test_results.json")
    selected = load_npz(run / "predictions.npz")[result["selected_id"]].mean(0)
    pairs = data["query_pairs"]
    additive = point_single[pairs[:, 0]] + point_single[pairs[:, 1]]
    extended_pred = additive.copy()
    extended_pred[:, keep] = selected
    records = []
    for selector in [0, 1]:
        s, y = halves[selector]
        residual = y - s[pairs[:, 0]] - s[pairs[:, 1]]
        other_y = halves[1 - selector][1]
        for i, row in enumerate(residual):
            panel = np.lexsort((genes, -np.abs(row)))[:20]
            records.append({"pair_index": int(data["query_indices"][i]), "selector_half": selector,
                "genes": genes[panel].tolist(), "outside_learned_schema": int((~keep[panel]).sum()),
                "selected_extension_mse": float(np.mean((extended_pred[i, panel] - other_y[i, panel]) ** 2)),
                "additive_mse": float(np.mean((additive[i, panel] - other_y[i, panel]) ** 2))})
    write_json(out / "panels.json", records)
    write_json(out / "summary.json", {"original_genes": len(genes), "learned_genes": int(keep.sum()), "panel_size": 20,
        "outside_schema_fraction": float(np.mean([r["outside_learned_schema"] / 20 for r in records])),
        "selected_extension_mse": float(np.mean([r["selected_extension_mse"] for r in records])),
        "additive_mse": float(np.mean([r["additive_mse"] for r in records])),
        "protocol": "evaluator-only absolute residual selector on one technical half, score response on opposite half; swap and average",
        "extension": "fixed selected prediction on its frozen schema; observed full-single additive response everywhere else",
        "interpretation": "measurement-sensitive diagnostic; no extended model fitting or independent biological confirmation"})
    print(args.dataset, "extended diagnostic complete", flush=True)


if __name__ == "__main__":
    main()
