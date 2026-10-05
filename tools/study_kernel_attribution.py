"""Bounded nested attribution and matched neighborhood-removal controls."""
from pathlib import Path
import time
import zipfile
import numpy as np
from opal.artifacts import load_npz, sha256, source_hash, write_json
from opal.models.kernel import global_kernel, single_kernel, symmetric_kernel, smoother, prediction_error
from tools.improve_residual_model import fold_sets


def kernel_components(singles, pairs, bandwidths, seed):
    global_term = global_kernel(singles, pairs)
    permutation = np.random.default_rng(seed).permutation(len(singles))
    output = {("shared_offset", 1.): np.zeros((len(pairs), len(pairs))), ("global", 1.): global_term}
    for bandwidth in bandwidths:
        single = single_kernel(singles, bandwidth=bandwidth)
        pair = symmetric_kernel(single, pairs, pairs, interaction=True)
        shuffled = single[np.ix_(permutation, permutation)]
        output[("single_sum", bandwidth)] = symmetric_kernel(single, pairs, pairs)
        output[("pair_product", bandwidth)] = pair
        output[("global_plus_pair", bandwidth)] = global_term + pair
        output[("shuffled_pair", bandwidth)] = global_term + symmetric_kernel(shuffled, pairs, pairs, interaction=True)
    return output, permutation


def normalized(kernel, train):
    scale = max(float(np.diag(kernel)[train].mean()), 1e-12)
    return kernel / scale, scale


def select(components, gram, train, folds, alphas, allowed=None):
    records = []
    for (family, bandwidth), kernel in components.items():
        if allowed is not None and family not in allowed:
            continue
        for alpha in ([1.] if family == "shared_offset" else alphas):
            numerator, count = 0., 0
            for ti, vi in folds:
                fit, val = train[ti], train[vi]
                scaled, _ = normalized(kernel, fit)
                weights = smoother(scaled, fit, val, alpha)
                numerator += prediction_error(weights, fit, val, gram) * len(val)
                count += len(val)
            records.append({"kind": family, "bandwidth": bandwidth, "alpha": alpha, "mse": numerator / count})
    if not records:
        raise ValueError("No eligible attribution candidates")
    return min(records, key=lambda r: r["mse"]), records


def predict_residual(components, selected, train, val, residual):
    kernel, scale = normalized(components[(selected["kind"], selected["bandwidth"])], train)
    weights = smoother(kernel, train, val, selected["alpha"])
    return weights @ residual[train], weights, scale


def study(dataset, output, config):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError("Attribution outputs are immutable")
    output.mkdir(parents=True)
    started = time.perf_counter()
    data = load_npz(dataset / "profiles.npz")
    singles, doubles = np.asarray(data["singles"], dtype=float), np.asarray(data["doubles"], dtype=float)
    pairs = np.asarray(data["pairs"], dtype=int)
    residual = doubles - singles[pairs[:, 0]] - singles[pairs[:, 1]]
    gram = residual @ residual.T / residual.shape[1]
    components, permutation = kernel_components(singles, pairs, config["bandwidths"], config["seed"])
    write_json(output / "protocol.json", {**config, "profiles_sha256": sha256(dataset / "profiles.npz"),
               "source_hash": source_hash(), "runner_sha256": sha256(__file__),
               "evaluator_access": False, "normalization": "mean kernel diagonal fitted on each inner/outer fitting set",
               "candidate_counts": {f: sum(1 if k[0] == "shared_offset" else len(config["alphas"]) for k in components if k[0] == f) for f in config["families"]},
               "feature_permutation": permutation.tolist(), "attribution_is_not_causal_mechanism": True})
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((root / "src/opal").rglob("*.py")):
            archive.write(path, path.relative_to(root).as_posix())
        for path in [Path(__file__), root / "tools/improve_residual_model.py"]:
            archive.write(path, path.relative_to(root).as_posix())
    output_results = {}
    for kind in ["random", "partner"]:
        folds = fold_sets(pairs, kind, config["seed"], config["outer_folds"])
        if len(folds) < 2:
            raise ValueError("Insufficient feasible outer folds; no redraw")
        records, totals, coverage = [], {}, np.zeros(len(pairs), dtype=int)
        accumulated = {}
        for f, (train, val) in enumerate(folds):
            inner = fold_sets(pairs[train], kind, config["seed"] + f + 1, config["inner_folds"])
            if len(inner) < 2 or any(not len(t) or not len(v) for t, v in inner):
                raise ValueError("Insufficient inner folds; no redraw")
            selected, search = select(components, gram, train, inner, config["alphas"])
            chosen = {family: min((r for r in search if r["kind"] == family), key=lambda r: r["mse"]) for family in config["families"]}
            chosen["selected"] = selected
            predictions = {"additive": np.zeros((len(val), residual.shape[1]))}
            influences = {}
            for label, selection in chosen.items():
                pred, weights, scale = predict_residual(components, selection, train, val, residual)
                predictions[label] = pred
                influences[label] = {"maximum_absolute_weight": float(np.abs(weights).max()), "training_kernel_scale": scale}
            # Remove equally many near and random training pairs. Similarity
            # consumes allowed references only, never validation responses.
            removal_count = min(max(1, int(len(train) * config["neighborhood_removal_fraction"])), max(0, len(train) - 6))
            similarity = components[("pair_product", config["bandwidths"][1])][np.ix_(val, train)].max(0)
            near = train[np.argsort(-similarity, kind="stable")[:removal_count]]
            random = np.random.default_rng(config["seed"] + 1000 + f).choice(train, size=removal_count, replace=False)
            removal_records = {}
            for label, removed in [("near_removed", near), ("random_removed", random)]:
                retained = train[~np.isin(train, removed)]
                small_inner = fold_sets(pairs[retained], kind, config["seed"] + f + 1, config["inner_folds"])
                if len(small_inner) < 2 or any(not len(t) or not len(v) for t, v in small_inner):
                    removal_records[label] = {"status": "infeasible", "removed": removed.tolist()}
                    continue
                selection, _ = select(components, gram, retained, small_inner, config["alphas"], set(config["families"]) - {"shuffled_pair"})
                predictions[label], _, _ = predict_residual(components, selection, retained, val, residual)
                removal_records[label] = {"status": "complete", "removed": removed.tolist(), "selection": selection}
            # Fixed-selection branch subtraction retains the full mixed
            # kernel's scaling and alpha; retuned families remain separate.
            mixed = chosen["global_plus_pair"]
            mixed_kernel = components[("global_plus_pair", mixed["bandwidth"])]
            _, mixed_scale = normalized(mixed_kernel, train)
            for label, component in [("fixed_global_only", components[("global", 1.)]),
                                     ("fixed_pair_only", components[("pair_product", mixed["bandwidth"])])]:
                weights = smoother(component / mixed_scale, train, val, mixed["alpha"])
                predictions[label] = weights @ residual[train]
            metrics = {label: float(np.mean((pred - residual[val]) ** 2)) for label, pred in predictions.items()}
            for label, pred in predictions.items():
                totals[label] = totals.get(label, [0., 0])
                totals[label][0] += metrics[label] * len(val)
                totals[label][1] += len(val)
                accumulated.setdefault(label, np.zeros_like(residual))[val] += pred
            coverage[val] += 1
            records.append({"fold": f, "train": train.tolist(), "validation": val.tolist(),
                            "inner_splits": [{"train": train[t].tolist(), "validation": train[v].tolist()} for t, v in inner],
                            "selection": selected, "search": search, "metrics": metrics, "influences": influences,
                            "neighborhood_controls": removal_records,
                            "validation_target_training_degrees": [[int(np.sum(np.any(pairs[train] == target, axis=1))) for target in pair] for pair in pairs[val]]})
        mse = {label: total / count for label, (total, count) in totals.items()}
        averaged = {}
        for label, predictions in accumulated.items():
            # Infeasible removal folds do not share the primary coverage.
            counts = np.zeros(len(pairs), dtype=int)
            for record in records:
                if label in record["metrics"]:
                    counts[record["validation"]] += 1
            covered = counts > 0
            predictions[covered] /= counts[covered, None]
            averaged[label] = float(np.mean((predictions[covered] - residual[covered]) ** 2))
        np.savez_compressed(output / f"{kind}_predictions.npz", **accumulated, coverage=coverage, pair_indices=data["pair_indices"])
        output_results[kind] = {"mse_per_occurrence": mse, "mse_averaged_predictions": averaged, "covered_pairs": int((coverage > 0).sum()),
                                "records": records, "selected_gain_over_global_percent": 100 * (1 - mse["selected"] / mse["global"]) if mse["global"] else None,
                                "interpretation": "dependent development folds; fixed branch removal and retuned families answer different questions"}
        print(dataset.name, kind, mse, flush=True)
    output_results["seconds"] = time.perf_counter() - started
    write_json(output / "results.json", output_results)
    return output_results
