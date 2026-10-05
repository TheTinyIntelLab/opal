"""Development selection is separate from the single frozen final evaluation."""
from pathlib import Path
import time
import csv
import numpy as np
from opal.artifacts import read_json, write_json, load_npz, fingerprint, array_hash, source_hash, environment, sha256
from opal.training import fit, predict, NEURAL
from opal.inference import export_members
from opal.evaluation import metrics, paired_interval, component_panel, geometry


def configurations(protocol):
    """Bounded registered search, not a full Cartesian product."""
    if protocol.get('selection_scope', 'neural') not in ['neural', 'competitive']:
        raise ValueError('selection_scope must be neural or competitive')
    common = {"embedding": protocol.get("embedding", 16)}
    configs = [{**common, "family": f} for f in ["control", "additive", "mean_response", "mean_residual"]]
    for alpha in protocol.get("ridge_alpha", [1, 100]):
        for family in ["ridge", "reduced_rank", "global"]:
            configs.append({**common, "family": family, "alpha": alpha, "rank": 4})
    configs.append({**common, "family": "global", "alpha": 100, "nonlinear": True})
    configs.append({**common, "family": "nearest", "neighbors": 3})
    neural = {**common, "epochs": protocol.get("epochs", 120), "lr": 0.01, "weight_decay": 0.01,
              "rank": 4, "features": 16, "width": 64, "depth": 1, "offset": True}
    for family in ["coordinate", "bilinear", "independent", "network", "sum_network"]:
        for rank in protocol.get("ranks", [2, 8]):
            configs.append({**neural, "family": family, "rank": rank})
    if protocol.get("ablations", True):
        for amendment in [{"direct": True}, {"finetune": True}, {"fixed_basis": True},
                          {"unrestricted": True}, {"offset": False}, {"random_init": True},
                          {"representation": "identity"}, {"representation": "encoder"}, {"shuffle": True}]:
            configs.append({**neural, "family": "bilinear", **amendment})
    for config in protocol.get('kernel_configs', []):
        configs.append({**config, 'family': 'kernel'})
    for c in configs:
        c["id"] = fingerprint(c)[:12]
    return configs


def cv_candidate(config, data, folds, seeds):
    pairs, doubles, singles = data["pairs"], data["doubles"], data["singles"]
    errors, epochs, oof = [], [], []
    for seed in seeds:
        prediction = np.zeros_like(doubles)
        coverage = np.zeros(len(doubles), int)
        fold_sets = [(np.array(f["development"]), np.array(f["test"])) for f in folds if f["feasible"]] if isinstance(folds, list) else [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in sorted(set(folds))]
        if len(fold_sets) < 2:
            raise ValueError("Fewer than two feasible inner holdouts; register infeasibility rather than redraw")
        for train, val in fold_sets:
            y = doubles[train].copy()
            if config.get("shuffle", False):
                rng = np.random.default_rng(seed)
                additive = singles[pairs[train, 0]] + singles[pairs[train, 1]]
                y = additive + (y - additive)[rng.permutation(len(y))]
            fitted = fit(config, singles, pairs[train], y, seed, (pairs[val], doubles[val]), data.get("single_partitions"))
            prediction[val] += predict(fitted, pairs[val])
            coverage[val] += 1
            epochs.append(fitted["epochs"])
        eligible = coverage > 0
        prediction[eligible] /= coverage[eligible, None]
        prediction[~eligible] = np.nan
        oof.append(prediction)
        errors.append(float(np.mean((prediction[eligible] - doubles[eligible]) ** 2)))
    return {"config": config, "cv_mse": float(np.mean(errors)), "seed_mse": errors,
            "refit_epochs": max(1, int(np.median(epochs))), "oof": np.stack(oof), "oof_covered_pairs": int(eligible.sum())}


def dataset_fingerprint(path):
    path = Path(path)
    files = ["profiles.npz", "source_manifest.json", "splits.json", "gene_schema.json"]
    files += [f for f in ["profile_uncertainty.npz", "cv_splits.json"] if (path / f).exists()]
    return fingerprint({f: sha256(path / f) for f in files})


def fitting_data(dataset):
    data = load_npz(Path(dataset) / "profiles.npz")
    uncertainty = Path(dataset) / "profile_uncertainty.npz"
    if uncertainty.exists():
        halves = load_npz(uncertainty)
        data["single_partitions"] = (halves["singles_0"], halves["singles_1"])
    return data


def development(dataset, output, protocol):
    dataset, output = Path(dataset), Path(output)
    if (output / "selection.json").exists():
        raise FileExistsError("A frozen development run exists; use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    evaluator_sha = sha256(dataset / "evaluator/profiles.npz")
    evaluator_manifest = dataset / "evaluation_manifest.json"
    if evaluator_manifest.exists() and read_json(evaluator_manifest)["profiles_sha256"] != evaluator_sha:
        raise ValueError("Evaluator changed since preparation")
    run_id = fingerprint({"data": dataset_fingerprint(dataset), "protocol": protocol, "code": source_hash()})
    write_json(output / "status.json", {"stage": "development", "status": "running", "run_id": run_id})
    started = time.perf_counter()
    data = fitting_data(dataset)
    split = read_json(dataset / "splits.json")
    folds = np.array([split["fold"][str(i)] for i in data["pair_indices"]])
    if (dataset / "cv_splits.json").exists():
        folds = read_json(dataset / "cv_splits.json")
    configs = configurations(protocol)
    write_json(output / "protocol.json", {**protocol, "configurations": configs, "endpoint": "equal-pair mean MSE", "minimum_skill": 0.05,
               "budget": {"maximum_configurations": len(configs), "threads": 1, "maximum_seconds": protocol.get("maximum_seconds", 7200)}})
    write_json(output / "environment.json", environment())
    import zipfile
    source_root = Path(__file__).parents[1]
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(source_root.rglob("*.py")):
            archive.write(file, "opal/" + str(file.relative_to(source_root)).replace("\\", "/"))
    screen = []
    try:
        for i, config in enumerate(configs):
            if time.perf_counter() - started > protocol.get("maximum_seconds", 7200):
                raise TimeoutError("Registered development wall-clock budget exhausted")
            record = cv_candidate(config, data, folds, protocol.get("screening_seeds", [11]))
            screen.append(record)
            np.savez_compressed(output / f"oof_{config['id']}.npz", predictions=record["oof"], pair_indices=data["pair_indices"])
            print(f"Development {i+1}/{len(configs)} {config['family']} MSE={record['cv_mse']:.6f}", flush=True)
        # Finalists are chosen separately within comparison families and ablations.
        finalists = []
        groups = {}
        for r in screen:
            c = r["config"]
            group = c["family"] + "/" + ",".join(k for k in ["direct", "finetune", "fixed_basis", "unrestricted", "random_init", "shuffle"] if c.get(k)) + "/" + c.get("representation", "pca") + "/" + str(c.get("offset", True))
            groups.setdefault(group, []).append(r)
        seeds = protocol.get("finalist_seeds", [11, 29, 47, 71, 101])
        for group, records in groups.items():
            for record in sorted(records, key=lambda r: r["cv_mse"])[:protocol.get("finalists_per_family", 2)]:
                if time.perf_counter() - started > protocol.get("maximum_seconds", 7200):
                    raise TimeoutError("Registered finalist wall-clock budget exhausted")
                # Closed-form estimates have no optimizer randomness.
                r = cv_candidate(record["config"], data, folds, seeds if record["config"]["family"] in NEURAL else seeds[:1])
                r["group"] = group
                finalists.append(r)
                np.savez_compressed(output / f"finalist_oof_{r['config']['id']}.npz", predictions=r["oof"], pair_indices=data["pair_indices"])
        eligible_families = NEURAL if protocol.get('selection_scope', 'neural') == 'neural' else {r['config']['family'] for r in finalists}
        candidates = [r for r in finalists if r["config"]["family"] in eligible_families and not any(r["config"].get(k) for k in ["direct", "shuffle"]) and r["config"].get("representation", "pca") != "identity"]
        selected = min(candidates, key=lambda r: r["cv_mse"])
        simpler = min([r for r in finalists if r["config"]["family"] in ["mean_residual", "global", "ridge", "reduced_rank", "nearest"]], key=lambda r: r["cv_mse"])
        # Save per-family and ablation selections, preserving all registered comparisons.
        champions = [min([r for r in finalists if r["group"] == g], key=lambda r: r["cv_mse"]) for g in groups]
        records = [{k: v for k, v in r.items() if k != "oof"} for r in champions]
        all_records = [{k: v for k, v in r.items() if k != "oof"} for r in finalists]
        write_json(output / "development_results.json", all_records)
        additive = data["singles"][data["pairs"][:, 0]] + data["singles"][data["pairs"][:, 1]]
        dev_scale = np.sqrt(np.mean(additive ** 2, axis=1))
        tau = max(1e-8, 0.01 * float(np.median(dev_scale)))
        residual_mse = np.mean((data["doubles"] - additive) ** 2, axis=1)
        write_json(output / "precision_plan.json", {"development_additive_mean_mse": float(residual_mse.mean()),
                   "development_pair_error_sd": float(residual_mse.std(ddof=1)), "planned_test_pairs": len(data["query_pairs"]),
                   "descriptive_error_mean_95_halfwidth": float(1.96 * residual_mse.std(ddof=1) / max(1, len(data["query_pairs"])) ** 0.5),
                   "caveat": "Planning approximation ignores pair-network dependence; not a power guarantee"})
        selection = {"schema": 2, "run_id": run_id, "data_fingerprint": dataset_fingerprint(dataset), "source_hash": source_hash(),
                     "evaluator_sha256": evaluator_sha,
                     "evaluator_freeze_origin": "preparation manifest" if evaluator_manifest.exists() else "development snapshot of legacy bundle",
                     "protocol_fingerprint": fingerprint(protocol), "selected_id": selected["config"]["id"],
                     "simpler_id": simpler["config"]["id"], "comparisons": records,
                     "ensemble_seeds": protocol.get("ensemble_seeds", [11, 29, 47, 71, 101]), "tau": tau,
                     "strong_threshold": float(np.quantile(geometry(data["singles"], data["pairs"], data["doubles"], tau)["nonadditivity"], 0.75)),
                     "test_opened": False, "fit_access": {"development_pair_indices": data["pair_indices"].tolist(), "test_responses": False}}
        write_json(output / "selection.json", selection)
        write_json(output / "status.json", {"stage": "development", "status": "complete", "run_id": run_id, "seconds": time.perf_counter() - started})
        return selection
    except Exception as e:
        write_json(output / "status.json", {"stage": "development", "status": "failed", "error": str(e), "run_id": run_id})
        raise


def final(dataset, output):
    output = Path(output)
    if (output / "test_access.json").exists():
        raise FileExistsError("Test already accessed; preserve this run")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "final_fit_lock.json").open("x", encoding="utf-8") as lock:
        lock.write('{"scope":"exclusive final fitting; interrupted runs require a documented new run"}')
    write_json(output / "final_status.json", {"status": "running"})
    try:
        return _final_impl(dataset, output)
    except Exception as error:
        write_json(output / "final_status.json", {"status": "failed", "error": str(error),
            "test_access_recorded": (output / "test_access.json").exists()})
        raise


def _final_impl(dataset, output):
    dataset, output = Path(dataset), Path(output)
    selection = read_json(output / "selection.json")
    if (output / "test_access.json").exists():
        raise FileExistsError("Test already accessed for this run; preserve it and register any corrected rerun")
    if selection["source_hash"] != source_hash() or selection["data_fingerprint"] != dataset_fingerprint(dataset):
        raise ValueError("Frozen source or fitting data changed; create a documented new development run")
    evaluator_path = dataset / "evaluator/profiles.npz"
    if "evaluator_sha256" not in selection:
        raise ValueError("Evaluator identity missing from historical selection; restore its saved source snapshot")
    if sha256(evaluator_path) != selection["evaluator_sha256"]:
        raise ValueError("Evaluator checksum changed since development")
    data = fitting_data(dataset)
    query = data["query_pairs"]
    predictions, fitted_models, timing, counts = {}, {}, {}, {}
    started = time.perf_counter()
    for record in selection["comparisons"]:
        config = record["config"].copy()
        config["epochs"] = record["refit_epochs"]
        members, pred = [], []
        start = time.perf_counter()
        for seed in selection["ensemble_seeds"] if config["family"] in NEURAL else selection["ensemble_seeds"][:1]:
            y = data["doubles"].copy()
            if config.get("shuffle", False):
                additive = data["singles"][data["pairs"][:, 0]] + data["singles"][data["pairs"][:, 1]]
                y = additive + (y - additive)[np.random.default_rng(seed).permutation(len(y))]
            member = fit(config, data["singles"], data["pairs"], y, seed, single_partitions=data.get("single_partitions"))
            members.append(member)
            pred.append(predict(member, query))
        predictions[config["id"]] = np.stack(pred)
        fitted_models[config["id"]] = members
        timing[config["id"]] = time.perf_counter() - start
        m = members[0]
        counts[config["id"]] = {"members": len(members), "parameters_per_member": m.get("parameters", sum(v.size for v in m["model"].values() if isinstance(v, np.ndarray)) if m["type"] == "baseline" else 0),
                                "trainable_parameters_per_member": m.get("trainable_parameters")}
    np.savez_compressed(output / "predictions.npz", **predictions, pair_indices=data["query_indices"])
    from opal.training.studies import sample_predictions
    sample_pred, sample_records = sample_predictions(selection, data)
    np.savez_compressed(output / "sample_predictions.npz", **sample_pred)
    write_json(output / "sample_protocol.json", sample_records)
    # Prediction arrays and lock exist before the evaluator is read.
    write_json(output / "test_access.json", {"selection_sha256": sha256(output / "selection.json"),
               "evaluator_sha256": selection["evaluator_sha256"],
               "predictions_sha256": sha256(output / "predictions.npz"), "test_pair_indices": data["query_indices"].tolist(),
               "evaluation_batch": "registered internal comparisons", "first_access_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()})
    if sha256(evaluator_path) != selection["evaluator_sha256"]:
        raise ValueError("Evaluator changed during final fitting")
    evaluated = load_npz(evaluator_path)
    if not np.array_equal(evaluated["pair_indices"], data["query_indices"]):
        raise ValueError("Evaluator pair order differs from frozen queries")
    target = evaluated["doubles"]
    if target.shape != (len(query), len(data["genes"])) or not np.isfinite(target).all():
        raise ValueError("Evaluator shape or finite-value contract failed")
    additive = data["singles"][query[:, 0]] + data["singles"][query[:, 1]]
    for record in sample_records:
        record["metrics"] = metrics(sample_pred[record["key"]], target, additive)[0]
    write_json(output / "sample_efficiency.json", sample_records)
    panels = component_panel(data["singles"], query, data["genes"], control_sd=data.get("control_sd"))
    summaries, errors = {}, {}
    for cid, ensemble in predictions.items():
        p = ensemble.mean(0)
        summary, error = metrics(p, target, additive)
        summary["component_panel_mse"] = float(np.mean([np.mean((p[i, panel] - target[i, panel]) ** 2) for i, panel in enumerate(panels)]))
        summary["component_panel_rule"] = "single effect / (control SD + 0.1), stable gene-ID ties" if "control_sd" in data else "absolute single effect, stable ID ties"
        summary["single_member_mse"] = float(np.mean((ensemble[0] - target) ** 2))
        summary["ensemble_spread_mean"] = float(ensemble.std(0).mean())
        summary.update(counts[cid])
        summary["fit_seconds"] = timing[cid]
        summaries[cid], errors[cid] = summary, error
    selected = selection["selected_id"]
    additive_id = next(r["config"]["id"] for r in selection["comparisons"] if r["config"]["family"] == "additive")
    comparisons = {"additive": paired_interval(errors[additive_id] - errors[selected], query),
                   "development_selected_simpler": paired_interval(errors[selection["simpler_id"]] - errors[selected], query)}
    # No significance claim: these descriptive bootstraps omit measurement uncertainty.
    conclusion = "synthetic engineering evidence only" if read_json(dataset / "source_manifest.json").get("synthetic") else "internal benchmark; complete v1 inference requires public methods and measurement uncertainty"
    write_json(output / "test_results.json", {"models": summaries, "comparisons": comparisons, "selected_id": selected,
               "simpler_id": selection["simpler_id"], "conclusion_scope": conclusion})
    with (output / "per_condition.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "pair_index", "target_a", "target_b", "mse"])
        for cid, err in errors.items():
            for i, value in enumerate(err):
                writer.writerow([cid, int(data["query_indices"][i]), *data["targets"][query[i]], float(value)])
    source = read_json(dataset / "source_manifest.json")
    export_members(fitted_models[selected], data, output / "checkpoint", {"context": source.get("context", "synthetic"),
                   "scale": source.get("scale", "log2"), "synthetic": source.get("synthetic", False),
                   "data_fingerprint": selection["data_fingerprint"], "research_checkpoint": True,
                   "source_hash": selection["source_hash"], "test_results": "../test_results.json"})
    write_json(output / "final_status.json", {"status": "complete", "seconds": time.perf_counter() - started,
               "predictions_sha256": sha256(output / "predictions.npz")})
    return read_json(output / "test_results.json")
