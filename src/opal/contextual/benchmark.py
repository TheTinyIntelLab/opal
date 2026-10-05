"""Selection on fitting contexts, isolated calibration, predictions before labels."""
from pathlib import Path
import time
import numpy as np
from opal.artifacts import environment, fingerprint, load_npz, sha256, source_hash, write_json
from .contracts import load_dataset
from .model import ContextualKernel
from .inference import ContextPredictor, export_checkpoint
from .reliability import empirical_radius, selective_risk


def default_candidates():
    candidates = [{"kind": "shared_offset", "alpha": 1.}]
    for kind in ["global", "pooled_pair", "contextual_pair", "pooled_sum", "contextual_sum"]:
        for alpha in [.01, 1., 100.]:
            for bandwidth in ([1.] if kind == "global" else [.25, 1., 4.]):
                candidates.append({"kind": kind, "alpha": alpha, "bandwidth": bandwidth,
                                   "context_bandwidth": 1., "context_weight": 1.})
    return candidates


def select(bank, data, candidates):
    """Only this fitting bundle can influence selection."""
    contexts = np.unique(data["contexts"])
    if len(contexts) < 2:
        raise ValueError("Context selection needs at least two fitting contexts")
    groups = {int(c): bank.metadata['context_records'][str(bank.arrays['contexts'][c])].get('selection_group', str(bank.arrays['contexts'][c])) for c in contexts}
    if len(set(groups.values())) < 2:
        raise ValueError('Selection needs at least two independent context groups')
    rows = []
    for config in candidates:
        total, count, folds = 0., 0, []
        for group in sorted(set(groups.values())):
            held = [c for c in contexts if groups[int(c)] == group]
            fit = np.flatnonzero(~np.isin(data['contexts'], held))
            val = np.flatnonzero(np.isin(data['contexts'], held))
            model = ContextualKernel(bank, config).fit(data["pairs"][fit], data["contexts"][fit], data["responses"][fit])
            prediction = model.predict(data["pairs"][val], data["contexts"][val])
            mse = float(np.mean((prediction - data["responses"][val]) ** 2))
            folds.append({'held_context_group': group, 'held_contexts': bank.arrays['contexts'][held].tolist(), "fit": fit.tolist(), "validation": val.tolist(), "mse": mse})
            total += mse * len(val)
            count += len(val)
        rows.append({"config": config, "mse": total / count, "folds": folds})
    winner = min(rows, key=lambda row: row["mse"])
    return winner, rows


def run_benchmark(dataset, output, candidates=None, *, frozen_config=None):
    dataset, output = Path(dataset), Path(output)
    bank, data, query, protocol = load_dataset(dataset)
    if output.exists():
        raise FileExistsError("Preserve completed or interrupted runs; choose a new output directory")
    output.mkdir(parents=True)
    started = time.perf_counter()
    configs = default_candidates() if candidates is None else list(candidates)
    if not configs:
        raise ValueError("Candidate set cannot be empty")
    write_json(output / "run_protocol.json", {"dataset_protocol_hash": sha256(dataset / "protocol.json"),
               "source_hash": source_hash(), "candidates": configs, "environment": environment(),
               "endpoint": "equal-condition all-gene MSE", "selection_unit": "held fitting context",
               "synthetic": bank.metadata["synthetic"], "reference_access": protocol["reference_access"],
               "frozen_config": frozen_config, "material_skill_threshold": .05 if frozen_config else None,
               "scope": "synthetic engineering" if bank.metadata["synthetic"] else "exploratory until independent exposure/replication review"})
    write_json(output / "status.json", {"status": "running", "evaluator_access": False})
    try:
        winner, rows = select(bank, data, configs)
        if frozen_config is not None:
            # An independent study must not replace the previously frozen
            # procedure with the winner of a new source-context search.
            winner = {"config": dict(frozen_config), "mse": None, "selection": "frozen_before_evaluator_access"}
        write_json(output / "selection.json", {"winner": winner, "search": rows, "calibration_outcomes_used": False,
                   "evaluator_outcomes_used": False, "selection_hash": fingerprint(winner)})
        # Report the best fitting-selected model from each family, not models
        # chosen by the eventual test scores.
        chosen = {kind: min((r for r in rows if r["config"]["kind"] == kind), key=lambda r: r["mse"])
                  for kind in sorted({r["config"]["kind"] for r in rows})}
        chosen["selected"] = winner
        models, predictions = {}, {"additive": bank.additive(query["pairs"], query["contexts"])}
        for label, selected in chosen.items():
            model = ContextualKernel(bank, selected["config"]).fit(data["pairs"], data["contexts"], data["responses"])
            models[label] = model
            predictions[label] = model.predict(query["pairs"], query["contexts"])
        np.savez_compressed(output / "predictions.npz", **predictions, identities=query["identities"])
        prediction_hash = sha256(output / "predictions.npz")
        # Calibration is opened only after selection and predictions are frozen.
        calibration = load_npz(dataset / "calibration" / "profiles.npz")
        if calibration["identities"].tolist() != protocol["calibration_identities"]:
            raise ValueError("Calibration identity order mismatch")
        if len(calibration["pairs"]):
            calibration_predictions = models["selected"].predict(calibration["pairs"], calibration["contexts"])
            radius = empirical_radius(calibration_predictions, calibration["responses"], calibration["pairs"])
        else:
            radius = {"status": "unavailable", "radius": None, "formal_guarantee": False}
        write_json(output / "calibration.json", radius)
        export_checkpoint(models["selected"], output / "checkpoint", radius,
                          {"dataset_protocol_sha256": sha256(dataset / "protocol.json"), "selection_sha256": sha256(output / "selection.json")})
        predictor = ContextPredictor(output / "checkpoint")
        errors = []
        for i, (pair, ci) in enumerate(zip(query["pairs"], query["contexts"])):
            predicted = predictor.predict(*bank.arrays["targets"][pair], context=str(bank.arrays["contexts"][ci]))["effect"]
            errors.append(float(np.max(np.abs(predicted - predictions["selected"][i]))))
        export_error = max(errors, default=0.)
        if export_error > 1e-8:
            raise ValueError("Contextual export round-trip mismatch")
        # Freeze a durable access record before opening held-context outcomes.
        if sha256(dataset / "evaluator" / "profiles.npz") != protocol["files"]["evaluator/profiles.npz"] or sha256(output / "predictions.npz") != prediction_hash:
            raise ValueError("Evaluator or prediction fingerprint changed")
        write_json(output / "test_access.json", {"predictions_sha256": prediction_hash,
                   "evaluator_sha256": protocol["files"]["evaluator/profiles.npz"], "selection_sha256": sha256(output / "selection.json"),
                   "predictions_and_selection_saved_before_access": True})
        evaluator = load_npz(dataset / "evaluator" / "profiles.npz")
        if not np.array_equal(evaluator["identities"], query["identities"]):
            raise ValueError("Evaluator identity order mismatch")
        y = evaluator["responses"]
        if y.shape != predictions["selected"].shape or not np.isfinite(y).all():
            raise ValueError("Invalid evaluator responses")
        mse = {label: float(np.mean((p - y) ** 2)) for label, p in predictions.items()}
        reference_label = min((label for label in chosen if label in {"shared_offset", "global", "pooled_pair", "pooled_sum"}),
                              key=lambda label: chosen[label]["mse"], default=None)
        condition_errors = np.mean((predictions["selected"] - y) ** 2, axis=1)
        distance = [predictor.predict(*bank.arrays["targets"][p], context=str(bank.arrays["contexts"][c]))["support"]["nearest_training_context_distance"]
                    for p, c in zip(query["pairs"], query["contexts"])]
        result = {"status": "complete", "synthetic": bank.metadata["synthetic"], "mse": mse,
                  "selected_config": winner["config"], "export_maximum_absolute_difference": export_error,
                  "frozen_procedure": frozen_config is not None,
                  "material_skill_threshold": .05 if frozen_config else None,
                  "development_selected_simple_reference": reference_label,
                  "skill_over_development_selected_reference": 1 - mse["selected"] / mse[reference_label] if reference_label and mse[reference_label] > 0 else None,
                  "empirical_radius_coverage": float(np.mean(np.sqrt(condition_errors) <= radius["radius"])) if radius.get("radius") is not None else None,
                  "selective_risk_diagnostic": selective_risk(condition_errors, distance),
                  "per_context_mse": {str(bank.arrays["contexts"][c]): float(condition_errors[query["contexts"] == c].mean()) for c in np.unique(query["contexts"])},
                  "condition_squared_errors": condition_errors.tolist(), "evaluation_identities": query["identities"].tolist(),
                  "seconds": time.perf_counter() - started, "formal_coverage_guarantee": False,
                  "capability_claim": "none; synthetic engineering" if bank.metadata["synthetic"] else "requires independent source and replication review"}
        write_json(output / "results.json", result)
        write_json(output / "status.json", {"status": "complete", "evaluator_access": True, "predictions_sha256": prediction_hash})
        return result
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
                   "test_access_record_exists": (output / "test_access.json").exists()})
        raise
