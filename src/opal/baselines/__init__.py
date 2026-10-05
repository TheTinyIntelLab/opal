"""Common-information classical controls."""
import numpy as np
from opal.representations import pair_features


def fit_baseline(config, embedding, pairs, doubles, singles):
    additive = singles[pairs[:, 0]] + singles[pairs[:, 1]]
    residual = doubles - additive
    family = config["family"]
    model = {"config": config, "embedding": embedding, "singles": singles}
    if family == "mean_response":
        model["mean"] = doubles.mean(0)
    elif family == "mean_residual":
        model["mean"] = residual.mean(0)
    elif family in ["ridge", "reduced_rank", "global", "nearest"]:
        features = additive if family == "global" else pair_features(embedding, pairs)
        if family == "global" and config.get("nonlinear", False):
            features = np.concatenate([features, np.tanh(features), features ** 2], axis=1)
        mean, scale = features.mean(0), np.maximum(features.std(0), 1e-6)
        z = (features - mean) / scale
        model.update({"feature_mean": mean, "feature_scale": scale})
        if family == "nearest":
            model.update({"train_features": z, "train_residual": residual})
        else:
            y = doubles if config.get("direct", False) else residual
            center = y.mean(0)
            alpha = config.get("alpha", 1.0)
            # Dual solve is faster and regularizes only slope, not shared offset.
            dual = np.linalg.solve(z @ z.T + alpha * np.eye(len(z)), y - center)
            if family == "reduced_rank":
                _, _, vt = np.linalg.svd(y - center, full_matrices=False)
                basis = vt[:config.get("rank", 4)].T
                dual = dual @ basis @ basis.T
            model.update({"train_features": z, "dual_coefficients": dual, "mean": center})
    elif family not in ["control", "additive"]:
        raise ValueError("Unknown baseline")
    return model


def predict_baseline(model, pairs):
    config = model["config"]
    family = config["family"]
    singles = model["singles"]
    additive = singles[pairs[:, 0]] + singles[pairs[:, 1]]
    if family == "control":
        return np.zeros_like(additive)
    if family == "additive":
        return additive
    if family == "mean_response":
        return np.broadcast_to(model["mean"], additive.shape).copy()
    if family == "mean_residual":
        return additive + model["mean"]
    features = additive if family == "global" else pair_features(model["embedding"], pairs)
    if family == "global" and config.get("nonlinear", False):
        features = np.concatenate([features, np.tanh(features), features ** 2], axis=1)
    z = (features - model["feature_mean"]) / model["feature_scale"]
    if family == "nearest":
        dist = ((z[:, None] - model["train_features"][None]) ** 2).mean(-1)
        idx = np.argsort(dist, axis=1, kind="stable")[:, :config.get("neighbors", 3)]
        correction = model["train_residual"][idx].mean(1)
    else:
        correction = (z @ model["train_features"].T) @ model["dual_coefficients"] + model["mean"]
    return correction if config.get("direct", False) else additive + correction
