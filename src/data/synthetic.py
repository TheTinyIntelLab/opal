"""Known effect generators. Synthetic output is never biological evidence."""
from itertools import combinations
import numpy as np
from opal.artifacts import write_json, array_hash, sha256
from opal.splits import random_pair_split, partner_splits
from pathlib import Path

SCENARIOS = ["additive_log", "additive_count", "low_rank", "cross_coordinate", "high_rank",
             "global_distortion", "emergent", "noisy_controls"]


def generate(scenario="low_rank", seed=23, targets=24, genes=96, latent=8, rank=4, noise=0.015):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown synthetic scenario: {scenario}")
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(targets, latent))
    basis = np.linalg.qr(rng.normal(size=(genes, latent)))[0]
    singles = 0.6 * e @ basis.T
    pairs = np.array(list(combinations(range(targets), 2)), dtype=np.int64)
    additive = singles[pairs[:, 0]] + singles[pairs[:, 1]]
    control = np.full(genes, 6.0)
    if scenario == "additive_log":
        residual = np.zeros_like(additive)
    elif scenario == "additive_count":
        mu0 = np.exp2(control) - 1
        ma = np.exp2(control + singles[pairs[:, 0]]) - 1
        mb = np.exp2(control + singles[pairs[:, 1]]) - 1
        if np.any(ma + mb - mu0 <= 0):
            raise ValueError("Synthetic count-additive mean must be positive")
        residual = np.log2(ma + mb - mu0 + 1) - control - additive
    elif scenario == "high_rank":
        q = e[pairs[:, 0], :, None] * e[pairs[:, 1], None, :]
        q = (q + q.transpose(0, 2, 1)).reshape(len(pairs), -1)
        residual = 0.1 * q @ rng.normal(size=(latent * latent, genes)) / latent
    elif scenario == "global_distortion":
        residual = 0.3 * np.tanh(3 * additive) - 0.1 * additive
    else:
        projection = np.eye(latent) if scenario in ("low_rank", "noisy_controls") else rng.normal(size=(latent, latent)) / np.sqrt(latent)
        q = (e[pairs[:, 0]] @ projection) * (e[pairs[:, 1]] @ projection)
        b = np.linalg.qr(rng.normal(size=(genes, rank)))[0]
        residual = 0.4 * q @ rng.normal(size=(latent, rank)) @ b.T / np.sqrt(latent)
        if scenario == "emergent":
            singles[:, -genes // 4:] = 0
            additive = singles[pairs[:, 0]] + singles[pairs[:, 1]]
            residual[:, :-genes // 4] = 0
    truth = additive + residual
    observed_singles = singles.copy()
    if scenario == "noisy_controls":
        shared_control_error = rng.normal(0, 0.04, genes)
        observed_singles += rng.normal(0, 0.04, singles.shape) - shared_control_error
        observed = truth + rng.normal(0, noise, truth.shape) - shared_control_error
    else:
        observed = truth + rng.normal(0, noise, truth.shape)
    return {"singles": observed_singles, "doubles": observed, "pairs": pairs,
            "truth": truth, "true_residual": residual, "control": control,
            "targets": np.array([f"T{i:03d}" for i in range(targets)]),
            "genes": np.array([f"G{i:04d}" for i in range(genes)])}


def freeze_evaluator(path, indices):
    path = Path(path)
    write_json(path / "evaluation_manifest.json", {"schema": 1,
        "profiles_sha256": sha256(path / "evaluator/profiles.npz"),
        "pair_indices": np.asarray(indices).tolist(),
        "scope": "opaque evaluator identity; no response values exposed to fitting"})


def save_prepared(path, arrays, manifest, seed=1729, folds=5, split_override=None):
    """Physically separate held-out responses from the fitting bundle."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    pair_ids = arrays["targets"][arrays["pairs"]]
    split = random_pair_split(pair_ids, seed, folds=folds) if split_override is None else split_override
    dev_set, test_set = set(split["development"]), set(split["test"])
    if dev_set & test_set or dev_set | test_set != set(range(len(pair_ids))):
        raise ValueError("Split override must partition all pair indices")
    dev, test = np.array(split["development"]), np.array(split["test"])
    extras = {k: arrays[k] for k in ["control_sd"] if k in arrays}
    np.savez_compressed(path / "profiles.npz", singles=arrays["singles"], pairs=arrays["pairs"][dev],
                        doubles=arrays["doubles"][dev], pair_indices=dev,
                        query_pairs=arrays["pairs"][test], query_indices=test,
                        control=arrays["control"], targets=arrays["targets"], genes=arrays["genes"], **extras)
    evaluator = path / "evaluator"
    evaluator.mkdir(exist_ok=True)
    np.savez_compressed(evaluator / "profiles.npz", doubles=arrays["doubles"][test],
                        pair_indices=test, **({"truth": arrays["truth"][test]} if "truth" in arrays else {}))
    freeze_evaluator(path, test)
    write_json(path / "splits.json", split)
    write_json(path / "partner_splits.json", partner_splits(pair_ids))
    write_json(path / "source_manifest.json", manifest)
    write_json(path / "gene_schema.json", {"genes": arrays["genes"].tolist(), "hash": array_hash(arrays["genes"])})
    write_json(path / "fit_access_log.json", {"resources": ["controls", "all eligible singles"],
               "development_pair_indices": dev.tolist(), "test_double_expression_consumed": False})
    return split
