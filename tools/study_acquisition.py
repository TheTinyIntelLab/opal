"""Equal-budget retrospective label acquisition, with fitting-only tuning."""
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from opal.artifacts import load_npz, sha256, write_json
from opal.models.kernel import KernelResidual, single_kernel, symmetric_kernel
from tools.improve_residual_model import candidates, fold_sets, select


def choose_next(policy, pairs, acquired, available, geometry, kernel, alpha, rng):
    acquired, available = np.asarray(acquired, dtype=int), np.asarray(available, dtype=int)
    if not len(available) or set(acquired) & set(available):
        raise ValueError("Acquisition requires disjoint nonempty available identities")
    if policy == "uniform":
        return int(rng.choice(available))
    if policy == "target_balance":
        degree = np.bincount(pairs[acquired].ravel(), minlength=int(pairs.max()) + 1)
        scores = -degree[pairs[available]].sum(1)
    elif policy == "diversity":
        scores = np.diag(geometry)[available, None] + np.diag(geometry)[acquired][None] - 2 * geometry[np.ix_(available, acquired)]
        scores = np.maximum(scores, 0).min(1)
    elif policy == "kernel_variance":
        covariance = kernel[np.ix_(acquired, acquired)] + alpha * np.eye(len(acquired))
        cross = kernel[np.ix_(available, acquired)]
        scores = np.maximum(np.diag(kernel)[available] - np.sum(cross * solve(covariance, cross.T, assume_a="pos").T, axis=1), 0)
    else:
        raise ValueError("Unknown acquisition policy")
    ties = available[np.isclose(scores, scores.max(), rtol=1e-12, atol=1e-12)]
    return int(rng.choice(ties))


def study(dataset, output, config):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError("Acquisition outputs are immutable")
    output.mkdir(parents=True)
    data = load_npz(dataset / "profiles.npz")
    singles, pairs, responses = (data[k] for k in ["singles", "pairs", "doubles"])
    geometry = symmetric_kernel(single_kernel(singles), pairs, pairs)
    records = []
    write_json(output / "protocol.json", {**config, "profiles_sha256": sha256(dataset / "profiles.npz"),
               "evaluator_access": False, "hyperparameters": "selected on initial acquired labels only; fixed across policies and budgets",
               "kernel_variance": "conditional acquisition proxy, not calibrated predictive uncertainty"})
    for seed in config["seeds"]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pairs))
        nval = max(1, int(np.ceil(len(pairs) * config["validation_fraction"])))
        val, pool = order[:nval], order[nval:]
        if len(pool) < config["initial_labels"]:
            raise ValueError("Too few acquisition candidates")
        initial = pool[:config["initial_labels"]]
        observed = np.asarray(responses[initial], dtype=float) - singles[pairs[initial, 0]] - singles[pairs[initial, 1]]
        initial_configs = candidates(singles, pairs[initial])
        winner, search = select(initial_configs, observed @ observed.T / observed.shape[1], np.arange(len(initial)),
                                fold_sets(pairs[initial], "random", seed, 3))
        model_config = winner["config"]
        kernel = KernelResidual(singles, model_config).kernel(pairs, pairs)
        for policy in config["policies"]:
            acquired = initial.tolist()
            policy_rng = np.random.default_rng(seed)
            for budget in config["budgets"]:
                if budget > len(pool):
                    records.append({"seed": seed, "policy": policy, "budget": budget, "status": "infeasible", "available_pool": len(pool)})
                    continue
                while len(acquired) < budget:
                    available = pool[~np.isin(pool, acquired)]
                    chosen = choose_next(policy, pairs, acquired, available, geometry, kernel, model_config["alpha"], policy_rng)
                    acquired.append(chosen)
                if set(acquired) & set(val):
                    raise ValueError("Acquisition selected a held-out validation condition")
                indices = np.array(acquired)
                model = KernelResidual(singles, model_config).fit(pairs[indices], responses[indices])
                prediction = model.predict(pairs[val])
                stem = f"seed-{seed}_{policy}_{budget}"
                np.savez_compressed(output / f"{stem}.npz", predictions=prediction, validation_identities=data["pair_indices"][val])
                records.append({"seed": seed, "policy": policy, "budget": budget, "status": "complete", "initial": initial.tolist(),
                                "acquired": acquired.copy(), "validation": val.tolist(), "model_config": model_config,
                                "mse": float(np.mean((prediction - responses[val]) ** 2)),
                                "prediction_sha256": sha256(output / f"{stem}.npz")})
    summary = []
    for policy in config["policies"]:
        for budget in config["budgets"]:
            rows = [r for r in records if r["policy"] == policy and r["budget"] == budget and r["status"] == "complete"]
            if rows:
                summary.append({"policy": policy, "budget": budget, "mean_mse": float(np.mean([r["mse"] for r in rows])),
                                "dependent_seed_count": len(rows), "range_not_confidence_interval": [min(r["mse"] for r in rows), max(r["mse"] for r in rows)]})
    result = {"scope": config["scope"], "curves": summary, "records": records, "prospective_claim": False}
    write_json(output / "results.json", result)
    return result
