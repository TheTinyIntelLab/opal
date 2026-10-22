"""Frozen synthetic cross-initiation probe; no biological input interface."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    import numpy as np
    import psutil
    import torch
    from threadpoolctl import threadpool_limits
    from opal.block_completion import ScalarCompletion, fit_reference_grid
    from opal.evaluation_contracts import metrics_v2

    freeze_path = ROOT / "results/research-continuation/replication-freeze-v1.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for name, expected in freeze["code_and_protocol_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen input drift: " + name)
    output = ROOT / "results/research-continuation/replication-v1"
    output.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    process = psutil.Process()
    torch.set_num_threads(1)
    binding = {"freeze_sha256": digest(freeze_path), "tool_sha256": digest(__file__),
               "command": sys.argv, "python": sys.version, "synthetic_only": True,
               "versions": {n: importlib.metadata.version(n) for n in
                            ("numpy", "torch", "scipy", "psutil", "threadpoolctl")}}
    write(output / "started.json", binding)

    def check():
        peak = max(process.memory_info().rss, getattr(process.memory_info(), "peak_wset", 0))
        size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        if time.monotonic() - started > 600 or peak > 2 * 1024**3 or size > 10 * 1024**2:
            raise RuntimeError("Registered resource cap")
        if shutil.disk_usage(output).free < 1024**3:
            raise RuntimeError("Unchanged disk reserve")

    worlds = ("additive", "biological_persistent", "technical_persistent",
              "technical_redrawn", "sparse_independent_edges")
    all_cells = []
    pairs = np.array([(i, j) for i in range(60) for j in range(i + 1, 60)], dtype=np.int64)
    try:
        with threadpool_limits(1):
            for seed in (6101, 6102, 6103):
                rng = np.random.default_rng(seed)
                singles = rng.normal(0, .2, 60)
                def factors():
                    f = rng.normal(size=(60, 2))
                    signal = np.prod(f[pairs, 0], axis=1) - np.prod(f[pairs, 1], axis=1)
                    return signal * .3 / np.sqrt(np.mean(signal**2))
                persistent, redrawn = factors(), factors()
                sparse = (rng.random(len(pairs)) < .1) * -rng.uniform(.2, .6, len(pairs))
                sparse *= .3 / np.sqrt(np.mean(sparse**2))
                order = rng.permutation(len(pairs))
                tr, va, te = order[:885], order[885:1239], order[1239:]
                if not np.all(np.bincount(pairs[tr].ravel(), minlength=60)):
                    raise RuntimeError("Missing TRAIN gene support")
                # Streams are shared across worlds and noise levels, without outcome selection.
                errors = [np.sqrt(.5) * rng.normal(size=(len(pairs), 1)) +
                          np.sqrt(.5) * rng.normal(size=(len(pairs), 4)) for _ in range(2)]
                single_error = rng.normal(size=60) / 2
                for noise in (0., .1):
                    measured_singles = singles + noise * single_error
                    for world in worlds:
                        check()
                        label = f"{seed}-{noise:g}-{world}"
                        if world == "additive":
                            signals = [np.zeros(len(pairs))] * 2
                        elif world == "technical_redrawn":
                            signals = [persistent, redrawn]
                        elif world == "sparse_independent_edges":
                            signals = [sparse] * 2
                        else:
                            signals = [persistent] * 2
                        truth = np.stack([singles[pairs].sum(1) + s for s in signals])
                        observed = truth + noise * np.stack([e.mean(1) for e in errors])
                        inputs = output / (label + "-inputs.npz")
                        np.savez_compressed(inputs, pairs=pairs, singles=measured_singles,
                                            true_singles=singles, truth=truth, observed=observed,
                                            train=tr, validation=va, test=te)
                        grid = fit_reference_grid(measured_singles, pairs[tr], observed[0, tr], check=check)
                        scores = [metrics_v2(observed[0, va], ref.predict(pairs[va]))["mse"] for ref in grid]
                        choice = min(range(len(grid)), key=lambda i: (scores[i], i))
                        reference = grid[choice]
                        reference_path = output / (label + "-reference.npz")
                        reference.save(reference_path)
                        residual = observed[0, tr] - reference.predict(pairs[tr])
                        scale = max(float(observed[0, tr].std()), 1e-6)
                        predictions = {"reference": reference.predict(pairs[te])}
                        selected = {"world": world, "seed": seed, "noise": noise,
                                    "input_sha256": digest(inputs), "reference_grid_validation_mse": scores,
                                    "reference_choice": choice, "reference_kind": reference.kind,
                                    "reference_alpha": reference.alpha, "reference_sha256": digest(reference_path),
                                    "checkpoints": {}, "exact_replays": {}}
                        for name, target in (("aligned", residual), ("shuffled", residual[
                                np.random.default_rng(seed + 100000).permutation(len(tr))])):
                            model = ScalarCompletion(rank=2, penalty=.0001, steps=400, seed=seed).fit(
                                pairs[tr], target, 60, scale,
                                (pairs[va], observed[0, va] - reference.predict(pairs[va])), check=check)
                            path = output / (label + "-" + name + ".npz")
                            model.save(path)
                            prediction = model.predict(pairs[te])
                            exact = np.array_equal(prediction, ScalarCompletion.load(path).predict(pairs[te]))
                            if not exact:
                                raise RuntimeError("Checkpoint replay mismatch")
                            selected["exact_replays"][name] = bool(exact)
                            selected["checkpoints"][name] = {"sha256": digest(path), "best_step": model.best_step}
                            predictions[name] = predictions["reference"] + prediction
                        # Ideal structure oracle, not a data-accessible candidate. An unseen IID
                        # edge uses its distribution mean; a new technical draw has mean zero.
                        oracle = np.stack([singles[pairs[te]].sum(1)] * 2)
                        if world == "sparse_independent_edges":
                            oracle += sparse.mean()
                        elif world in ("biological_persistent", "technical_persistent"):
                            oracle += persistent[te]
                        elif world == "technical_redrawn":
                            oracle[0] += persistent[te]
                        predictions["conditional_mean_oracle"] = oracle
                        pred_path = output / (label + "-predictions.npz")
                        np.savez_compressed(pred_path, **predictions)
                        selected["prediction_sha256"] = digest(pred_path)
                        write(output / (label + "-selection.json"), selected)
                        all_cells.append((label, selected))
            # Persist every candidate before the first TEST metric; scoring cannot alter selection.
            write(output / "all-predictions-frozen.json", {
                "cells": [label for label, _ in all_cells],
                "sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
                "test_metrics_computed": False})
            results = []
            for label, selected in all_cells:
                with np.load(output / (label + "-inputs.npz"), allow_pickle=False) as inputs:
                    latent, observed = inputs["truth"][:, inputs["test"]], inputs["observed"][:, inputs["test"]]
                with np.load(output / (label + "-predictions.npz"), allow_pickle=False) as predictions:
                    record = dict(selected)
                    record["metrics"] = {name: {kind: [metrics_v2(y[k], predictions[name][k] if
                                                   predictions[name].ndim == 2 else predictions[name])["mse"] for k in (0, 1)]
                                                for kind, y in (("observed", observed), ("noiseless", latent))}
                                         for name in predictions.files}
                results.append(record)
            observational_alias = all(
                np.array_equal(np.load(output / f"{s}-{n:g}-biological_persistent-predictions.npz")["aligned"],
                               np.load(output / f"{s}-{n:g}-technical_persistent-predictions.npz")["aligned"])
                for s in (6101, 6102, 6103) for n in (0., .1))
            noisy = [c for c in results if c["world"] == "biological_persistent" and c["noise"] == .1]
            gains = [1 - c["metrics"]["aligned"]["observed"][1] /
                     c["metrics"]["reference"]["observed"][1] for c in noisy]
            check()
            result = {"cells": results, "persistent_noisy_second_initiation_gains": gains,
                      "registered_synthetic_recovery_gate": bool(all(g >= .05 for g in gains)),
                      "biological_and_technical_predictions_exactly_identical": bool(observational_alias),
                      "biological_validation": False, "wall_seconds": time.monotonic() - started,
                      "cpu_seconds": time.process_time() - cpu,
                      "peak_bytes": max(process.memory_info().rss, process.memory_info().peak_wset)}
            write(output / "results.json", result)
            print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=2))
    except Exception as exc:
        write(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                       "completed_cells": [label for label, _ in all_cells]})
        raise


if __name__ == "__main__":
    main()
