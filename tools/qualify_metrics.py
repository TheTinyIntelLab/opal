"""Evaluator-only null/positive controls; no candidate fitting or selection."""
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import read_json, write_json, load_npz
from opal.evaluation import metrics


def main():
    with threadpool_limits(limits=1):
        for dataset in ["norman", "wessels"]:
            bank = ROOT / f"data/prepared/{dataset}-log2"
            run = ROOT / f"results/runs/{dataset}-log2-primary"
            data = load_npz(bank / "profiles.npz")
            y = load_npz(bank / "evaluator/profiles.npz")["doubles"]
            q = data["query_pairs"]
            additive = data["singles"][q[:, 0]] + data["singles"][q[:, 1]]
            train_additive = data["singles"][data["pairs"][:, 0]] + data["singles"][data["pairs"][:, 1]]
            offset = (data["doubles"] - train_additive).mean(0)
            controls = {"zero_effect": np.zeros_like(y), "additive": additive,
                        "development_mean_residual": additive + offset,
                        "oracle_response_positive_control_not_a_predictor": y}
            summaries = {name: metrics(pred, y, additive)[0] for name, pred in controls.items()}
            rng = np.random.default_rng(8411)
            sham = [metrics(additive + (y - additive)[rng.permutation(len(y))], y, additive)[0] for _ in range(100)]
            summaries["permuted_test_residual_diagnostic_not_a_predictor"] = {
                "draws": 100, "mse_mean": float(np.mean([r["mse"] for r in sham])),
                "response_pearson_mean": float(np.mean([r["pearson"] for r in sham])),
                "residual_cosine_mean": float(np.mean([r["cosine"] for r in sham]))}
            write_json(run / "metric_qualification.json", {"controls": summaries,
                "interpretation": "positive and shuffled evaluator-only controls qualify metric behavior, never compete in model selection",
                "residual_and_response_mse_max_difference": float(np.max(np.abs((additive - y) ** 2 - (-(y - additive)) ** 2)))})


if __name__ == "__main__":
    main()
