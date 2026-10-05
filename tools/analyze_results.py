"""Evidence figures and diagnostics from already frozen prediction artifacts."""
from pathlib import Path
import sys
import shutil
import time
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import read_json, write_json, load_npz
from opal.evaluation import geometry, cosine
from opal.inference import Predictor


def analyze(run, dataset):
    run, dataset = Path(run), Path(dataset)
    selection = read_json(run / "selection.json")
    results = read_json(run / "test_results.json")
    data = load_npz(dataset / "profiles.npz")
    test = load_npz(dataset / "evaluator/profiles.npz")["doubles"]
    arrays = load_npz(run / "predictions.npz")
    selected = arrays[results["selected_id"]].mean(0)
    pairs = data["query_pairs"]
    additive = data["singles"][pairs[:, 0]] + data["singles"][pairs[:, 1]]
    geom = geometry(data["singles"], pairs, test, selection["tau"])
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    records = selection["comparisons"]
    labels, scores = [], []
    for r in records:
        c = r["config"]
        suffix = ",".join(k for k in ["direct", "finetune", "fixed_basis", "unrestricted", "random_init", "shuffle"] if c.get(k))
        if c.get("representation", "pca") != "pca":
            suffix += c["representation"]
        labels.append(c["family"] + (" (" + suffix + ")" if suffix else ""))
        scores.append(results["models"][c["id"]]["mse"])
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(scores)), scores, color="#25656e")
    ax.set_yticks(range(len(scores)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Equal-pair held-out MSE, registered effect scale")
    ax.set_title(run.name + ": internal comparators")
    fig.tight_layout()
    fig.savefig(figures / "baselines.pdf")
    plt.close(fig)
    error = ((selected - test) ** 2).mean(1)
    base_error = ((additive - test) ** 2).mean(1)
    simple_pred = arrays[results["simpler_id"]].mean(0)
    simple_error = ((simple_pred - test) ** 2).mean(1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].scatter(geom["nonadditivity"], base_error - error, s=20, label="vs additive")
    axes[0].scatter(geom["nonadditivity"], simple_error - error, s=20, label="vs selected simpler", marker="x")
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set(xlabel="Normalized residual RMS", ylabel="Paired MSE improvement")
    axes[0].legend(fontsize=7)
    pred_res, true_res = selected - additive, test - additive
    axes[1].scatter(np.sqrt(np.mean(true_res ** 2, axis=1)), np.sqrt(np.mean(pred_res ** 2, axis=1)), s=20)
    bound = max(np.sqrt(np.mean(true_res ** 2, axis=1)).max(), np.sqrt(np.mean(pred_res ** 2, axis=1)).max())
    axes[1].plot([0, bound], [0, bound], color="black", linewidth=0.7)
    axes[1].set(xlabel="Observed residual RMS", ylabel="Predicted residual RMS")
    fig.tight_layout()
    fig.savefig(figures / "residual_geometry.pdf")
    plt.close(fig)
    diagnostics = {"test_pairs": len(test), "geometry_threshold": selection["strong_threshold"],
                   "strong_pairs": int(sum(geom["nonadditivity"] >= selection["strong_threshold"])),
                   "geometry_warning": "Additive error and observed residual energy are mathematically coupled"}
    distance = np.mean((selected[:, None] - test[None]) ** 2, axis=2)
    retrieved = np.argmin(distance, axis=1)
    diagnostics["retrieval"] = {"candidate_set": "all held-out pairs in this run", "candidates": len(test),
                                "accuracy": float(np.mean(retrieved == np.arange(len(test)))), "chance": 1 / len(test),
                                "limitation": "Similar conditions and unequal signal strength can make exact identification ambiguous"}
    spread = arrays[results["selected_id"]].std(0).mean(1)
    dev_oof = load_npz(run / f"finalist_oof_{selection['selected_id']}.npz")["predictions"]
    dev_pred = dev_oof.mean(0)
    dev_error = np.mean((dev_pred - data["doubles"]) ** 2, axis=1)
    dev_spread = dev_oof.std(0).mean(1)
    valid = np.isfinite(dev_error) & np.isfinite(dev_spread)
    diagnostics["spread_error"] = {"development_oof_pearson": float(np.corrcoef(dev_error[valid], dev_spread[valid])[0, 1]) if valid.sum() > 2 and dev_spread[valid].std() > 1e-12 else None,
                                    "test_pearson": float(np.corrcoef(error, spread)[0, 1]) if len(error) > 2 and spread.std() > 1e-12 else None,
                                    "calibrated_probability": False, "risk_threshold": None}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].scatter(dev_spread[valid], dev_error[valid], s=18)
    axes[0].set(title="Development OOF", xlabel="Member disagreement", ylabel="MSE")
    axes[1].scatter(spread, error, s=18)
    axes[1].set(title="Held out", xlabel="Member disagreement", ylabel="MSE")
    fig.tight_layout()
    fig.savefig(figures / "reliability.pdf")
    plt.close(fig)
    costs, devscores = [], []
    for r in records:
        m = results["models"][r["config"]["id"]]
        costs.append(m["parameters_per_member"])
        devscores.append(r["cv_mse"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(costs, devscores, s=25)
    ax.set(xscale="log", xlabel="Stored parameters / array entries per member", ylabel="Development CV MSE")
    fig.tight_layout()
    fig.savefig(figures / "capacity.pdf")
    plt.close(fig)
    curve = read_json(run / "sample_efficiency.json")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for seed in sorted(set(r["subset_order_seed"] for r in curve)):
        rows = [r for r in curve if r["subset_order_seed"] == seed]
        ax.plot([r["pairs"] for r in rows], [r["metrics"]["mse"] for r in rows], marker="o", label=str(seed))
    ax.axhline(base_error.mean(), color="black", linewidth=0.7, label="additive")
    ax.set(xlabel="Training pairs, all singles remain observed", ylabel="Held-out MSE", title="Full-development-fixed hyperparameters")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "sample_efficiency.pdf")
    plt.close(fig)
    uncertainty = dataset / "evaluator/profile_uncertainty.npz"
    if uncertainty.exists():
        halves = load_npz(uncertainty)
        residual = [halves[f"doubles_{i}"] - halves[f"singles_{i}"][pairs[:, 0]] - halves[f"singles_{i}"][pairs[:, 1]] for i in [0, 1]]
        alignment = cosine(*residual)
        diagnostics["split_half_reliability"] = {"residual_cosine": float(np.nanmean(alignment)), "defined_pairs": int(np.isfinite(alignment).sum()),
                                                "warning": "Technical halves are not biological replicates or an accuracy ceiling"}
        emergent = []
        for select_half, score_half in [(0, 1), (1, 0)]:
            idx = np.argsort(-np.abs(residual[select_half]), axis=1, kind="stable")[:, :20]
            row = np.arange(len(idx))[:, None]
            emergent.append(float(np.mean((selected[row, idx] - halves[f"doubles_{score_half}"][row, idx]) ** 2)))
        diagnostics["emergent_panel"] = {"swapped_half_mse": float(np.mean(emergent)), "genes_per_pair": min(20, test.shape[1]),
                                        "coverage": "primary universe only; extended-output training pending", "scope": "evaluator-only post-prediction diagnostic"}
        devhalves = load_npz(dataset / "profile_uncertainty.npz")
        dr = [devhalves[f"doubles_{i}"] - devhalves[f"singles_{i}"][data["pairs"][:, 0]] - devhalves[f"singles_{i}"][data["pairs"][:, 1]] for i in [0, 1]]
        spectrum = [np.linalg.svd(r - r.mean(0), compute_uv=False) for r in dr]
        diagnostics["development_residual_singular_values"] = [s.tolist() for s in spectrum]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for i, s in enumerate(spectrum):
            ax.plot(np.arange(1, len(s) + 1), s, label=f"measurement half {i+1}")
        ax.set(xlabel="Singular-value index", ylabel="Residual singular value", title="Development only; noise spectrum not yet adjusted")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "residual_spectrum.pdf")
        plt.close(fig)
    # Qualitative selection is an explicit best/median/worst rule.
    ordered = np.argsort(error, kind="stable")
    examples = [int(ordered[0]), int(ordered[len(ordered) // 2]), int(ordered[-1])]
    diagnostics["examples"] = [{"rule": rule, "pair": data["targets"][pairs[i]].tolist(), "mse": float(error[i])} for rule, i in zip(["best", "median", "worst"], examples)]
    model = Predictor(run / "checkpoint")
    a, b = data["targets"][pairs[0]]
    samples = []
    for _ in range(100):
        started = time.perf_counter()
        model.predict(a, b)
        samples.append(time.perf_counter() - started)
    timing_path = run / "inference_timing.json"
    diagnostics["inference"] = read_json(timing_path) if timing_path.exists() else {"cold_load_seconds": model.cold_seconds, "warm_single_median_seconds": float(np.median(samples)),
                                "warm_single_p95_seconds": float(np.quantile(samples, 0.95)), "iterations": 100, "threads": 1,
                                "checkpoint_bytes": sum(p.stat().st_size for p in (run / "checkpoint").iterdir()),
                                "peak_native_memory": "not measured"}
    if not timing_path.exists():
        write_json(timing_path, diagnostics["inference"])
    write_json(run / "diagnostics.json", diagnostics)
    return diagnostics


def manuscript_assets():
    evidence = ROOT / "paper/evidence"
    evidence.mkdir(exist_ok=True)
    synthetic = ROOT / "results/synthetic"
    for source, dest in [(synthetic / "table.tex", evidence / "table.tex"),
                         (synthetic / "figures/synthetic_skill.pdf", evidence / "synthetic_skill.pdf")]:
        if source.exists():
            shutil.copyfile(source, dest)
    synthetic_rows = (evidence / "table.tex").read_text(encoding="utf-8")
    (evidence / "table.tex").write_text("\\begin{tabular}{llrrr}\n\\toprule\nScenario & Selected family & MSE & Skill & Simpler MSE \\\\\n\\midrule\n" + synthetic_rows + "\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    rows, records = [], []
    for name in ["norman-log2-primary", "wessels-log2-primary", "norman-count-sensitivity", "wessels-count-sensitivity", "chemical-exploratory"]:
        run = ROOT / "results/runs" / name
        if not (run / "test_results.json").exists():
            continue
        selection, results = read_json(run / "selection.json"), read_json(run / "test_results.json")
        selected = results["models"][results["selected_id"]]
        simple = results["models"][results["simpler_id"]]
        config = next(r["config"] for r in selection["comparisons"] if r["config"]["id"] == results["selected_id"])
        additive_id = next(r["config"]["id"] for r in selection["comparisons"] if r["config"]["family"] == "additive")
        records.append({"run": name, "selected_config": config, "selected": selected, "simpler": simple, "comparisons": results["comparisons"]})
        labels = {"norman-log2-primary": "Norman (log)", "wessels-log2-primary": "Wessels (log)",
                  "norman-count-sensitivity": "Norman (count)", "wessels-count-sensitivity": "Wessels (count)", "chemical-exploratory": "Chemical (exploratory)"}
        rows.append(f"{labels[name]} & {config['family'].replace('_', ' ')} & {results['models'][additive_id]['mse']:.5f} & {selected['mse']:.5f} & {simple['mse']:.5f} & {selected['skill_additive']:.3f} \\\\")
        if (run / "figures/baselines.pdf").exists():
            shutil.copyfile(run / "figures/baselines.pdf", evidence / f"{name}-baselines.pdf")
        if name in ["norman-log2-primary", "wessels-log2-primary"]:
            for diagnostic in ["capacity", "reliability", "residual_geometry", "residual_spectrum", "sample_efficiency"]:
                source = run / f"figures/{diagnostic}.pdf"
                if source.exists():
                    shutil.copyfile(source, evidence / f"{name}-{diagnostic}.pdf")
    (evidence / "biological_table.tex").write_text("\\begin{tabular}{llrrrr}\n\\toprule\nDataset/scale & Neural family & Additive MSE & Neural MSE & Simpler MSE & Skill \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    write_json(ROOT / "results/research_summary.json", records)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    for name, dataset in [("norman-log2-primary", "norman-log2"), ("wessels-log2-primary", "wessels-log2"),
                          ("norman-count-sensitivity", "norman-count"), ("wessels-count-sensitivity", "wessels-count"),
                          ("chemical-exploratory", "chemical-exploratory")]:
        run = ROOT / "results/runs" / name
        if (run / "final_status.json").exists():
            print(f"Analyzing {name}", flush=True)
            analyze(run, ROOT / "data/prepared" / dataset)
    manuscript_assets()
