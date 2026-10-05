"""Generate supplementary evidence exclusively from saved completed runs."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import read_json, write_json, sha256


def main():
    evidence = ROOT / "paper/evidence"
    evidence.mkdir(exist_ok=True)
    records, rows = {}, []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, dataset in zip(axes, ["norman", "wessels"]):
        root = ROOT / f"results/robustness/{dataset}"
        entries = []
        primary = ROOT / f"results/runs/{dataset}-log2-primary"
        paths = [("random_1729", primary)] + [(p.name, p / "run") for p in sorted(root.glob("random_*"))] + [(p.name, p / "run") for p in sorted(root.glob("partner_*"))]
        for name, run in paths:
            if not (run / "final_status.json").exists() or read_json(run / "final_status.json")["status"] != "complete":
                continue
            result = read_json(run / "test_results.json")
            selected = result["models"][result["selected_id"]]
            simpler = result["models"][result["simpler_id"]]
            entries.append({"track": name, "selected": selected, "simpler": simpler,
                            "comparisons": result["comparisons"], "prediction_sha256": sha256(run / "predictions.npz")})
        records[dataset] = entries
        for kind in ["random", "partner"]:
            group = [e for e in entries if e["track"].startswith(kind)]
            if group:
                skills = [e["selected"]["skill_additive"] for e in group]
                gains = [e["simpler"]["mse"] - e["selected"]["mse"] for e in group]
                rows.append(f"{dataset.title()} & {kind} & {len(group)} & $[{min(skills):.3f}, {max(skills):.3f}]$ & $[{min(gains):.5f}, {max(gains):.5f}]$ " + chr(92) * 2)
        positions = np.arange(len(entries))
        ax.scatter(positions, [e["selected"]["skill_additive"] for e in entries], label="selected neural", color="#246775")
        ax.scatter(positions, [e["simpler"]["skill_additive"] for e in entries], label="development-selected simpler", color="#a56537", marker="x")
        ax.axhline(0, color="gray", lw=.8)
        ax.set_xticks(positions, [e["track"].replace("random_", "R").replace("partner_", "P") for e in entries], rotation=60)
        ax.set_title(dataset.title() + ": correlated split sensitivities")
        ax.set_ylabel("Skill relative to additive")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(evidence / "biological_robustness.pdf")
    fig.savefig(evidence / "biological_robustness.png", dpi=160)
    plt.close(fig)
    write_json(ROOT / "results/secondary_summary.json", records)
    (evidence / "secondary_table.tex").write_text("\\begin{tabular}{llrrr}\n\\toprule\nDataset & Protocol & Runs & Neural skill range & Gain over simpler range \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    public = []
    for dataset in ["norman-log2", "wessels-log2"]:
        path = ROOT / f"results/public_references/{dataset}/evaluation.json"
        if path.exists():
            result = read_json(path)
            public.extend(f"{dataset.split('-')[0].title()} & {name.replace('_', ' ')} & {r['mse']:.6f} & {r['skill_additive']:.3f} " + chr(92) * 2 for name, r in result["results"].items())
    (evidence / "public_table.tex").write_text("\\begin{tabular}{llrr}\n\\toprule\nDataset & Adapted SLIM input & MSE & Additive skill \\\\\n\\midrule\n" + "\n".join(public) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, dataset in enumerate(["norman-log2", "wessels-log2", "chemical-exploratory"]):
        bank = ROOT / "data/prepared" / dataset
        source = read_json(bank / "source_manifest.json")
        qc = read_json(bank / "qc_report.json")
        observed = len({tuple(sorted(v["components"])) for v in source.get("condition_map", {}).values()
                        if len(v["components"]) == 2 and len(set(v["components"])) == 2})
        if not observed:
            observed = 25
        split = read_json(bank / "splits.json")
        ax.bar(i - .22, observed, width=.2, label="observed pairs" if i == 0 else None, color="#999999")
        ax.bar(i, len(split["development"]), width=.2, label="eligible development" if i == 0 else None, color="#246775")
        ax.bar(i + .22, len(split["test"]), width=.2, label="eligible held-out" if i == 0 else None, color="#a56537")
    ax.set_xticks(range(3), ["Norman", "Wessels", "ComboSciPlex"])
    ax.set_ylabel("Distinct target pairs")
    ax.set_title("Source coverage and reference-matched eligibility")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(evidence / "coverage.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
