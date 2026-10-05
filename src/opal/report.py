"""Generate evidence tables and figures from saved experiments only."""
from pathlib import Path
import csv
import numpy as np
from opal.artifacts import read_json, write_json


def render_report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    summary = read_json(root / "summary.json")
    experiments = summary["experiments"]
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "selected_family", "test_mse", "additive_skill", "simpler_test_mse"])
        for e in experiments:
            w.writerow([e["scenario"], e["selected_family"], e["selected"]["mse"], e["selected"]["skill_additive"], e["simpler"]["mse"]])
    fig, ax = plt.subplots(figsize=(9, 4))
    skills = [e["selected"]["skill_additive"] for e in experiments]
    ax.bar(range(len(experiments)), [np.nan if v is None else v for v in skills], color="#25656e")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(range(len(experiments)), [e["scenario"].replace("_", "\n") for e in experiments], fontsize=8)
    ax.set_ylabel("Additive-relative skill on noisy held-out effects")
    ax.set_title("Synthetic checks, not biological validation")
    fig.tight_layout()
    fig.savefig(figures / "synthetic_skill.pdf")
    fig.savefig(figures / "synthetic_skill.png", dpi=180)
    plt.close(fig)
    rows = []
    for e in experiments:
        skill = e["selected"]["skill_additive"]
        rows.append(f"{e['scenario'].replace('_', ' ')} & {e['selected_family'].replace('_', ' ')} & {e['selected']['mse']:.5f} & {skill:.3f} & {e['simpler']['mse']:.5f} \\\\" if skill is not None else f"{e['scenario'].replace('_', ' ')} & {e['selected_family']} & {e['selected']['mse']:.5f} & undefined & {e['simpler']['mse']:.5f} \\\\")
    (root / "table.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"summary": str(root / "summary.csv"), "figure": str(figures / "synthetic_skill.pdf")}
