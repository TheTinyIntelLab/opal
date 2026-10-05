"""One-command orchestration with verified completed stages and preserved runs."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import read_json, write_json, source_hash, sha256
from opal.training.benchmark import dataset_fingerprint


def run(*arguments):
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Run all registered internal Opal experiments")
    p.add_argument("--config", default="configs/experiments/research.json")
    p.add_argument("--skip-synthetic", action="store_true")
    p.add_argument("--development-only", action="store_true")
    p.add_argument("--skip-secondary", action="store_true", help="Skip biological robustness and supplementary diagnostics")
    p.add_argument("--skip-architecture", action="store_true", help="Skip the post-primary architecture amendment")
    p.add_argument("--skip-paper", action="store_true", help="Keep TeX source without requiring a local compiler")
    p.add_argument("--tectonic", help="Existing Tectonic executable for the paper build")
    args = p.parse_args(argv)
    protocol = read_json(ROOT / args.config)
    log = {"config": args.config, "scope": protocol["scope"], "stages": [], "status": "running"}
    write_json(ROOT / "results/pipeline_status.json", log)
    try:
        if not args.skip_synthetic:
            synthetic = read_json(ROOT / protocol["synthetic"])
            root = ROOT / synthetic["output"]
            if (root / "summary.json").exists() and (not synthetic.get("secondary_studies") or (root / "secondary/summary.json").exists()):
                log["stages"].append({"synthetic": "existing completed evidence", "summary_sha256": sha256(root / "summary.json")})
            else:
                run("main.py", "reproduce", "--config", protocol["synthetic"])
        for name in protocol["datasets"]:
            dataset = ROOT / "data/prepared" / name
            if not (dataset / "profiles.npz").exists():
                if name == "chemical-exploratory":
                    for resource, path in [("chemical-original", "data/raw/GSE206741_count_matrix.mtx.gz"),
                                           ("chemical-cells", "data/raw/source_metadata/combosciplex_cell_metadata.tsv.gz"),
                                           ("chemical-genes", "data/raw/source_metadata/combosciplex_gene_metadata.tsv.gz")]:
                        if not (ROOT / path).exists():
                            run("tools/download.py", resource)
                    run("tools/prepare_chemical.py")
                    log["stages"].append({"dataset": name, "status": "exploratory preparation complete"})
                    continue
                source, scale = name.split("-")
                resources = [("norman", "data/raw/NormanWeissman2019_filtered.h5ad")] if source == "norman" else [
                    ("wessels-metadata", "data/raw/GSE213957_THP1-CaRPool-seq.metadata.tsv.gz"),
                    ("wessels-counts", "data/raw/GSE213957_THP1-CaRPool-seq_and_GSE213957_HEX293FTstabRNA.tar.gz")]
                for resource, path in resources:
                    if not (ROOT / path).exists():
                        run("tools/download.py", resource)
                run("tools/prepare_sources.py", source, "--scale", scale)
            log["stages"].append({"dataset": name, "fitting_data_fingerprint": dataset_fingerprint(dataset)})
            write_json(ROOT / "results/pipeline_status.json", log)
        for file in protocol["experiments"]:
            config = read_json(ROOT / file)
            output = ROOT / config["output"]
            if not (output / "selection.json").exists():
                run("main.py", "benchmark", "--config", file, "--stage", "development")
            selection = read_json(output / "selection.json")
            if selection["data_fingerprint"] != dataset_fingerprint(ROOT / config["dataset"]):
                raise ValueError(f"Fitting data changed for {file}; register a new run")
            if args.development_only:
                state = "development complete"
            elif (output / "final_status.json").exists():
                access = read_json(output / "test_access.json")
                if access["predictions_sha256"] != sha256(output / "predictions.npz"):
                    raise ValueError("Completed predictions changed")
                state = "existing completed evaluation preserved"
            else:
                if selection["source_hash"] != source_hash():
                    raise ValueError("Code changed since selection; restore saved source snapshot or register a new development run")
                run("main.py", "benchmark", "--config", file, "--stage", "final")
                state = "final complete"
            log["stages"].append({"experiment": file, "status": state})
            write_json(ROOT / "results/pipeline_status.json", log)
        if not args.development_only and not args.skip_secondary:
            for source in protocol.get("biological_secondary", []):
                if not (ROOT / f"results/robustness/{source}/summary.json").exists():
                    run("tools/robustness_real.py", source)
            for source in protocol.get("measurement_bootstrap", []):
                if not (ROOT / f"results/runs/{source}-log2-primary/measurement_bootstrap/summary.json").exists():
                    run("tools/measurement_bootstrap.py", source)
                if not (ROOT / f"results/runs/{source}-log2-primary/extended_gene_diagnostic/summary.json").exists():
                    run("tools/extended_gene_diagnostic.py", source)
                if not (ROOT / f"results/runs/{source}-log2-primary/inference_memory.json").exists():
                    run("tools/measure_inference_memory.py", source)
            for name in protocol.get("adapted_public_references", []):
                if not (ROOT / f"results/public_references/{name}/evaluation.json").exists():
                    if not (ROOT / "data/public_methods/slim_local_manifest.json").exists():
                        run("tools/fetch_public_sources.py", "SLIM")
                    run("tools/run_slim_reference.py", name)
            run("tools/analyze_results.py")
            run("tools/qualify_metrics.py")
            run("tools/summarize_secondary.py")
            run("tools/audit_saved_runs.py")
            if not args.skip_architecture:
                run("main.py", "frontier", "--skip-paper")
            if not args.skip_paper:
                run("tools/build_paper.py", *(["--tectonic", args.tectonic] if args.tectonic else []))
        log["status"] = "registered internal batch complete"
    except Exception as e:
        log.update({"status": "failed", "error": str(e)})
        raise
    finally:
        write_json(ROOT / "results/pipeline_status.json", log)
    print(log["status"])


if __name__ == "__main__":
    main()
