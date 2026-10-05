"""Reproduction, preparation, development locking, evaluation, and inference."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from opal.artifacts import read_json, write_json, environment, load_npz


def config_file(path):
    path = Path(path)
    if path.suffix in [".yaml", ".yml"]:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return read_json(path)


def reproduce(config):
    from opal.data.synthetic import generate, save_prepared, SCENARIOS
    from opal.training.benchmark import development, final
    root = Path(config.get("output", "results/synthetic"))
    if (root / "summary.json").exists():
        raise FileExistsError("Reproduction output already exists; use --output to preserve previous artifacts")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "suite_protocol.json", config)
    write_json(root / "environment.json", environment())
    summaries = []
    for scenario in config.get("scenarios", SCENARIOS):
        arrays = generate(scenario, **config.get("generator", {}))
        dataset, run = root / scenario / "dataset", root / scenario / "run"
        save_prepared(dataset, arrays, {"synthetic": True, "scenario": scenario, "context": "synthetic", "scale": "log2", "generator": config.get("generator", {})}, config.get("outer_seed", 1729))
        print(f"Starting synthetic scenario {scenario}", flush=True)
        development(dataset, run, config["training"])
        result = final(dataset, run)
        cid = result["selected_id"]
        selected = next(r["config"] for r in read_json(run / "selection.json")["comparisons"] if r["config"]["id"] == cid)
        summaries.append({"scenario": scenario, "selected_family": selected["family"], "selected_config": selected,
                          "selected": result["models"][cid], "comparisons": result["comparisons"],
                          "simpler": result["models"][result["simpler_id"]], "run": str(run)})
        write_json(root / "progress.json", summaries)
    write_json(root / "summary.json", {"scope": "synthetic engineering experiments; no biological validation", "experiments": summaries})
    if config.get("secondary_studies", False):
        from opal.training.studies import run_secondary_suite
        run_secondary_suite(root / "secondary", generate("low_rank", **config.get("generator", {})), config["training"])
    from opal.report import render_report
    render_report(root)
    return str(root / "summary.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Opal population mean combination prediction")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("reproduce")
    p.add_argument("--config", default="configs/experiments/synthetic.json")
    p.add_argument("--output")
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--dataset", required=True)
    p = sub.add_parser("benchmark")
    p.add_argument("--config", required=True)
    p.add_argument("--stage", choices=["development", "final"], required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--run", required=True)
    p.add_argument("--split", choices=["test"], default="test")
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pair", nargs=2)
    p.add_argument("--single")
    p.add_argument("--context")
    p.add_argument("--batch", help="CSV with target_a,target_b columns")
    p.add_argument("--output", required=True)
    p = sub.add_parser("export")
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("report")
    p.add_argument("--results", default="results/synthetic")
    args = parser.parse_args(argv)
    if args.command == "reproduce":
        config = config_file(args.config)
        if args.output:
            config["output"] = args.output
        result = reproduce(config)
    elif args.command == "prepare":
        from opal.data.prepare import prepare
        result = prepare(config_file(args.config))
    elif args.command == "audit":
        from opal.training.benchmark import dataset_fingerprint
        path = Path(args.dataset)
        data = load_npz(path / "profiles.npz")
        if not np.isfinite(data["singles"]).all() or not np.isfinite(data["doubles"]).all():
            raise ValueError("Nonfinite profiles")
        if set(data["pair_indices"]).intersection(data["query_indices"]):
            raise ValueError("Train/test overlap")
        result = {"data_fingerprint": dataset_fingerprint(path), "singles": len(data["singles"]),
                  "development_pairs": len(data["pairs"]), "test_queries": len(data["query_pairs"]), "genes": len(data["genes"]),
                  "source": read_json(path / "source_manifest.json")}
    elif args.command == "benchmark":
        from opal.training.benchmark import development, final
        config = config_file(args.config)
        result = development(config["dataset"], config["output"], config["training"]) if args.stage == "development" else final(config["dataset"], config["output"])
    elif args.command == "evaluate":
        # Reading existing immutable evaluation is not a second test fit/access.
        result = read_json(Path(args.run) / "test_results.json")
    elif args.command == "predict":
        from opal.inference import Predictor
        predictor = Predictor(args.checkpoint)
        if args.batch:
            import csv
            with open(args.batch, encoding="utf-8", newline="") as f:
                pairs = [(r["target_a"], r["target_b"]) for r in csv.DictReader(f)]
            outputs = predictor.predict_batch(pairs, args.context)
            np.savez_compressed(args.output, genes=predictor.reference["genes"], pairs=np.asarray(pairs),
                                **{k: np.stack([r[k] for r in outputs]) for k in ["effect", "additive", "residual", "ensemble_spread"]})
        else:
            if args.pair and args.single:
                raise ValueError("Choose pair or single route")
            prediction = predictor.predict(*(args.pair or [args.single, None]), context=args.context)
            np.savez_compressed(args.output, **{k: v for k, v in prediction.items() if isinstance(v, np.ndarray)})
        result = {"output": args.output, "context": predictor.manifest["metadata"]["context"], "units": predictor.manifest["metadata"]["scale"]}
    elif args.command == "export":
        from opal.inference import Predictor
        source = Path(args.run) / "checkpoint"
        Predictor(source)
        if Path(args.output).exists():
            raise FileExistsError("Export destination exists")
        shutil.copytree(source, args.output)
        result = {"output": args.output}
    else:
        from opal.report import render_report
        result = render_report(Path(args.results))
    print(json.dumps(result, indent=2, allow_nan=False))
