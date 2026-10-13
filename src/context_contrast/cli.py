"""Explicit nonnumerical planning and separately authorized numerical commands."""
import argparse
import json
import os
from pathlib import Path
from ..artifacts import fingerprint, read_json, sha256, write_json
from .contracts import PROTOCOL, ContractError, verify_source, qualify_graph, reviewed, load_config
from .resources import inventory, authorize, Budget


def parser():
    p = argparse.ArgumentParser(description="Opal fixed-context successor; numerical work needs owner acceptance")
    p.add_argument("--config", type=Path)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    source = sub.add_parser("source")
    source.add_argument("--root", type=Path, default=Path.cwd())
    source.add_argument("--output", type=Path, required=True)
    graph = sub.add_parser("qualify")
    graph.add_argument("--metadata", type=Path, required=True)
    graph.add_argument("--review", type=Path, required=True)
    graph.add_argument("--output", type=Path, required=True)
    meta = sub.add_parser("metadata")
    meta.add_argument("--root", type=Path, default=Path.cwd())
    meta.add_argument("--review", type=Path, required=True)
    meta.add_argument("--output", type=Path, required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--pairs", type=int, required=True)
    inv.add_argument("--genes", type=int, required=True)
    inv.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "predict", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--input", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--acceptance", type=Path, required=True)
        cmd.add_argument("--inventory", type=Path, required=True)
        if name == "prepare":
            cmd.add_argument("--root", type=Path, default=Path.cwd())
        if name == "evaluate":
            cmd.add_argument("--integrity-review", type=Path)
    for name in ("readiness-template", "readiness"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--output", type=Path, required=True)
        if name == "readiness":
            cmd.add_argument("--input", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--input", type=Path, required=True)
    return p


def new_file(path, value):
    if path.exists():
        raise FileExistsError(path)
    write_json(path, value)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.config:
        load_config(args.config)
    if args.command == "protocol":
        print(json.dumps(PROTOCOL, indent=2))
        return 0
    if args.command == "source":
        new_file(args.output, verify_source(args.root))
        return 0
    if args.command == "inventory":
        from .model import base_menu
        if args.pairs < 36 or not 256 <= args.genes <= 512:
            raise ContractError("Inventory must preserve qualification minima")
        new_file(args.output, inventory(args.pairs, args.genes, len(base_menu())))
        return 0
    if args.command == "metadata":
        import h5py
        from .preparation import metadata
        identity = verify_source(args.root)
        if not identity["qualified"]:
            raise ContractError("Source identity mismatch")
        with h5py.File(identity["path"], "r") as file:
            meta = metadata(file, read_json(args.review))
        new_file(args.output, dict(source_identity=identity, rows=meta["rows"], genes=meta["genes"],
            excluded_cells=meta["excluded_cells"], expression_accessed=False))
        return 0
    if args.command == "qualify":
        meta, review = read_json(args.metadata), read_json(args.review)
        new_file(args.output, qualify_graph(meta["rows"], review, args.review.parent))
        return 0
    if args.command in ("readiness-template", "readiness"):
        from .readiness import blank_package, validate_readiness
        value = blank_package() if args.command == "readiness-template" else validate_readiness(read_json(args.input), args.input.parent)
        new_file(args.output, value)
        return 0
    if args.command == "replay":
        import numpy as np
        from .workflow import verify_freeze
        from .artifacts import load_state
        from .model import predict
        from ..artifacts import array_hash
        seal = verify_freeze(args.input)
        for r in seal["records"]:
            for arm, expected in r["replay"].items():
                state = load_state(args.input/"inference"/(r["role"]+"-"+str(r["fold"])+"-"+arm))
                result = predict(state["model"], state["query"]["singles"], state["query"]["pair_ids"])
                if array_hash(result) != expected["prediction_hash"]:
                    raise ContractError("Replay mismatch")
        print("complete inference replay verified; no outcomes loaded")
        return 0
    operation = args.command
    input_path = args.input
    if operation == "predict":
        input_path = args.input/"prepared_manifest.json"
    elif operation == "evaluate":
        input_path = args.input/"prediction_freeze.json"
    accepted = authorize(args.acceptance, operation, args.output, sha256(input_path), sha256(args.inventory))
    plan = read_json(args.inventory)
    from .model import base_menu
    if plan != inventory(plan.get("pairs", 0), plan.get("genes", 0), len(base_menu())):
        raise ContractError("Complete procedure inventory mismatch")
    if operation == "predict":
        manifest = read_json(input_path)
        if len(manifest["folds"]) != plan["pairs"] or len(manifest["genes"]) != plan["genes"]:
            raise ContractError("Inventory dimensions differ from prepared inputs")
    elif operation == "evaluate":
        manifest = read_json(args.input/"input_manifest.json")
        if len(manifest["folds"]) != plan["pairs"] or len(manifest["genes"]) != plan["genes"]:
            raise ContractError("Evaluation inventory dimensions differ from inputs")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    budget = Budget(args.output, accepted)
    try:
        with threadpool_limits(limits=1):
            if operation == "prepare":
                from .preparation import prepare_raw
                source = verify_source(args.root)
                prepare_raw(source["path"], read_json(args.input), args.input.parent, source, args.output, budget)
            elif operation == "predict":
                from .workflow import run_predictions
                run_predictions(args.input, args.output, budget)
            else:
                from .workflow import evaluate_run
                integrity = None
                if args.integrity_review:
                    review = read_json(args.integrity_review)
                    if not reviewed(review, args.integrity_review.parent):
                        raise ContractError("Software integrity evidence must be hash-reviewed")
                    integrity = review.get("details")
                evaluate_run(args.input, args.output, budget, integrity)
            budget.check()
    except Exception as error:
        write_json(args.output/"failure.json", dict(operation=operation, reason=str(error),
            experiment_id=PROTOCOL["experiment_id"], partial_assets_retained=True, experiment_completed=False))
        budget.receipt("failed_partial_attempt")
        raise
    budget.receipt("completed_"+operation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
