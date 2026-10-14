"""Run frozen bridge inference or explicitly labeled synthetic verification."""
import argparse
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from threadpoolctl import threadpool_limits

from opal.frontier_bridge.contracts import load_config, read_json, write_json
from opal.frontier_bridge.preparation import CASES, prepare_banks, synthetic_banks
from opal.frontier_bridge.resources import Budget
from opal.frontier_bridge.workflow import failure_receipt, replay, run

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/opal_frontier_bridge_v1.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepared", type=Path)
    mode.add_argument("--replay", type=Path)
    mode.add_argument("--synthetic", choices=CASES)
    parser.add_argument("--verify-compatible-code", action="store_true",
                        help="Explicitly verify saved predictions under changed code; reports every code hash difference, with no refitting")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    budget = None
    try:
        budget = Budget(ROOT, cfg, args.output.resolve())
        budget.check()
        with threadpool_limits(limits=1):
            if args.replay:
                result = replay(args.replay, args.output, budget, args.verify_compatible_code)
            else:
                if args.synthetic:
                    banks, genes = synthetic_banks(args.synthetic)
                    prepared_dir = args.output / "prepared"
                    prepared_dir.mkdir(exist_ok=False)
                    prepared = prepare_banks(banks, genes, cfg, prepared_dir,
                                             dict(qualified=False, role="synthetic_engineering", case=args.synthetic),
                                             budget, synthetic_case=args.synthetic)
                else:
                    prepared = read_json(args.prepared)
                result = run(prepared, cfg, ROOT, args.output, budget)
        print(result)
        return 0
    except Exception as error:
        failure_receipt(args.output, cfg, f"{type(error).__name__}: {error}", budget)
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
