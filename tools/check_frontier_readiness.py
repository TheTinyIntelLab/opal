"""Validate a local independent-measurement evidence package without scoring."""
import argparse
from pathlib import Path

from opal.frontier_bridge.contracts import load_config, read_json, write_json
from opal.frontier_bridge.resources import Budget

ROOT = Path(__file__).resolve().parents[1]
from opal.frontier_bridge.readiness import blank_package, validate_readiness


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--package", type=Path)
    choice.add_argument("--template", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    budget = Budget(ROOT, load_config(ROOT / "configs/experiments/opal_frontier_bridge_v1.json"), args.output)
    budget.check()
    if args.template:
        package = blank_package()
        budget.write_json(args.output / "measurement_package.json", package)
        root = args.output
    else:
        package = read_json(args.package)
        root = args.package.parent
    result = validate_readiness(package, root)
    budget.write_json(args.output / "measurement_readiness.json", result)
    (args.output / "report.md").write_text(
        "# Independent measurement readiness\n\n" + result["status"] + ".\n\n" +
        "This checks reviewed local evidence; it does not authorize laboratory execution or establish biological confirmation.\n\n" +
        "\n".join("- " + reason for reason in result["reasons"]) + "\n", encoding="utf-8")
    budget.check()
    budget.write_json(args.output / "resource_ledger.json", budget.evidence())
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
