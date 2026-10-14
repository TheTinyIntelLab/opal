"""Qualify the fixed exposed source, then prepare role-separated inputs."""
import argparse
import os
from pathlib import Path
import sys

# Pin numerical runtimes before importing numpy or Opal.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

from opal.frontier_bridge.contracts import (ContractError, fingerprint, load_config,
                                           qualify, read_json, sha256, write_json)
from opal.frontier_bridge.preparation import prepare_banks
from opal.frontier_bridge.resources import Budget
from opal.frontier_bridge.workflow import failure_receipt

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/opal_frontier_bridge_v1.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qualify-only", action="store_true")
    group.add_argument("--qualification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    budget = None
    try:
        budget = Budget(ROOT, cfg, args.output.resolve())
        budget.check()
        with threadpool_limits(limits=1):
            result = qualify(ROOT, cfg)
        result["configuration_hash"] = fingerprint(cfg)
        budget.write_json(args.output / "qualification.json", result)
        budget.write_json(args.output / "qualification_manifest.json", dict(path=str((args.output / "qualification.json").resolve()), sha256=sha256(args.output / "qualification.json")))
        if not result["qualified"]:
            reason = f"Frozen support failed: {result['common_pair_count']} common pairs, {result['common_gene_count']} common reference-selected genes; minima are 36 and 256. {', '.join(result['reasons'])}."
            failure_receipt(args.output, cfg, reason, budget, result)
            print(reason)
            return 0  # Completed feasibility decision, not a failed command.
        if args.qualify_only:
            budget.write_json(args.output / "resource_ledger.json", budget.evidence())
            print("Qualified public metadata; no response scoring.")
            return 0
        supplied = read_json(args.qualification)
        if "path" in supplied and "sha256" in supplied:
            if sha256(supplied["path"]) != supplied["sha256"]:
                raise ContractError("Qualification manifest tampered")
            supplied = read_json(supplied["path"])
        if supplied != result:
            raise ContractError("Qualification changed before preparation")
        banks = {}
        for context, record in result["banks"].items():
            with np.load(record["path"], allow_pickle=False) as original:
                indices = [original["genes"].tolist().index(g) for g in result["common_genes"]]
                mask = np.isin(original["pair_ids"], result["common_pairs"])
                banks[context] = dict(singles=original["singles"][mask][:, :, indices].astype(np.float64),
                                      responses=original["responses"][mask][:, indices].astype(np.float64),
                                      pair_ids=original["pair_ids"][mask], strata=original["culture_ids"][mask])
        prepare_banks(banks, result["common_genes"], cfg, args.output, result, budget)
        budget.write_json(args.output / "resource_ledger.json", budget.evidence())
        return 0
    except Exception as error:
        failure_receipt(args.output, cfg, f"{type(error).__name__}: {error}", budget)
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
