"""Separate source/bridge/reference files from evaluator-only outcomes."""
import itertools
import json
from pathlib import Path

import numpy as np

from .contracts import (ContractError, assert_roles, canonical_pair, fingerprint,
                        sha256, write_json, query_bundle)
from .splits import manifest


def save_npz(path, budget=None, **arrays):
    if budget:
        budget.check(sum(np.asarray(a).nbytes for a in arrays.values()) + 8192)
    np.savez_compressed(path, **arrays)
    if budget:
        budget.check()
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def load_npz(record):
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ContractError("Role file tampered")
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def prepare_banks(banks, genes, cfg, output, qualification, budget=None, synthetic_case=None):
    """Only called after successful qualification or for explicit synthesis."""
    if synthetic_case is None and not qualification.get("qualified"):
        raise ContractError("Public preparation requires a qualified source")
    output = Path(output)
    if not isinstance(genes, list) or len(genes) != len(set(genes)) or any(not isinstance(g, str) or not g for g in genes):
        raise ContractError("Unique ordered literal gene schema required")
    if set(banks) != set(cfg["contexts"]):
        raise ContractError("Exact registered context banks required")
    normalized, panels = {}, []
    scale = "synthetic_expression_change" if synthetic_case else "log2_library_10000_change"
    for context, bank in banks.items():
        references = query_bundle(bank["singles"], bank["pair_ids"], genes, scale, genes, scale, cfg.get("aliases"))
        ids, x = references["pair_ids"], references["singles"]
        strata = np.asarray(bank["strata"]).astype(str)
        y = np.asarray(bank["responses"], dtype=np.float64)
        if strata.shape != (len(ids),) or any(not st for st in strata) or y.shape != (len(ids), len(genes)) or not np.isfinite(y).all():
            raise ContractError("Finite matched pair/stratum response schema required")
        if len(set(zip(ids, strata))) != len(ids):
            raise ContractError("Duplicate canonical condition rows")
        if not len(ids):
            raise ContractError("Empty context bank")
        panels.append(set.intersection(*(set(ids[strata == st]) for st in sorted(set(strata)))))
        normalized[context] = dict(singles=x, responses=y, pair_ids=ids, strata=strata)
    banks = normalized
    pairs = sorted(set.intersection(*panels))
    if len(pairs) < cfg["minimum_common_pairs"] or len(genes) < cfg["minimum_common_genes"]:
        raise ContractError("Frozen common support minima failed")
    split = manifest(pairs, cfg)
    records = dict(contexts={}, folds=split, genes=genes, genes_hash=fingerprint(genes),
                   scale="synthetic_expression_change" if synthetic_case else "log2_library_10000_change",
                   experiment_id=cfg["experiment_id"], configuration_hash=fingerprint(cfg),
                   role="synthetic_engineering" if synthetic_case else "exposed_public_development",
                   synthetic_case=synthetic_case, source_qualification=qualification)
    conditions = []
    for context, bank in banks.items():
        ids, strata, x, y = [bank[k] for k in ("pair_ids", "strata", "singles", "responses")]
        if len(set(zip(ids, strata))) != len(ids):
            raise ContractError("Duplicate canonical condition rows")
        per_context = dict(folds=[], strata=sorted(set(strata)))
        for fold in split:
            assert_roles(fold["source"], fold["bridges"], fold["query"])
            source_mask, bridge_mask, query_mask = [np.isin(ids, fold[role]) for role in ("source", "bridges", "query")]
            prefix = context.replace("+", "positive").replace("-", "negative") + f"-fold-{fold['fold']}"
            source = save_npz(output / f"{prefix}-source.npz", budget,
                              singles=x[source_mask], responses=y[source_mask], pair_ids=ids[source_mask], strata=strata[source_mask])
            reference = save_npz(output / f"{prefix}-references.npz", budget,
                                 singles=x[query_mask], pair_ids=ids[query_mask], strata=strata[query_mask])
            bridge = save_npz(output / f"{prefix}-bridges.npz", budget,
                              singles=x[bridge_mask], responses=y[bridge_mask], pair_ids=ids[bridge_mask], strata=strata[bridge_mask])
            evaluator = save_npz(output / f"{prefix}-evaluator.npz", budget,
                                 responses=y[query_mask], pair_ids=ids[query_mask], strata=strata[query_mask])
            per_context["folds"].append(dict(fold=fold["fold"], source=source, bridges=bridge, references=reference, evaluator=evaluator))
            for role, mask in [("source_fitting", source_mask), ("target_bridge", bridge_mask), ("evaluator_only", query_mask)]:
                for p, st in zip(ids[mask], strata[mask]):
                    conditions.append(dict(study_id="PertSpectra-inhouse" if not synthetic_case else "synthetic",
                                           context_id=context, context_definition=context,
                                           stratum_id=str(st), culture_id=None, culture_provenance_status="synthetic" if synthetic_case else "technical SampleIndex; independent initiation unknown",
                                           canonical_pair=str(p), fold=fold["fold"], label_access_role=role,
                                           exposure_role=records["role"], genes_hash=records["genes_hash"], scale_id=records["scale"],
                                           assay_id="synthetic" if synthetic_case else "PertSpectra-count-derived-scRNA",
                                           dose=None, dose_unit=None, simultaneous_sequential=None, duration=None,
                                           source_hash=qualification.get("original_source_hash"), source_revision="prepared count bank",
                                           guide_equivalence=None, original_condition_rows=None,
                                           qc_support="synthetic complete support" if synthetic_case else "historical ten-cell matched control/single/pair QC"))
        records["contexts"][context] = per_context
    records["conditions"] = conditions
    (budget.write_json if budget else write_json)(output / "folds.json", split)
    (budget.write_json if budget else write_json)(output / "prepared_manifest.json", records)
    return records


CASES = ("affine", "null", "transfer", "nontransfer", "rank_deficient")


def synthetic_banks(case, seed=20261011, genes=256):
    if case not in CASES:
        raise ContractError("Unregistered synthetic engineering case")
    rng = np.random.default_rng(seed)
    identities = [f"T{i:02d}" for i in range(15)]
    pairs = list(itertools.combinations(identities, 2))
    ids = np.asarray([canonical_pair(p) for p in pairs])
    profiles = rng.normal(size=(len(identities), 4))
    if case == "rank_deficient":
        profiles[:, 1:] = profiles[:, :1]
    basis, _ = np.linalg.qr(rng.normal(size=(genes, 4)))
    modes, _ = np.linalg.qr(rng.normal(size=(genes, 2)))
    other_modes, _ = np.linalg.qr(rng.normal(size=(genes, 2)))
    singles = np.asarray([[profiles[identities.index(c)] @ basis.T * np.sqrt(genes) for c in p] for p in pairs])
    coefficient = np.asarray([[profiles[identities.index(a), 0] * profiles[identities.index(b), 0],
                               profiles[identities.index(a), 1] * profiles[identities.index(b), 1]] for a, b in pairs])
    interaction = coefficient @ modes.T * np.sqrt(genes) * .8
    result = {}
    for context, scale in [("TNFA+", 1.), ("TNFA-", .8)]:
        x = singles * scale
        y = x.sum(1)
        if case == "affine":
            y = .7 * y + rng.normal(size=genes) * .2
        elif case in ("transfer", "rank_deficient"):
            y = y + interaction * (1. if context == "TNFA+" else 1.3)
        elif case == "nontransfer":
            y = y + (interaction if context == "TNFA+" else coefficient @ other_modes.T * np.sqrt(genes) * .8)
        result[context] = dict(singles=x.astype(np.float64), responses=y.astype(np.float64),
                               pair_ids=ids.copy(), strata=np.repeat("synthetic-stratum", len(ids)))
    return result, [f"G{i:04d}" for i in range(genes)]
