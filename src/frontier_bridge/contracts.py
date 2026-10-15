"""Hash-bound inputs, exact roles, and fail-closed qualification."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import EXPERIMENT_ID


class ContractError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config(path, root=None):
    cfg = read_json(path)
    fixed = dict(experiment_id=EXPERIMENT_ID, schema_version=1,
                 role="exposed_public_development",
                 primary_metric="equal_pair_equal_stratum_response_mse",
                 retrieval_metric="query_panel_centered_cosine_normalized_inverse_rank",
                 contexts=["TNFA+", "TNFA-"], minimum_common_pairs=36,
                 minimum_common_genes=256, minimum_source_pairs=20,
                 minimum_inner_pairs=12, minimum_inner_queries=4,
                 outer_folds=3, inner_folds=3, fold_seed=20261011,
                 bridge_count=4, diagnostic_bridge_count=0, input_rank=4,
                 output_rank=2, source_penalties=[0.1, 1, 10], target_penalty=1,
                 rank_relative_tolerance=1e-8, maximum_condition_number=10000,
                 mode_rms_floor=1e-8, shuffle_seeds=[20261021, 20261022, 20261023],
                 bootstrap_seed=20261031, bootstrap_draws=2000,
                 minimum_relative_mse_gain=0.05)
    for key, expected in fixed.items():
        if cfg.get(key) != expected:
            raise ContractError(f"Frozen contract mismatch: {key}")
    resources = cfg["resources"]
    maxima = dict(workers=1, threads=1, cpu_core_hours=24,
                  new_retained_bytes=60_000_000, artifact_cap_bytes=3_000_000_000)
    if resources.get("amendment"):
        # Configurations live in <research-root>/configs/experiments, while
        # this module may be imported from an installed wheel. Never resolve
        # research authorization records relative to site-packages. An
        # explicitly supplied root supports separately located config files.
        asset_root = Path(root).resolve() if root is not None else Path(path).resolve().parents[2]
        amendment_path = checked_asset(asset_root, resources["amendment"])
        amendment = read_json(amendment_path)
        authorization = amendment.get("authorization_basis")
        if amendment.get("experiment_id") != EXPERIMENT_ID or not isinstance(authorization, str) or not authorization.strip():
            raise ContractError("Invalid owner resource amendment")
        for key in maxima:
            if key in amendment:
                if isinstance(amendment[key], bool) or not isinstance(amendment[key], (int, float)) or not math.isfinite(amendment[key]) or amendment[key] <= 0:
                    raise ContractError(f"Invalid amendment resource allowance: {key}")
                if key in ("workers", "threads", "cpu_core_hours") and amendment[key] > maxima[key]:
                    raise ContractError("Amendment changes frozen CPU execution scope")
                maxima[key] = amendment[key]
        if any(amendment.get(k) for k in ("raw_downloads", "environment_installations", "gpu_training")):
            raise ContractError("Amendment cannot substitute raw assets or environments")
    for key, maximum in [("workers", 1), ("threads", 1), ("cpu_core_hours", 24),
                         ("new_retained_bytes", 60_000_000), ("artifact_cap_bytes", 3_000_000_000)]:
        value = resources[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maxima[key]:
            raise ContractError(f"Invalid resource allowance: {key}")
    if resources["reserve_bytes"] < 1073741824 or any(resources[k] for k in
            ["raw_downloads", "environment_installations", "gpu_training"]):
        raise ContractError("Forbidden resource expansion")
    return cfg


def checked_asset(root, record):
    path = Path(root) / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ContractError(f"Asset missing or tampered: {path}")
    return path


def canonical_pair(value, aliases=None):
    parts = json.loads(value) if isinstance(value, (str, np.str_)) else list(value)
    if len(parts) != 2 or any(not isinstance(p, str) or not p.strip() for p in parts):
        raise ContractError("Two literal intervention assignments required")
    aliases = aliases or {}
    resolved = []
    for part in parts:
        seen = set()
        while part in aliases:
            if part in seen:
                raise ContractError("Cyclic intervention alias")
            seen.add(part)
            part = aliases[part]
        resolved.append(part)
    if len(set(resolved)) != 2:
        raise ContractError("Self pairs are unsupported")
    # Reagent/dose-qualified tokens remain literal. Unknown assignments are
    # never made equivalent by stripping punctuation or case.
    return json.dumps(sorted(resolved), separators=(",", ":"))


def query_bundle(singles, pair_ids, genes, scale, expected_genes, expected_scale, aliases=None):
    if list(genes) != list(expected_genes) or scale != expected_scale:
        raise ContractError("Mismatched ordered genes or response scale")
    x = np.asarray(singles, dtype=np.float64).copy()
    ids = np.asarray([canonical_pair(p, aliases) for p in pair_ids])
    if x.shape != (len(ids), 2, len(genes)) or not np.isfinite(x).all():
        raise ContractError("Finite matched measured singles required")
    for i, p in enumerate(pair_ids):
        original = json.loads(str(p))
        resolved = json.loads(canonical_pair(original, aliases))
        # Resolve each component through aliases before ordering its single.
        left = original[0]
        while left in (aliases or {}):
            left = aliases[left]
        if left != resolved[0]:
            x[i] = x[i, ::-1]
    return {"singles": x, "pair_ids": ids}


def assert_roles(source_ids, bridge_ids, query_ids):
    source, bridge, query = [set(map(canonical_pair, ids)) for ids in (source_ids, bridge_ids, query_ids)]
    if source & query or bridge & query or not bridge <= source:
        raise ContractError("Global query exclusion or bridge membership violated")


def qualify(root, cfg):
    """Establish metadata infeasibility before reopening any prepared bank."""
    records = {k: checked_asset(root, cfg[k]) for k in
               ("source_registry", "prepared_source", "source_support", "source_access")}
    registry = read_json(records["source_registry"])
    prepared = read_json(records["prepared_source"])
    support = read_json(records["source_support"])
    access = read_json(records["source_access"])
    if not prepared.get("counts_verified") or prepared.get("matrix_key") != "layers/counts":
        raise ContractError("Count-derived endpoint required")
    if access["full_file_sha256"] != prepared["full_file_sha256"]:
        raise ContractError("Source provenance hash mismatch")
    banks, pair_sets, gene_sets, manifests = {}, [], [], []
    for context in cfg["contexts"]:
        entry = prepared["contexts"][context]
        if entry["units"] != "log2_library_10000_change":
            raise ContractError("Mismatched response scale")
        genes = entry["genes"]
        if not genes or len(genes) != len(set(genes)) or any(not isinstance(g, str) or not g for g in genes):
            raise ContractError("Ordered reference gene schema mismatch")
        if not entry["strata"] or len(entry["strata"]) != len(set(entry["strata"])):
            raise ContractError("Stratum registry mismatch")
        registered = [s for s in registry["sources"] if "/PertSpectra/" in s["source_version"]
                      and ("TNFA-positive" if context == "TNFA+" else "TNFA-negative") in s["source_version"]]
        if len(registered) != 1:
            raise ContractError("Exact source registry membership required")
        checkpoint = checked_asset(root, {"path": registered[0]["manifest"], "sha256": registered[0]["manifest_sha256"]})
        if not any(Path(r["path"]).is_relative_to(Path(entry["local"])) for r in read_json(checkpoint)):
            raise ContractError("Prepared bank not linked to registered source")
        panels = []
        for st in entry["strata"]:
            evidence = {}
            for row in support:
                if row["context"] != context or str(row["sample_index"]) != st:
                    continue
                targets = tuple(sorted(row["targets"]))
                if targets in evidence:
                    raise ContractError("Duplicate condition support metadata")
                evidence[targets] = row
            panel = set()
            for parts, row in evidence.items():
                if len(parts) != 2 or not row.get("included_pair"):
                    continue
                for identity in ((), (parts[0],), (parts[1],), parts):
                    matched = evidence.get(identity)
                    if not matched or matched["cells"] < 10 or not matched.get("guide_mixtures"):
                        raise ContractError("Missing matched reference/support/guide provenance")
                    if len(identity) < 2 and not matched["reference_supported"]:
                        raise ContractError("Unsupported matched reference")
                panel.add(canonical_pair(parts, cfg["aliases"]))
            panels.append(panel)
        panel = set.intersection(*panels)
        if len(panel) != entry["pairs"] or any(stpanel != panel for stpanel in panels):
            raise ContractError("Incomplete fixed source panel metadata")
        path = Path(entry["local"]) / "canonical_bank.npz"
        banks[context] = dict(path=str(path), sha256=entry["bank_sha256"], strata=entry["strata"],
                              original_genes_hash=fingerprint(genes), scale=entry["units"],
                              current_bank_available=path.is_file(), current_bank_hash_verified=False)
        pair_sets.append(panel)
        gene_sets.append(set(genes))
        manifests.append(dict(path=str(checkpoint), sha256=sha256(checkpoint)))
    common_pairs = sorted(set.intersection(*pair_sets))
    common_genes = sorted(set.intersection(*gene_sets))
    failures = []
    if len(common_pairs) < cfg["minimum_common_pairs"]:
        failures.append("insufficient_common_pairs")
    if len(common_genes) < cfg["minimum_common_genes"]:
        failures.append("insufficient_common_reference_selected_genes")
    # A proven metadata support failure is sufficient: opening arrays cannot
    # remedy it, and absent historical banks must not hide that scientific fact.
    metadata_support_passed = not failures
    if metadata_support_passed:
        for context, record in banks.items():
            path = Path(record["path"])
            if not record["current_bank_available"]:
                failures.append(f"prepared_bank_unavailable:{context}")
                continue
            if sha256(path) != record["sha256"]:
                raise ContractError("Prepared bank hash mismatch")
            entry = prepared["contexts"][context]
            with np.load(path, allow_pickle=False) as bank:
                if set(bank.files) != {"singles", "responses", "pair_ids", "culture_ids", "genes"}:
                    raise ContractError("Unexpected bank schema")
                if bank["genes"].astype(str).tolist() != entry["genes"]:
                    raise ContractError("Ordered reference gene schema mismatch")
                ids = [canonical_pair(p, cfg["aliases"]) for p in bank["pair_ids"].astype(str)]
                strata = bank["culture_ids"].astype(str).tolist()
                if sorted(set(strata)) != entry["strata"] or len(set(zip(strata, ids))) != len(ids):
                    raise ContractError("Duplicate or mismatched canonical pair/stratum rows")
                expected = pair_sets[cfg["contexts"].index(context)]
                if any({p for p, s in zip(ids, strata) if s == st} != expected for st in entry["strata"]):
                    raise ContractError("Prepared bank differs from fixed metadata panel")
                singles = bank["singles"]
                if singles.shape != (len(ids), 2, len(entry["genes"])) or not np.isfinite(singles).all():
                    raise ContractError("Unsupported single reference schema")
            record["current_bank_hash_verified"] = True
    return dict(experiment_id=EXPERIMENT_ID, configuration_hash=fingerprint(cfg), qualified=not failures,
                status="qualified_public_prototype" if not failures else "infeasible_public_prototype",
                reasons=failures, common_pairs=common_pairs, common_genes=common_genes,
                common_pair_count=len(common_pairs), common_gene_count=len(common_genes),
                genes_hash=fingerprint(common_genes), banks=banks, registry_manifests=manifests,
                original_source_hash=prepared["full_file_sha256"], metadata_hashes={k: sha256(p) for k, p in records.items()},
                support_evidence_scope="hash-verified prepared metadata and included-pair/reference support records",
                metadata_support_passed=metadata_support_passed,
                current_bank_hashes_verified=all(b["current_bank_hash_verified"] for b in banks.values()),
                current_bank_arrays_opened=metadata_support_passed and any(b["current_bank_hash_verified"] for b in banks.values()),
                historical_bank_hashes_are_expected_not_current_measurements=True,
                query_expression_accessed=False, prior_exposure="both contexts exposed in model-series and information development",
                normalization_order=["library_total_10000", "condition_mean", "log2_1_plus", "matched_control_subtraction"],
                guide_equivalence="unknown; assignment-level guide mixtures", physical_dose=None,
                initiation=None, treatment_duration=None, original_condition_rows=None,
                culture_provenance_status="SampleIndex technical/source strata; independent initiation unknown",
                independent_biological_confirmation=False, frontier_claim=False)
