"""Immutable successor protocol, literal identities and outcome-blind eligibility."""
import json
from pathlib import Path
import numpy as np
from ..artifacts import fingerprint, read_json, sha256


class ContractError(ValueError):
    pass


class Unsupported(ContractError):
    def __init__(self, reason, diagnostic=None):
        super().__init__(reason)
        self.diagnostic = diagnostic


PROTOCOL = dict(experiment_id="opal_context_contrast_v1", schema_version=1,
    contexts=["TNFA+", "TNFA-"], role="exposed_public_development",
    minimum_common_genes=256, maximum_genes=512, minimum_common_pairs=36,
    input_rank=4, output_rank=2, outer_folds=3, inner_folds=3,
    fold_seed=20261012, penalties=[0.1, 1, 10], rank_relative_tolerance=1e-8,
    maximum_condition_number=1e4, scale_floor=1e-6,
    shuffle_seeds=[20261021, 20261022, 20261023], bootstrap_draws=2000,
    bootstrap_seed=20261031, retrieval_permutations=99, retrieval_seed=20261033,
    minimum_relative_mse_gain=0.05, normalization="library_10000_mean_log2_control_subtract",
    primary_metric="equal_pair_context_aggregate_response_mse",
    raw_path="data/raw/pertspectra-model-series/inhouse.h5ad",
    raw_bytes=3142132144,
    raw_sha256="7b72d474be90d3e236e9b4d7dce3fdf9806e5329f84f3eaa973ec605f52ebd12")


def load_config(path):
    cfg = read_json(path)
    if cfg != PROTOCOL:
        raise ContractError("Successor protocol mismatch; register a new procedure")
    return cfg


def literal_identity(value, aliases=None):
    aliases = aliases or {}
    seen = set()
    while value in aliases:
        if value in seen:
            raise ContractError("Alias cycle")
        seen.add(value)
        value = aliases[value]
    if not isinstance(value, str) or not value:
        raise ContractError("Literal alias identities required")
    return value


def canonical_pair(value, aliases=None):
    parts = json.loads(value) if isinstance(value, (str, np.str_)) else list(value)
    if len(parts) != 2 or any(not isinstance(c, str) or not c for c in parts):
        raise ContractError("Two literal intervention identities required")
    mapped = [literal_identity(c, aliases) for c in parts]
    if len(set(mapped)) != 2:
        raise ContractError("Self pairs forbidden")
    return json.dumps(sorted(mapped), separators=(",", ":"))


def folds(ids):
    ids = sorted(set(map(canonical_pair, ids)))
    order = np.random.default_rng(PROTOCOL["fold_seed"]).permutation(ids)
    return {str(p): i % 3 for i, p in enumerate(order)}


def masks(ids):
    assignment = folds(ids)
    labels = np.asarray([assignment[canonical_pair(p)] for p in ids])
    for k in range(3):
        if not np.any(labels == k) or not np.any(labels != k):
            raise Unsupported("Three nonempty grouped folds required")
        yield labels != k, labels == k


def shared_panel(genes, effects):
    """effects[context] is unique-single by literal gene, never pair responses."""
    genes = np.asarray(genes, str)
    if len(set(genes)) != len(genes):
        raise ContractError("Duplicate literal genes")
    if set(effects) != set(PROTOCOL["contexts"]):
        raise ContractError("Exact fixed contexts required")
    arrays = [np.asarray(effects[c], np.float64) for c in PROTOCOL["contexts"]]
    if any(a.ndim != 2 or a.shape[1] != len(genes) or len(a) < 2 for a in arrays):
        raise ContractError("Unique matched reference effects required")
    eligible = np.logical_and.reduce([np.isfinite(a).all(0) for a in arrays])
    if eligible.sum() < 256:
        raise Unsupported("At least 256 common reference-eligible genes required")
    scores = np.mean([a[:, eligible].var(0) for a in arrays], axis=0)
    positions = np.flatnonzero(eligible)
    selected = positions[np.lexsort((genes[eligible], -scores))[:512]]
    return selected, dict(genes=genes[selected].tolist(), genes_hash=fingerprint(genes[selected].tolist()),
        eligible_genes=int(eligible.sum()), scores=scores[np.lexsort((genes[eligible], -scores))[:512]].tolist(),
        rule="equal_context_average_within_context_single_effect_variance_lexical_ties")


def verify_source(root, cfg=PROTOCOL):
    path = Path(root) / cfg["raw_path"]
    if not path.is_file():
        return dict(qualified=False, status="source_unavailable", path=str(path))
    size = path.stat().st_size
    digest = sha256(path) if size == cfg["raw_bytes"] else None
    ok = size == cfg["raw_bytes"] and digest == cfg["raw_sha256"]
    return dict(qualified=ok, status="source_identity_verified" if ok else "source_identity_mismatch",
        path=str(path.resolve()), bytes=size, sha256=digest, copied=False, downloaded=False)


def reviewed(record, root):
    if not isinstance(record, dict) or record.get("approved") is not True:
        return False
    if not record.get("reviewer") or not record.get("reviewed_at") or not record.get("evidence"):
        return False
    for e in record["evidence"]:
        path = (Path(root) / e.get("path", "")).resolve()
        if not path.is_file() or not path.is_relative_to(Path(root).resolve()) or sha256(path) != e.get("sha256"):
            return False
    return True


def qualify_graph(rows, review, root):
    """Only label metadata and reference support enter the graph decision."""
    reasons, graph = [], []
    if not reviewed(review.get("rights"), root) or review.get("rights", {}).get("research_use_authorized") is not True:
        reasons.append("rights_unresolved")
    if not reviewed(review.get("assignments"), root):
        assignment = review.get("assignments", {})
        audited = isinstance(assignment, dict) and assignment.get("reviewer") and assignment.get("reviewed_at") and assignment.get("evidence")
        reasons.append("assignment_equivalence_unqualified" if audited else "assignment_equivalence_unreviewed")
    required = ("reagent_ids", "dose", "dose_unit", "duration", "duration_unit", "initiation_mode", "assay_id")
    signatures, support, excluded, incompatible = {}, {}, [], set()
    for row in rows:
        r = dict(row)
        r["pair_id"] = canonical_pair(r["pair_id"], review.get("aliases"))
        if r.get("context") not in PROTOCOL["contexts"] or not r.get("stratum"):
            raise ContractError("Unknown context or empty stratum")
        key = (r["pair_id"], r["context"], r["stratum"])
        if key in support:
            raise ContractError("Duplicate canonical pair/context/stratum")
        support[key] = r
        row_reasons = []
        if any(r.get(k) is None or r.get(k) == "" or r.get(k) == [] for k in required) or r.get("assignment_match") is not True:
            row_reasons.append("inexact_assignment")
        if not isinstance(r.get("reagent_ids"), list) or len(r.get("reagent_ids", [])) != 2 or r.get("initiation_mode") not in ("simultaneous", "sequential"):
            row_reasons.append("invalid_reagent_or_initiation")
        signature = fingerprint({k: r.get(k) for k in required})
        if r["pair_id"] in signatures and signatures[r["pair_id"]] != signature:
            incompatible.add(r["pair_id"])
            row_reasons.append("assignment_mismatch")
        signatures[r["pair_id"]] = signature
        if any(not isinstance(r.get(k), int) or isinstance(r.get(k), bool) or r[k] < 10
               for k in ("control_cells", "single_a_cells", "single_b_cells", "pair_cells")):
            row_reasons.append("insufficient_matching_cells")
        r["eligible"] = not row_reasons
        r["exclusion_reasons"] = row_reasons
        if row_reasons:
            excluded.append(dict(pair_id=r["pair_id"], context=r["context"], stratum=r["stratum"], reasons=row_reasons))
        graph.append(r)
    strata = {c: sorted({r["stratum"] for r in graph if r["context"] == c}) for c in PROTOCOL["contexts"]}
    complete = sorted(p for p in signatures if p not in incompatible and all(strata[c] and all((p, c, s) in support and support[(p, c, s)]["eligible"] for s in strata[c])
        for c in PROTOCOL["contexts"]))
    if len(complete) < 36:
        reasons.append("ineligible_graph:fewer_than_36_complete_pairs")
    components = {c for p in complete for c in json.loads(p)}
    reachable = {min(components)} if components else set()
    while True:
        expanded = reachable | {c for p in complete if reachable.intersection(json.loads(p)) for c in json.loads(p)}
        if expanded == reachable:
            break
        reachable = expanded
    if reachable != components:
        reasons.append("ineligible_graph:disconnected_components")
    status = "eligible_graph" if not reasons else "rights_unresolved" if "rights_unresolved" in reasons else "ineligible_graph"
    return dict(qualified=not reasons, status=status,
        reasons=sorted(set(reasons)), graph=graph, complete_pairs=complete, strata=strata,
        excluded_pairs=sorted(set(signatures) - set(complete)), exposure="exposed_TNF_history",
        excluded_conditions=excluded, incompatible_assignments=sorted(incompatible),
        independent_cultures=False, review_hash=fingerprint(review))
