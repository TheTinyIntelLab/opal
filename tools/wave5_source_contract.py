"""Experimental response-free source contracts, not an ingestion or access grant."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import wave4_source_contract as old

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"activity", "GEMINI", "GI", "held_double", "all_partner_fit", "absolute_fitness_rank"}
OPERATIONS = ["predict_construct", "equal_construct_within_culture", "equal_culture", "equal_pair_score"]


def schema(row, keys):
    if not isinstance(row, dict) or set(row) != set(keys):
        raise ValueError("Unexpected or incomplete metadata schema")


def annotation(row):
    schema(row, {"state", "value", "basis"})
    old.text(row["basis"], "annotation basis")
    if row["state"] not in {"documented", "unknown", "not_applicable"}:
        raise ValueError("Unknown annotation state")
    if row["state"] == "documented":
        old.text(row["value"], "documented annotation")
    elif row["value"] is not None:
        raise ValueError("Unknown or inapplicable annotation cannot invent value")


def validate_metadata(data):
    """Check invented structures, without claiming observed controls or coverage."""
    schema(data, {"source", "control_role", "guides", "samples", "constructs", "matches", "anchors", "assignments", "selection", "aggregation"})
    if data["source"] not in {"spidr", "shen2017"} or data["control_role"] not in {"nt", "scramble"}:
        raise ValueError("Explicit source and control role required")
    if data["control_role"] != {"spidr": "nt", "shen2017": "scramble"}[data["source"]]:
        raise ValueError("Source-specific control role mismatch")
    guides = old.index_unique(data["guides"], "id", {"id", "gene", "sequence", "arm", "role", "transcript", "mismatch"}, "guide")
    samples = old.index_unique(data["samples"], "id", {"id", "context", "time_origin", "time", "culture", "initiation"}, "sample")
    constructs = old.index_unique(data["constructs"], "id", {"id", "left", "right"}, "construct")
    if not guides or not samples or not constructs or sum(map(len, (guides, samples, constructs))) > 512:
        raise ValueError("Empty or oversized fixture")
    for g in guides.values():
        for key in ("gene", "sequence", "arm", "role"):
            old.text(g[key], key)
        if g["arm"] not in {"arm1", "arm2"} or g["role"] not in {"target", data["control_role"]}:
            raise ValueError("Unknown arm or role")
        if any(c not in "ACGT" for c in g["sequence"]):
            raise ValueError("Exact uppercase DNA required")
        for key in ("transcript", "mismatch"):
            annotation(g[key])
        if data["source"] == "spidr" and g["role"] == "target" and any(g[k]["state"] != "documented" for k in ("transcript", "mismatch")):
            raise ValueError("SPIDR exact identity annotations unresolved")
    culture_lineage, sample_keys = {}, set()
    for s in samples.values():
        for key in s:
            old.text(s[key], key)
        lineage = s["context"], s["time_origin"], s["initiation"]
        if s["culture"] in culture_lineage and culture_lineage[s["culture"]] != lineage:
            raise ValueError("Contradictory culture lineage/context/time origin")
        culture_lineage[s["culture"]] = lineage
        key = s["culture"], s["time"]
        if key in sample_keys:
            raise ValueError("Ambiguous sample/time")
        sample_keys.add(key)
    for culture in culture_lineage:
        if len([k for k in sample_keys if k[0] == culture]) < 2:
            raise ValueError("Missing time mapping")

    def exact(g):
        return json.dumps({k: v for k, v in g.items() if k != "id"}, sort_keys=True)

    exact_keys, pairs, doubles = set(), set(), set()
    for c in constructs.values():
        if c["left"] not in guides or c["right"] not in guides:
            raise ValueError("Unknown guide key")
        left, right = guides[c["left"]], guides[c["right"]]
        if (left["arm"], right["arm"]) != ("arm1", "arm2"):
            raise ValueError("Ordered arm mismatch")
        key = exact(left), exact(right)
        if key in exact_keys:
            raise ValueError("Duplicate exact ordered construct")
        exact_keys.add(key)
        if left["role"] == right["role"] == "target":
            if left["gene"] == right["gene"]:
                raise ValueError("Distinct gene pairs only")
            pairs.add(tuple(sorted((left["gene"], right["gene"]))))
            doubles.add(c["id"])
    if not doubles:
        raise ValueError("No double design")
    seen = set()
    for m in data["matches"]:
        schema(m, {"double", "sample", "left_single", "right_single", "provenance"})
        if m["provenance"] != "explicit_control_design":
            raise ValueError("Outcome-derived single provenance")
        key = m["double"], m["sample"]
        if key in seen or m["double"] not in doubles or m["sample"] not in samples:
            raise ValueError("Duplicate or unknown matched link")
        seen.add(key)
        d = constructs[m["double"]]
        for side, other in (("left", "right"), ("right", "left")):
            ref = m[side + "_single"]
            schema(ref, {"construct", "sample"})
            if ref["sample"] != m["sample"] or ref["construct"] not in constructs:
                raise ValueError("Single sample/time/culture or key mismatch")
            s = constructs[ref["construct"]]
            if exact(guides[s[side]]) != exact(guides[d[side]]) or guides[s[other]]["role"] != data["control_role"]:
                raise ValueError("Exact guide/arm/control single mismatch")
    if seen != {(d, s) for d in doubles for s in samples}:
        raise ValueError("Incomplete double/sample single links")
    anchors = set()
    for a in data["anchors"]:
        schema(a, {"sample", "construct"})
        if a["sample"] not in samples or a["sample"] in anchors or a["construct"] not in constructs:
            raise ValueError("Invalid control anchor key")
        c = constructs[a["construct"]]
        if any(guides[c[side]]["role"] != data["control_role"] for side in ("left", "right")):
            raise ValueError("Control-control anchor required")
        anchors.add(a["sample"])
    if anchors != set(samples):
        raise ValueError("Missing control anchor")
    assignments, variants = {}, set()
    for a in data["assignments"]:
        schema(a, {"gene_a", "gene_b", "variant", "split"})
        for v in a.values():
            old.text(v, "assignment")
        pair = tuple(sorted((a["gene_a"], a["gene_b"])))
        key = pair, a["variant"]
        if pair not in pairs or key in variants or a["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unlinked or duplicate canonical assignment")
        variants.add(key)
        if pair in assignments and assignments[pair] != a["split"]:
            raise ValueError("Canonical variants/orientations cross splits")
        assignments[pair] = a["split"]
    if set(assignments) != pairs:
        raise ValueError("Missing canonical assignment")
    schema(data["selection"], {"features", "eligibility", "weights", "tuning"})
    if any(value != "response_free_design_only" for value in data["selection"].values()):
        raise ValueError("Response-derived selection or provenance forbidden")
    if data["aggregation"] != OPERATIONS:
        raise ValueError("Construct-first equal aggregation order required")
    return dict(design_consistent=True, designed_pairs=len(pairs), designed_double_constructs=len(doubles),
                matched_design_links=len(seen), observed_eligibility=None, observed_singles=None,
                response_access_authorized=False, response_header_access_authorized=False)


def validate_endpoint(row):
    schema(row, {"source", "endpoint", "input_units", "operations", "selection_basis", "actual_invocation", "reference_membership", "time_zero_missingness"})
    for key in ("actual_invocation", "reference_membership", "time_zero_missingness"):
        old.text(row[key], key)
    if row["selection_basis"] != "prospective_response_free":
        raise ValueError("Endpoint selection requires response-free provenance")
    expected = {
        "spidr": ("normalized_T14_T0_LFC", "normalized_counts", ["documented_normalization_already_applied", "pseudocount_in_normalized_units", "late_early_log_ratio", "documented_NT_centering"]),
        "shen2017": ("prospective_control_relative_growth_slope", "source_specific_counts", ["control_relative_abundance", "fixed_time_log_slope", "explicit_control_centering"]),
    }
    if row["source"] not in expected or (row["endpoint"], row["input_units"], row["operations"]) != expected[row["source"]]:
        raise ValueError("Incompatible endpoint units/operation order")
    return {"declaration_consistent": True, "actual_invocation_verified": False, "numeric_endpoint_tested": False, "response_access_authorized": False}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_authorization(freeze_path, authorization_path):
    f = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    a = json.loads(Path(authorization_path).read_text(encoding="utf-8"))
    if f.get("schema") != "opal-wave5-plan-freeze-v1" or a.get("schema") != "opal-wave5-execution-authorization-v1" or a.get("freeze_sha256") != digest(freeze_path) or a.get("B_source_authorized") is not True:
        raise ValueError("Root authorization mismatch")
    for scope in (f.get("permissions", {}), a.get("scope", {})):
        if any(scope.get(k) is not False for k in ("reserved_response_access", "reserved_response_header_access", "biological_fitting", "new_biological_scoring", "scientific_successor")) or scope.get("source_primary_narrative_metadata_only") is not True:
            raise ValueError("Unsafe authorization scope")
    required = "docs/research_reset/wave5/B_revised.md"
    if required not in f.get("files", {}):
        raise ValueError("Revised proposal not bound")
    for name, expected in f["files"].items():
        p = (ROOT / name).resolve()
        if not p.is_relative_to(ROOT) or digest(p) != expected:
            raise ValueError(f"Frozen proposal/input drift: {name}")
    return f


def run(output, freeze_path, authorization_path, binding_path):
    import psutil
    output = Path(output)
    f = verify_authorization(freeze_path, authorization_path)
    binding = json.loads(Path(binding_path).read_text(encoding="utf-8"))
    if binding.get("freeze_sha256") != digest(freeze_path):
        raise ValueError("Experimental binding freeze mismatch")
    required = {"tools/wave5_source_contract.py", "tests/test_wave5_source_contract.py", "tools/wave4_source_contract.py", "docs/research_reset/wave3_spidr_results.md", "docs/research_reset/wave3_source_decision.md"}
    if not required.issubset(binding.get("files", {})):
        raise ValueError("Incomplete experimental binding")
    for name, expected in binding["files"].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f"Experimental code/input drift: {name}")
    if output.exists():
        raise FileExistsError("Preserve existing report")
    started, cpu = time.monotonic(), time.process_time()
    old.write_new(output.with_suffix(".start.json"), {"freeze_sha256": digest(freeze_path), "binding_sha256": digest(binding_path), "code_sha256": digest(__file__)})
    try:
        sources = old.reviewed_snapshot()
        memory = psutil.Process().memory_info()
        receipt = dict(protocol="wave5-source-contract-v1", sources=sources, source_truth_certified=False,
                       actual_source_mapping_validated=False, response_access=False, response_header_access=False,
                       fits=0, scores=0, network_bytes=0, planned_matrix="B1-B8 and D1-D6; documentary acquisitions reported separately",
                       resources=dict(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                                      peak_bytes=max(memory.rss, getattr(memory, "peak_wset", 0)), guard="cooperative stage checks, not interrupting watchdog"),
                       freeze_sha256=digest(freeze_path), binding_sha256=digest(binding_path))
        limits = f["limits"]["B"]
        if receipt["resources"]["wall_seconds"] > min(120, limits["wall_seconds_per_command"]) or receipt["resources"]["peak_bytes"] > min(512*1024**2, limits["peak_bytes"]) or len(json.dumps(receipt).encode()) > min(10*1024**2, limits["output_bytes"]):
            raise RuntimeError("Source resource cap exceeded")
        old.write_new(output, receipt)
        return receipt
    except Exception as exc:
        old.write_new(output.with_suffix(".failure.json"), {"error": repr(exc), "wall_seconds": time.monotonic()-started})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freeze", type=Path, default=ROOT/"results/research-wave5/freeze.json")
    p.add_argument("--authorization", type=Path, default=ROOT/"docs/research_reset/wave5/EXECUTION_AUTHORIZED.json")
    p.add_argument("--binding", type=Path, default=ROOT/"results/research-wave5/B-experiment-binding.json")
    args = p.parse_args()
    run(args.output, args.freeze, args.authorization, args.binding)
    print("Response-free documentary snapshot written; both sources remain blocked")
