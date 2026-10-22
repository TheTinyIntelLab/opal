"""Response-free documentary and synthetic metadata checks, never count ingestion.

No network or response loader exists here. Passing checks establish internal
consistency of supplied metadata, not the truth of unavailable source content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = {
    "rights": ("item_notice", "specific_supplement"),
    "sample_columns": ("actual_column_map", "actual_response_columns"),
    "ordered_key": ("ordered_key_map", "actual_response_keys"),
    "invocation": ("actual_processing_call", "actual_invocation"),
    "normalization": ("normalization_contract", "reference_pseudocount_centering"),
    "endpoint": ("endpoint_registration", "units_qc_zero_missingness"),
    "controls": ("explicit_control_map", "control_roles"),
    "single_design": ("exact_control_construct_map", "exact_guide_arm_design"),
    "lineage": ("initiation_record", "culture_initiation"),
    "evaluator": ("evaluator_receipt", "equal_information_leakage_strict_replay"),
    "resources": ("full_design_rehearsal", "source_scale_resources"),
    "budget": ("budget_registration", "unique_pair_union_singles_controls"),
    "procedure": ("procedure_registration", "complete_learning_procedure"),
}
STATES = {"supported", "blocked", "unknown", "not_run"}
TEXT_FIELDS = {"claim", "artifact", "kind", "scope", "locator", "status"}


def text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} requires explicit nonblank text")
    return value


def fields(row, names, name):
    if not isinstance(row, dict) or set(row) != set(names):
        raise ValueError(f"Invalid {name} schema")


def assess_evidence(evidence, *, initiations, independent_confirmation=True):
    """Check evidence classification without certifying documentary truth."""
    if not isinstance(evidence, dict) or set(evidence) != set(CLAIMS):
        raise ValueError("Incomplete or unexpected evidence claims")
    if type(independent_confirmation) is not bool:
        raise ValueError("Explicit independent-confirmation claim required")
    if not isinstance(initiations, list):
        raise ValueError("Initiation list required")
    for identity in initiations:
        text(identity, "initiation")
    blockers = []
    for claim, row in evidence.items():
        fields(row, TEXT_FIELDS, "evidence")
        for key, value in row.items():
            text(value, key)
        if row["claim"] != claim or row["status"] not in STATES:
            raise ValueError("Unknown status or mismatched evidence claim")
        if row["status"] == "supported":
            if (row["kind"], row["scope"]) != CLAIMS[claim]:
                raise ValueError(f"Unsupported documentary scope for {claim}")
            if row["artifact"].startswith("synthetic:"):
                # Synthetic support is valid only for software fixtures, never
                # evidence used by the real-source snapshot below.
                text(row["locator"], "synthetic locator")
        else:
            blockers.append(claim)
    lineage_known = evidence["lineage"]["status"] == "supported"
    if independent_confirmation and (not lineage_known or len(set(initiations)) < 2):
        blockers.append("independent_initiation_claim")
    return {
        "status": "blocked" if blockers else "documentary_review_only",
        "blockers": sorted(blockers),
        "independent_initiations_documented": lineage_known and len(set(initiations)) >= 2,
        "observed_single_availability": None,
        "observed_eligibility": None,
        "response_access_authorized": False,
        "response_header_access_authorized": False,
        "limitation": "Evidence classification and consistency do not certify source truth or authorize access",
    }


def index_unique(rows, key, names, name):
    if not isinstance(rows, list):
        raise ValueError(f"{name} must be a list")
    result = {}
    for row in rows:
        fields(row, names, name)
        identity = text(row[key], key)
        if identity in result:
            raise ValueError(f"Duplicate {name} key")
        result[identity] = row
    return result


GUIDE_FIELDS = {"id", "gene", "sequence", "transcript", "arm", "mismatch", "role"}
SAMPLE_FIELDS = {"id", "culture", "time", "initiation"}
CONSTRUCT_FIELDS = {"id", "left", "right"}
MATCH_FIELDS = {"double", "sample", "left_single", "right_single", "provenance", "weighting"}


def validate_design(metadata):
    """Validate response-free structures; invented fixtures are not real headers."""
    fields(metadata, {"control_role", "guides", "samples", "constructs", "matches", "assignments"}, "design")
    role = metadata["control_role"]
    if role not in {"nt", "scramble"}:
        raise ValueError("Explicit NT or scramble role required; safe-editing is not NT")
    guides = index_unique(metadata["guides"], "id", GUIDE_FIELDS, "guide")
    samples = index_unique(metadata["samples"], "id", SAMPLE_FIELDS, "sample")
    constructs = index_unique(metadata["constructs"], "id", CONSTRUCT_FIELDS, "construct")
    if not guides or not samples or not constructs:
        raise ValueError("Empty design cannot establish mappings")
    for guide in guides.values():
        for field in ("gene", "sequence", "transcript"):
            text(guide[field], field)
        if (guide["arm"] not in {"arm1", "arm2"} or guide["role"] not in {"target", role}
                or (guide["role"] == "target" and type(guide["mismatch"]) is not bool)
                or (guide["role"] == role and guide["mismatch"] is not None)):
            raise ValueError("Unknown arm, mismatch or control role")
        if any(base not in "ACGT" for base in guide["sequence"]):
            raise ValueError("Exact uppercase DNA required")
    cultures = {}
    culture_times = set()
    for sample in samples.values():
        for field in ("culture", "time", "initiation"):
            text(sample[field], field)
        if sample["time"] not in {"early", "late"}:
            raise ValueError("Unknown time role")
        culture = sample["culture"]
        if culture in cultures and cultures[culture] != sample["initiation"]:
            raise ValueError("Contradictory culture initiation")
        cultures[culture] = sample["initiation"]
        key = culture, sample["time"]
        if key in culture_times:
            raise ValueError("Ambiguous culture/time mapping")
        culture_times.add(key)
    if any((culture, time_role) not in culture_times for culture in cultures for time_role in ("early", "late")):
        raise ValueError("Incomplete early/late culture mapping")
    exact_constructs = set()

    def exact(guide):
        return tuple(guide[k] for k in ("gene", "sequence", "transcript", "arm", "mismatch", "role"))

    for construct in constructs.values():
        if construct["left"] not in guides or construct["right"] not in guides:
            raise ValueError("Unknown construct guide")
        left, right = guides[construct["left"]], guides[construct["right"]]
        if left["arm"] != "arm1" or right["arm"] != "arm2":
            raise ValueError("Ordered construct arm mismatch")
        key = exact(left), exact(right)
        if key in exact_constructs:
            raise ValueError("Duplicate exact ordered construct")
        exact_constructs.add(key)
    seen_matches = set()
    for match in metadata["matches"]:
        fields(match, MATCH_FIELDS, "match")
        if match["provenance"] != "measured_control_design" or match["weighting"] != "equal_construct_equal_culture":
            raise ValueError("Outcome-derived singles, eligibility or weights forbidden")
        if match["double"] not in constructs or match["sample"] not in samples:
            raise ValueError("Unknown double/sample mapping")
        key = match["double"], match["sample"]
        if key in seen_matches:
            raise ValueError("Duplicate matched-single assignment")
        seen_matches.add(key)
        double = constructs[match["double"]]
        if any(guides[double[side]]["role"] != "target" for side in ("left", "right")):
            raise ValueError("Double requires two targets")
        for side, other in (("left", "right"), ("right", "left")):
            reference = match[f"{side}_single"]
            fields(reference, {"construct", "sample"}, "single reference")
            if reference["sample"] != match["sample"]:
                raise ValueError("Single must match exact sample/time/culture")
            if reference["construct"] not in constructs:
                raise ValueError("Unknown single construct")
            single = constructs[reference["construct"]]
            if (exact(guides[single[side]]) != exact(guides[double[side]])
                    or guides[single[other]]["role"] != role):
                raise ValueError("Single requires exact guide/arm and explicit control partner")
    assignments = {}
    for row in metadata["assignments"]:
        fields(row, {"gene_a", "gene_b", "variant", "split"}, "assignment")
        for key in row:
            text(row[key], key)
        if row["gene_a"] == row["gene_b"] or row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Invalid distinct pair/split")
        pair = tuple(sorted((row["gene_a"], row["gene_b"])))
        if pair in assignments and assignments[pair] != row["split"]:
            raise ValueError("Canonical pair orientations/variants cross splits")
        assignments[pair] = row["split"]
    return {"design_consistent": True, "matched_design_links": len(seen_matches),
            "cultures": len(cultures), "initiations": len(set(cultures.values())),
            "observed_eligibility": None, "response_access_authorized": False,
            "limitation": "Mapping consistency only; no observed coverage or actual-source schema certified"}


def reviewed_snapshot():
    """Existing documentary record only, with no fabricated actual-source map."""
    sources = {}
    spidr = "docs/research_reset/wave3_spidr_results.md"
    shen = "docs/research_reset/wave3_source_decision.md"
    for source, artifact in (("spidr", spidr), ("shen2017", shen)):
        evidence = {}
        for claim in CLAIMS:
            evidence[claim] = dict(claim=claim, status="not_run" if claim in {"evaluator", "resources", "budget", "procedure", "endpoint"} else "unknown",
                artifact=artifact, kind="existing_review", scope="unresolved_prerequisite",
                locator="Official metadata review and unresolved blockers" if source == "spidr" else "A suitable alternative design, with a separate endpoint")
        if source == "spidr":
            evidence["lineage"] = dict(claim="lineage", status="supported", artifact=spidr,
                kind="initiation_record", scope="culture_initiation", locator="Rights and Lineage blockers: two split cultures after common transduction")
        sources[source] = {"evidence": evidence,
            "assessment": assess_evidence(evidence, initiations=["documented_shared_transduction"] if source == "spidr" else []),
            "design_evidence": "121198 perfect/perfect two-construct designed pairs; observed eligibility unknown" if source == "spidr" else "73-gene dense design; 657 gene/scramble designs do not prove both-arm observed singles",
            "endpoint": "normalized T14/T0 LFC, not per-doubling gamma" if source == "spidr" else "separate prospective knockout control-based growth slope, not published all-partner fitness",
            "forbidden_priors": ["activity", "GEMINI", "GI", "outcome_selected_guides"] if source == "spidr" else ["all_partner_fitted_singles", "absolute_fitness_probe_ranks", "derived_weights"],
        }
    return sources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_freeze(path):
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = freeze.get("metadata", {})
    scope = freeze.get("scope", {})
    if (freeze.get("schema") != "opal-wave4-freeze-v1" or metadata.get("authorized") is not True
            or metadata.get("metadata_only") is not True or scope.get("metadata_only") is not True
            or scope.get("source_response_access") is not False or scope.get("source_response_header_access") is not False
            or scope.get("biological_fit") is not False or scope.get("test_scoring") is not False
            or freeze.get("successor_authorized") is not False):
        raise ValueError("Root freeze does not authorize this metadata-only scope")
    for name, expected in freeze.get("files", {}).items():
        bound = (ROOT / name.replace("\\", "/")).resolve()
        if not bound.is_relative_to(ROOT) or digest(bound) != expected:
            raise ValueError(f"Frozen proposal/input drift: {name}")
    required = "docs/research_reset/wave4/B_revised.md"
    if required not in {name.replace("\\", "/") for name in freeze.get("files", {})}:
        raise ValueError("B revised proposal not bound")
    return freeze


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(output, freeze_path):
    import psutil
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing source-contract output")
    freeze = verify_freeze(freeze_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    started, cpu = time.monotonic(), time.process_time()
    process = psutil.Process()
    memory = process.memory_info()
    peak = max(memory.rss, getattr(memory, "peak_wset", 0))
    start = output.with_suffix(".start.json")
    failure = output.with_suffix(".failure.json")
    if failure.exists():
        raise FileExistsError("Preserve previous failed attempt")
    write_new(start, {"code_sha256": digest(__file__), "freeze_sha256": digest(freeze_path),
        "bound_record_sha256": {name: digest(ROOT/name) for name in ("docs/research_reset/wave3_spidr_results.md", "docs/research_reset/wave3_source_decision.md")},
        "response_access": False, "response_header_access": False})
    try:
        sources = reviewed_snapshot()
        memory = process.memory_info()
        peak = max(peak, memory.rss, getattr(memory, "peak_wset", 0))
        receipt = dict(protocol="wave4-source-evidence-contract-v1", sources=sources,
            response_access=False, response_header_access=False, new_source_outcome_access=False,
            fits=0, scores=0, actual_source_mapping_validated=False, requests=[],
            freeze_sha256=digest(freeze_path), code_sha256=digest(__file__),
            resources=dict(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                peak_rss_bytes=peak, network_bytes=0, guard="cooperative stage checks; not interrupting watchdog"))
        limits = freeze["metadata"]
        encoded = json.dumps(receipt, allow_nan=False).encode()
        if (receipt["resources"]["wall_seconds"] > min(120, limits["wall_seconds"])
                or peak > min(512*1024**2, limits["peak_bytes"])
                or len(encoded) > min(10*1024**2, limits["output_bytes"])):
            raise RuntimeError("Metadata resource cap exceeded")
        write_new(output, receipt)
        return receipt
    except Exception as exc:
        write_new(failure, {"error": repr(exc), "wall_seconds": time.monotonic()-started,
            "cpu_seconds": time.process_time()-cpu, "peak_rss_bytes": peak})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=ROOT/"results/research-wave4/freeze.json")
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    result = run(args.output, args.freeze)
    print(json.dumps({name: row["assessment"] for name, row in result["sources"].items()}, sort_keys=True))
