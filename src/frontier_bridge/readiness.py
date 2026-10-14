"""Check local evidence for drafting a separate confirmation contract.

This is a metadata/evidence completeness check, never a power analysis or
permission to acquire cultures, execute an assay, score outcomes, or publish.
"""
from pathlib import Path
import math

from .contracts import canonical_pair, fingerprint, sha256


SCHEMA = "frontier_bridge_measurement_readiness_v1"
GATES = {
    "independent_initiation": ("cultures", "initiation_provenance_reviewed"),
    "pretreatment_contexts": ("contexts", "definitions_frozen_before_treatment"),
    "randomization": ("plate_block_assignments", "randomized_assignment_reviewed"),
    "literal_interventions": ("assignments", "assignment_equivalence_reviewed"),
    "independent_references": ("reference_matches", "independently_measured", "shared_reference_covariance_retained"),
    "measurement_counts": ("counts", "counts_supported_by_review", "campaign_costs"),
    "release_rights": ("license_id", "permitted_uses", "research_use_authorized", "redistribution_review_completed"),
    "prospective_information": ("held_query_pair_ids", "bridge_pair_ids", "global_exclusion_context_ids", "allowed_query_references", "freeze_record", "query_outcomes_unscored"),
    "pilot_covariance": ("pilot_id", "covariance_method", "shared_reference_covariance_retained", "empirical_covariance_reviewed"),
    "power_review": ("useful_margin", "margin_metric", "functional_endpoint", "sample_size_by_context", "empirical_power_reviewed", "assay_cost_reviewed", "biological_collaborator"),
    "interval_validation": ("estimand", "coverage_validation", "null_type_i_validation", "shared_reference_covariance_retained", "claim_specific_support_reviewed"),
    "owner_arrangements": ("measurement_owner", "laboratory_arrangements_reviewed", "compute_arrangements_reviewed", "registration_arrangements_reviewed"),
}
TRUE_FIELDS = {key for fields in GATES.values() for key in fields if key.endswith("reviewed") or key.endswith("retained")} | {
    "definitions_frozen_before_treatment", "independently_measured", "counts_supported_by_review",
    "research_use_authorized", "redistribution_review_completed", "query_outcomes_unscored"}


def blank_package():
    return dict(schema=SCHEMA, candidate_contract_id="frontier_bridge_confirmation_v1",
                status="unreviewed_template", gates={name: dict(approved=False, reviewer=None,
                    reviewed_at=None, evidence=[], details={key: None for key in fields})
                    for name, fields in GATES.items()})


def present(value):
    return value is not None and value != "" and value != [] and value != {}


def validate_readiness(package, package_root):
    """Fail closed on missing, contradictory, unreviewed or tampered evidence."""
    root = Path(package_root).resolve()
    failures, receipts = [], {}
    if not isinstance(package, dict):
        package = {}
        failures.append("package: JSON object required")
    if package.get("schema") != SCHEMA:
        failures.append("schema: unsupported readiness package")
    if package.get("candidate_contract_id") != "frontier_bridge_confirmation_v1":
        failures.append("candidate_contract_id: separate confirmation contract required")
    for name, fields in GATES.items():
        gates = package.get("gates", {})
        gate = gates.get(name, {}) if isinstance(gates, dict) else {}
        gate = gate if isinstance(gate, dict) else {}
        problems, verified = [], []
        if gate.get("approved") is not True:
            problems.append("review approval missing")
        for key in ("reviewer", "reviewed_at"):
            if not present(gate.get(key)):
                problems.append(f"{key} missing")
        details = gate.get("details", {})
        details = details if isinstance(details, dict) else {}
        for key in fields:
            value = details.get(key)
            if key in TRUE_FIELDS:
                if value is not True:
                    problems.append(f"{key} must be explicitly true")
            elif not present(value):
                problems.append(f"{key} missing")
        evidence = gate.get("evidence", [])
        if not isinstance(evidence, list) or not evidence:
            problems.append("reviewed evidence files missing")
            evidence = []
        for record in evidence:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"] or not isinstance(record.get("sha256"), str) or not record["sha256"]:
                problems.append("evidence path/hash missing")
                continue
            path = (root / record["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                problems.append(f"evidence unavailable inside package: {record['path']}")
            elif sha256(path) != record["sha256"]:
                problems.append(f"evidence hash mismatch: {record['path']}")
            else:
                verified.append(dict(path=record["path"], sha256=record["sha256"]))
        list_fields = {"independent_initiation": ("cultures",), "pretreatment_contexts": ("contexts",),
                       "literal_interventions": ("assignments",), "independent_references": ("reference_matches",),
                       "prospective_information": ("held_query_pair_ids", "bridge_pair_ids", "global_exclusion_context_ids"),
                       "release_rights": ("permitted_uses",)}
        for field in list_fields.get(name, ()):
            if not isinstance(details.get(field), list) or not details[field]:
                problems.append(f"{field} must be a nonempty list")
        if name == "independent_initiation" and isinstance(details.get("cultures"), list):
            cultures = details["cultures"]
            for c in cultures:
                if not isinstance(c, dict) or any(not present(c.get(k)) for k in ("culture_id", "context_id", "initiation_id", "initiation_date", "provenance_record")):
                    problems.append("each culture needs independent initiation and context provenance")
            complete = [c for c in cultures if isinstance(c, dict)]
            for key in ("culture_id", "initiation_id"):
                ids = [c.get(key) for c in complete]
                if any(not isinstance(value, str) or not value for value in ids):
                    problems.append(f"{key} must be a literal identifier")
                elif len(ids) != len(set(ids)):
                    problems.append(f"duplicate {key}; technical strata cannot establish independent cultures")
        if name == "literal_interventions" and isinstance(details.get("assignments"), list):
            for a in details["assignments"]:
                keys = ("assignment_id", "intervention_id", "reagent_id", "dose", "dose_unit", "treatment_duration", "duration_unit", "initiation_mode")
                if not isinstance(a, dict) or any(not present(a.get(k)) for k in keys):
                    problems.append("literal reagent/dose/time/initiation assignment missing")
                elif a["initiation_mode"] not in ("simultaneous", "sequential"):
                    problems.append("initiation_mode must be simultaneous or sequential")
        if name == "pretreatment_contexts" and isinstance(details.get("contexts"), list):
            for c in details["contexts"]:
                if not isinstance(c, dict) or any(not present(c.get(k)) for k in ("context_id", "pretreatment_definition", "definition_freeze_record")):
                    problems.append("context needs a pretreatment definition and freeze record")
        if name == "independent_references" and isinstance(details.get("reference_matches"), list):
            for r in details["reference_matches"]:
                if not isinstance(r, dict) or any(not present(r.get(k)) for k in ("culture_id", "control_match_ids", "single_match_ids", "reference_initiation_ids", "endpoint_scale", "normalization_order")):
                    problems.append("reference matching, initiation, endpoint and normalization provenance missing")
        if name == "measurement_counts":
            counts = details.get("counts", {})
            for key in ("contexts", "cultures", "pairs", "genes", "control_outcomes", "single_outcomes", "bridge_outcomes"):
                value = counts.get(key) if isinstance(counts, dict) else None
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    problems.append(f"counts.{key} must be an observed positive integer")
            cost = details.get("campaign_costs", {})
            for key in ("distinct_bridge_identities", "repeated_bridge_outcomes", "control_outcomes", "single_outcomes", "assay_cost", "currency"):
                if not isinstance(cost, dict) or not present(cost.get(key)):
                    problems.append(f"campaign_costs.{key} missing")
        if name == "prospective_information":
            try:
                query = {canonical_pair(p) for p in details.get("held_query_pair_ids", []) or []}
                bridges = {canonical_pair(p) for p in details.get("bridge_pair_ids", []) or []}
                if not query or not bridges:
                    problems.append("nonempty held query and bridge identities required")
                if query & bridges:
                    problems.append("held query and bridge identities overlap")
            except (ValueError, TypeError):
                problems.append("invalid canonical query/bridge identities")
        if name == "power_review":
            margin = details.get("useful_margin")
            if not isinstance(margin, (int, float)) or isinstance(margin, bool) or not math.isfinite(margin) or margin <= 0:
                problems.append("reviewed useful margin must be finite and positive")
            sizes = details.get("sample_size_by_context")
            if not isinstance(sizes, dict) or not sizes or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in sizes.values()):
                problems.append("reviewed sample size requires positive culture counts per context")
        if name == "interval_validation":
            for key in ("coverage_validation", "null_type_i_validation"):
                record = details.get(key)
                if not isinstance(record, dict) or record.get("passed") is not True or not isinstance(record.get("replicates"), int) or isinstance(record.get("replicates"), bool) or record.get("replicates", 0) <= 0 or any(not present(record.get(k)) for k in ("empirical_result", "reviewed_acceptance_criterion", "replicates")):
                    problems.append(f"{key} needs empirical result, reviewed criterion, replicates and passed=true")
        receipts[name] = dict(passed=not problems, reasons=problems, verified_evidence=verified)
        failures.extend(f"{name}: {p}" for p in problems)
    ready = not failures
    return dict(schema=SCHEMA, ready_to_draft_confirmation_contract=ready,
                status="evidence_complete_for_contract_review" if ready else "measurement_not_ready",
                package_hash=fingerprint(package), gates=receipts, reasons=failures,
                independent_biological_confirmation=False, frontier_claim=False,
                laboratory_execution_authorized=False, expression_scoring_performed=False,
                next_action="draft separately reviewed claim-specific confirmation contract" if ready else "obtain missing reviewed measurement and empirical-power evidence",
                scope="local evidence completeness and consistency; does not verify scientific adequacy of reviewer judgments")
