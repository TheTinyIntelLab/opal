"""Readiness fixtures are software metadata, never measurement evidence."""
import copy
import subprocess
import sys
from pathlib import Path

from opal.frontier_bridge.contracts import read_json, sha256
from opal.frontier_bridge.readiness import GATES, blank_package, validate_readiness


def test_blank_package_honestly_not_ready(tmp_path):
    result = validate_readiness(blank_package(), tmp_path)
    assert result["status"] == "measurement_not_ready"
    assert not result["ready_to_draft_confirmation_contract"]
    assert set(result["gates"]) == set(GATES)
    assert all(not g["passed"] for g in result["gates"].values())
    assert not result["independent_biological_confirmation"]
    assert not result["laboratory_execution_authorized"]


def test_evidence_hash_and_release_review_are_required(tmp_path):
    package = blank_package()
    evidence = tmp_path / "software_fixture.txt"
    evidence.write_text("Software test fixture, not a real license or measurement review.")
    package["gates"]["release_rights"] = dict(approved=True, reviewer="software fixture", reviewed_at="2026-10-12",
        evidence=[dict(path=evidence.name, sha256=sha256(evidence))], details=dict(
            license_id="software_fixture", permitted_uses=["unit_test"], research_use_authorized=True,
            redistribution_review_completed=True))
    result = validate_readiness(package, tmp_path)
    assert result["gates"]["release_rights"]["passed"]
    assert not result["ready_to_draft_confirmation_contract"]
    evidence.write_text("Changed test fixture")
    result = validate_readiness(package, tmp_path)
    assert not result["gates"]["release_rights"]["passed"]
    assert any("hash mismatch" in r for r in result["gates"]["release_rights"]["reasons"])


def test_technical_strata_do_not_establish_independent_initiation(tmp_path):
    package = blank_package()
    package["gates"]["independent_initiation"]["details"]["cultures"] = [
        dict(culture_id=f"software-stratum-{i}", context_id="fixture", initiation_id="same-initiation",
             initiation_date="fixture", provenance_record="software fixture") for i in range(2)]
    result = validate_readiness(package, tmp_path)
    assert any("duplicate initiation_id" in r for r in result["reasons"])


def test_reversed_query_bridge_overlap_is_rejected(tmp_path):
    package = blank_package()
    details = package["gates"]["prospective_information"]["details"]
    details.update(held_query_pair_ids=['["A","B"]'], bridge_pair_ids=['["B","A"]'])
    result = validate_readiness(package, tmp_path)
    assert any("identities overlap" in r for r in result["reasons"])


def test_claim_coverage_requires_empirical_results_and_reviewed_criteria(tmp_path):
    package = blank_package()
    details = package["gates"]["interval_validation"]["details"]
    details["coverage_validation"] = dict(passed=True)
    details["null_type_i_validation"] = dict(passed=True)
    result = validate_readiness(package, tmp_path)
    assert any("empirical result" in r for r in result["reasons"])


def test_cli_template_is_not_ready_and_refuses_overwrite(tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "template"
    command = [sys.executable, str(root / "tools/check_frontier_readiness.py"), "--template", "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    receipt = read_json(output / "measurement_readiness.json")
    assert not receipt["ready_to_draft_confirmation_contract"]
    before = sha256(output / "measurement_readiness.json")
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert sha256(output / "measurement_readiness.json") == before


def test_malformed_gate_objects_fail_closed(tmp_path):
    package = blank_package()
    package["gates"]["power_review"] = None
    assert not validate_readiness(package, tmp_path)["ready_to_draft_confirmation_contract"]
    assert not validate_readiness([], tmp_path)["ready_to_draft_confirmation_contract"]


def test_nonempty_strings_cannot_bypass_record_list_validation(tmp_path):
    package = blank_package()
    gate = package["gates"]["independent_initiation"]
    evidence = tmp_path / "fixture.txt"
    evidence.write_text("Software completeness fixture; no actual culture evidence.")
    gate.update(approved=True, reviewer="fixture", reviewed_at="fixture", evidence=[dict(path=evidence.name, sha256=sha256(evidence))])
    gate["details"].update(cultures="not a culture list", initiation_provenance_reviewed=True)
    result = validate_readiness(package, tmp_path)
    assert not result["gates"]["independent_initiation"]["passed"]
    assert any("nonempty list" in r for r in result["reasons"])


def test_malformed_evidence_paths_and_unhashable_culture_identifiers_fail_closed(tmp_path):
    package = blank_package()
    gate = package["gates"]["independent_initiation"]
    gate["evidence"] = [dict(path=123, sha256="hash")]
    gate["details"]["cultures"] = [dict(culture_id=[], context_id="x", initiation_id=[], initiation_date="x", provenance_record="x")]
    result = validate_readiness(package, tmp_path)
    assert not result["ready_to_draft_confirmation_contract"]
    assert any("literal identifier" in r for r in result["reasons"])


def test_interval_replicates_require_positive_observed_counts(tmp_path):
    package = blank_package()
    details = package["gates"]["interval_validation"]["details"]
    details["coverage_validation"] = dict(passed=True, empirical_result="fixture", reviewed_acceptance_criterion="fixture", replicates=True)
    result = validate_readiness(package, tmp_path)
    assert any("coverage_validation needs" in r for r in result["reasons"])
