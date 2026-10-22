"""Adversarial synthetic metadata only; no real reserved header or outcome."""
import copy
import json

import pytest

from tools import wave4_source_contract as q


def evidence():
    return {claim: dict(claim=claim, status="supported", artifact=f"synthetic:{claim}",
                       kind=kind, scope=scope, locator="fixture section 1")
            for claim, (kind, scope) in q.CLAIMS.items()}


def fixture():
    def guide(identity, gene, seq, arm, role="target", mismatch=False, transcript="synthetic_tx"):
        return dict(id=identity, gene=gene, sequence=seq, transcript=transcript,
                    arm=arm, role=role, mismatch=mismatch)
    return dict(control_role="nt", guides=[
        guide("a", "A", "AAAA", "arm1"), guide("b", "B", "CCCC", "arm2"),
        guide("n1", "NT", "GGGG", "arm1", "nt", None),
        guide("n2", "NT", "TTTT", "arm2", "nt", None)],
        samples=[dict(id=f"synthetic_{c}_{t}", culture=c, time=t, initiation="shared")
                 for c in ("culture1", "culture2") for t in ("early", "late")],
        constructs=[dict(id="ab", left="a", right="b"), dict(id="an", left="a", right="n2"),
                    dict(id="nb", left="n1", right="b"), dict(id="nn", left="n1", right="n2")],
        matches=[dict(double="ab", sample="synthetic_culture1_early",
                      left_single=dict(construct="an", sample="synthetic_culture1_early"),
                      right_single=dict(construct="nb", sample="synthetic_culture1_early"),
                      provenance="measured_control_design", weighting="equal_construct_equal_culture")],
        assignments=[dict(gene_a="A", gene_b="B", variant="perfect", split="test"),
                     dict(gene_a="B", gene_b="A", variant="mixed", split="test")])


def test_complete_fixture_still_denies_access_and_observed_eligibility():
    result = q.validate_design(fixture())
    assert result["design_consistent"]
    assert result["cultures"] == 2 and result["initiations"] == 1
    assert result["observed_eligibility"] is None
    assert result["response_access_authorized"] is False
    result = q.assess_evidence(evidence(), initiations=["one", "two"])
    assert result["status"] == "documentary_review_only"
    assert result["response_access_authorized"] is False
    assert result["response_header_access_authorized"] is False
    assert result["observed_single_availability"] is None


@pytest.mark.parametrize("claim", sorted(q.CLAIMS))
@pytest.mark.parametrize("status", ["blocked", "unknown", "not_run"])
def test_each_unresolved_gate_blocks(claim, status):
    rows = evidence()
    rows[claim]["status"] = status
    assert claim in q.assess_evidence(rows, initiations=["one", "two"])["blockers"]


@pytest.mark.parametrize("claim,kind,scope", [
    ("sample_columns", "archive_titles", "actual_response_columns"),
    ("invocation", "function_definition", "actual_invocation"),
    ("rights", "article_license", "specific_supplement"),
    ("single_design", "gene_name_overlap", "exact_guide_arm_design"),
    ("evaluator", "metadata_tests", "equal_information_leakage_strict_replay"),
    ("budget", "old_failed_budget", "unique_pair_union_singles_controls")])
def test_generic_or_wrong_scope_evidence_cannot_support_specific_claim(claim, kind, scope):
    rows = evidence()
    rows[claim].update(kind=kind, scope=scope)
    with pytest.raises(ValueError, match="scope"):
        q.assess_evidence(rows, initiations=["one", "two"])


def test_split_cultures_do_not_establish_independent_initiation_but_task_claim_is_explicit():
    rows = evidence()
    assert "independent_initiation_claim" in q.assess_evidence(rows, initiations=["shared", "shared"])["blockers"]
    result = q.assess_evidence(rows, initiations=["shared"], independent_confirmation=False)
    assert not result["independent_initiations_documented"]
    assert result["status"] == "documentary_review_only"
    assert not result["response_access_authorized"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "empty_locator", "bad_status", "wrong_claim"])
def test_malformed_evidence_rejected(mutation):
    rows = evidence()
    if mutation == "missing":
        del rows["rights"]
    elif mutation == "extra":
        rows["new"] = rows["rights"]
    elif mutation == "empty_locator":
        rows["rights"]["locator"] = ""
    elif mutation == "bad_status":
        rows["rights"]["status"] = True
    else:
        rows["rights"]["claim"] = "controls"
    with pytest.raises(ValueError):
        q.assess_evidence(rows, initiations=["one", "two"])


@pytest.mark.parametrize("section", ["guides", "samples", "constructs"])
def test_duplicate_keys_rejected(section):
    data = fixture()
    data[section].append(copy.deepcopy(data[section][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        q.validate_design(data)


def test_renamed_exact_duplicate_construct_rejected():
    data = fixture()
    data["guides"].append(dict(data["guides"][0], id="renamed"))
    data["constructs"].append(dict(id="duplicate", left="renamed", right="b"))
    with pytest.raises(ValueError, match="exact ordered"):
        q.validate_design(data)


@pytest.mark.parametrize("field,value", [("sequence", "ACAC"), ("transcript", "other_tx"), ("mismatch", True)])
def test_same_gene_wrong_exact_single_fails(field, value):
    data = fixture()
    data["guides"].append(dict(data["guides"][0], id="other", **{field: value}))
    data["constructs"][1]["left"] = "other"
    with pytest.raises(ValueError, match="exact guide"):
        q.validate_design(data)


@pytest.mark.parametrize("sample", ["synthetic_culture1_late", "synthetic_culture2_early", "missing"])
def test_wrong_time_culture_or_unknown_single_sample_rejected(sample):
    data = fixture()
    data["matches"][0]["left_single"]["sample"] = sample
    with pytest.raises(ValueError, match="sample/time/culture"):
        q.validate_design(data)


@pytest.mark.parametrize("mutation", ["arm_swap", "safe_editing", "unknown_mismatch", "non_dna", "missing_late", "ambiguous_time", "lineage_conflict"])
def test_ambiguous_or_incompatible_metadata_rejected(mutation):
    data = fixture()
    if mutation == "arm_swap":
        data["constructs"][0].update(left="b", right="a")
    elif mutation == "safe_editing":
        data["guides"][2]["role"] = "safe_editing"
    elif mutation == "unknown_mismatch":
        data["guides"][0]["mismatch"] = None
    elif mutation == "non_dna":
        data["guides"][0]["sequence"] = "AAAN"
    elif mutation == "missing_late":
        del data["samples"][1]
    elif mutation == "ambiguous_time":
        data["samples"].append(dict(data["samples"][0], id="duplicate_time"))
    else:
        data["samples"][1]["initiation"] = "different"
    with pytest.raises(ValueError):
        q.validate_design(data)


@pytest.mark.parametrize("forbidden", ["all_partner_fit", "absolute_fitness_rank", "activity", "GEMINI", "GI", "held_double"])
def test_outcome_derived_single_or_eligibility_provenance_rejected(forbidden):
    data = fixture()
    data["matches"][0]["provenance"] = forbidden
    with pytest.raises(ValueError, match="Outcome-derived"):
        q.validate_design(data)


def test_outcome_weights_and_split_leakage_rejected():
    data = fixture()
    data["matches"][0]["weighting"] = "absolute_fitness_rank"
    with pytest.raises(ValueError, match="weights"):
        q.validate_design(data)
    data = fixture()
    data["assignments"][1]["split"] = "train"
    with pytest.raises(ValueError, match="cross splits"):
        q.validate_design(data)


def test_metadata_order_and_variant_order_do_not_change_consistency():
    data = fixture()
    expected = q.validate_design(data)
    for section in ("guides", "samples", "constructs", "assignments"):
        data[section].reverse()
    assert q.validate_design(data) == expected


def test_existing_record_snapshot_is_blocked_and_does_not_invent_actual_map():
    result = q.reviewed_snapshot()
    assert set(result) == {"spidr", "shen2017"}
    for source in result.values():
        assert source["assessment"]["status"] == "blocked"
        assert source["assessment"]["observed_eligibility"] is None
        assert "sample_columns" in source["assessment"]["blockers"]
        assert "invocation" in source["assessment"]["blockers"]
        assert "rights" in source["assessment"]["blockers"]
    assert "independent_initiation_claim" in result["spidr"]["assessment"]["blockers"]


def test_no_response_fields_or_unknown_schema_are_accepted():
    data = fixture()
    data["counts"] = [1, 2]
    with pytest.raises(ValueError, match="schema"):
        q.validate_design(data)


def test_freeze_required_and_writes_refuse_overwrite(tmp_path):
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({"scope": {"metadata_only": True}}))
    with pytest.raises(ValueError, match="Root freeze"):
        q.verify_freeze(path)
    out = tmp_path / "result.json"
    q.write_new(out, {"first": True})
    with pytest.raises(FileExistsError):
        q.write_new(out, {"replacement": True})


def test_custody_drift_is_rejected_before_review(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    bound = tmp_path / "docs/research_reset/wave4/B_revised.md"
    bound.parent.mkdir(parents=True)
    bound.write_text("changed")
    freeze = dict(schema="opal-wave4-freeze-v1", successor_authorized=False,
        metadata=dict(authorized=True, metadata_only=True),
        scope=dict(metadata_only=True, source_response_access=False, source_response_header_access=False,
                   biological_fit=False, test_scoring=False),
        files={"docs/research_reset/wave4/B_revised.md": "0"*64})
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="drift"):
        q.verify_freeze(path)
