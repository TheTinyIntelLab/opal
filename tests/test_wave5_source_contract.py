"""Invented response-free identities only, never actual reserved headers."""
import copy
import json

import pytest

from tools import wave5_source_contract as q


def fixture(source="spidr"):
    role = "nt" if source == "spidr" else "scramble"
    def ann(value):
        return dict(state="documented", value=value, basis="synthetic fixture")
    def guide(identity, gene, sequence, arm, target=True):
        return dict(id=identity, gene=gene, sequence=sequence, arm=arm,
                    role="target" if target else role, transcript=ann("tx"), mismatch=ann("perfect"))
    samples = [dict(id=f"invented_{c}_{t}", context="invented_cell", time_origin="transduction",
                    time=t, culture=c, initiation="shared") for c in ("c1", "c2") for t in ("d3", "d14", "d28")]
    return dict(source=source, control_role=role,
        guides=[guide("a", "A", "AAAA", "arm1"), guide("b", "B", "CCCC", "arm2"),
                guide("n1", "CONTROL", "GGGG", "arm1", False), guide("n2", "CONTROL", "TTTT", "arm2", False)],
        samples=samples,
        constructs=[dict(id="ab", left="a", right="b"), dict(id="an", left="a", right="n2"),
                    dict(id="nb", left="n1", right="b"), dict(id="nn", left="n1", right="n2")],
        matches=[dict(double="ab", sample=s["id"], left_single=dict(construct="an", sample=s["id"]),
                      right_single=dict(construct="nb", sample=s["id"]), provenance="explicit_control_design") for s in samples],
        anchors=[dict(sample=s["id"], construct="nn") for s in samples],
        assignments=[dict(gene_a="A", gene_b="B", variant="perfect", split="test"),
                     dict(gene_a="B", gene_b="A", variant="mixed", split="test")],
        selection={k: "response_free_design_only" for k in ("features", "eligibility", "weights", "tuning")},
        aggregation=list(q.OPERATIONS))


@pytest.mark.parametrize("source", ["spidr", "shen2017"])
def test_positive_design_is_not_observed_coverage_or_permission(source):
    r = q.validate_metadata(fixture(source))
    assert r["designed_pairs"] == 1 and r["matched_design_links"] == 6
    assert r["observed_eligibility"] is None and r["observed_singles"] is None
    assert r["response_access_authorized"] is r["response_header_access_authorized"] is False


@pytest.mark.parametrize("field", ["transcript", "mismatch"])
def test_shen_unknown_annotation_is_explicit_and_not_invented(field):
    d = fixture("shen2017")
    d["guides"][0][field].update(state="unknown", value=None)
    assert q.validate_metadata(d)["design_consistent"]
    d["guides"][0][field]["value"] = "invented value"
    with pytest.raises(ValueError, match="invent"):
        q.validate_metadata(d)
    d = fixture()
    d["guides"][0][field].update(state="unknown", value=None)
    with pytest.raises(ValueError, match="unresolved"):
        q.validate_metadata(d)


@pytest.mark.parametrize("section", ["guides", "samples", "constructs", "matches", "anchors", "assignments"])
def test_duplicate_keys_and_links_reject(section):
    d = fixture()
    d[section].append(copy.deepcopy(d[section][0]))
    with pytest.raises(ValueError):
        q.validate_metadata(d)


@pytest.mark.parametrize("section", ["matches", "anchors", "assignments"])
def test_incomplete_required_links_reject(section):
    d = fixture()
    d[section] = []
    with pytest.raises(ValueError, match="Incomplete|Missing"):
        q.validate_metadata(d)


@pytest.mark.parametrize("field", ["sequence", "transcript", "mismatch"])
def test_same_gene_wrong_exact_single_reject(field):
    d = fixture()
    other = copy.deepcopy(d["guides"][0])
    other["id"] = "other"
    if field == "sequence":
        other[field] = "ACAC"
    else:
        other[field]["value"] = "other"
    d["guides"].append(other)
    d["constructs"][1]["left"] = "other"
    with pytest.raises(ValueError, match="Exact"):
        q.validate_metadata(d)


@pytest.mark.parametrize("mutation", ["arm", "unknown_guide", "context", "time_origin", "initiation", "duplicate_time", "wrong_sample", "non_control_anchor", "unlinked_pair", "split", "missing_time", "role"])
def test_adversarial_joins_and_assignment_reject(mutation):
    d = fixture()
    if mutation == "arm":
        d["constructs"][0].update(left="b", right="a")
    elif mutation == "unknown_guide":
        d["constructs"][0]["left"] = "missing"
    elif mutation in {"context", "time_origin", "initiation"}:
        d["samples"][1][mutation] = "contradiction"
    elif mutation == "duplicate_time":
        d["samples"][1]["time"] = "d3"
    elif mutation == "wrong_sample":
        d["matches"][0]["left_single"]["sample"] = "invented_c2_d3"
    elif mutation == "non_control_anchor":
        d["anchors"][0]["construct"] = "an"
    elif mutation == "unlinked_pair":
        d["assignments"][0]["gene_a"] = "UNKNOWN"
    elif mutation == "split":
        d["assignments"][1]["split"] = "train"
    elif mutation == "missing_time":
        d["samples"] = [d["samples"][0]]
    else:
        d["guides"][2]["role"] = "safe_editing"
    with pytest.raises(ValueError):
        q.validate_metadata(d)


@pytest.mark.parametrize("forbidden", sorted(q.FORBIDDEN))
@pytest.mark.parametrize("location", ["features", "eligibility", "weights", "tuning", "single"])
def test_response_derived_provenance_reject(forbidden, location):
    d = fixture()
    if location == "single":
        d["matches"][0]["provenance"] = forbidden
    else:
        d["selection"][location] = forbidden
    with pytest.raises(ValueError, match="provenance"):
        q.validate_metadata(d)


@pytest.mark.parametrize("operations", [q.OPERATIONS[::-1], q.OPERATIONS[1:], ["pool_constructs", "equal_pair_score"], ["predict_construct", "fitness_weighted_culture", "equal_pair_score"]])
def test_aggregation_order_and_weights_reject(operations):
    d = fixture()
    d["aggregation"] = operations
    with pytest.raises(ValueError, match="aggregation"):
        q.validate_metadata(d)


def test_metadata_order_permutation_preserves_result():
    d = fixture()
    expected = q.validate_metadata(d)
    for section in ("guides", "samples", "constructs", "matches", "anchors", "assignments"):
        d[section].reverse()
    assert q.validate_metadata(d) == expected


@pytest.mark.parametrize("field", ["counts", "response", "header", "fitness"])
def test_response_fields_reject(field):
    d = fixture()
    d[field] = []
    with pytest.raises(ValueError, match="schema"):
        q.validate_metadata(d)


def endpoint(source="spidr"):
    return dict(source=source, endpoint="normalized_T14_T0_LFC" if source == "spidr" else "prospective_control_relative_growth_slope",
        input_units="normalized_counts" if source == "spidr" else "source_specific_counts",
        operations=["documented_normalization_already_applied", "pseudocount_in_normalized_units", "late_early_log_ratio", "documented_NT_centering"] if source == "spidr" else ["control_relative_abundance", "fixed_time_log_slope", "explicit_control_centering"],
        selection_basis="prospective_response_free", actual_invocation="synthetic locator, not source verified",
        reference_membership="synthetic explicit control design", time_zero_missingness="synthetic prospective fixed rule")


@pytest.mark.parametrize("source", ["spidr", "shen2017"])
def test_consistent_endpoint_declaration_is_not_actual_verification(source):
    r = q.validate_endpoint(endpoint(source))
    assert not r["actual_invocation_verified"] and not r["numeric_endpoint_tested"]


@pytest.mark.parametrize("op", ["round_counts", "normalize_again", "raw_T0_gt_50", "all_partner_fit", "fitness_rank_weighting"])
def test_forbidden_endpoint_operations_reject(op):
    d = endpoint()
    d["operations"].append(op)
    with pytest.raises(ValueError, match="units/operation"):
        q.validate_endpoint(d)


@pytest.mark.parametrize("key,value", [("endpoint", "gamma"), ("input_units", "raw_integer"), ("selection_basis", "held_double"), ("actual_invocation", "")])
def test_endpoint_units_selection_and_invocation_required(key, value):
    d = endpoint()
    d[key] = value
    with pytest.raises(ValueError):
        q.validate_endpoint(d)


def test_missing_authorization_and_overwrite_reject(tmp_path):
    f, a = tmp_path/"freeze.json", tmp_path/"authorization.json"
    f.write_text("{}")
    a.write_text("{}")
    with pytest.raises(ValueError, match="authorization"):
        q.verify_authorization(f, a)
    p = tmp_path/"out.json"
    q.old.write_new(p, {"first": True})
    with pytest.raises(FileExistsError):
        q.old.write_new(p, {"second": True})


def test_portable_synthetic_authorization_and_custody_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    p = tmp_path/"docs/research_reset/wave5/B_revised.md"
    p.parent.mkdir(parents=True)
    p.write_text("synthetic proposal")
    scope = dict(source_primary_narrative_metadata_only=True, reserved_response_access=False,
                 reserved_response_header_access=False, biological_fitting=False, new_biological_scoring=False, scientific_successor=False)
    f, a = tmp_path/"freeze.json", tmp_path/"authorization.json"
    f.write_text(json.dumps(dict(schema="opal-wave5-plan-freeze-v1", permissions=scope, files={p.relative_to(tmp_path).as_posix(): q.digest(p)})))
    a.write_text(json.dumps(dict(schema="opal-wave5-execution-authorization-v1", freeze_sha256=q.digest(f), B_source_authorized=True, scope=scope)))
    assert q.verify_authorization(f, a)
    p.write_text("drift")
    with pytest.raises(ValueError, match="drift"):
        q.verify_authorization(f, a)


def test_snapshot_preserves_unexecuted_source_gates():
    for row in q.old.reviewed_snapshot().values():
        for claim in ("evaluator", "resources", "budget", "procedure"):
            assert row["evidence"][claim]["status"] == "not_run"
        assert row["assessment"]["observed_eligibility"] is None
        assert not row["assessment"]["response_access_authorized"]
