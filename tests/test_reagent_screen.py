"""Synthetic exact-guide QA; these counts are not a biological result."""
from dataclasses import replace
import json
import math

import pytest

from src.reagent_screen import (Construct, CountObservation, Guide, ReagentScreen,
                                ScreenContract, aggregate, pair_split, strict_bool)


def synthetic_screen():
    contract = ScreenContract("synthetic_fixture", "synthetic_context", ("r1", "r2"),
                              "split_cultures_not_independent_initiations", "synthetic_only")
    a1 = Guide("A1", "A", "AAAA", "txA", "arm1", True, False)
    a2 = replace(a1, name="A2", sequence="AAAC")
    b1 = Guide("B1", "B", "CCCC", "txB", "arm2", True, False)
    b2 = replace(b1, name="B2", sequence="CCCG")
    n1 = Guide("NT1", "NT", "GGGG", "not_applicable", "arm1", False, True)
    n2 = replace(n1, name="NT2", sequence="GGGT")
    m1, m2 = replace(n1, arm="arm2"), replace(n2, arm="arm2")
    constructs, observations = [], []

    def add(identity, left, right, effect):
        constructs.append(Construct(identity, left, right))
        for rep in contract.replicates:
            observations.append(CountObservation(identity, rep, "T0", "T14", 110,
                                                 int(120 * 2 ** effect - 10)))

    add("nt1", n1, m1, 0)
    add("nt2", n2, m2, 0)
    for guide, effects in ((a1, (-3, -1)), (a2, (0, 0))):
        for j, (control, effect) in enumerate(zip((m1, m2), effects)):
            add(f"single_{guide.name}_{j}", guide, control, effect)
    for guide, effects in ((b1, (0, 0)), (b2, (-1, -3))):
        for j, (control, effect) in enumerate(zip((n1, n2), effects)):
            add(f"single_{guide.name}_{j}", control, guide, effect)
    add("pair1", a1, b1, -1)
    add("pair2", a2, b2, -1)
    return ReagentScreen(contract, tuple(constructs), tuple(observations))


def test_minimum_is_per_construct_before_equal_partner_construct_and_replicate_averaging():
    output = aggregate(synthetic_screen())
    pair = output["pairs"][0]
    assert pair["role"] == "primary"
    assert pair["stratum"] == "both_mismatched"
    assert pair["prediction"] == -2
    assert pair["target"] == -1
    # Taking min after averaging guide-specific singles would produce -1.
    assert pair["prediction"] != -1
    assert [p["constructs"] for p in pair["replicates"]] == [2, 2]
    assert output["counts"]["primary_pairs"] == 1
    assert output["exclusions"] == []


@pytest.mark.parametrize("value", [None, "", "unknown", "false", "0", 0, 1])
def test_unknown_boolean_annotation_fails_closed(value):
    with pytest.raises(ValueError, match="boolean"):
        strict_bool(value)


def test_explicit_boolean_parser_and_dataclass_never_coerce_strings():
    assert strict_bool("TRUE") is True
    assert strict_bool("FALSE") is False
    with pytest.raises(ValueError, match="Boolean"):
        replace(synthetic_screen().constructs[-1].left, mismatched="FALSE")


@pytest.mark.parametrize("changes", [dict(count_kind="normalized_counts"),
                                     dict(normalization="library_size"), dict(units="gamma"),
                                     dict(schema_status="unknown")])
def test_unknown_real_count_schema_normalization_units_and_status_stop(changes):
    with pytest.raises(ValueError):
        replace(synthetic_screen().contract, **changes)


def test_real_source_requires_schema_hash_and_explicit_control_mapping():
    contract = synthetic_screen().contract
    with pytest.raises(ValueError, match="Real source"):
        replace(contract, schema_status="qualified_source_metadata")
    real = replace(contract, schema_status="qualified_source_metadata",
                   source_schema_sha256="a" * 64, controls_evidence="reviewed_source_metadata.json")
    assert real.units == "log2_t14_over_t0"


@pytest.mark.parametrize("value", ["", "100", -1, 0.5, True, float("nan")])
def test_raw_count_schema_rejects_blanks_fractional_normalized_and_nonfinite(value):
    with pytest.raises(ValueError, match="Raw counts"):
        CountObservation("c", "r1", "T0", "T14", value, 10)


def test_missing_is_unavailable_and_both_t0_samples_must_be_above_50():
    screen = synthetic_screen()
    for replacement, reason in ((dict(t0=None), "missing_count_or_replicate"),
                                (dict(endpoint=None), "missing_count_or_replicate"),
                                (dict(t0=50), "t0_not_above_50_in_both_replicates")):
        rows = tuple(replace(o, **replacement) if o.construct == "pair1" and o.replicate == "r2"
                     else o for o in screen.observations)
        out = aggregate(replace(screen, observations=rows))
        assert not out["pairs"]
        assert {e["reason"] for e in out["exclusions"]} == {
            reason, "fewer_than_two_eligible_ordered_constructs"}


def test_observed_endpoint_zero_is_a_valid_count_not_missing():
    screen = synthetic_screen()
    rows = tuple(replace(o, endpoint=0) if o.construct == "pair1" else o
                 for o in screen.observations)
    out = aggregate(replace(screen, observations=rows))
    assert len(out["pairs"]) == 1
    assert math.isfinite(out["pairs"][0]["target"])


@pytest.mark.parametrize("mismatch", ["sequence", "transcript", "orientation"])
def test_single_matches_exact_sequence_transcript_and_vector_arm(mismatch):
    screen = synthetic_screen()
    constructs = []
    for c in screen.constructs:
        if c.identity.startswith("single_A1"):
            if mismatch == "orientation":
                c = replace(c, left=replace(c.right, arm="arm1"),
                            right=replace(c.left, arm="arm2"))
            else:
                c = replace(c, left=replace(c.left, name="different_guide", **{
                    mismatch: "TTTT" if mismatch == "sequence" else "other_transcript"}))
        constructs.append(c)
    out = aggregate(replace(screen, constructs=tuple(constructs)))
    assert not out["pairs"]
    assert any(e.get("construct") == "pair1" and e["reason"].startswith("missing_exact")
               for e in out["exclusions"])


def test_controls_are_required_and_are_never_inferred_from_gene_name():
    screen = synthetic_screen()
    rows = tuple(o for o in screen.observations if not o.construct.startswith("nt"))
    with pytest.raises(ValueError, match="non-targeting-pair controls"):
        aggregate(replace(screen, observations=rows))


def test_only_explicit_control_can_have_not_applicable_mismatch_annotation():
    screen = synthetic_screen()
    nt = screen.constructs[0].left
    assert replace(nt, mismatched=None).mismatched is None
    with pytest.raises(ValueError, match="Boolean"):
        replace(screen.constructs[-1].left, mismatched=None)
    with pytest.raises(ValueError, match="Mismatch strata"):
        _ = screen.constructs[0].stratum


def test_duplicates_unknown_rows_and_time_mismatches_are_rejected():
    screen = synthetic_screen()
    with pytest.raises(ValueError, match="Duplicate construct identity"):
        replace(screen, constructs=screen.constructs + screen.constructs[:1])
    with pytest.raises(ValueError, match="Duplicate ordered"):
        replace(screen, constructs=screen.constructs + (replace(screen.constructs[-1], identity="alias"),))
    with pytest.raises(ValueError, match="Duplicate construct/replicate"):
        replace(screen, observations=screen.observations + screen.observations[:1])
    for row, error in ((replace(screen.observations[0], construct="unknown"), "Unknown construct"),
                       (replace(screen.observations[0], endpoint_time="T7"), "time-point")):
        with pytest.raises(ValueError, match=error):
            replace(screen, observations=(row,) + screen.observations[1:])


def test_source_contract_and_aggregation_round_trip_are_exact_and_deterministic():
    screen = synthetic_screen()
    serialized = screen.to_json()
    restored = ReagentScreen.from_json(serialized)
    assert restored.to_json() == serialized
    assert aggregate(restored) == aggregate(screen)
    reordered = replace(screen, constructs=tuple(reversed(screen.constructs)),
                        observations=tuple(reversed(screen.observations)))
    assert aggregate(reordered) == aggregate(screen)
    altered = json.loads(serialized)
    altered["constructs"][0]["left"]["mismatched"] = "unknown"
    with pytest.raises(ValueError):
        ReagentScreen.from_json(json.dumps(altered))
    altered = json.loads(serialized)
    altered["constructs"][0]["activity"] = 3
    with pytest.raises(TypeError):
        ReagentScreen.from_json(json.dumps(altered))


def test_zero_sum_and_frozen_callable_prediction_api_and_nonfinite_refusal():
    screen = synthetic_screen()
    assert aggregate(screen, "zero")["pairs"][0]["prediction"] == 0
    assert aggregate(screen, "sum")["pairs"][0]["prediction"] == -2
    assert aggregate(screen, lambda a, b, c: min(a, b) + 1)["pairs"][0]["prediction"] == -1
    with pytest.raises(ValueError, match="finite scalar"):
        aggregate(screen, lambda *_: float("nan"))
    with pytest.raises(ValueError, match="Unknown reference"):
        aggregate(screen, "unknown")


def test_control_median_is_per_replicate_and_equal_replicates_follow_construct_averaging():
    screen = synthetic_screen()
    rows = []
    for o in screen.observations:
        # Multiplying endpoint+pseudocount shifts every raw r2 LFC by +1.
        # Its own NT median must remove this, including singles and pairs.
        if o.replicate == "r2":
            o = replace(o, endpoint=2 * (o.endpoint + 10) - 10)
        # A genuine r2 pair response change survives that centering.
        if o.replicate == "r2" and o.construct.startswith("pair"):
            o = replace(o, endpoint=2 * (o.endpoint + 10) - 10)
        rows.append(o)
    out = aggregate(replace(screen, observations=tuple(rows)))
    assert out["control_centers"] == {"r1": 0, "r2": 1}
    pair = out["pairs"][0]
    assert [r["target"] for r in pair["replicates"]] == [-1, 0]
    assert pair["target"] == -0.5
    assert pair["prediction"] == -2


def test_canonical_pair_split_groups_orientations_and_all_strata_and_replicates():
    assert pair_split("source", ("A", "B")) == pair_split("source", ("B", "A"))
    screen = synthetic_screen()
    # Reclassify only the double and its exact single guides, without using outcomes.
    for flags, stratum in (((False, False), "perfect_perfect"), ((True, False), "mixed")):
        constructs = []
        for c in screen.constructs:
            left = replace(c.left, mismatched=flags[0]) if not c.left.is_control else c.left
            right = replace(c.right, mismatched=flags[1]) if not c.right.is_control else c.right
            constructs.append(replace(c, left=left, right=right))
        out = aggregate(replace(screen, constructs=tuple(constructs)))
        assert out["pairs"][0]["stratum"] == stratum
        assert out["pairs"][0]["role"] == "secondary"
        assert out["pairs"][0]["split"] == aggregate(screen)["pairs"][0]["split"]
