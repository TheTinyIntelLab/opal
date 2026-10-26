"""Invented-fixture checks for the source-free design-contract prerequisite.

These tests do not train a model, open a biological file, or qualify an assay.
"""
import hashlib
import math

import numpy as np
import pytest

from opal.frontier_design_contracts import (
    CONTRACT, FORMULA, LIMITS, SUPPORT_MEANING, Contrast, DeclaredCandidate,
    Dependence, DesignSplit, Provenance, apply_held_genes, apply_train_scaler,
    assign_construct_partitions, assign_hash_partitions, control_dependence,
    declared_log_contrast, declared_reference_class, declare_eligibility,
    declare_library, fit_digest, fit_train_scaler, make_fit_view, partition_index,
    role_meaning, seal_prediction, select_declared, support_reason, train_graph,
    unperturbed_class,
)


OK = Provenance(True, True, False, False)


def _guide(guide_id, sequence, target_id, role):
    return dict(guide_id=guide_id, sequence=sequence, target_id=target_id, role=role)


def _construct(construct_id, guide_a, guide_b, orientation):
    return dict(construct_id=construct_id, guide_a=guide_a, guide_b=guide_b, orientation=orientation)


def variant_library():
    guides = [
        _guide("a1", "ACGT", "A", "perturbation"),
        _guide("a2", "ACGA", "A", "perturbation"),
        _guide("b", "CCGT", "B", "perturbation"),
        _guide("c", "TCGT", "C", "perturbation"),
    ]
    constructs = [
        _construct("ab1", "a1", "b", "forward"),
        _construct("ab2", "b", "a2", "reverse"),
        _construct("ac", "a1", "c", "forward"),
    ]
    return declare_library(guides, constructs)


def distinct_library():
    guides = [
        _guide("a", "AAAA", "A", "perturbation"),
        _guide("b", "CCCC", "B", "perturbation"),
        _guide("c", "GGGG", "C", "perturbation"),
        _guide("d", "TTTT", "D", "perturbation"),
    ]
    constructs = [
        _construct("ab", "a", "b", "forward"),
        _construct("cd", "c", "d", "forward"),
        _construct("ac", "a", "c", "forward"),
    ]
    return declare_library(guides, constructs)


def screen():
    guides = [
        _guide("gA1", "AAAA", "A", "perturbation"),
        _guide("gA2", "CCCC", "A", "perturbation"),
        _guide("gB1", "GGGG", "B", "perturbation"),
        _guide("gC1", "TTTT", "C", "perturbation"),
        _guide("gD1", "AAAC", "D", "perturbation"),
        _guide("gN1", "AACC", "NTC1", "ntc"),
        _guide("gN2", "CCGG", "NTC1", "ntc"),
        _guide("gE1", "GGAA", "NE", "nonessential_target"),
    ]
    constructs = [
        _construct("ab", "gA1", "gB1", "forward"),
        _construct("ab-rev", "gB1", "gA1", "reverse"),
        _construct("ab-var", "gA2", "gB1", "forward"),
        _construct("bc", "gB1", "gC1", "forward"),
        _construct("cd", "gC1", "gD1", "forward"),
        _construct("a-ntc", "gA1", "gN1", "forward"),
        _construct("a-ntc-var", "gA1", "gN2", "forward"),
        _construct("a-ne", "gA1", "gE1", "forward"),
        _construct("b-ntc", "gB1", "gN1", "forward"),
        _construct("ntc", "gN1", "gN2", "forward"),
    ]
    library = declare_library(guides, constructs)
    labels = ["train", "train", "train", "train", "train", "train", "train", "train", "validation", "validation"]
    split = apply_held_genes(library, assign_construct_partitions(library, labels), ["B"])
    return library, split


def _fit(library, split, measured=None, labels=None, missing=None):
    genes = library.gene_ids
    if measured is None:
        measured = [True] * len(genes)
    by_gene = dict(zip(genes, measured))
    singles = [0.0 if bit else None for bit in measured]

    def pack(name):
        index = [int(i) for i in partition_index(split, name)]
        stored, gaps, support = [], [], []
        for position in index:
            construct = library.constructs[position]
            stored.append(0.0 if labels is None else labels.get(position, 0.0))
            gaps.append(False if missing is None else missing.get(position, False))
            ready = construct.class_name == "perturbation_pair" and all(by_gene[gene] for gene in construct.endpoints)
            support.append(ready)
        return index, stored, gaps, support

    train, validation = pack("train"), pack("validation")
    return make_fit_view(
        library, split, OK, genes, singles, measured,
        train[0], train[1], train[2], train[3],
        validation[0], validation[1], validation[2], validation[3],
    )


def _bucket(salt, key):
    digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) % 10
    if value < 6:
        return "train"
    if value < 8:
        return "validation"
    return "test"


def test_limits_stay_unqualified():
    assert dict(LIMITS) == {
        "colo1_qualified": False,
        "source_endpoint_qualified": False,
        "guide_mapping_qualified": False,
        "independence_qualified": False,
        "guide_nuisance_recovery_claim": False,
        "model_claim": False,
        "historical_all_task_gate": "failed",
        "provenance_is_proof": False,
        "shared_controls_are_replicates": False,
        "formula_is_verified_source_processing": False,
        "frozen_train_partner_support_enforced": False,
        "library_total_excludes_held_outcomes": False,
        "ntc_pair_proven_unperturbed": False,
    }
    with pytest.raises(TypeError):
        LIMITS["colo1_qualified"] = True


def test_roles_are_declared_and_ntc_is_not_a_proven_control():
    library, split = screen()
    classes = {construct.class_name for construct in library.constructs}
    assert "single_vs_ntc" in classes and "single_vs_nonessential" in classes and "ntc_pair" in classes
    assert role_meaning("ntc") == "non_targeting"
    assert role_meaning("nonessential_target") == "intervention"
    assert unperturbed_class("ntc_pair") is False
    assert unperturbed_class("single_vs_nonessential") is False
    assert declared_reference_class("ntc_pair") is True
    assert declared_reference_class("single_vs_ntc") is False
    assert declared_reference_class("single_vs_nonessential") is False
    assert library.gene_ids == ("A", "B", "C", "D", "NE")
    assert dict(zip(library.gene_ids, library.gene_roles))["NE"] == "nonessential_target"
    ntc = library.constructs[split.partition.index("validation")]
    assert support_reason(library, split, dict(zip(library.gene_ids, [True] * 5)), split.partition.index("validation")) == (
        False, "ntc_pair_is_not_a_perturbation",
    )
    assert ntc.class_name == "ntc_pair"


def test_public_split_cannot_place_one_variant_in_train_and_test():
    library = variant_library()
    forged = DesignSplit(CONTRACT, ("train", "test", "validation"), ())
    with pytest.raises(ValueError, match="canonical key is assigned to more than one partition"):
        make_fit_view(
            library, forged, OK, ("A", "B", "C"), [0.0, 0.0, 0.0], [True, True, True],
            [0], [1.0], [False], [True], [2], [2.0], [False], [True],
        )


def test_split_reused_from_another_library_cannot_cross_test():
    donor = distinct_library()
    reused = DesignSplit(CONTRACT, ("train", "test", "validation"), ())
    view = _fit(donor, reused)
    assert view.train.keys == ("pair|A|B",)
    assert list(view.test_index) == [1]
    with pytest.raises(ValueError, match="canonical key is assigned to more than one partition"):
        _fit(variant_library(), reused)


def test_forged_held_labels_and_unknown_genes_fail():
    library = distinct_library()
    with pytest.raises(ValueError, match="held partition requires incidence"):
        _fit(library, DesignSplit(CONTRACT, ("held", "train", "validation"), ()))
    with pytest.raises(ValueError, match="incident construct remains outside the held partition"):
        _fit(library, DesignSplit(CONTRACT, ("test", "train", "validation"), ("A",)))
    with pytest.raises(ValueError, match="held partition requires incidence"):
        _fit(library, DesignSplit(CONTRACT, ("held", "held", "held"), ("A",)))
    with pytest.raises(ValueError, match="unknown held gene"):
        _fit(library, DesignSplit(CONTRACT, ("train", "train", "train"), ("NOPE",)))


def test_assign_refuses_orientation_and_variant_disagreement():
    library = variant_library()
    with pytest.raises(ValueError, match="variants and orientations must share one split"):
        assign_construct_partitions(library, ["train", "test", "validation"])


def test_hash_and_held_gene_keep_every_variant_together():
    library, split = screen()
    hashed = assign_hash_partitions(library, "design-salt")
    groups = {}
    for construct, part in zip(library.constructs, hashed.partition):
        assert part == _bucket("design-salt", construct.canonical_key)
        groups.setdefault(construct.canonical_key, set()).add(part)
    assert all(len(parts) == 1 for parts in groups.values())
    assert split.partition == (
        "held", "held", "held", "held", "train", "train", "train", "train", "held", "validation",
    )
    assert split.held_genes == ("B",)
    held_keys = {library.constructs[i].canonical_key for i, part in enumerate(split.partition) if part == "held"}
    assert held_keys == {"pair|A|B", "pair|B|C", "single_ntc|B"}
    assert partition_index(split, "train").tolist() == [4, 5, 6, 7]
    assert partition_index(split, "validation").tolist() == [9]
    graph = train_graph(library, split)
    assert graph.components == (("A",), ("C", "D"))
    assert dict(graph.degrees) == {"A": 0, "C": 1, "D": 1}
    assert graph.n_components == 2


def test_prerequisite_support_is_not_frozen_partner_coverage():
    library, split = screen()
    measured = dict(zip(library.gene_ids, [True, True, True, True, False]))
    cd = split.partition.index("train")
    assert library.constructs[cd].canonical_key == "pair|C|D"
    assert support_reason(library, split, measured, cd) == (True, None)
    assert dict(train_graph(library, split).degrees)["D"] == 1
    assert LIMITS["frozen_train_partner_support_enforced"] is False
    held = split.partition.index("held")
    assert support_reason(library, split, measured, held) == (False, "held_incident_excluded")
    ne = [i for i, construct in enumerate(library.constructs) if construct.class_name == "single_vs_nonessential"][0]
    assert support_reason(library, split, measured, ne) == (False, "nonessential_target_is_an_intervention")
    ntc_single = [i for i, construct in enumerate(library.constructs) if construct.class_name == "single_vs_ntc" and "A" in construct.endpoints][0]
    assert support_reason(library, split, measured, ntc_single) == (False, "ntc_contrast_is_not_a_perturbation_pair")
    view = _fit(library, split, measured=[True, True, True, True, False])
    assert view.support_meaning == SUPPORT_MEANING
    assert view.frozen_partner_support_included is False
    assert view.declarations_are_proof is False
    assert view.train.support.tolist() == [True, False, False, False]
    assert view.validation.support.tolist() == [False]
    with pytest.raises(TypeError, match="test_labels"):
        make_fit_view(
            library, split, OK, library.gene_ids, [0.0] * 5, [True] * 5,
            [4], [1.0], [False], [True], [9], [1.0], [False], [False], test_labels=[9.0],
        )


def test_missing_labels_do_not_enter_the_fit_digest():
    library, split = screen()
    first = _fit(library, split, labels={4: 5.0, 7: 1e9}, missing={7: True})
    second = _fit(library, split, labels={4: 5.0, 7: None}, missing={7: True})
    changed = _fit(library, split, labels={4: 6.0, 7: None}, missing={7: True})
    assert math.isnan(first.train.labels[first.train.index.tolist().index(7)])
    assert fit_digest(first) == fit_digest(second)
    assert fit_digest(first) != fit_digest(changed)
    with pytest.raises(ValueError):
        first.train.labels[0] = 3.0


def test_provenance_and_eligibility_refuse_outcome_dependence():
    with pytest.raises(ValueError, match="held-double-derived"):
        make_fit_view_provenance(Provenance(True, True, True, False))
    with pytest.raises(ValueError, match="not independently measured"):
        make_fit_view_provenance(Provenance(False, True, False, False))
    with pytest.raises(ValueError, match="response-selected"):
        make_fit_view_provenance(Provenance(True, True, False, True))
    with pytest.raises(ValueError, match="depends on outcomes"):
        make_fit_view_provenance(Provenance(True, False, False, False))
    mask = declare_eligibility([True, False], OK)
    assert mask.tolist() == [True, False]
    with pytest.raises(TypeError, match="labels"):
        declare_eligibility([True], OK, labels=[1.0])


def make_fit_view_provenance(provenance):
    library = distinct_library()
    split = DesignSplit(CONTRACT, ("train", "validation", "test"), ())
    return make_fit_view(
        library, split, provenance, library.gene_ids, [0.0] * 4, [True] * 4,
        [0], [1.0], [False], [True], [1], [1.0], [False], [True],
    )


def test_log_contrast_hand_oracle_zero_missing_and_declared_total():
    blocked = declared_log_contrast(
        [0], [4], [False], [0, object()], [4, 4], [False, True],
        ["c0", "c1"], ["independent_control", "independent_control"], ["train", "train"],
        pseudocount=1, scale=4,
    )
    assert blocked.missing.tolist() == [True] and math.isnan(blocked.values[0])
    clean = declared_log_contrast(
        [0, 3, 0], [4, 4, 4], [False, False, True],
        [0, 3, object()], [4, 4, object()], [False, False, False],
        ["c0", "c1", "held"], ["independent_control", "independent_control", "held_pair"],
        ["train", "train", "held"], pseudocount=1, scale=4,
    )
    assert clean.values[0] == -1
    assert clean.values[1] == 1
    assert clean.values[1] - clean.values[0] == 2
    assert clean.missing.tolist() == [False, False, True]
    assert math.isnan(clean.values[2])
    assert clean.formula == FORMULA
    assert clean.train_reference_ids == ("c0", "c1")
    assert clean.independent_replicates is False
    assert clean.source_processing_verified is False
    assert clean.library_total_is_declaration is True
    assert clean.held_outcomes_absent_from_denominator is False
    other_total = declared_log_contrast(
        [1], [2], [False], [0, 3], [4, 4], [False, False],
        ["c0", "c1"], ["independent_control", "independent_control"], ["train", "train"],
        pseudocount=1, scale=4,
    )
    assert other_total.values[0] == math.log2(3) - 1
    assert other_total.values[0] != 0
    poisoned = declared_log_contrast(
        [0], [4], [False], [0, 3, 10**9], [4, 4, 4], [False, False, False],
        ["c0", "c1", "held"], ["independent_control", "independent_control", "held_pair"],
        ["train", "train", "held"], pseudocount=1, scale=4,
    )
    assert poisoned.values[0] == clean.values[0]
    with pytest.raises(ValueError, match="not a declared TRAIN independent control"):
        declared_log_contrast(
            [0], [4], [False], [0, 3], [4, 4], [False, False],
            ["c0", "pair"], ["independent_control", "pair_outcome"], ["train", "train"],
            pseudocount=1, scale=4,
        )
    with pytest.raises(ValueError):
        Contrast(np.array([0.0]), np.array([False]), FORMULA, ("c0",), False, False, True, True)
    with pytest.raises(TypeError, match="held_outcomes"):
        declared_log_contrast(
            [0], [4], [False], [0], [4], [False], ["c0"],
            ["independent_control"], ["train"], pseudocount=1, scale=4, held_outcomes=[9],
        )


def test_train_scaler_ignores_held_rows_and_missing_values():
    fitted = fit_train_scaler([0, 4, object()], [False, False, False], ["train_reference", "train_reference", "held_pair"])
    poisoned = fit_train_scaler([0, 4, 1e9], [False, False, False], ["train_reference", "train_reference", "held_pair"])
    assert fitted.center == 2 and fitted.scale == 2 and fitted.n == 2 and fitted.degenerate is False
    assert poisoned.center == fitted.center and poisoned.scale == fitted.scale
    applied = apply_train_scaler([0, 4, object()], [False, False, True], fitted)
    assert applied.tolist()[:2] == [-1, 1] and math.isnan(applied[2])
    dishonest = fit_train_scaler([0, 4, 12], [False, False, False], ["train_reference"] * 3)
    assert dishonest.center != fitted.center
    zero = fit_train_scaler([0, 0], [False, False], ["train_reference", "train_reference"])
    assert zero.degenerate is True and zero.scale == 1
    assert apply_train_scaler([0], [False], zero).tolist() == [0]
    with pytest.raises(TypeError, match="test_values"):
        fit_train_scaler([0, 4], [False, False], ["train_reference", "train_reference"], test_values=[9])


def test_shared_control_ids_are_dependence_keys():
    linked = control_dependence([("c1",), ("c1", "c2"), ("c2",), ("c3",)])
    assert linked.component.tolist() == [0, 0, 0, 1]
    assert linked.dependence_keys == (("c1", "c2"), ("c3",))
    assert linked.independent_replicates is False
    assert not hasattr(linked, "n_replicates")
    with pytest.raises(ValueError, match="not independent replicates"):
        Dependence(np.array([0]), (("c1",),), True)
    with pytest.raises(ValueError, match="explicit control ids"):
        control_dependence([()])


def test_selection_tie_break_has_no_test_score():
    low = DeclaredCandidate("c", "full", 5, 0.5)
    early = DeclaredCandidate("a", "simple", 1, 1.0)
    late = DeclaredCandidate("b", "simple", 1, 1.0)
    chosen = select_declared([late, low, early])
    assert chosen.winner == "c" and chosen.order == ("c", "a", "b")
    tied = select_declared([late, early])
    assert tied.winner == "a" and tied.tie_break == "validation_score,complexity,name,family"
    with pytest.raises(TypeError, match="test_scores"):
        select_declared([early], test_scores=[0.0])


def test_prediction_abstention_is_not_stored_as_zero():
    assert seal_prediction("cd", supported=True, value=0.0).value == 0.0
    with pytest.raises(ValueError, match="no numeric stand-in"):
        seal_prediction("cd", supported=False, value=0.0, abstention="held_incident_excluded")
    refused = seal_prediction("ab", supported=False, abstention="held_incident_excluded")
    assert refused.value is None and refused.abstention == "held_incident_excluded"


def test_declaration_failures():
    with pytest.raises(ValueError, match="self-pairs"):
        declare_library(
            [_guide("a1", "AAAA", "A", "perturbation"), _guide("a2", "CCCC", "A", "perturbation")],
            [_construct("aa", "a1", "a2", "forward")],
        )
    with pytest.raises(ValueError, match="orientation"):
        declare_library(
            [_guide("a", "AAAA", "A", "perturbation"), _guide("b", "CCCC", "B", "perturbation")],
            [_construct("ab", "a", "b", "reverse")],
        )
    with pytest.raises(ValueError, match="same sequence"):
        declare_library(
            [_guide("a", "AAAA", "A", "perturbation"), _guide("b", "AAAA", "B", "perturbation")],
            [_construct("ab", "a", "b", "forward")],
        )
    with pytest.raises(ValueError, match="whole nonnegative"):
        declared_log_contrast(
            [1.5], [4], [False], [0], [4], [False], ["c0"],
            ["independent_control"], ["train"], pseudocount=1, scale=4,
        )
    with pytest.raises(ValueError, match="positive constant"):
        declared_log_contrast(
            [0], [1], [False], [0], [1], [False], ["c0"],
            ["independent_control"], ["train"], pseudocount=0, scale=1,
        )
