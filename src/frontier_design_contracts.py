"""Source-free design contracts for guide identity, splits, and fixture arithmetic.

``opal-frontier-design-contracts-v1`` is a prerequisite API. It does not qualify
COLO1, verify an assay endpoint, prove that singles were measured independently,
recover guide nuisance, or fit a model. Provenance flags and TRAIN reference
membership are caller declarations, not evidence that those statements are true.

The only arithmetic helper is the caller-declared recipe
``caller-declared-raw-count-normalized-log2-median-reference-v1``:

    norm(count, total) = log2(pseudocount + count * scale / total)
    contrast = norm(sample) - median(norm of declared TRAIN independent controls)

``pseudocount`` and ``scale`` have no defaults. An even-sized median is the mean
of the two central sorted norms. A raw count of zero is observed. A missing flag
is not a zero and is not read. Held-pair columns and rows are not read, so they
cannot move the reference median or the TRAIN center. A dishonest tag that calls
a held outcome an independent TRAIN control is not detectable from the numbers.

``library_total`` and provenance flags are caller declarations. This module
cannot inspect a precomputed denominator, so it cannot prove that held-pair
outcomes were left out of that total. Fit-view support means both perturbation
endpoints have a declared measured single. It is not the frozen tournament rule
that each query gene must have two distinct TRAIN partners, and it does not
establish frozen-policy coverage. An NTC/NTC construct is only a declared
reference class, not proof of an untreated or biologically unperturbed control.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np


CONTRACT = "opal-frontier-design-contracts-v1"
FORMULA = "caller-declared-raw-count-normalized-log2-median-reference-v1"
LIMITS = MappingProxyType({
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
})
SUPPORT_MEANING = "declared_measured_single_prerequisite"
_ROLES = {"perturbation": 0, "nonessential_target": 1, "ntc": 2}
_FIT_PARTITIONS = ("train", "validation", "test")
_ALL_PARTITIONS = _FIT_PARTITIONS + ("held",)
_CLASSES = (
    "perturbation_pair", "single_vs_ntc", "single_vs_nonessential",
    "ntc_pair", "nonessential_pair", "ntc_vs_nonessential",
)


@dataclass(frozen=True)
class Guide:
    guide_id: str
    sequence: str
    target_id: str
    role: str


@dataclass(frozen=True)
class Construct:
    construct_id: str
    guide_a: str
    guide_b: str
    orientation: str
    canonical_key: str
    class_name: str
    endpoints: tuple


@dataclass(frozen=True)
class Library:
    contract: str
    guides: tuple
    constructs: tuple
    gene_ids: tuple
    gene_roles: tuple


@dataclass(frozen=True)
class Provenance:
    """Four explicit declarations. Accepted values are not measurement proof.

    In particular, they do not show that held-pair outcomes are absent from a
    precomputed library total or any other caller-supplied denominator.
    """

    singles_independently_measured: bool
    eligibility_outcome_independent: bool
    singles_derived_from_held_doubles: bool
    eligibility_response_selected: bool

    def __post_init__(self):
        for name in (
            "singles_independently_measured", "eligibility_outcome_independent",
            "singles_derived_from_held_doubles", "eligibility_response_selected",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an explicit bool declaration")


@dataclass(frozen=True)
class DesignSplit:
    contract: str
    partition: tuple
    held_genes: tuple


@dataclass(frozen=True)
class FitRows:
    """``support`` is the measured-single prerequisite, not frozen partner coverage."""

    index: np.ndarray
    labels: np.ndarray
    missing: np.ndarray
    support: np.ndarray
    keys: tuple
    classes: tuple
    orientations: tuple
    arms: tuple


@dataclass(frozen=True)
class FitView:
    """TRAIN and VALIDATION rows only. TEST and held indices carry no labels."""

    contract: str
    train: FitRows
    validation: FitRows
    single_ids: tuple
    single_values: np.ndarray
    single_measured: np.ndarray
    test_index: np.ndarray
    held_index: np.ndarray
    provenance: tuple
    declarations_are_proof: bool
    source_endpoint_qualified: bool
    support_meaning: str
    frozen_partner_support_included: bool

    def __post_init__(self):
        if self.declarations_are_proof or self.source_endpoint_qualified or self.frozen_partner_support_included:
            raise ValueError("a fit view cannot claim proof, source qualification, or frozen partner coverage")
        if self.support_meaning != SUPPORT_MEANING:
            raise ValueError("fit-view support is only the declared measured-single prerequisite")
        if self.provenance != (True, True, False, False):
            raise ValueError("fit view provenance is not the accepted declaration")


@dataclass(frozen=True)
class Contrast:
    values: np.ndarray
    missing: np.ndarray
    formula: str
    train_reference_ids: tuple
    independent_replicates: bool
    source_processing_verified: bool
    library_total_is_declaration: bool = True
    held_outcomes_absent_from_denominator: bool = False

    def __post_init__(self):
        if self.formula != FORMULA or self.independent_replicates or self.source_processing_verified:
            raise ValueError("contrast cannot claim a verified source endpoint or independent replicates")
        if not self.library_total_is_declaration or self.held_outcomes_absent_from_denominator:
            raise ValueError("library_total is a caller declaration; held outcomes in a precomputed denominator are not detectable")


@dataclass(frozen=True)
class TrainScaler:
    center: float
    scale: float
    degenerate: bool
    n: int
    formula: str
    source_processing_verified: bool

    def __post_init__(self):
        if self.formula != FORMULA or self.source_processing_verified:
            raise ValueError("scaler cannot claim verified source processing")
        if not math.isfinite(self.center) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("TRAIN scaler arithmetic must remain finite with positive scale")


@dataclass(frozen=True)
class Dependence:
    """Shared control ids are dependence keys, never independent replicates."""

    component: np.ndarray
    dependence_keys: tuple
    independent_replicates: bool

    def __post_init__(self):
        if self.independent_replicates:
            raise ValueError("shared control ids are dependence keys, not independent replicates")


@dataclass(frozen=True)
class TrainGraph:
    """Distinct TRAIN canonical pairs. Construct copies do not add degree."""

    components: tuple
    degrees: tuple
    n_components: int


@dataclass(frozen=True)
class DeclaredCandidate:
    name: str
    family: str
    complexity: int
    validation_score: float


@dataclass(frozen=True)
class Selection:
    winner: str
    order: tuple
    tie_break: str


@dataclass(frozen=True)
class Prediction:
    key: str
    supported: bool
    value: float | None
    abstention: str | None


def role_meaning(role):
    """Return the declared role label, not a biological proof."""
    if role == "ntc":
        return "non_targeting"
    if role == "nonessential_target":
        return "intervention"
    if role == "perturbation":
        return "perturbation"
    raise ValueError("unknown role")


def unperturbed_class(class_name):
    """No class is proven untreated or biologically unperturbed, including NTC/NTC."""
    if class_name not in _CLASSES:
        raise ValueError("unknown construct class")
    return False


def declared_reference_class(class_name):
    """NTC/NTC is only a declared reference class, not an untreated proof."""
    if class_name not in _CLASSES:
        raise ValueError("unknown construct class")
    return class_name == "ntc_pair"


def declare_library(guides, constructs):
    """Require an explicit guide id, sequence, and target or control on both arms."""
    if type(guides) not in (list, tuple) or type(constructs) not in (list, tuple):
        raise TypeError("guides and constructs must be explicit sequences")
    if not guides or not constructs:
        raise ValueError("a declared library needs guides and constructs")
    guide_rows = tuple(sorted((_guide(item) for item in guides), key=lambda g: g.guide_id))
    if len({g.guide_id for g in guide_rows}) != len(guide_rows):
        raise ValueError("guide ids must be unique")
    seen_sequence = {}
    roles = {}
    for guide in guide_rows:
        previous = seen_sequence.get(guide.sequence)
        identity = (guide.target_id, guide.role)
        if previous is not None and previous != identity:
            raise ValueError("the same sequence cannot map to two targets or roles")
        seen_sequence[guide.sequence] = identity
        if guide.target_id in roles and roles[guide.target_id] != guide.role:
            raise ValueError("one target id cannot carry two roles")
        roles[guide.target_id] = guide.role
    by_id = {g.guide_id: g for g in guide_rows}
    rows = []
    seen_constructs = set()
    for item in constructs:
        raw = _fields(item, ("construct_id", "guide_a", "guide_b", "orientation"))
        if raw["construct_id"] in seen_constructs:
            raise ValueError("construct ids must be unique")
        seen_constructs.add(raw["construct_id"])
        if raw["guide_a"] not in by_id or raw["guide_b"] not in by_id:
            raise ValueError("both arms need a declared guide")
        left, right = by_id[raw["guide_a"]], by_id[raw["guide_b"]]
        orientation = _orientation(left, right)
        if raw["orientation"] != orientation:
            raise ValueError("orientation does not match the declared arms")
        class_name, key, endpoints = _classify(left, right)
        rows.append(Construct(
            raw["construct_id"], left.guide_id, right.guide_id, orientation,
            key, class_name, endpoints,
        ))
    gene_ids = tuple(sorted(gene for gene, role in roles.items() if role != "ntc"))
    return Library(
        CONTRACT, guide_rows, tuple(rows), gene_ids, tuple(roles[gene] for gene in gene_ids),
    )


def assign_hash_partitions(library, salt):
    """Assign each canonical key once. Variants and orientations share that split.

    The bucket is ``sha256(salt + '|' + key)`` as 8 hex digits, modulo 10:
    0-5 TRAIN, 6-7 VALIDATION, 8-9 TEST. The salt is design text, not an outcome.
    """
    _library(library)
    if type(salt) is not str or not salt or salt != salt.strip():
        raise ValueError("split salt must be explicit nonblank text")
    assigned = {}
    parts = []
    for construct in library.constructs:
        part = assigned.get(construct.canonical_key)
        if part is None:
            part = _bucket(salt, construct.canonical_key)
            assigned[construct.canonical_key] = part
        parts.append(part)
    return DesignSplit(CONTRACT, tuple(parts), ())


def assign_construct_partitions(library, partitions):
    """Accept construct labels only when every copy of a canonical key agrees."""
    _library(library)
    if type(partitions) not in (list, tuple) or len(partitions) != len(library.constructs):
        raise ValueError("one partition label is required per construct")
    if any(part not in _FIT_PARTITIONS for part in partitions):
        raise ValueError("partitions must be train, validation, or test")
    agreed = {}
    for construct, part in zip(library.constructs, partitions):
        previous = agreed.get(construct.canonical_key)
        if previous is not None and previous != part:
            raise ValueError("variants and orientations must share one split")
        agreed[construct.canonical_key] = part
    return DesignSplit(CONTRACT, tuple(partitions), ())


def apply_held_genes(library, split, held_genes):
    """Move every incident canonical key, including variants, out of TRAIN, VALIDATION, and TEST."""
    _library(library)
    split = _split(split, library)
    if split.held_genes or any(part not in _FIT_PARTITIONS for part in split.partition):
        raise ValueError("held-gene exclusion starts from a train/validation/test split")
    if type(held_genes) not in (list, tuple) or not held_genes:
        raise ValueError("a held-gene split needs at least one gene")
    if len(set(held_genes)) != len(held_genes):
        raise ValueError("held genes must be unique")
    roles = dict(zip(library.gene_ids, library.gene_roles))
    for gene in held_genes:
        if gene not in roles:
            raise ValueError("unknown held gene")
    held = tuple(sorted(held_genes))
    held_set = set(held)
    incident = {
        construct.canonical_key for construct in library.constructs
        if held_set.intersection(construct.endpoints)
    }
    parts = tuple(
        "held" if construct.canonical_key in incident else part
        for construct, part in zip(library.constructs, split.partition)
    )
    return _split(DesignSplit(CONTRACT, parts, held), library)


def partition_index(split, name):
    """Return construct positions for one partition, in declaration order."""
    if name not in _ALL_PARTITIONS:
        raise ValueError("unknown partition")
    if type(split) is not DesignSplit or split.contract != CONTRACT:
        raise TypeError("design split required")
    if any(part not in _ALL_PARTITIONS for part in split.partition):
        raise ValueError("unknown partition label")
    return _freeze_int([i for i, part in enumerate(split.partition) if part == name])


def train_graph(library, split):
    """Report TRAIN perturbation-pair connectivity after held genes are removed.

    Degree counts distinct canonical pairs. It is not frozen two-partner support.
    """
    _library(library)
    split = _split(split, library)
    held = set(split.held_genes)
    genes = [
        gene for gene, role in zip(library.gene_ids, library.gene_roles)
        if role == "perturbation" and gene not in held
    ]
    edges = []
    seen = set()
    for construct, part in zip(library.constructs, split.partition):
        incident = bool(held.intersection(construct.endpoints))
        if incident and part in ("train", "validation"):
            raise ValueError("an incident construct remains in TRAIN or VALIDATION")
        if part != "train" or construct.class_name != "perturbation_pair" or construct.canonical_key in seen:
            continue
        seen.add(construct.canonical_key)
        edges.append(construct.endpoints)
    parent = {gene: gene for gene in genes}
    degree = {gene: 0 for gene in genes}

    def find(gene):
        while parent[gene] != gene:
            parent[gene] = parent[parent[gene]]
            gene = parent[gene]
        return gene

    for left, right in edges:
        if left not in parent or right not in parent:
            raise ValueError("TRAIN edge is outside the unheld perturbation genes")
        degree[left] += 1
        degree[right] += 1
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left
    groups = {}
    for gene in genes:
        groups.setdefault(find(gene), []).append(gene)
    components = tuple(
        tuple(sorted(members)) for _, members in sorted(groups.items(), key=lambda item: sorted(item[1])[0])
    )
    return TrainGraph(components, tuple((gene, degree[gene]) for gene in sorted(degree)), len(components))


def require_provenance(provenance):
    """Refuse unsafe declarations. A passing declaration is still not proof.

    It also does not prove that a precomputed denominator excludes held outcomes.
    """
    if type(provenance) is not Provenance:
        raise TypeError("provenance must be an explicit declaration")
    if provenance.singles_derived_from_held_doubles:
        raise ValueError("refusing declared held-double-derived singles")
    if not provenance.singles_independently_measured:
        raise ValueError("refusing a declaration that singles were not independently measured")
    if provenance.eligibility_response_selected:
        raise ValueError("refusing declared response-selected eligibility")
    if not provenance.eligibility_outcome_independent:
        raise ValueError("refusing a declaration that eligibility depends on outcomes")
    return provenance


def declare_eligibility(mask, provenance, **extra):
    """Record an eligibility mask. Outcomes are not arguments."""
    if extra:
        raise TypeError("forbidden arguments: " + ", ".join(sorted(extra)))
    require_provenance(provenance)
    bits = _bools(mask, None, "eligibility")
    return _freeze_bool(bits)


def make_fit_view(
    library, split, provenance, single_ids, single_values, single_measured,
    train_index, train_labels, train_missing, train_support,
    validation_index, validation_labels, validation_missing, validation_support,
    **extra,
):
    """Seal TRAIN/VALIDATION indices, labels, masks, and arm metadata.

    TEST targets, held labels, and other keyword payloads are refused.
    ``train_support`` and ``validation_support`` record only whether a
    perturbation pair's endpoints have declared measured singles.
    """
    if extra:
        raise TypeError("forbidden arguments: " + ", ".join(sorted(extra)))
    _library(library)
    split = _split(split, library)
    accepted = require_provenance(provenance)
    values, measured = _singles(library, single_ids, single_values, single_measured)
    measured_by_gene = dict(zip(library.gene_ids, measured.tolist()))
    train = _fit_rows(
        library, split, "train", train_index, train_labels, train_missing,
        train_support, measured_by_gene,
    )
    validation = _fit_rows(
        library, split, "validation", validation_index, validation_labels,
        validation_missing, validation_support, measured_by_gene,
    )
    if set(train.index.tolist()) & set(validation.index.tolist()):
        raise ValueError("TRAIN and VALIDATION overlap")
    train_keys = set(train.keys)
    validation_keys = set(validation.keys)
    if train_keys & validation_keys:
        raise ValueError("a canonical key crosses TRAIN and VALIDATION")
    held_index = partition_index(split, "held")
    held_keys = {library.constructs[i].canonical_key for i in held_index.tolist()}
    if held_keys & (train_keys | validation_keys):
        raise ValueError("a held canonical key remains in TRAIN or VALIDATION")
    return FitView(
        CONTRACT, train, validation, library.gene_ids, values, measured,
        partition_index(split, "test"), held_index,
        (
            accepted.singles_independently_measured,
            accepted.eligibility_outcome_independent,
            accepted.singles_derived_from_held_doubles,
            accepted.eligibility_response_selected,
        ),
        False, False, SUPPORT_MEANING, False,
    )


def fit_digest(view):
    """Hash the sealed fit view. Labels that were missing are absent."""
    if type(view) is not FitView or view.contract != CONTRACT:
        raise TypeError("fit view required")
    payload = {
        "contract": view.contract,
        "declarations_are_proof": view.declarations_are_proof,
        "source_endpoint_qualified": view.source_endpoint_qualified,
        "support_meaning": view.support_meaning,
        "frozen_partner_support_included": view.frozen_partner_support_included,
        "provenance": list(view.provenance),
        "single_ids": list(view.single_ids),
        "single_values": [_json_number(value) for value in view.single_values.tolist()],
        "single_measured": [bool(value) for value in view.single_measured.tolist()],
        "test_index": view.test_index.tolist(),
        "held_index": view.held_index.tolist(),
        "train": _row_payload(view.train),
        "validation": _row_payload(view.validation),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def select_declared(candidates, **extra):
    """Choose the lowest validation score, then complexity, name, and family.

    TEST scores are not arguments. This is a tie break, not a fitted model.
    """
    if extra:
        raise TypeError("forbidden arguments: " + ", ".join(sorted(extra)))
    if type(candidates) not in (list, tuple) or not candidates:
        raise ValueError("at least one declared candidate is required")
    names = []
    for candidate in candidates:
        if type(candidate) is not DeclaredCandidate:
            raise TypeError("declared candidates required")
        _token(candidate.name, "candidate name")
        _token(candidate.family, "candidate family")
        if type(candidate.complexity) is not int or isinstance(candidate.complexity, bool) or candidate.complexity < 0:
            raise ValueError("complexity must be a nonnegative integer")
        if type(candidate.validation_score) not in (int, float) or isinstance(candidate.validation_score, bool):
            raise TypeError("validation score must be numeric")
        if not math.isfinite(float(candidate.validation_score)):
            raise ValueError("validation score must be finite")
        names.append(candidate.name)
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    ranked = tuple(sorted(
        candidates,
        key=lambda item: (float(item.validation_score), item.complexity, item.name, item.family),
    ))
    return Selection(ranked[0].name, tuple(item.name for item in ranked), "validation_score,complexity,name,family")


def support_reason(library, split, measured, index):
    """Return the measured-single prerequisite, or an explicit abstention.

    True means a perturbation pair has declared measured singles. It does not
    apply the frozen two-distinct-TRAIN-partner rule and fits nothing.
    """
    _library(library)
    split = _split(split, library)
    if type(measured) is not dict or set(measured) != set(library.gene_ids):
        raise ValueError("measured singles must be declared for every gene id")
    if any(type(bit) is not bool for bit in measured.values()):
        raise TypeError("measured singles must be explicit bools")
    if type(index) is not int or isinstance(index, bool) or not 0 <= index < len(library.constructs):
        raise ValueError("construct index is outside the library")
    construct = library.constructs[index]
    if set(split.held_genes).intersection(construct.endpoints):
        return False, "held_incident_excluded"
    if construct.class_name == "perturbation_pair":
        if _both_measured(construct, measured):
            return True, None
        return False, "missing_independent_single"
    if construct.class_name == "single_vs_ntc":
        return False, "ntc_contrast_is_not_a_perturbation_pair"
    if construct.class_name in ("single_vs_nonessential", "nonessential_pair", "ntc_vs_nonessential"):
        return False, "nonessential_target_is_an_intervention"
    if construct.class_name == "ntc_pair":
        return False, "ntc_pair_is_not_a_perturbation"
    raise ValueError("unknown construct class")


def seal_prediction(key, *, supported, value=None, abstention=None):
    """Keep a finite supported value, or an abstention with no numeric stand-in."""
    _token(key, "prediction key")
    if type(supported) is not bool:
        raise TypeError("support must be an explicit bool")
    if supported:
        if abstention is not None:
            raise ValueError("a supported prediction has no abstention")
        number = _finite_number(value, "prediction")
        return Prediction(key, True, number, None)
    if value is not None or type(abstention) is not str or not abstention.strip():
        raise ValueError("an abstention needs an explicit reason and no numeric stand-in")
    return Prediction(key, False, None, abstention)


def declared_log_contrast(
    count, library_total, missing, reference_counts, reference_totals, reference_missing,
    reference_ids, reference_kinds, reference_partitions, *, pseudocount, scale, **extra,
):
    """Apply the caller-declared log contrast. This is not verified source processing.

    A column is withheld only when its kind is ``held_pair`` and its partition is
    ``held``; that column is not read. Every other column must be a TRAIN
    ``independent_control``. A half-tag is refused, not silently included.
    If any declared TRAIN control is missing, every contrast is missing; the
    observed subset is not reused and held outcomes are not substitutes.
    ``library_total`` is used exactly as declared. A precomputed total can
    already contain held-pair outcomes; this function cannot detect that.
    """
    if extra:
        raise TypeError("forbidden arguments: " + ", ".join(sorted(extra)))
    pseudocount = _positive_constant(pseudocount, "pseudocount")
    scale = _positive_constant(scale, "scale")
    sample_missing = _bools(missing, None, "sample missing")
    n = len(sample_missing)
    _aligned(count, n, "count")
    _aligned(library_total, n, "library total")
    members = _reference_members(
        reference_counts, reference_totals, reference_missing, reference_ids,
        reference_kinds, reference_partitions,
    )
    train_ids = tuple(item["id"] for item in members if item["train_control"])
    if not train_ids:
        raise ValueError("TRAIN independent-control membership is empty")
    blocked = any(item["train_control"] and item["missing"] for item in members)
    values = [math.nan] * n
    out_missing = [True] * n
    if not blocked:
        center = _median([
            _norm(item["count"], item["total"], pseudocount, scale)
            for item in members if item["train_control"]
        ])
        for i, is_missing in enumerate(sample_missing):
            if is_missing:
                continue
            values[i] = _norm(count[i], library_total[i], pseudocount, scale) - center
            out_missing[i] = False
    return Contrast(
        _freeze_float(values), _freeze_bool(out_missing), FORMULA, train_ids, False, False,
        True, False,
    )


def fit_train_scaler(values, missing, membership, **extra):
    """Center and scale declared TRAIN reference rows. Held rows are not read.

    ``n`` counts those declared rows. It is not a replicate count. Scale is the
    root mean square deviation around the mean, with divisor ``n``. A zero scale
    is reported as degenerate and stored as 1 so later application still centers.
    """
    if extra:
        raise TypeError("forbidden arguments: " + ", ".join(sorted(extra)))
    missing_bits = _bools(missing, None, "scaler missing")
    n = len(missing_bits)
    _aligned(values, n, "scaler values")
    if type(membership) not in (list, tuple) or len(membership) != n:
        raise ValueError("scaler membership must align with values")
    used = []
    for i, (label, is_missing) in enumerate(zip(membership, missing_bits)):
        if label == "held_pair":
            continue
        if label != "train_reference":
            raise ValueError("scaler membership must be train_reference or held_pair")
        if not is_missing:
            used.append(i)
    if not used:
        raise ValueError("no observed TRAIN reference values")
    observed = [_finite_number(values[i], "TRAIN reference") for i in used]
    center = sum(observed) / len(observed)
    scale = math.sqrt(sum((item - center) ** 2 for item in observed) / len(observed))
    degenerate = scale == 0.0
    return TrainScaler(center, 1.0 if degenerate else scale, degenerate, len(observed), FORMULA, False)


def apply_train_scaler(values, missing, scaler, **extra):
    """Apply a frozen TRAIN scaler. Missing positions are not read or filled."""
    if extra:
        raise TypeError("forbidden arguments: " + ", ".join(sorted(extra)))
    if type(scaler) is not TrainScaler or scaler.formula != FORMULA or scaler.source_processing_verified:
        raise TypeError("TRAIN scaler required")
    missing_bits = _bools(missing, None, "scaler missing")
    n = len(missing_bits)
    _aligned(values, n, "scaler values")
    out = [math.nan] * n
    for i, is_missing in enumerate(missing_bits):
        if is_missing:
            continue
        out[i] = _finite_number(
            (_finite_number(values[i], "scaler input") - scaler.center) / scaler.scale,
            "scaled output",
        )
    return _freeze_float(out)


def control_dependence(control_ids):
    """Connect rows that share a control id, including transitive sharing.

    Component size is dependence, not replication. ``independent_replicates``
    is always false.
    """
    if type(control_ids) not in (list, tuple) or not control_ids:
        raise ValueError("control membership is required")
    parent = list(range(len(control_ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    first = {}
    rows = []
    for i, ids in enumerate(control_ids):
        if type(ids) not in (list, tuple) or not ids:
            raise ValueError("each row needs explicit control ids")
        row = []
        for control_id in ids:
            _token(control_id, "control id")
            if control_id in row:
                raise ValueError("a row repeats a control id")
            row.append(control_id)
            if control_id in first:
                union(i, first[control_id])
            else:
                first[control_id] = i
        rows.append(tuple(row))
    roots = [find(i) for i in range(len(rows))]
    order = {}
    component = []
    for root in roots:
        if root not in order:
            order[root] = len(order)
        component.append(order[root])
    buckets = [set() for _ in range(len(order))]
    for i, row in enumerate(rows):
        buckets[component[i]].update(row)
    keys = tuple(tuple(sorted(bucket)) for bucket in buckets)
    return Dependence(_freeze_int(component), keys, False)


def _guide(item):
    raw = _fields(item, ("guide_id", "sequence", "target_id", "role"))
    if raw["role"] not in _ROLES:
        raise ValueError("role must be perturbation, nonessential_target, or ntc")
    if type(raw["sequence"]) is not str or not raw["sequence"] or any(base not in "ACGT" for base in raw["sequence"]):
        raise ValueError("sequence must be an explicit declared ACGT string")
    return Guide(
        _token(raw["guide_id"], "guide id"), raw["sequence"],
        _token(raw["target_id"], "target id"), raw["role"],
    )


def _fields(item, names):
    if type(item) is not dict or set(item) != set(names):
        raise ValueError("exact fields required: " + ",".join(names))
    return item


def _token(value, field):
    if type(value) is not str or not value or value != value.strip() or "|" in value or any(ch.isspace() for ch in value):
        raise ValueError(f"{field} must be explicit nonblank text without '|'")
    return value


def _orientation(left, right):
    if left.guide_id == right.guide_id:
        raise ValueError("the same guide cannot occupy both arms")
    if left.role == right.role:
        if left.role == "ntc":
            ordered = ((left.target_id, left.guide_id), (right.target_id, right.guide_id))
            return "forward" if ordered[0] < ordered[1] else "reverse"
        if left.target_id == right.target_id:
            raise ValueError("self-pairs are refused")
        return "forward" if left.target_id < right.target_id else "reverse"
    return "forward" if _ROLES[left.role] < _ROLES[right.role] else "reverse"


def _classify(left, right):
    roles = {left.role, right.role}
    if roles == {"perturbation"}:
        lo, hi = sorted((left.target_id, right.target_id))
        return "perturbation_pair", f"pair|{lo}|{hi}", (lo, hi)
    if roles == {"perturbation", "ntc"}:
        gene = left.target_id if left.role == "perturbation" else right.target_id
        return "single_vs_ntc", f"single_ntc|{gene}", (gene,)
    if roles == {"perturbation", "nonessential_target"}:
        if left.role == "perturbation":
            gene, other = left.target_id, right.target_id
        else:
            gene, other = right.target_id, left.target_id
        return "single_vs_nonessential", f"single_nonessential|{gene}|{other}", (gene, other)
    if roles == {"ntc"}:
        return "ntc_pair", "ntc_pair", ()
    if roles == {"nonessential_target"}:
        lo, hi = sorted((left.target_id, right.target_id))
        return "nonessential_pair", f"nonessential_pair|{lo}|{hi}", (lo, hi)
    other = left.target_id if left.role == "nonessential_target" else right.target_id
    return "ntc_vs_nonessential", f"ntc_vs_nonessential|{other}", (other,)


def _bucket(salt, key):
    digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) % 10
    if value < 6:
        return "train"
    if value < 8:
        return "validation"
    return "test"


def _library(library):
    if type(library) is not Library or library.contract != CONTRACT:
        raise TypeError("declared library required")
    return library


def _split(split, library):
    """Reject forged splits, including a canonical key shared by TRAIN and TEST."""
    if type(split) is not DesignSplit or split.contract != CONTRACT:
        raise TypeError("design split required")
    if type(split.partition) is not tuple or len(split.partition) != len(library.constructs):
        raise ValueError("split does not match the construct table")
    if any(type(part) is not str or part not in _ALL_PARTITIONS for part in split.partition):
        raise ValueError("unknown partition label")
    if type(split.held_genes) is not tuple or any(type(gene) is not str for gene in split.held_genes):
        raise TypeError("held genes must be an explicit tuple of ids")
    if len(set(split.held_genes)) != len(split.held_genes):
        raise ValueError("held genes must be unique")
    known = set(library.gene_ids)
    if any(gene not in known for gene in split.held_genes):
        raise ValueError("unknown held gene")
    held = set(split.held_genes)
    agreed = {}
    for construct, part in zip(library.constructs, split.partition):
        previous = agreed.get(construct.canonical_key)
        if previous is not None and previous != part:
            raise ValueError("canonical key is assigned to more than one partition")
        agreed[construct.canonical_key] = part
        incident = bool(held.intersection(construct.endpoints))
        if incident and part != "held":
            raise ValueError("incident construct remains outside the held partition")
        if part == "held" and not incident:
            raise ValueError("held partition requires incidence to a declared held gene")
    return split


def _singles(library, single_ids, single_values, single_measured):
    if tuple(single_ids) != library.gene_ids:
        raise ValueError("single ids must match the declared gene ids in order")
    measured = _bools(single_measured, len(library.gene_ids), "single measured")
    _aligned(single_values, len(library.gene_ids), "single values")
    values = [math.nan] * len(measured)
    for i, is_measured in enumerate(measured):
        if is_measured:
            values[i] = _finite_number(single_values[i], "single")
    return _freeze_float(values), _freeze_bool(measured)


def _fit_rows(library, split, name, index, labels, missing, support, measured):
    positions = _ints(index, name + " index")
    if any(i >= len(library.constructs) for i in positions):
        raise ValueError(name + " index is outside the construct table")
    if len(set(positions)) != len(positions) or sorted(positions) != partition_index(split, name).tolist():
        raise ValueError(name + " index must be the split's construct positions")
    missing_bits = _bools(missing, len(positions), name + " missing")
    support_bits = _bools(support, len(positions), name + " support")
    _aligned(labels, len(positions), name + " labels")
    stored = [math.nan] * len(positions)
    keys, classes, orientations, arms = [], [], [], []
    guides = {guide.guide_id: guide for guide in library.guides}
    held = set(split.held_genes)
    for i, position in enumerate(positions):
        construct = library.constructs[position]
        if split.partition[position] != name:
            raise ValueError(name + " index disagrees with the split")
        if held.intersection(construct.endpoints):
            raise ValueError("an incident construct remains in TRAIN or VALIDATION")
        if not missing_bits[i]:
            stored[i] = _finite_number(labels[i], name + " label")
        expected = construct.class_name == "perturbation_pair" and _both_measured(construct, measured)
        if support_bits[i] is not expected:
            raise ValueError("prerequisite support mask disagrees with declared measured singles")
        left, right = guides[construct.guide_a], guides[construct.guide_b]
        keys.append(construct.canonical_key)
        classes.append(construct.class_name)
        orientations.append(construct.orientation)
        arms.append((
            left.guide_id, left.sequence, left.target_id, left.role,
            right.guide_id, right.sequence, right.target_id, right.role,
        ))
    return FitRows(
        _freeze_int(positions), _freeze_float(stored), _freeze_bool(missing_bits),
        _freeze_bool(support_bits), tuple(keys), tuple(classes), tuple(orientations), tuple(arms),
    )


def _both_measured(construct, measured):
    return all(measured[gene] for gene in construct.endpoints)


def _reference_members(counts, totals, missing, ids, kinds, partitions):
    missing_bits = _bools(missing, None, "reference missing")
    width = len(missing_bits)
    for name, values in (
        ("reference count", counts), ("reference total", totals), ("reference id", ids),
        ("reference kind", kinds), ("reference partition", partitions),
    ):
        _aligned(values, width, name)
    if len(set(ids)) != width:
        raise ValueError("reference ids must be unique")
    members = []
    for i, is_missing in enumerate(missing_bits):
        kind, part = kinds[i], partitions[i]
        if type(kind) is not str or type(part) is not str:
            raise TypeError("reference kind and partition must be explicit text")
        held = kind == "held_pair" or part == "held"
        train_control = kind == "independent_control" and part == "train"
        if held and train_control:
            raise ValueError("contradictory reference membership")
        if held:
            if kind != "held_pair" or part != "held":
                raise ValueError("contradictory reference membership")
            members.append({"id": _token(ids[i], "reference id"), "train_control": False, "missing": True})
            continue
        if not train_control:
            raise ValueError("reference member is not a declared TRAIN independent control")
        member = {"id": _token(ids[i], "reference id"), "train_control": True, "missing": is_missing}
        if not is_missing:
            member["count"] = counts[i]
            member["total"] = totals[i]
        members.append(member)
    return members


def _norm(count, total, pseudocount, scale):
    count_n = _whole(count, "count")
    total_n = _whole(total, "library total")
    if total_n <= 0 or count_n > total_n:
        raise ValueError("library total must cover the raw count")
    value = math.log2(pseudocount + count_n * (scale / total_n))
    return _finite_number(value, "normalized log count")


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _whole(value, name):
    number = _finite_number(value, name)
    if number < 0 or number != math.floor(number):
        raise ValueError(f"{name} must be a whole nonnegative raw count")
    return number


def _positive_constant(value, name):
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be a declared positive constant")
    return number


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _bools(values, n, name):
    if isinstance(values, np.ndarray):
        if values.dtype != np.dtype(bool) or values.ndim != 1:
            raise ValueError(name + " must be an explicit bool mask")
        bits = values.tolist()
    elif type(values) in (list, tuple):
        bits = list(values)
    else:
        raise TypeError(name + " must be an explicit bool mask")
    if n is not None and len(bits) != n:
        raise ValueError(name + " has the wrong length")
    if any(type(bit) is not bool for bit in bits):
        raise ValueError(name + " must be an explicit bool mask")
    return bits


def _ints(values, name):
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError(name + " must be integer positions")
        nums = values.tolist()
    elif type(values) in (list, tuple):
        nums = list(values)
    else:
        raise TypeError(name + " must be integer positions")
    if any(type(num) is not int or num < 0 for num in nums):
        raise ValueError(name + " must be integer positions")
    return nums


def _aligned(values, n, name):
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or len(values) != n:
            raise ValueError(name + " must align")
        return values
    if type(values) not in (list, tuple) or len(values) != n:
        raise ValueError(name + " must align")
    return values


def _freeze_int(values):
    array = np.array(values, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


def _freeze_float(values):
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _freeze_bool(values):
    array = np.array(values, dtype=bool, copy=True)
    array.setflags(write=False)
    return array


def _json_number(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _row_payload(rows):
    return {
        "index": rows.index.tolist(),
        "labels": [_json_number(value) for value in rows.labels.tolist()],
        "missing": [bool(value) for value in rows.missing.tolist()],
        "support": [bool(value) for value in rows.support.tolist()],
        "keys": list(rows.keys),
        "classes": list(rows.classes),
        "orientations": list(rows.orientations),
        "arms": [list(arm) for arm in rows.arms],
    }
