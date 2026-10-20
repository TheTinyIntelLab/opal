"""Exact-reagent scalar-screen adapter, independently of any source file parser.

This contract is executable on synthetic counts. It does not qualify SPIDR's
unopened response table, its count semantics, or biological replication.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from statistics import mean, median
from typing import Callable


SCHEMA_VERSION = 1
NORMALIZATION = "raw_log2_plus10_minus_ntpair_median"
UNITS = "log2_t14_over_t0"
STRATA = ("both_mismatched", "perfect_perfect", "mixed")


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} requires explicit nonblank text")


def strict_bool(value):
    """Parse only explicit booleans; unknown flags must never become False."""
    if type(value) is bool:
        return value
    if value in ("TRUE", "FALSE"):
        return value == "TRUE"
    raise ValueError("Unknown boolean/mismatch annotation")


@dataclass(frozen=True)
class ScreenContract:
    source: str
    context: str
    replicates: tuple[str, str]
    initiation_lineage: str
    schema_status: str
    count_kind: str = "raw_integer_counts"
    normalization: str = NORMALIZATION
    units: str = UNITS
    t0_time: str = "T0"
    endpoint_time: str = "T14"
    source_schema_sha256: str = "synthetic"
    controls_evidence: str = "synthetic_explicit_control_flags"

    def __post_init__(self):
        for field in ("source", "context", "initiation_lineage", "controls_evidence",
                      "t0_time", "endpoint_time"):
            _text(getattr(self, field), field)
        if (type(self.replicates) is not tuple or len(self.replicates) != 2 or len(set(self.replicates)) != 2
                or any(not isinstance(r, str) or not r.strip() for r in self.replicates)):
            raise ValueError("Exactly two distinct declared replicate IDs required")
        if self.t0_time != "T0" or self.endpoint_time != "T14":
            raise ValueError("This LFC contract requires declared T0/T14 times")
        if self.schema_status not in ("synthetic_only", "qualified_source_metadata"):
            raise ValueError("Unknown or unqualified source schema: stop before responses")
        if self.schema_status == "qualified_source_metadata":
            digest = self.source_schema_sha256
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or self.controls_evidence == "synthetic_explicit_control_flags"):
                raise ValueError("Real source needs reviewed schema hash/control evidence")
        if (self.count_kind != "raw_integer_counts" or self.normalization != NORMALIZATION
                or self.units != UNITS):
            raise ValueError("Unsupported counts/normalization/units; resolve raw schema first")


@dataclass(frozen=True)
class Guide:
    name: str
    gene: str
    sequence: str
    transcript: str
    arm: str
    mismatched: bool | None
    is_control: bool

    def __post_init__(self):
        for field in ("name", "gene", "sequence", "transcript"):
            _text(getattr(self, field), field)
        if self.arm not in ("arm1", "arm2"):
            raise ValueError("Unknown vector arm/orientation")
        if (type(self.is_control) is not bool
                or (type(self.mismatched) is not bool
                    and not (self.is_control and self.mismatched is None))):
            raise ValueError("Boolean/mismatch fields must be explicit booleans")
        if any(base not in "ACGT" for base in self.sequence):
            raise ValueError("Guide sequence must be explicit uppercase DNA")

    @property
    def exact_key(self):
        # A renamed identical guide is still the same reagent. Preserve dose
        # annotation, transcript and vector arm; gene name alone never matches.
        return (self.gene, self.sequence, self.transcript, self.arm,
                self.mismatched, self.is_control)


@dataclass(frozen=True)
class Construct:
    identity: str
    left: Guide
    right: Guide

    def __post_init__(self):
        _text(self.identity, "construct identity")
        if self.left.arm != "arm1" or self.right.arm != "arm2":
            raise ValueError("Construct must preserve ordered arm1/arm2 orientation")

    @property
    def pair(self):
        return tuple(sorted((self.left.gene, self.right.gene)))

    @property
    def stratum(self):
        if self.left.is_control or self.right.is_control:
            raise ValueError("Mismatch strata apply only to targeted double constructs")
        n = int(self.left.mismatched) + int(self.right.mismatched)
        return ("perfect_perfect", "mixed", "both_mismatched")[n]


@dataclass(frozen=True)
class CountObservation:
    construct: str
    replicate: str
    t0_time: str
    endpoint_time: str
    t0: int | None
    endpoint: int | None

    def __post_init__(self):
        for field in ("construct", "replicate", "t0_time", "endpoint_time"):
            _text(getattr(self, field), field)
        for value in (self.t0, self.endpoint):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Raw counts must be nonnegative integers or None, not blanks/normalized values")


@dataclass(frozen=True)
class ReagentScreen:
    contract: ScreenContract
    constructs: tuple[Construct, ...]
    observations: tuple[CountObservation, ...]

    def __post_init__(self):
        constructs = {c.identity: c for c in self.constructs}
        if len(constructs) != len(self.constructs):
            raise ValueError("Duplicate construct identity")
        ordered = {(c.left.exact_key, c.right.exact_key) for c in self.constructs}
        if len(ordered) != len(self.constructs):
            raise ValueError("Duplicate ordered guide construct, including renamed aliases")
        names = {}
        for c in self.constructs:
            for guide in (c.left, c.right):
                name = (guide.name, guide.arm)
                if name in names and names[name] != guide.exact_key:
                    raise ValueError("Inconsistent guide identity metadata")
                names[name] = guide.exact_key
        seen = set()
        for row in self.observations:
            key = (row.construct, row.replicate)
            if key in seen:
                raise ValueError("Duplicate construct/replicate observation")
            seen.add(key)
            if row.construct not in constructs or row.replicate not in self.contract.replicates:
                raise ValueError("Unknown construct or replicate")
            if (row.t0_time != self.contract.t0_time
                    or row.endpoint_time != self.contract.endpoint_time):
                raise ValueError("Unmatched time-point/endpoint contract")

    def to_json(self):
        return json.dumps(dict(schema_version=SCHEMA_VERSION, **asdict(self)),
                          sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text):
        value = json.loads(text)
        if value.pop("schema_version", None) != SCHEMA_VERSION:
            raise ValueError("Unknown reagent-screen schema version")
        contract = dict(value.pop("contract"))
        contract["replicates"] = tuple(contract["replicates"])
        constructs = []
        for raw in value.pop("constructs"):
            c = dict(raw)
            c["left"], c["right"] = Guide(**c["left"]), Guide(**c["right"])
            constructs.append(Construct(**c))
        observations = tuple(CountObservation(**o) for o in value.pop("observations"))
        if value:
            raise ValueError("Unknown reagent-screen fields")
        return cls(ScreenContract(**contract), tuple(constructs), observations)


def pair_split(source, pair):
    """All guide, orientation, stratum and replicate variants share this split."""
    a, b = sorted(pair)
    payload = f"opal-reset-v2|{source}|{a}|{b}".encode("utf-8")
    bucket = int(hashlib.sha256(payload).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else "validation" if bucket < 8 else "test"


def aggregate(screen: ReagentScreen, reference: str | Callable = "minimum"):
    """Normalize and predict each construct, then equally average constructs/reps.

    The callback receives exact-guide singles and the ordered Construct. It may
    use only already-frozen fitted state. This adapter never trains a predictor.
    A single missing T0/endpoint or T0 <=50 in either culture excludes a construct
    in both cultures. None means unavailable; an observed endpoint zero is valid.
    """
    if not callable(reference) and reference not in ("zero", "sum", "minimum"):
        raise ValueError("Unknown reference prediction API")
    reps = screen.contract.replicates
    rows = {(o.construct, o.replicate): o for o in screen.observations}
    exclusions, eligible = [], []
    for c in sorted(screen.constructs, key=lambda x: x.identity):
        matched = [rows.get((c.identity, r)) for r in reps]
        if any(o is None or o.t0 is None or o.endpoint is None for o in matched):
            exclusions.append(dict(construct=c.identity, reason="missing_count_or_replicate"))
        elif any(o.t0 <= 50 for o in matched):
            exclusions.append(dict(construct=c.identity, reason="t0_not_above_50_in_both_replicates"))
        else:
            eligible.append(c)
    raw = {(c.identity, r): math.log2((rows[c.identity, r].endpoint + 10)
                                    / (rows[c.identity, r].t0 + 10))
           for c in eligible for r in reps}
    nt = [c for c in eligible if c.left.is_control and c.right.is_control]
    if not nt:
        raise ValueError("No eligible explicit non-targeting-pair controls; stop")
    centers = {r: median(raw[c.identity, r] for c in nt) for r in reps}
    lfc = {key: value - centers[key[1]] for key, value in raw.items()}
    singles = {}
    for c in eligible:
        if c.left.is_control == c.right.is_control:
            continue
        guide = c.right if c.left.is_control else c.left
        for r in reps:
            singles.setdefault((guide.exact_key, r), []).append(lfc[c.identity, r])
    singles = {key: mean(values) for key, values in singles.items()}
    predictions, groups = [], {}
    for c in eligible:
        if c.left.is_control or c.right.is_control:
            continue
        if c.left.gene == c.right.gene:
            exclusions.append(dict(construct=c.identity, reason="same_gene_not_pair_task"))
            continue
        if any((g.exact_key, r) not in singles for g in (c.left, c.right) for r in reps):
            exclusions.append(dict(construct=c.identity, reason="missing_exact_arm_transcript_sequence_single"))
            continue
        for r in reps:
            a, b = singles[c.left.exact_key, r], singles[c.right.exact_key, r]
            pred = (reference(a, b, c) if callable(reference) else
                    0.0 if reference == "zero" else a + b if reference == "sum" else min(a, b))
            if isinstance(pred, bool) or not isinstance(pred, (int, float)) or not math.isfinite(pred):
                raise ValueError("Prediction API requires a finite scalar")
            item = dict(construct=c.identity, pair=list(c.pair), stratum=c.stratum,
                        replicate=r, single_left=a, single_right=b,
                        target=lfc[c.identity, r], prediction=float(pred))
            predictions.append(item)
            groups.setdefault((c.pair, c.stratum, r), []).append(item)
    pair_results = []
    for pair, stratum in sorted({(pair, stratum) for pair, stratum, _ in groups}):
        by_rep = [groups[pair, stratum, r] for r in reps]
        if any(len(items) < 2 for items in by_rep):
            exclusions.append(dict(pair=list(pair), stratum=stratum,
                                   reason="fewer_than_two_eligible_ordered_constructs"))
            continue
        summaries = [dict(replicate=r, constructs=len(items),
                          target=mean(item["target"] for item in items),
                          prediction=mean(item["prediction"] for item in items))
                     for r, items in zip(reps, by_rep)]
        pair_results.append(dict(pair=list(pair), stratum=stratum,
                                 role="primary" if stratum == "both_mismatched" else "secondary",
                                 split=pair_split(screen.contract.source, pair),
                                 target=mean(s["target"] for s in summaries),
                                 prediction=mean(s["prediction"] for s in summaries),
                                 replicates=summaries))
    return dict(schema_version=SCHEMA_VERSION, units=screen.contract.units,
                evidence=screen.contract.schema_status, independent_initiations="unestablished",
                control_centers=centers, pairs=pair_results,
                construct_predictions=predictions, exclusions=exclusions,
                counts=dict(input_constructs=len(screen.constructs),
                            input_guide_arm_identities=len({g.exact_key for c in screen.constructs
                                                           for g in (c.left, c.right)}),
                            count_eligible_constructs=len(eligible),
                            matched_single_reagents=len({key[0] for key in singles}),
                            primary_pairs=sum(p["role"] == "primary" for p in pair_results),
                            secondary_pair_strata=sum(p["role"] == "secondary" for p in pair_results)))
