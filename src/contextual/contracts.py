"""Versioned references and physically isolated combination outcomes."""
from dataclasses import dataclass
from pathlib import Path
import copy
import numpy as np
from opal.artifacts import array_hash, load_npz, read_json, sha256, write_json

SCHEMA = "opal-contextual-observed-singles-v2"
REFERENCE_KEYS = {"singles", "available", "controls", "features", "targets", "genes", "contexts"}


def _identifiers(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "US" or not len(values):
        raise ValueError(f"{name} must be a nonempty string vector")
    if any(not str(v).strip() for v in values) or len(set(values.tolist())) != len(values):
        raise ValueError(f"{name} must contain unique nonempty identifiers")
    return values.astype(str)


@dataclass
class ReferenceBank:
    """Only allowed controls and measured singles; no combination responses."""
    arrays: dict
    metadata: dict

    def __post_init__(self):
        if set(self.arrays) != REFERENCE_KEYS:
            raise ValueError("Reference fields must exactly match the observed-single schema")
        self.arrays = {k: np.array(v, copy=True) for k, v in self.arrays.items()}
        self.metadata = copy.deepcopy(self.metadata)
        for name in ["targets", "genes", "contexts"]:
            self.arrays[name] = _identifiers(self.arrays[name], name)
        c, t, g = (len(self.arrays[k]) for k in ["contexts", "targets", "genes"])
        for name, shape in [("singles", (c, t, g)), ("available", (c, t)), ("controls", (c, g))]:
            if self.arrays[name].shape != shape:
                raise ValueError(f"Invalid {name} shape")
        if self.arrays["available"].dtype.kind != "b":
            raise ValueError("Availability must be a boolean mask, not inferred from effect values")
        f = self.arrays["features"]
        if f.ndim != 2 or f.shape[0] != c or not f.shape[1]:
            raise ValueError("Context features require one finite nonempty row per context")
        for name in ["singles", "controls", "features"]:
            if self.arrays[name].dtype.kind not in "fi" or not np.isfinite(self.arrays[name]).all():
                raise ValueError(f"Nonfinite or nonnumeric {name}")
        if np.any(self.arrays["singles"][~self.arrays["available"]] != 0):
            raise ValueError("Unavailable profiles must use zero storage and a false availability mask")
        if self.metadata.get("schema") != SCHEMA or self.metadata.get("scale") not in {"log2", "count"}:
            raise ValueError("A supported reference schema and effect scale are required")
        records = self.metadata.get("context_records", {})
        if set(records) != set(self.arrays["contexts"].tolist()):
            raise ValueError("Context metadata must exactly cover the context identifiers")
        for record in records.values():
            if any(not isinstance(record.get(k), str) or not record[k].strip()
                   for k in ["biological_state", "modality", "assay", "study", "source"]):
                raise ValueError("Contexts require biological state, modality, assay, study and source")
        if len({r["modality"] for r in records.values()}) != 1:
            raise ValueError("Cross-modality pooling needs a separately evaluated contract")
        if len({r["assay"] for r in records.values()}) != 1:
            raise ValueError("Cross-assay pooling needs a separately evaluated contract")
        interventions = self.metadata.get("intervention_records", {})
        if set(interventions) != set(self.arrays["targets"].tolist()):
            raise ValueError("Every intervention needs canonical target, reagent, modality, dose and time metadata")
        modality = next(iter(records.values()))["modality"]
        for record in interventions.values():
            if any(not isinstance(record.get(k), str) or not record[k].strip()
                   for k in ["canonical_target", "reagent", "modality", "dose", "time"]):
                raise ValueError("Incomplete intervention condition metadata")
            if record["modality"] != modality:
                raise ValueError("Intervention modality differs from context contract")
        names = self.metadata.get("feature_names", [])
        if not isinstance(names, list) or len(names) != f.shape[1] or any(not isinstance(v, str) or not v for v in names) or len(set(names)) != len(names):
            raise ValueError("Context features require unique declared feature names")
        if self.metadata.get("reference_access") != "controls_and_measured_singles_all_declared_contexts":
            raise ValueError("Target-context reference access must be declared explicitly")
        if self.metadata.get("context_features_source") not in {"controls_and_metadata", "synthetic_generator"}:
            raise ValueError("Context features cannot derive from combination outcomes")
        if not isinstance(self.metadata.get("synthetic"), bool):
            raise ValueError("Synthetic status must be explicit")
        if self.metadata["context_features_source"] == "synthetic_generator" and not self.metadata["synthetic"]:
            raise ValueError("Synthetic features cannot be labeled biological evidence")
        for value in self.arrays.values():
            value.flags.writeable = False

    @property
    def fingerprint(self):
        return array_hash(*(self.arrays[k] for k in sorted(REFERENCE_KEYS)))

    def validate_rows(self, pairs, contexts):
        pairs, contexts = np.asarray(pairs), np.asarray(contexts)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or contexts.shape != (len(pairs),):
            raise ValueError("Pairs and context rows have incompatible dimensions")
        if pairs.dtype.kind not in "iu" or contexts.dtype.kind not in "iu":
            raise ValueError("Pair and context indices must be integers")
        if (np.any(pairs < 0) or np.any(pairs >= len(self.arrays["targets"])) or
                np.any(contexts < 0) or np.any(contexts >= len(self.arrays["contexts"]))):
            raise ValueError("Pair or context index out of range")
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError("Repeated-target combinations are unsupported")
        if not self.arrays["available"][contexts[:, None], pairs].all():
            raise ValueError("Both measured singles must be available in the query context")
        return np.sort(pairs.astype(np.int64), axis=1), contexts.astype(np.int64)

    def additive(self, pairs, contexts):
        pairs, contexts = self.validate_rows(pairs, contexts)
        return self.arrays["singles"][contexts, pairs[:, 0]] + self.arrays["singles"][contexts, pairs[:, 1]]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "references.npz", **self.arrays)
        write_json(directory / "references.json", {"metadata": self.metadata,
                   "reference_hash": self.fingerprint, "arrays_sha256": sha256(directory / "references.npz")})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = read_json(directory / "references.json")
        if sha256(directory / "references.npz") != manifest["arrays_sha256"]:
            raise ValueError("Reference checksum mismatch")
        bank = cls(load_npz(directory / "references.npz"), manifest["metadata"])
        if bank.fingerprint != manifest["reference_hash"]:
            raise ValueError("Reference fingerprint mismatch")
        return bank


def prepare_dataset(directory, bank, pairs, contexts, responses, *, held_contexts, calibration_fraction=.2, seed=1729):
    """Hold out contexts by identity and separate their outcomes before fitting."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Prepared contextual datasets are immutable; choose a new directory")
    pairs, contexts = bank.validate_rows(pairs, contexts)
    responses = np.asarray(responses, dtype=np.float64)
    if responses.shape != (len(pairs), len(bank.arrays["genes"])) or not np.isfinite(responses).all():
        raise ValueError("Combination responses require finite registered gene-order values")
    keys = [(int(c), int(a), int(b)) for (a, b), c in zip(pairs, contexts)]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate pair/context observations must be aggregated through a declared replicate policy")
    held_contexts = list(held_contexts)
    context_map = {str(v): i for i, v in enumerate(bank.arrays["contexts"])}
    if not held_contexts or len(set(held_contexts)) != len(held_contexts) or any(c not in context_map for c in held_contexts):
        raise ValueError("Held-out contexts must be distinct supported identifiers")
    if not 0 <= calibration_fraction < .5:
        raise ValueError("Calibration fraction must be in [0, 0.5)")
    test = np.flatnonzero(np.isin(contexts, [context_map[c] for c in held_contexts]))
    development = np.flatnonzero(~np.isin(contexts, [context_map[c] for c in held_contexts]))
    if not len(test) or len(np.unique(contexts[development])) < 2:
        raise ValueError("Need outcomes in a held-out context and at least two fitting contexts")
    # Group repeated intervention pairs across contexts so calibration labels
    # cannot also appear in fitting under a different context.
    groups = sorted(set(map(tuple, pairs[development].tolist())))
    if len(groups) < 3:
        raise ValueError("Too few independent pair groups for contextual fitting")
    order = np.random.default_rng(seed).permutation(len(groups))
    ncal = int(np.ceil(len(groups) * calibration_fraction))
    if len(groups) - ncal < 3:
        raise ValueError("Too few pair groups after reserving calibration")
    calibration_keys = {groups[i] for i in order[:ncal]}
    calibration = np.array([i for i in development if tuple(pairs[i]) in calibration_keys], dtype=int)
    fitting = np.array([i for i in development if tuple(pairs[i]) not in calibration_keys], dtype=int)
    bank.save(directory)
    for label, rows in [("development", fitting), ("calibration", calibration)]:
        target = directory if label == "development" else directory / "calibration"
        target.mkdir(exist_ok=True)
        np.savez_compressed(target / "profiles.npz", pairs=pairs[rows], contexts=contexts[rows],
                            responses=responses[rows], identities=rows)
    np.savez_compressed(directory / "queries.npz", pairs=pairs[test], contexts=contexts[test], identities=test)
    (directory / "evaluator").mkdir()
    np.savez_compressed(directory / "evaluator" / "profiles.npz", responses=responses[test], identities=test)
    files = ["references.npz", "references.json", "profiles.npz", "calibration/profiles.npz",
             "queries.npz", "evaluator/profiles.npz"]
    write_json(directory / "protocol.json", {"schema": SCHEMA, "seed": seed, "held_contexts": held_contexts,
               "calibration_fraction": calibration_fraction, "calibration_group": "unordered_intervention_pair",
               "track": "new_context_with_measured_references", "synthetic": bank.metadata["synthetic"],
               "reference_access": bank.metadata["reference_access"], "files": {f: sha256(directory / f) for f in files},
               "fitting_identities": fitting.tolist(), "calibration_identities": calibration.tolist(),
               "evaluation_identities": test.tolist(), "confirmation_status": "synthetic" if bank.metadata["synthetic"] else "requires_exposure_and_replication_review"})
    return directory


def load_dataset(directory):
    """Load fitting outcomes and query identities, never evaluator/calibration labels."""
    directory = Path(directory)
    protocol = read_json(directory / "protocol.json")
    if protocol.get("schema") != SCHEMA:
        raise ValueError("Unsupported contextual dataset schema")
    for file, expected in protocol["files"].items():
        if Path(file).as_posix() not in {"references.npz", "references.json", "profiles.npz", "calibration/profiles.npz", "queries.npz", "evaluator/profiles.npz"}:
            raise ValueError("Unexpected dataset file")
        if sha256(directory / file) != expected:
            raise ValueError(f"Dataset checksum mismatch: {file}")
    bank = ReferenceBank.load(directory)
    development, query = load_npz(directory / "profiles.npz"), load_npz(directory / "queries.npz")
    if set(query) != {"pairs", "contexts", "identities"}:
        raise ValueError("Query bundle must contain identities only")
    bank.validate_rows(development["pairs"], development["contexts"])
    bank.validate_rows(query["pairs"], query["contexts"])
    fitting, cal, test = (set(protocol[k]) for k in ["fitting_identities", "calibration_identities", "evaluation_identities"])
    if fitting & cal or fitting & test or cal & test:
        raise ValueError("Fitting, calibration and evaluation identities overlap")
    if development["identities"].tolist() != protocol["fitting_identities"] or query["identities"].tolist() != protocol["evaluation_identities"]:
        raise ValueError("Dataset identity order mismatch")
    held = {np.flatnonzero(bank.arrays["contexts"] == c)[0] for c in protocol["held_contexts"]}
    if set(development["contexts"]) & held or not set(query["contexts"]) <= held:
        raise ValueError("Held-context isolation violated")
    return bank, development, query, protocol
