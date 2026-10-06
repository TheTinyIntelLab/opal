"""Safe contextual exports, observed-single routes and honest support metadata."""
from pathlib import Path
import numpy as np
from opal.artifacts import load_npz, read_json, sha256, write_json
from .contracts import ReferenceBank, SCHEMA
from .model import ContextualKernel


def export_checkpoint(model, directory, calibration=None, provenance=None):
    directory = Path(directory)
    model.bank.save(directory)
    np.savez_compressed(directory / "fitted.npz", **model.state())
    write_json(directory / "calibration.json", calibration or {"status": "unavailable", "formal_guarantee": False})
    files = ["references.npz", "references.json", "fitted.npz", "calibration.json"]
    manifest = {"schema": SCHEMA, "checkpoint_state_version": 3, "config": model.config,
                "files": {f: sha256(directory / f) for f in files}, "reference_hash": model.bank.fingerprint,
                "gene_order": model.bank.arrays["genes"].tolist(), "population_mean_only": True,
                "combination_outcomes_included": False, "synthetic": model.bank.metadata["synthetic"],
                "provenance": provenance or {}, "reliability": "diagnostic_only_no_coverage_guarantee"}
    write_json(directory / "manifest.json", manifest)
    return manifest


class ContextPredictor:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = read_json(directory / "manifest.json")
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported contextual checkpoint schema")
        version = self.manifest.get('checkpoint_state_version', 2)
        if version not in {2, 3}:
            raise ValueError('Unsupported contextual fitted-state version')
        expected = {"references.npz", "references.json", "fitted.npz", "calibration.json"}
        if set(self.manifest.get("files", {})) != expected:
            raise ValueError("Unexpected checkpoint files")
        for name, checksum in self.manifest["files"].items():
            if sha256(directory / name) != checksum:
                raise ValueError("Contextual checkpoint checksum mismatch")
        self.bank = ReferenceBank.load(directory)
        if self.bank.fingerprint != self.manifest["reference_hash"] or self.bank.arrays["genes"].tolist() != self.manifest["gene_order"]:
            raise ValueError("Checkpoint reference or gene-order mismatch")
        state = load_npz(directory / 'fitted.npz')
        if version == 3 and not {'gene_scale', 'reference_distance'} <= set(state):
            raise ValueError('Fitted-state version 3 requires persisted reference scales')
        self.model = ContextualKernel(self.bank, self.manifest["config"]).restore(state)
        self.calibration = read_json(directory / "calibration.json")
        self.targets = {str(t): i for i, t in enumerate(self.bank.arrays["targets"])}
        self.contexts = {str(c): i for i, c in enumerate(self.bank.arrays["contexts"])}
        self.train_keys = {(int(c), *sorted(map(int, p))) for p, c in zip(self.model.train_pairs, self.model.train_contexts)}

    def predict(self, a=None, b=None, *, context, genes=None):
        if context not in self.contexts:
            raise ValueError("Unsupported context; compatible measured references must be registered first")
        if genes is not None and not np.array_equal(np.asarray(genes), self.bank.arrays["genes"]):
            raise ValueError("Gene order mismatch")
        ci = self.contexts[context]
        if a is None and b is not None:
            raise ValueError("Single queries require the first intervention")
        for target in [a, b]:
            if target is not None and target not in self.targets:
                raise ValueError("Unknown intervention")
            if target is not None and not self.bank.arrays["available"][ci, self.targets[target]]:
                raise ValueError("Measured single is unavailable; missing-single prediction is unsupported")
        g = len(self.bank.arrays["genes"])
        if a is None:
            effect, additive, route = np.zeros(g), np.zeros(g), "control_reference"
            support = {"reference_route": True}
        elif b is None:
            effect = self.bank.arrays["singles"][ci, self.targets[a]].copy()
            additive, route, support = effect.copy(), "single_reference", {"reference_route": True}
        else:
            pair = np.array([[self.targets[a], self.targets[b]]])
            cx = np.array([ci])
            effect = self.model.predict(pair, cx)[0]
            additive = self.bank.additive(pair, cx)[0]
            key = (ci, *sorted(pair[0].tolist()))
            observed_context = ci in self.model.train_contexts
            route = "fitting_combination" if key in self.train_keys else "unmeasured_pair_in_fitting_context" if observed_context else "new_context_with_measured_references"
            z = (self.bank.arrays["features"] - self.model.context_mean) / self.model.context_scale
            source = np.unique(self.model.train_contexts)
            weights = self.model.weights(pair, cx)[0]
            support = {"training_conditions": len(self.model.train_pairs), "training_contexts": len(source),
                       "same_context_training_conditions": int(np.sum(self.model.train_contexts == ci)),
                       "pair_seen_in_other_fitting_context": any(sorted(p.tolist()) == sorted(pair[0].tolist()) and int(c) != ci
                                                                 for p, c in zip(self.model.train_pairs, self.model.train_contexts)),
                       "nearest_training_context_distance": float(np.min(np.linalg.norm(z[source] - z[ci], axis=1))),
                       "maximum_absolute_influence_weight": float(np.max(np.abs(weights))),
                       "influence_is_not_a_mechanism": True}
        return {"effect": effect, "additive": additive, "residual": effect - additive,
                "single_estimation_corrections": np.zeros((2, g)), "genes": self.bank.arrays["genes"].copy(),
                "context": context, "scale": self.bank.metadata["scale"], "route": route,
                "support": support, "calibration": self.calibration,
                "reliability_status": "reference" if b is None else "synthetic_engineering_only" if self.bank.metadata["synthetic"] else "exploratory_no_validated_abstention",
                "synthetic": self.bank.metadata["synthetic"], "reference_hash": self.bank.fingerprint}
