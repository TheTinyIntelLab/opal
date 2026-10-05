"""No-pickle exports containing observed singles and development-only weights."""
from pathlib import Path
import time
import numpy as np
import torch
from opal.artifacts import write_json, read_json, load_npz, sha256, array_hash
from opal.models import ResidualNetwork
from opal.training import predict


def export_members(members, data, output, metadata):
    output = Path(output)
    if output.exists() and (output / "manifest.json").exists():
        raise FileExistsError("Export exists; choose a new immutable output directory")
    output.mkdir(parents=True, exist_ok=True)
    reference = {k: data[k] for k in ["singles", "targets", "genes", "control"]}
    np.savez_compressed(output / "references.npz", **reference)
    configs = []
    for i, member in enumerate(members):
        if member['type'] == 'noise_kernel':
            arrays = member['model'].state()
            config = {'type': 'noise_kernel', 'config': member['config'], 'seed': member['seed']}
        elif member['type'] == 'prior_kernel':
            arrays = member['model'].state()
            config = {'type': 'prior_kernel', 'config': member['config'], 'seed': member['seed']}
        elif member['type'] == 'feature_ridge':
            arrays = member['model'].state()
            config = {'type': 'feature_ridge', 'config': member['config'], 'seed': member['seed']}
        elif member['type'] == 'advanced':
            arrays = member['model'].state()
            config = {'type': 'advanced', 'config': member['config'], 'seed': member['seed']}
        elif member['type'] == 'kernel':
            arrays = member['model'].state()
            config = {'type': 'kernel', 'config': member['config'], 'seed': member['seed']}
        elif member["type"] == "neural":
            arrays = {k: v.detach().cpu().numpy() for k, v in member["model"].state_dict().items()}
            config = {"type": "neural", "config": member["config"], "epochs": member["epochs"], "seed": member["seed"]}
        else:
            arrays = {k: v for k, v in member["model"].items() if isinstance(v, np.ndarray) and k != "singles"}
            config = {"type": "baseline", "config": member["model"]["config"], "seed": member["seed"]}
        np.savez_compressed(output / f"member_{i}.npz", **arrays)
        configs.append(config)
    files = {p.name: sha256(p) for p in output.glob("*.npz")}
    manifest = {"schema": 1, "members": configs, "files": files, "metadata": metadata,
                "reference_hash": array_hash(data["singles"], data["targets"], data["genes"]),
                "gene_order": data["genes"].tolist(), "training_pairs": data["targets"][data["pairs"]].tolist(),
                "benchmark_pairs": data["targets"][data["query_pairs"]].tolist() if "query_pairs" in data else [],
                "test_double_expression_included": False, "population_mean_only": True}
    write_json(output / "manifest.json", manifest)
    return manifest


class Predictor:
    def __init__(self, path):
        started = time.perf_counter()
        path = Path(path)
        self.manifest = read_json(path / "manifest.json")
        for name, expected in self.manifest["files"].items():
            if Path(name).name != name or sha256(path / name) != expected:
                raise ValueError("Checkpoint checksum or file name mismatch")
        self.reference = load_npz(path / "references.npz")
        if self.manifest["gene_order"] != self.reference["genes"].tolist():
            raise ValueError("Gene schema mismatch")
        if array_hash(self.reference["singles"], self.reference["targets"], self.reference["genes"]) != self.manifest["reference_hash"]:
            raise ValueError("Profile-bank fingerprint mismatch")
        self.members = []
        for i, config in enumerate(self.manifest["members"]):
            arrays = load_npz(path / f"member_{i}.npz")
            if config['type'] == 'noise_kernel':
                from opal.models.noise_kernel import NoiseAwareKernel
                model = NoiseAwareKernel(self.reference['singles'], config['config']).restore(arrays)
                fitted = {'type': 'noise_kernel', 'model': model}
            elif config['type'] == 'prior_kernel':
                from opal.models.prior_kernel import PriorKernelResidual
                model = PriorKernelResidual(self.reference['singles'], arrays['bank'], arrays['mask'], config['config'])
                model.restore(arrays)
                fitted = {'type': 'prior_kernel', 'model': model}
            elif config['type'] == 'feature_ridge':
                from opal.models.experts import FeatureRidgeResidual
                model = FeatureRidgeResidual.from_state(arrays, config['config']['alpha'])
                if not np.array_equal(model.singles, self.reference['singles']):
                    raise ValueError('Feature-ridge observed-single bank mismatch')
                fitted = {'type': 'feature_ridge', 'model': model}
            elif config['type'] == 'advanced':
                from opal.models.advanced import AdvancedResidual
                model = AdvancedResidual.from_state(arrays, config['config'])
                if not np.array_equal(model.singles, self.reference['singles']):
                    raise ValueError('Advanced observed-single bank mismatch')
                fitted = {'type': 'advanced', 'model': model, 'config': config['config']}
            elif config['type'] == 'kernel':
                from opal.models.kernel import KernelResidual
                model = KernelResidual(self.reference['singles'], config['config']).restore(arrays)
                fitted = {'type': 'kernel', 'model': model}
            elif config["type"] == "neural":
                model = ResidualNetwork(arrays["embedding"], len(self.reference["genes"]), config["config"])
                model.load_state_dict({k: torch.from_numpy(v) for k, v in arrays.items()}, strict=True)
                model.eval()
                fitted = {"type": "neural", "model": model, "singles": self.reference["singles"], "config": config["config"]}
            else:
                fitted = {"type": "baseline", "model": {**arrays, "config": config["config"], "singles": self.reference["singles"]}}
            self.members.append(fitted)
        self.target_index = {t: i for i, t in enumerate(self.reference["targets"])}
        self.training_pairs = {tuple(sorted(p)) for p in self.manifest["training_pairs"]}
        self.benchmark_pairs = {tuple(sorted(p)) for p in self.manifest.get("benchmark_pairs", [])}
        self.cold_seconds = time.perf_counter() - started

    def predict(self, a=None, b=None, context=None, genes=None):
        supported = self.manifest["metadata"]["context"]
        if context is not None and context != supported:
            raise ValueError(f"Unsupported context; checkpoint supports {supported}")
        if genes is not None and list(genes) != self.reference["genes"].tolist():
            raise ValueError("Input gene order differs from frozen schema")
        zero = np.zeros(len(self.reference["genes"]))
        if a is None and b is None:
            effect, additive, spread = zero, zero, zero
            support = "observed_control_route"
        else:
            if a not in self.target_index or (b is not None and b not in self.target_index):
                raise ValueError("Missing observed single profile for query")
            if b is None:
                effect = additive = self.reference["singles"][self.target_index[a]]
                spread = zero
                support = "observed_single_route"
            else:
                if a == b:
                    raise ValueError("Same-target pairs are not registered")
                pair = np.array([[self.target_index[a], self.target_index[b]]])
                ensemble = np.stack([predict(m, pair)[0] for m in self.members])
                effect, spread = ensemble.mean(0), ensemble.std(0)
                additive = self.reference["singles"][pair[0]].sum(0)
                identity = tuple(sorted((a, b)))
                support = "development_combination" if identity in self.training_pairs else (
                    "registered_test_combination" if identity in self.benchmark_pairs else "other_pair_in_supported_context")
        return {"effect": effect, "additive": additive, "residual": effect - additive,
                "ensemble_spread": spread, "genes": self.reference["genes"],
                "units": self.manifest["metadata"].get("scale", "log2"),
                "combination_support": support,
                "support_interpretation": "registered pair identities only; not calibrated reliability or evidence of an unmeasured source pair",
                "spread_interpretation": "uncalibrated member disagreement"}

    def predict_batch(self, pairs, context=None):
        return [self.predict(a, b, context) for a, b in pairs]
