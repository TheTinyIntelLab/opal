"""Pinned native NAIAD adapter with externally supplied splits and no test labels.

Architecture, datasets, optimizer, scheduler and trainer execute from the official
checkout. This module only supplies inputs, selects its native validation-best
checkpoint, and exports numerical state without executable pickle objects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REVISION = "9207e9e855e865160a196a6989392f6db622b0d6"
SOURCE_HASHES = {
    "naiad/__init__.py": "ebdd38c70ebbc919854a4c32607e23d61a6153f0bc15f8f15e5a9bd192d45628",
    "naiad/active_learning.py": "9b10d098779c44dd40b5ab6a020ecf7861b6f0ebf101810b92e786d99419d309",
    "naiad/models.py": "9d93ae89651adba44c059a0a9b07dfd5b29d70051c47d19d6d7a83148d761c6a",
    "naiad/naiad.py": "8d1a30698b2afcbe22870bcb0d2ab9c3386a529669c887c46ba42408dde0fb98",
    "naiad/utils.py": "4b9f60323ac972f8d4f217882b94766c9fabf4a16c102c4bcb62bf05469581b4",
}


@dataclass(frozen=True)
class NativeConfig:
    seed: int = 20261022
    epochs: int = 100
    batch_size: int = 1024


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_official(source):
    """Refuse a different or modified implementation before importing it."""
    source = Path(source).resolve()
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != REVISION:
        raise ValueError("NAIAD revision mismatch")
    for name, expected in SOURCE_HASHES.items():
        if file_hash(source / name) != expected:
            raise ValueError(f"NAIAD source hash mismatch: {name}")
    name = "_opal_official_naiad_9207e9e"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, source / "naiad/__init__.py",
            submodule_search_locations=[str(source / "naiad")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    del sys.modules[key]
            raise
    return sys.modules[name]


def _validate(genes, pairs, singles):
    genes = np.asarray(genes)
    pairs = np.asarray(pairs)
    singles = np.asarray(singles)
    if genes.ndim != 1 or len(set(genes.tolist())) != len(genes):
        raise ValueError("Unique one-dimensional gene identities required")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("Integer pair matrix required")
    if np.any(pairs < 0) or np.any(pairs >= len(genes)) or np.any(pairs[:, 0] >= pairs[:, 1]):
        raise ValueError("Distinct canonical pairs required")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate canonical pairs")
    if singles.shape != (len(genes),) or not np.isfinite(singles).all():
        raise ValueError("One finite measured single per gene required")
    return genes, pairs, singles


class NativeNAIAD:
    def __init__(self, source, config=NativeConfig()):
        self.source = str(Path(source).resolve())
        self.config = config

    def fit(self, genes, pairs, singles, targets, split, check=None):
        """Fit only split 0/1. Split-2 targets may be NaN or sentinel values."""
        genes, pairs, singles = _validate(genes, pairs, singles)
        targets, split = np.asarray(targets), np.asarray(split)
        if targets.shape != (len(pairs),) or split.shape != (len(pairs),):
            raise ValueError("Aligned target/split vectors required")
        if set(np.unique(split)) != {0, 1, 2}:
            raise ValueError("Nonempty train, validation and test splits required")
        train, val = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
        if len(train) < 2 or len(val) < 3:
            raise ValueError("Native constructor needs two train and three validation rows")
        used = np.r_[train, val]
        if not np.isfinite(targets[used]).all():
            raise ValueError("Finite training/validation outcomes required")
        support = np.bincount(pairs[train].ravel(), minlength=len(genes)) > 0
        if not support[pairs].all():
            raise ValueError("Native lookup embeddings require train-supported genes")
        if self.config.epochs < 1 or self.config.batch_size < 1:
            raise ValueError("Positive epochs and batch size required")
        native = load_official(self.source)
        import pandas as pd
        def frame(rows):
            return pd.DataFrame(dict(id1=genes[pairs[rows, 0]].astype(str),
                                     id2=genes[pairs[rows, 1]].astype(str),
                                     id1_score=singles[pairs[rows, 0]].astype(np.float32),
                                     id2_score=singles[pairs[rows, 1]].astype(np.float32),
                                     comb_score=targets[rows].astype(np.float32)))
        train_frame, val_frame = frame(train), frame(val)
        data = pd.concat([train_frame, val_frame], ignore_index=True)
        # Constructor requires positive n_test. Its unused counts are reconciled
        # immediately; no test pair or test outcome ever enters this object.
        runner = native.NAIAD(data, n_train=len(train), n_test=2,
                             batch_size=self.config.batch_size)
        runner.n_val, runner.n_test = len(val), 0
        runner.set_seed(self.config.seed)
        runner.data = {"train": train_frame, "val": val_frame}
        runner.prepare_dataloaders()  # intentionally never prepare_data/random split
        runner.initialize_model(device="cpu")
        runner.setup_trainer(n_epoch=self.config.epochs)
        hook = None
        if check is not None:
            check()
            hook = runner.model.register_forward_pre_hook(lambda module, args: check())
        try:
            runner.train_model()
        finally:
            if hook is not None:
                hook.remove()
                # Native validation selection deep-copies the model, including
                # hooks. Never retain the caller's guard inside a checkpoint.
                if hasattr(runner, "best_model"):
                    runner.best_model._forward_pre_hooks.clear()
        if check is not None:
            check()
        self.genes = genes.astype(str).copy()
        self.singles = singles.astype(np.float32).copy()
        self.support = support.copy()
        self.treatments = list(runner.treatments)
        self.model_args = dict(runner.model_args)
        self.model = runner.best_model.eval()
        self.training_metrics = {k: list(v) for k, v in runner.training_metrics.items()}
        self.best_epoch = int(np.argmin(self.training_metrics["val_loss"])) + 1
        self.fit_rows = {"train": train.tolist(), "val": val.tolist()}
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        return self

    def predict(self, pairs):
        """Symmetrize the native phenotype head over the two single orders."""
        _, pairs, _ = _validate(self.genes, pairs, self.singles)
        if not self.support[pairs].all():
            raise ValueError("Unsupported gene query")
        mapping = {name: i for i, name in enumerate(self.treatments)}
        idx = np.array([[mapping[g] for g in row] for row in self.genes[pairs]], dtype=np.int64)
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(pairs), self.config.batch_size):
                sl = slice(start, start + self.config.batch_size)
                ids = torch.from_numpy(idx[sl])
                phenos = torch.from_numpy(self.singles[pairs[sl]])
                a = self.model(ids, phenos).reshape(-1)
                b = self.model(ids, phenos.flip(1)).reshape(-1)
                outputs.append(((a + b) / 2).numpy())
        result = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite native predictions")
        return result

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {f"state_{i}": v.detach().numpy() for i, v in enumerate(self.model.state_dict().values())}
        arrays.update(genes=self.genes, singles=self.singles, support=self.support)
        np.savez(directory / "state.npz", **arrays)
        meta = dict(revision=REVISION, config=asdict(self.config), model_args=self.model_args,
                    treatments=self.treatments, state_keys=list(self.model.state_dict()),
                    training_metrics=self.training_metrics, best_epoch=self.best_epoch,
                    fit_rows=self.fit_rows, parameter_count=self.parameter_count,
                    state_sha256=file_hash(directory / "state.npz"))
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, source, directory):
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if meta["revision"] != REVISION or file_hash(directory / "state.npz") != meta["state_sha256"]:
            raise ValueError("Native artifact signature mismatch")
        native = load_official(source)
        result = cls(source, NativeConfig(**meta["config"]))
        result.model = sys.modules[native.__name__ + ".models"].model_loader(
            len(meta["treatments"]), **meta["model_args"])
        with np.load(directory / "state.npz", allow_pickle=False) as arrays:
            result.model.load_state_dict({key: torch.from_numpy(arrays[f"state_{i}"].copy())
                                          for i, key in enumerate(meta["state_keys"])})
            result.genes = arrays["genes"].copy()
            result.singles = arrays["singles"].copy()
            result.support = arrays["support"].copy()
        for key in ["model_args", "treatments", "training_metrics", "best_epoch", "fit_rows", "parameter_count"]:
            setattr(result, key, meta[key])
        result.model.eval()
        return result


def fit_native(source, genes, pairs, singles, targets, split, config=NativeConfig(), check=None):
    return NativeNAIAD(source, config).fit(genes, pairs, singles, targets, split, check=check)
