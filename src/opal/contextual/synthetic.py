"""Explicit synthetic multi-context fixtures; never biological confirmation."""
from itertools import combinations
import numpy as np
from .contracts import ReferenceBank, SCHEMA, prepare_dataset


def generate(seed=29, targets=10, genes=32, contexts=4):
    if targets < 4 or genes < 4 or contexts < 3:
        raise ValueError("Synthetic contextual fixtures need >=4 targets/genes and >=3 contexts")
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(targets, 4))
    decoder = rng.normal(size=(4, genes)) / np.sqrt(genes)
    states = np.linspace(-1, 1, contexts)
    singles = np.stack([latent @ decoder * (.8 + .15 * c) for c in states])
    ids = np.array([f"synthetic_state_{i}" for i in range(contexts)])
    pairs0 = np.array(list(combinations(range(targets), 2)), dtype=int)
    pairs = np.tile(pairs0, (contexts, 1))
    cx = np.repeat(np.arange(contexts), len(pairs0))
    pair_features = latent[pairs[:, 0]] * latent[pairs[:, 1]]
    correction_decoder = rng.normal(size=(4, genes)) / np.sqrt(genes)
    residual = (.15 + .08 * states[cx, None]) * (pair_features @ correction_decoder)
    bank = ReferenceBank({"singles": singles, "available": np.ones((contexts, targets), dtype=bool),
                          "controls": np.full((contexts, genes), 6.) + states[:, None] * np.linspace(-.1, .1, genes),
                          "features": states[:, None], "targets": np.array([f"T{i:03d}" for i in range(targets)]),
                          "genes": np.array([f"G{i:04d}" for i in range(genes)]), "contexts": ids},
                         {"schema": SCHEMA, "scale": "log2", "synthetic": True,
                          "reference_access": "controls_and_measured_singles_all_declared_contexts",
                          "context_features_source": "synthetic_generator",
                          "feature_names": ["known_synthetic_state"],
                          "intervention_records": {f"T{i:03d}": {"canonical_target": f"T{i:03d}", "reagent": "synthetic_reagent",
                                                                "modality": "synthetic", "dose": "not_applicable", "time": "synthetic_fixed_time"} for i in range(targets)},
                          "context_records": {str(c): {"biological_state": str(c), "modality": "synthetic",
                                                       "assay": "synthetic_log_expression", "study": "synthetic_generator",
                                                       "source": "known synthetic generator"} for c in ids}})
    responses = bank.additive(pairs, cx) + residual + rng.normal(0, .01, residual.shape)
    return bank, pairs, cx, responses


def prepare_fixture(directory, seed=29):
    bank, pairs, contexts, responses = generate(seed)
    return prepare_dataset(directory, bank, pairs, contexts, responses,
                           held_contexts=[str(bank.arrays["contexts"][-1])], seed=seed)
