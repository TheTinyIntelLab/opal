"""Single-only fitted representations with recorded gene order."""
import numpy as np
import torch
from torch import nn
from opal.artifacts import array_hash

_CACHE = {}


def fit_representation(singles, dimension, kind="pca", seed=0, epochs=100, partitions=None):
    x = np.asarray(singles, np.float64)
    key = (array_hash(x), dimension, kind, seed if kind == "encoder" else 0, epochs,
           array_hash(*partitions) if partitions is not None else None)
    if key in _CACHE:
        return _CACHE[key]
    mean = x.mean(0)
    dim = min(dimension, len(x) - 1, x.shape[1])
    if dim < 1:
        raise ValueError("At least two singles required")
    if kind == "identity":
        result = {"kind": kind, "embedding": np.eye(len(x)), "mean": mean,
                  "matrix": np.zeros((len(x), x.shape[1]))}
        _CACHE[key] = result
        return result
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    matrix = vt[:dim]
    if kind == "encoder":
        torch.manual_seed(seed)
        encoder = nn.Linear(x.shape[1], dim, bias=False, dtype=torch.float64)
        decoder = nn.Linear(dim, x.shape[1], bias=False, dtype=torch.float64)
        with torch.no_grad():
            encoder.weight.copy_(torch.from_numpy(matrix))
            decoder.weight.copy_(torch.from_numpy(matrix.T))
        opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=0.003, weight_decay=0.01)
        tx = torch.from_numpy(x - mean)
        halves = None if partitions is None else [torch.as_tensor(p - mean, dtype=torch.float64) for p in partitions]
        for _ in range(epochs):
            if halves is None:
                noisy = tx + 0.01 * torch.randn_like(tx)
                loss = (decoder(encoder(noisy)) - tx).square().mean()
            else:
                loss = ((decoder(encoder(halves[0])) - halves[1]).square().mean()
                        + (decoder(encoder(halves[1])) - halves[0]).square().mean()) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        matrix = encoder.weight.detach().numpy().copy()
    elif kind != "pca":
        raise ValueError("Unknown representation")
    result = {"kind": kind, "embedding": (x - mean) @ matrix.T, "mean": mean, "matrix": matrix}
    if len(_CACHE) >= 32:
        _CACHE.clear()
    _CACHE[key] = result
    return result


def pair_features(embedding, pairs, products=True):
    a, b = embedding[pairs[:, 0]], embedding[pairs[:, 1]]
    parts = [a + b, np.abs(a - b)]
    if products:
        parts.append(a * b)
    return np.concatenate(parts, axis=1)
