"""Metadata-only global folds and fixed degree-balanced bridges."""
from collections import Counter
import json

import numpy as np

from .contracts import ContractError, canonical_pair


def folds(pair_ids, seed=20261011, count=3, aliases=None):
    ids = sorted({canonical_pair(p, aliases) for p in pair_ids})
    degree = [Counter() for _ in range(count)]
    sizes = [0] * count
    # Seeded traversal, then smallest fold and smallest resulting squared
    # component-degree sum. All ties use fold index, never outcomes.
    order = np.random.default_rng(seed).permutation(ids)
    assignment = {}
    for p in order:
        parts = json.loads(p)
        k = min(range(count), key=lambda f: (sizes[f], sum(2 * degree[f][c] + 1 for c in parts), f))
        assignment[p] = k
        sizes[k] += 1
        degree[k].update(parts)
    return {p: assignment[p] for p in ids}


def bridges(pair_ids, count=4):
    candidates = sorted({canonical_pair(p) for p in pair_ids})
    if len(candidates) < count:
        raise ContractError("Insufficient metadata bridge candidates")
    selected, degree = [], Counter()
    for _ in range(count):
        p = min((p for p in candidates if p not in selected),
                key=lambda p: (sum(2 * degree[c] + 1 for c in json.loads(p)), p))
        selected.append(p)
        degree.update(json.loads(p))
    return selected


def manifest(pair_ids, cfg):
    assignment = folds(pair_ids, cfg["fold_seed"], cfg["outer_folds"], cfg.get("aliases"))
    result = []
    for k in range(cfg["outer_folds"]):
        query = [p for p in assignment if assignment[p] == k]
        source = [p for p in assignment if assignment[p] != k]
        anchors = bridges(source, cfg["bridge_count"])
        components = {c for p in anchors for c in json.loads(p)}
        overlap = Counter(sum(c in components for c in json.loads(p)) for p in query)
        result.append(dict(fold=k, source=source, query=query, bridges=anchors,
                           component_overlap_counts={str(i): overlap[i] for i in range(3)},
                           disjoint_component_claim_permitted=overlap[0] > 0))
    return result


def source_permutation(pair_ids, seed):
    """One mapping reused across every source stratum."""
    ids = sorted({canonical_pair(p) for p in pair_ids})
    permuted = np.random.default_rng(seed).permutation(ids)
    return dict(zip(ids, map(str, permuted)))


def bridge_permutation(pair_ids, seed):
    ids = sorted({canonical_pair(p) for p in pair_ids})
    shift = 1 + int(np.random.default_rng(seed).integers(len(ids) - 1))
    return dict(zip(ids, ids[shift:] + ids[:shift]))
