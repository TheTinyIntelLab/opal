"""Fail-closed adapter for explicit condition/context/stratum mappings."""
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import sparse
from opal.artifacts import read_json, write_json, sha256, array_hash
from opal.profiles import validate_counts, library_normalize, mean_profile, effect, eligible_genes
from opal.data.synthetic import save_prepared
from opal.splits import canonical_pair, preserved_pair_split


def load_raw(config):
    path = Path(config["input"])
    if not path.exists():
        raise FileNotFoundError(f"Raw data missing: {path}. Follow data/source_registry.json; do not substitute synthetic data.")
    digest = sha256(path)
    if digest != config["sha256"]:
        raise ValueError("Raw input checksum disagrees with source manifest")
    if not config.get("reuse_terms") or not config.get("count_provenance"):
        raise ValueError("Reuse terms and raw-count provenance must be documented")
    if path.suffix == ".h5ad":
        import anndata
        adata = anndata.read_h5ad(path)
        location = config["count_location"]
        if location.startswith("layers/"):
            x = adata.layers[location.split("/", 1)[1]]
        elif location == "X":
            x = adata.X
        elif location == "raw/X" and adata.raw is not None:
            x = adata.raw.X
        else:
            raise ValueError("Explicit count location not found")
        gene_source = adata.raw.var if location == "raw/X" else adata.var
        genes = np.asarray(gene_source.index if config.get("gene_column") is None else gene_source[config["gene_column"]], str)
        labels = np.asarray(adata.obs[config["condition_column"]], str)
        strata_columns = config.get("stratum_columns", [])
        strata = np.array(["|".join(map(str, row)) for row in adata.obs[strata_columns].itertuples(index=False, name=None)]) if strata_columns else np.full(x.shape[0], "matched")
        cells = np.asarray(adata.obs_names, str)
        source_cells = len(cells)
        allowed = np.ones(len(cells), bool)
        for field, rule in config.get("cell_filters", {}).items():
            values = np.asarray(adata.obs[field])
            if "equals" in rule:
                allowed &= values == rule["equals"]
            if "maximum" in rule:
                allowed &= values <= rule["maximum"]
        x, labels, strata, cells = x[allowed], labels[allowed], strata[allowed], cells[allowed]
    elif path.suffix == ".npz":
        x = sparse.load_npz(path)
        meta = read_json(config["metadata"])
        genes, labels, strata, cells = [np.asarray(meta[k], str) for k in ["genes", "conditions", "strata", "cells"]]
        source_cells = len(cells)
    else:
        raise ValueError("Supported count inputs: sparse NPZ plus metadata JSON, or H5AD with explicit location")
    validate_counts(x, config["count_provenance"])
    if len(genes) != x.shape[1] or any(len(a) != x.shape[0] for a in (labels, strata, cells)):
        raise ValueError("Metadata dimensions disagree with count matrix")
    if len(set(genes)) != len(genes) or len(set(cells)) != len(cells):
        raise ValueError("Gene and cell identifiers must be unique")
    return sparse.csr_matrix(x), genes, labels, strata, cells, digest, source_cells


def prepare(config):
    x, genes, labels, strata, cells, digest, source_cells = load_raw(config)
    mapping = config["condition_map"]
    if set(labels) - set(mapping):
        raise ValueError("Every raw condition needs an explicit component/guide/context mapping")
    context = config["context"]
    totals = np.asarray(x.sum(axis=1)).ravel()
    q = config["qc"]
    valid = (totals >= q["minimum_library"]) & (totals <= q["maximum_library"])
    valid &= np.array([mapping[l]["context"] == context and not mapping[l].get("exclude", False) for l in labels])
    retention = {l: {"before": int(sum(labels == l)), "after": int(sum((labels == l) & valid))} for l in sorted(set(labels))}
    x, labels, strata, cells = x[valid], labels[valid], strata[valid], cells[valid]
    components = {}
    for label in set(labels):
        comp = mapping[label]["components"]
        if len(comp) > 2 or (len(comp) == 2 and comp[0] == comp[1]):
            raise ValueError("Unsupported repeated-target or higher-order condition")
        if "guide_ids" not in mapping[label]:
            raise ValueError("Guide/compound identity must be retained explicitly, even if unavailable")
        components[label] = tuple(sorted(comp))
    control_labels = {l for l, c in components.items() if len(c) == 0}
    if not control_labels:
        raise ValueError("No mapped controls")
    control_rows = np.array([l in control_labels for l in labels])
    normalizer = float(np.median(np.asarray(x[control_rows].sum(axis=1)).ravel()))
    xn = library_normalize(x, normalizer)
    reference_rows = np.array([len(components[l]) <= 1 for l in labels])
    keep = eligible_genes(xn[reference_rows], labels[reference_rows], config.get("minimum_mean", 0.1), config.get("minimum_detection", 0.01))
    if not keep.any():
        raise ValueError("No eligible output genes")
    minimum = q["minimum_cells"]
    rng = np.random.default_rng(config.get("measurement_seed", 31415))
    partitions = np.zeros(len(labels), int)
    for label in sorted(set(labels)):
        for s in sorted(set(strata[labels == label])):
            idx = np.flatnonzero((labels == label) & (strata == s))
            partitions[rng.permutation(idx)[::2]] = 1
    # Each label/stratum mean receives equal weight; no pooling by cell depth.
    def estimate(rows):
        outputs, support = {}, {}
        for label in sorted(set(labels)):
            values = []
            for s in sorted(set(strata[labels == label])):
                ri = np.flatnonzero(rows & (labels == label) & (strata == s))
                ci = np.flatnonzero(rows & control_rows & (strata == s))
                if len(ri) == 0:
                    continue
                if len(ci) == 0:
                    raise ValueError(f"No matched control for {label}, stratum {s}")
                values.append(effect(mean_profile(xn[ri]), mean_profile(xn[ci]), config.get("scale", "log2")))
            if values:
                outputs[label] = np.mean(values, axis=0)
                support[label] = sorted(set(strata[(labels == label) & rows]))
        return outputs, support
    profiles, support = estimate(np.ones(len(labels), bool))
    halves = [estimate(partitions == i)[0] for i in [0, 1]]
    eligible = {l for l in profiles if retention[l]["after"] >= minimum}
    singles_by_target = {}
    for l in sorted(eligible):
        if len(components[l]) == 1:
            t = components[l][0]
            if t in singles_by_target:
                raise ValueError("Multiple single constructs for a target require an explicit aggregation amendment")
            singles_by_target[t] = l
    targets = sorted(singles_by_target)
    target_index = {t: i for i, t in enumerate(targets)}
    pair_labels, excluded = [], {}
    seen_pairs = set()
    for l in sorted(components):
        if len(components[l]) != 2:
            continue
        a, b = canonical_pair(*components[l])
        if l not in eligible or a not in target_index or b not in target_index:
            excluded[l] = "minimum cells or matched single missing"
            continue
        sa, sb = singles_by_target[a], singles_by_target[b]
        if support[l] != support[sa] or support[l] != support[sb]:
            excluded[l] = "unmatched stratum support between pair and singles"
            continue
        if (a, b) in seen_pairs:
            if config.get("target_pair_aggregation") != "equal-condition":
                raise ValueError("Multiple pair constructs require an explicit target-level aggregation amendment")
        seen_pairs.add((a, b))
        pair_labels.append(l)
    grouped_pairs = {}
    for l in pair_labels:
        grouped_pairs.setdefault(components[l], []).append(l)
    pair_groups = list(grouped_pairs.values())
    arrays = {"singles": np.stack([profiles[singles_by_target[t]][keep] for t in targets]),
              "doubles": np.stack([np.mean([profiles[l][keep] for l in group], axis=0) for group in pair_groups]),
              "pairs": np.array([[target_index[t] for t in components[group[0]]] for group in pair_groups]),
              "control": np.log2(mean_profile(xn[control_rows])[keep] + 1) if config.get("scale", "log2") == "log2" else mean_profile(xn[control_rows])[keep],
              "targets": np.array(targets), "genes": genes[keep]}
    log_control = xn[control_rows].tocsr(copy=True)
    log_control.data = np.log2(log_control.data + 1)
    variance = mean_profile(log_control.multiply(log_control)) - mean_profile(log_control) ** 2
    arrays["control_sd"] = np.sqrt(np.maximum(variance, 0))[keep]
    output = Path(config["output"])
    manifest = {k: config[k] for k in ["input", "sha256", "count_provenance", "reuse_terms", "context", "condition_map"]}
    manifest.update({"synthetic": False, "observed_sha256": digest, "scale": config.get("scale", "log2")})
    split_override = None
    if "preserved_split" in config:
        roles = config["preserved_split"]
        split_override = preserved_pair_split(arrays["targets"][arrays["pairs"]], roles["development"], roles["protected"],
                                              config.get("split_seed", 1729), config.get("folds", 5))
        manifest["preserved_split"] = roles
    split = save_prepared(output, arrays, manifest, config.get("split_seed", 1729), config.get("folds", 5), split_override)
    write_json(output / "preprocessing.json", {"scale": manifest["scale"], "normalizer": normalizer,
               "qc": q, "gene_selection_resources": "controls and singles only", "stratum_weight": "equal", "gene_mask": keep.tolist()})
    write_json(output / "qc_report.json", {"retention": retention, "exclusions": excluded, "singles": len(targets),
               "source_cells": source_cells, "metadata_filtered_cells": source_cells - len(valid),
               "eligible_pair_conditions": len(pair_labels), "eligible_pairs": len(pair_groups),
               "pair_groups": pair_groups, "pair_aggregation": config.get("target_pair_aggregation", "none"),
               "observed_coverage": len(pair_groups) / max(1, len(targets) * (len(targets) - 1) / 2)})
    all_half = {}
    for i, half in enumerate(halves):
        all_half[f"singles_{i}"] = np.stack([half[singles_by_target[t]][keep] for t in targets])
        all_half[f"doubles_{i}"] = np.stack([np.mean([half[l][keep] for l in group], axis=0) for group in pair_groups])
    dev, test = split["development"], split["test"]
    np.savez_compressed(output / "profile_uncertainty.npz", **{k: v[dev] if k.startswith("doubles") else v for k, v in all_half.items()})
    np.savez_compressed(output / "evaluator" / "profile_uncertainty.npz", **{k: v[test] if k.startswith("doubles") else v for k, v in all_half.items()})
    # Extended responses stay in evaluator storage; extended training requires a separate prepared run.
    np.savez_compressed(output / "evaluator" / "extended_profiles.npz", genes=genes,
                        doubles=np.stack([np.mean([profiles[l] for l in group], axis=0) for group in pair_groups]))
    metadata = [{"condition": l, "components": list(components[l]), "context": context,
                 "strata": support[l], "guide_ids": mapping[l]["guide_ids"], **retention[l]} for l in profiles]
    write_json(output / "condition_metadata.json", metadata)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(metadata), output / "condition_metadata.parquet")
    except ImportError:
        write_json(output / "metadata_format.json", {"format": "json", "reason": "Install opal-research[data] for Parquet"})
    return {"output": str(output), "data_hash": array_hash(arrays["singles"], arrays["pairs"], arrays["doubles"])}
