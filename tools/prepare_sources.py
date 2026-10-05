"""Build explicit dataset contracts from inspected original source metadata.

This script never guesses counts from a normalized matrix. Archive members are
read directly rather than extracted to arbitrary filesystem paths.
"""
import argparse
from pathlib import Path
import sys
import json
import gzip
import tarfile
import io
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import sha256, write_json, read_json
from opal.data.prepare import prepare


def norman(output, scale="log2"):
    import anndata
    path = ROOT / "data/raw/NormanWeissman2019_filtered.h5ad"
    registry = read_json(ROOT / "data/source_registry.json")["datasets"]["norman_scperturb_v1_1"]
    if sha256(path) != registry["checksum_sha256"]:
        raise ValueError("Norman source identity differs from the pinned accepted release")
    a = anndata.read_h5ad(path, backed="r")
    obs = a.obs
    context = "K562/CRISPRa/Norman2019"
    mapping = {}
    for label in sorted(set(obs.perturbation.astype(str))):
        rows = obs.perturbation.astype(str) == label
        guides = sorted(set(obs.loc[rows, "guide_id"].astype(str)))
        controls_verified = label != "control" or all(all(t.startswith("NegCtrl") for t in g.split(";")[0].split("_")) for g in guides)
        mapping[label] = {"components": [] if label == "control" else label.split("_"), "context": context,
                          "guide_ids": guides, "exclude": not controls_verified,
                          "construct_warning": "Target-level profile retains all listed variants; control-guide burden may differ."}
    config = {"input": str(path.relative_to(ROOT)), "count_location": "X", "condition_column": "perturbation", "gene_column": "ensemble_id",
              "stratum_columns": ["gemgroup"], "output": str(output), "sha256": sha256(path),
              "reuse_terms": "CC-BY-4.0, Zenodo 7278143 v1.1", "count_provenance": "scPerturb processing notebook loads original Norman counts.mtx.gz with scipy.mmread, subsets barcodes, and writes X without normalization; original GEO GSE133344.",
              "context": context, "scale": scale, "condition_map": mapping,
              "cell_filters": {"good_coverage": {"equals": True}, "number_of_cells": {"equals": 1}},
              "target_pair_aggregation": "equal-condition", "qc": {"minimum_library": 500, "maximum_library": 20000, "minimum_cells": 50},
              "split_seed": 1729, "folds": 5}
    write_json(ROOT / f"configs/datasets/norman-{scale}.json", config)
    a.file.close()
    return prepare(config)


def wessels(output, scale="log2"):
    path = ROOT / "data/raw/GSE213957_THP1-CaRPool-seq_and_GSE213957_HEX293FTstabRNA.tar.gz"
    metadata_path = ROOT / "data/raw/GSE213957_THP1-CaRPool-seq.metadata.tsv.gz"
    registry = read_json(ROOT / "data/source_registry.json")["datasets"]["wessels_original"]
    expected = {f["filename"]: f["checksum_sha256"] for f in registry["files"]}
    if sha256(path) != expected[path.name] or sha256(metadata_path) != expected[metadata_path.name]:
        raise ValueError("Wessels original archive or metadata identity changed")
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)
    pieces, ordered_obs, schema = [], [], None
    archive_digest = sha256(path)
    with tarfile.open(path, "r:gz") as archive:
        for lane in range(1, 5):
            prefix = f"THP1-CaRPool-seq_and_HEK293FTstabRNA.GEXGDO{lane}"
            def text_member(name):
                with gzip.GzipFile(fileobj=archive.extractfile(name)) as f:
                    return f.read().decode().splitlines()
            features = [r.split("\t") for r in text_member(prefix + ".features.tsv.gz")]
            gene_idx = np.array([i for i, row in enumerate(features) if row[2] == "Gene Expression"])
            genes = [features[i][0] for i in gene_idx]
            if schema is not None and genes != schema:
                raise ValueError("Gene order differs across lanes")
            schema = genes
            barcodes = [f"L{lane}_{b}" for b in text_member(prefix + ".barcodes.tsv.gz")]
            row_indices = np.array([i for i, barcode in enumerate(barcodes) if barcode in metadata.index])
            with gzip.GzipFile(fileobj=archive.extractfile(prefix + ".matrix.mtx.gz")) as f:
                matrix = mmread(f).tocsr()
            # Only metadata-confirmed THP1 cells and GEX rows; never pool HEK293.
            pieces.append(matrix[gene_idx][:, row_indices].T.tocsr())
            ordered_obs.append(metadata.loc[[barcodes[i] for i in row_indices]])
            print(f"Wessels lane {lane}: retained {len(row_indices)} metadata-matched THP1 cells", flush=True)
    counts = sparse.vstack(pieces, format="csr")
    obs = pd.concat(ordered_obs)
    if set(obs.index) != set(metadata.index) or obs.index.duplicated().any():
        raise ValueError("Incomplete or duplicate Wessels barcode join")
    context = "THP1/Cas13d/Wessels2023"
    mapping = {}
    for label in sorted(set(obs.GenePair)):
        components = [t for t in label.split("_") if t != "NT"]
        rows = obs.GenePair == label
        mapping[label] = {"components": components, "context": context,
                          "guide_ids": sorted(set(obs.loc[rows, "CRISPR.Array"])),
                          "exclude": len(components) > 2 or len(set(components)) != len(components),
                          "construct_warning": "Multiple arrays retained at target-pair level; Single/Dual guide confidence must be inspected."}
    derived = ROOT / "data/raw/wessels_thp1_counts.npz"
    sparse.save_npz(derived, counts)
    meta = ROOT / "data/raw/wessels_thp1_metadata.json"
    write_json(meta, {"genes": schema, "cells": obs.index.tolist(), "conditions": obs.GenePair.tolist(), "strata": obs["TenX.Lane"].tolist()})
    totals = np.asarray(counts.sum(1)).ravel()
    refs = np.array([len(mapping[l]["components"]) <= 1 for l in obs.GenePair])
    # Frozen from controls/singles only; source-specific broad retention, not Norman's cap.
    maximum = int(np.ceil(np.quantile(totals[refs], 0.995)))
    config = {"input": str(derived.relative_to(ROOT)), "metadata": str(meta.relative_to(ROOT)), "output": str(output), "sha256": sha256(derived),
              "reuse_terms": "GEO public scientific reuse with original attribution; original deposit redistribution license not specified, do not redistribute raw files.",
              "count_provenance": "Original GSE213957 GEXGDO 10x Matrix Market UMI counts; GEX features only; barcode-joined THP1 metadata.",
              "source_archive_sha256": archive_digest, "source_metadata_sha256": sha256(metadata_path),
              "context": context, "scale": scale, "condition_map": mapping,
              "target_pair_aggregation": "equal-condition", "qc": {"minimum_library": 500, "maximum_library": maximum, "minimum_cells": 50},
              "split_seed": 1729, "folds": 5, "qc_amendment": "Maximum library is 99.5th percentile of controls/singles. Initial minimum 50 retained cells may exclude many singles; no test-response based adjustment."}
    write_json(ROOT / f"configs/datasets/wessels-{scale}.json", config)
    return prepare(config)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["norman", "wessels"])
    p.add_argument("--scale", choices=["log2", "count"], default="log2")
    p.add_argument("--output")
    args = p.parse_args()
    output = Path(args.output or f"data/prepared/{args.dataset}-{args.scale}")
    result = globals()[args.dataset](output, args.scale)
    print(json.dumps(result, indent=2))
