"""Original ComboSciPlex feasibility track with equal-well profile weighting.

Seven matched pairs support an exploratory threefold development batch, not
the chemical-validation gate specified in the full research roadmap.
"""
from pathlib import Path
import sys
import gzip
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import write_json, sha256, read_json
from opal.profiles import validate_counts, library_normalize, mean_profile, effect, eligible_genes
from opal.data.synthetic import save_prepared


def main():
    cells_path = ROOT / "data/raw/source_metadata/combosciplex_cell_metadata.tsv.gz"
    genes_path = ROOT / "data/raw/source_metadata/combosciplex_gene_metadata.tsv.gz"
    counts_path = ROOT / "data/raw/GSE206741_count_matrix.mtx.gz"
    expected = read_json(ROOT / "data/source_registry.json")["datasets"]["combosciplex_original"]["checksum_sha256"]
    if sha256(counts_path) != expected:
        raise ValueError("Chemical count source identity changed")
    obs = pd.read_csv(cells_path, sep="\t")
    var = pd.read_csv(genes_path, sep="\t")
    with gzip.open(counts_path, "rb") as f:
        counts = mmread(f).T.tocsr()
    if counts.shape != (len(obs), len(var)) or obs.Cell.duplicated().any() or var.id.duplicated().any():
        raise ValueError("Source metadata dimensions or identifiers disagree")
    validate_counts(counts, "GSE206741 original UMI Matrix Market")
    keys = [tuple(sorted(t for t in [a, b] if t != "DMSO")) for a, b in zip(obs.Drug1, obs.Drug2)]
    totals = np.asarray(counts.sum(1)).ravel()
    reference = np.array([len(k) <= 1 for k in keys])
    maximum = int(np.ceil(np.quantile(totals[reference], 0.995)))
    keep_cells = (totals >= 500) & (totals <= maximum)
    counts, obs = counts[keep_cells], obs.loc[keep_cells].reset_index(drop=True)
    keys = [k for k, keep in zip(keys, keep_cells) if keep]
    control_rows = np.array([len(k) == 0 for k in keys])
    normalizer = float(np.median(np.asarray(counts[control_rows].sum(1)).ravel()))
    xn = library_normalize(counts, normalizer)
    reference = np.array([len(k) <= 1 for k in keys])
    labels = np.array(["|".join(k) if k else "control" for k in keys])
    gene_mask = eligible_genes(xn[reference], labels[reference])
    wells = obs.Well.to_numpy()
    control_means = [mean_profile(xn[np.flatnonzero(control_rows & (wells == w))]) for w in sorted(set(wells[control_rows]))]
    control_mean = np.mean(control_means, axis=0)
    profiles, cell_counts, well_counts = {}, {}, {}
    unique = sorted(set(keys))
    for k in unique:
        rows = np.array([t == k for t in keys])
        values = []
        for w in sorted(set(wells[rows])):
            idx = np.flatnonzero(rows & (wells == w))
            values.append(effect(mean_profile(xn[idx]), control_mean))
        profiles[k] = np.mean(values, axis=0)
        cell_counts[k], well_counts[k] = int(rows.sum()), len(values)
    singles = sorted(k[0] for k in unique if len(k) == 1 and cell_counts[k] >= 50)
    pair_keys = [k for k in unique if len(k) == 2 and len(set(k)) == 2 and all(t in singles for t in k) and cell_counts[k] >= 50]
    arrays = {"singles": np.stack([profiles[(t,)][gene_mask] for t in singles]),
              "doubles": np.stack([profiles[k][gene_mask] for k in pair_keys]),
              "pairs": np.array([[singles.index(t) for t in k] for k in pair_keys]),
              "targets": np.array(singles), "genes": var.id.to_numpy(dtype=str)[gene_mask], "control": np.log2(control_mean[gene_mask] + 1)}
    output = ROOT / "data/prepared/chemical-exploratory"
    source = {"synthetic": False, "context": "A549/chemical/1uM_each/24h/GSE206741", "scale": "log2",
              "input": str(counts_path), "sha256": sha256(counts_path), "cell_metadata_sha256": sha256(cells_path),
              "gene_metadata_sha256": sha256(genes_path), "replicate_weighting": "equal wells within condition; shared equal-control-well plate reference",
              "dose_provenance": "CPA primary paper Methods: final concentration 1uM for each compound, 24h",
              "reuse_terms": "GEO scientific reuse with attribution; raw redistribution terms unspecified",
              "source_version": "GSE206741 original deposit", "status": "exploratory_small_matched_suite_not_chemical_release_validation"}
    save_prepared(output, arrays, source, folds=3)
    write_json(output / "preprocessing.json", {"normalizer": normalizer, "maximum_library": maximum, "minimum_library": 500,
               "minimum_cells": 50, "gene_mask": gene_mask.tolist(), "stratum": "one source sample/plate", "weighting": source["replicate_weighting"]})
    write_json(output / "qc_report.json", {"source_cells": len(keep_cells), "retained_cells": int(keep_cells.sum()), "singles": len(singles),
               "eligible_pairs": len(pair_keys), "eligible_pair_ids": [list(k) for k in pair_keys],
               "conditions": [{"components": list(k), "cells": cell_counts[k], "wells": well_counts[k]} for k in unique],
               "limitation": "Too few pairs for the planned chemical confirmatory protocol; threefold exploratory amendment before fitting."})
    print({"output": str(output), "singles": len(singles), "pairs": len(pair_keys), "genes": int(gene_mask.sum())})


if __name__ == "__main__":
    main()
