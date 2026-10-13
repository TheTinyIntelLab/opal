"""Existing-cache streaming, full-library normalization and isolated role files."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import numpy as np
from ..artifacts import fingerprint, write_json
from .artifacts import save_npz
from .contracts import PROTOCOL, ContractError, Unsupported, canonical_pair, literal_identity, folds, shared_panel, qualify_graph


def column(node):
    import h5py
    if isinstance(node, h5py.Group):
        values = column(node["categories"])
        return np.asarray([values[i] if i >= 0 else "nan" for i in node["codes"][:]])
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in node[:]])


def label(value):
    parts = value.split("|")
    if len(parts) > 2 or any(not p or p == "nan" or p.startswith("INTERGENIC") for p in parts):
        return None
    targets = [p for p in parts if not re.fullmatch(r"CTRL00\w*", p)]
    return tuple(sorted(targets)) if len(set(targets)) == len(targets) else None


def metadata(file, review):
    genes = column(file["var"][file["var"].attrs.get("_index", "_index")]).tolist()
    contexts, strata = column(file["obs/Treatment"]), column(file["obs/SampleIndex"])
    targets, guides = column(file["obs/target_gene_name"]), column(file["obs/feature_call"])
    if not (len(contexts) == len(strata) == len(targets) == len(guides)):
        raise ContractError("Metadata lengths mismatch")
    labels, keys, mixtures, counts, half_counts = [], [], defaultdict(Counter), Counter(), Counter()
    for i, (target, guide, c, s) in enumerate(zip(targets, guides, contexts, strata)):
        assigned = label(target)
        if len(target.split("|")) != len(guide.split("|")) or c not in PROTOCOL["contexts"]:
            assigned = None
        labels.append(assigned)
        key = (str(c), str(s), assigned)
        keys.append(key)
        if assigned is not None:
            counts[key] += 1
            half_counts[(key, i % 2)] += 1
            mixtures[key][guide] += 1
    rows = []
    declared = review.get("condition_assignments", {})
    original_rows = defaultdict(list)
    for i, k in enumerate(keys):
        original_rows[k].append(i)
    for (c, s, p), count in sorted(counts.items()):
        if len(p) != 2:
            continue
        pair = canonical_pair(p, review.get("aliases"))
        identity = json.dumps([pair, c, s], separators=(",", ":"))
        row = dict(declared.get(identity, {}), pair_id=pair, context=c, stratum=s,
            pair_cells=count, control_cells=counts[(c, s, ())],
            single_a_cells=counts[(c, s, (p[0],))], single_b_cells=counts[(c, s, (p[1],))],
            original_targets=list(p), guide_mixture=dict(mixtures[(c, s, p)]),
            control_guide_mixture=dict(mixtures[(c, s, ())]),
            single_guide_mixtures=[dict(mixtures[(c, s, (component,))]) for component in p],
            reference_original_rows=dict(control=original_rows[(c, s, ())],
                single_a=original_rows[(c, s, (p[0],))], single_b=original_rows[(c, s, (p[1],))]),
            canonical_component_order=[literal_identity(t, review.get("aliases")) for t in p],
            independent_initiation=None, original_rows=original_rows[(c, s, p)])
        rows.append(row)
    return dict(genes=genes, keys=keys, labels=labels, counts=counts, half_counts=half_counts,
        rows=rows, excluded_cells=sum(p is None for p in labels))


def columns(matrix, start, stop, rows, row_mask=None):
    from scipy import sparse
    encoding = matrix.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding == "csc_matrix":
        ptr = matrix["indptr"][start:stop+1]
        left, right = int(ptr[0]), int(ptr[-1])
        if row_mask is not None:
            indices = matrix["indices"][left:right]
            # CSC row indices are structural metadata. Read expression only
            # at permitted reference entries, before any pair outcome decode.
            keep = np.flatnonzero(row_mask[indices])
            data = matrix["data"][left+keep] if len(keep) else np.empty(0)
            gene_columns = np.repeat(np.arange(stop-start), np.diff(ptr))
            return sparse.csc_matrix((data, (indices[keep], gene_columns[keep])), shape=(rows, stop-start))
        return sparse.csc_matrix((matrix["data"][left:right], matrix["indices"][left:right], ptr-left), shape=(rows, stop-start))
    if encoding == "csr_matrix":
        # Bounded rows for CSR, no full matrix read or densification.
        pieces = []
        for r in range(0, rows, 1024):
            if row_mask is not None:
                block_rows = []
                for i in range(r, min(r+1024, rows)):
                    if not row_mask[i]:
                        block_rows.append(sparse.csr_matrix((1, stop-start)))
                        continue
                    left, right = map(int, matrix["indptr"][i:i+2])
                    indices = matrix["indices"][left:right]
                    keep = np.flatnonzero((indices >= start) & (indices < stop))
                    data = matrix["data"][left+keep] if len(keep) else np.empty(0)
                    block_rows.append(sparse.csr_matrix((data, indices[keep]-start, [0, len(keep)]), shape=(1, stop-start)))
                pieces.append(sparse.vstack(block_rows, format="csr"))
                continue
            ptr = matrix["indptr"][r:min(r+1024, rows)+1]
            left, right = int(ptr[0]), int(ptr[-1])
            x = sparse.csr_matrix((matrix["data"][left:right], matrix["indices"][left:right], ptr-left), shape=(len(ptr)-1, int(matrix.attrs["shape"][1])))
            pieces.append(x[:, start:stop])
        return sparse.vstack(pieces, format="csc")
    if hasattr(matrix, "shape"):
        if row_mask is not None:
            permitted = np.flatnonzero(row_mask)
            values = sparse.coo_matrix(matrix[permitted, start:stop])
            return sparse.csc_matrix((values.data, (permitted[values.row], values.col)), shape=(rows, stop-start))
        return sparse.csc_matrix(matrix[:, start:stop])
    raise ContractError("Unsupported backed count encoding")


def aggregate(matrix, meta, selected, references_only, budget):
    from scipy import sparse
    n, g = len(meta["keys"]), len(meta["genes"])
    library = np.zeros(n)
    reference_rows = np.asarray([k[2] is not None and len(k[2]) < 2 for k in meta["keys"]]) if references_only else None
    for start in range(0, g, 128):
        budget.check()
        values = columns(matrix, start, min(start+128, g), n, reference_rows)
        if not np.isfinite(values.data).all() or np.any(values.data < 0) or not np.allclose(values.data, np.rint(values.data), atol=1e-6, rtol=0):
            raise ContractError("Verified finite nonnegative integer counts required")
        library += np.asarray(values.sum(1)).ravel()
    groups = sorted(k for k in meta["counts"] if not references_only or len(k[2]) < 2)
    index = {k: i for i, k in enumerate(groups)}
    matrices, denominators = {}, {}
    for partition in ("primary", "a", "b"):
        row, col, weight = [], [], []
        counts = np.zeros(len(groups), int)
        for i, k in enumerate(meta["keys"]):
            if k not in index or library[i] <= 0 or (partition != "primary" and i % 2 != (0 if partition == "a" else 1)):
                continue
            row.append(index[k]); col.append(i); weight.append(10000/library[i]); counts[index[k]] += 1
        matrices[partition] = sparse.csr_matrix((weight, (row, col)), shape=(len(groups), n))
        denominators[partition] = counts
    means = {p: np.zeros((len(groups), len(selected))) for p in matrices}
    for start in range(0, g, 128):
        positions = np.flatnonzero((selected >= start) & (selected < min(start+128, g)))
        if not len(positions):
            continue
        budget.check()
        values = columns(matrix, start, min(start+128, g), n, reference_rows)[:, selected[positions]-start]
        for partition in means:
            means[partition][:, positions] = (matrices[partition] @ values).toarray()
    result = {}
    for partition, sums in means.items():
        result[partition] = {k: np.log2(1+sums[i]/denominators[partition][i])
            for i, k in enumerate(groups) if denominators[partition][i] >= 10}
    return result, {p: {str(k): int(denominators[p][i]) for i, k in enumerate(groups)} for p in denominators}


def matched_effects(means, graph, panel=None):
    pairs = graph["complete_pairs"]
    x, y, strata_records = [], [], []
    for p in pairs:
        per_x, per_y = [], []
        for c in PROTOCOL["contexts"]:
            xs, ys = [], []
            for s in graph["strata"][c]:
                row = next(r for r in graph["graph"] if r["pair_id"] == p and r["context"] == c and r["stratum"] == s)
                original = tuple(row["original_targets"])
                keys = [(c, s, ()), (c, s, (original[0],)), (c, s, (original[1],)), (c, s, original)]
                if any(k not in means for k in keys):
                    raise Unsupported("Missing matched ten-cell partition references/outcomes")
                control, a, b, outcome = [means[k] if panel is None else means[k][panel] for k in keys]
                components = list(original)
                canonical = json.loads(p)
                # Aliases are already frozen in graph; original order maps via row.
                effects = np.stack((a-control, b-control))
                order = row.get("canonical_component_order", components)
                if sorted(order) != canonical:
                    raise ContractError("Exact canonical component order missing for aliases")
                effects = effects[[order.index(t) for t in canonical]]
                xs.append(effects); ys.append(outcome-control)
                strata_records.append(dict(pair_id=p, context=c, stratum=s, response=(outcome-control).tolist()))
            per_x.append(np.mean(xs, axis=0)); per_y.append(np.mean(ys, axis=0))
        x.append(per_x); y.append(per_y)
    return np.asarray(x), np.asarray(y), strata_records


def prepare_roles(output, x, y, ids, genes, qualification, partitions=None, stratum_records=None, budget=None, metric_positives=None, program=None):
    """Separate outcome-free queries from source labels in each exclusive fold."""
    output = Path(output)
    write = budget.write_json if budget is not None else write_json
    output.mkdir(parents=True, exist_ok=False)
    ids = np.asarray([canonical_pair(p) for p in ids])
    if len(set(ids)) != len(ids) or len(ids) < 36 or len(genes) < 256:
        raise ContractError("Unchanged successor support minima")
    if x.shape != (len(ids), 2, 2, len(genes)) or y.shape != (len(ids), 2, len(genes)):
        raise ContractError("Role array shape mismatch")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ContractError("Finite role arrays required")
    assignment = folds(ids)
    records = []
    partitions = dict(primary=(x, y), **(partitions or {}))
    for role, arrays in partitions.items():
        rx, ry = arrays[:2]
        ey = arrays[2] if len(arrays) == 3 else ry
        if rx.shape != x.shape or ry.shape != y.shape or ey.shape != y.shape or not all(np.isfinite(a).all() for a in (rx, ry, ey)):
            raise ContractError("Reference swaps must retain identical queries")
        for fold in range(3):
            train = np.asarray([assignment[p] != fold for p in ids]); query = ~train
            prefix = f"{role}-{fold}"
            positives = {} if metric_positives is None else {"metric_positive_a": metric_positives[0][train], "metric_positive_b": metric_positives[1][train]}
            source = save_npz(output / (prefix+"-sourcefit.npz"), budget, singles=rx[train], responses=ry[train], pair_ids=ids[train], **positives)
            refs = save_npz(output / (prefix+"-references.npz"), budget, singles=rx[query], pair_ids=ids[query])
            evaluator = save_npz(output / (prefix+"-evaluator.npz"), budget, responses=ey[query], pair_ids=ids[query])
            records.append(dict(role=role, fold=fold, sourcefit=source, references=refs, evaluator=evaluator,
                source_ids=ids[train].tolist(), query_ids=ids[query].tolist()))
    manifest = dict(experiment_id=PROTOCOL["experiment_id"], protocol_hash=fingerprint(PROTOCOL),
        genes=list(genes), genes_hash=fingerprint(list(genes)), contexts=PROTOCOL["contexts"],
        folds=assignment, records=records, qualification=qualification, reference_swaps_available=len(partitions) == 3,
        exposure="exposed_TNF_history", source_identity=qualification.get("source_identity"),
        stratum_evaluator=None, program=program)
    if stratum_records:
        write(output / "stratum_evaluator.json", stratum_records)
        from ..artifacts import sha256
        manifest["stratum_evaluator"] = dict(path="stratum_evaluator.json", sha256=sha256(output / "stratum_evaluator.json"))
    write(output / "prepared_manifest.json", manifest)
    write(output / "exposure_ledger.json", dict(exposure="exposed_public_development", records=records,
        irreversible=True, independent_cultures=False, query_outcomes_only_in_evaluator=True))
    write(output / "label_ledger.json", dict(unique_pairs=len(ids), identities=ids.tolist(),
        sourcefit_and_evaluator_overlap_across_folds=True, independent_samples=False))
    return manifest


def prepare_raw(raw, review, review_root, source_identity, output, budget):
    import h5py
    write_json = budget.write_json
    if not source_identity.get("qualified"):
        raise ContractError("Verified source identity required before preparation")
    with h5py.File(raw, "r") as file:
        meta = metadata(file, review)
        graph = qualify_graph(meta["rows"], review, review_root)
        write_json(Path(output) / "qualification.json", dict(graph, source_identity=source_identity))
        if not graph["qualified"]:
            raise Unsupported("Source metadata/rights qualification failed")
        if "layers/counts" not in file:
            raise ContractError("Verified counts layer required; no upstream-X substitution")
        references, reference_counts = aggregate(file["layers/counts"], meta, np.arange(len(meta["genes"])), True, budget)
        single_effects = {}
        for c in PROTOCOL["contexts"]:
            targets = sorted({k[2][0] for k in references["primary"] if k[0] == c and len(k[2]) == 1})
            effects = []
            for target in targets:
                values = [references["primary"][(c, s, (target,))]-references["primary"][(c, s, ())]
                    for s in graph["strata"][c] if (c, s, (target,)) in references["primary"] and (c, s, ()) in references["primary"]]
                if len(values) == len(graph["strata"][c]):
                    effects.append(np.mean(values, axis=0))
            single_effects[c] = np.asarray(effects)
        panel, panel_record = shared_panel(meta["genes"], single_effects)
        # Freeze reference-only selection before reading any pair expression.
        write_json(Path(output) / "panel.json", panel_record)
        program = None
        if review.get("program") is not None:
            from .contracts import reviewed
            p = review["program"]
            if not reviewed(p, review_root) or p.get("source_only_or_biologically_prespecified") is not True:
                raise ContractError("Program requires reviewed pre-scoring source/prespecified evidence")
            weights = np.asarray([p["gene_weights"].get(g, 0) for g in panel_record["genes"]], float)
            if not np.isfinite(weights).all() or np.linalg.norm(weights) == 0:
                raise ContractError("Nonzero finite source-defined program weights required")
            program = dict(weights=weights.tolist(), top_k=p["top_k"], pool=graph["complete_pairs"],
                frozen_before_outcomes=True, provenance=fingerprint(p))
            write_json(Path(output)/"program_freeze.json", program)
        means, counts = aggregate(file["layers/counts"], meta, panel, False, budget)
        x, y, strata = matched_effects(means["primary"], graph)
        swaps, swap_reasons, metric_positives = {}, [], None
        try:
            xa, ya, _ = matched_effects(means["a"], graph)
            xb, yb, _ = matched_effects(means["b"], graph)
            metric_positives = (ya.copy(), yb.copy())
        except Unsupported as error:
            swap_reasons.append("metric_positive:"+str(error))
        try:
            # The pair count estimator is shared; only references/control anchors differ.
            # Pair means come from primary for both swaps, not disjoint pair labels.
            for key in means["primary"]:
                if len(key[2]) == 2:
                    means["a"][key] = means["primary"][key]
                    means["b"][key] = means["primary"][key]
            xa, ya, _ = matched_effects(means["a"], graph)
            xb, yb, _ = matched_effects(means["b"], graph)
            swaps = dict(a_to_b=(xa, ya, yb), b_to_a=(xb, yb, ya))
        except Unsupported as error:
            swap_reasons.append("reference_swap:"+str(error))
        qualification = dict(graph, source_identity=source_identity, reference_partition_reasons=swap_reasons,
            reference_partitions_disjoint=bool(swaps), panel=panel_record, counts=counts,
            reference_counts=reference_counts, normalization=PROTOCOL["normalization"],
            reference_partition_rule="source_cell_row_index_modulo_two",
            pair_estimator_in_both_swaps="same_primary_condition_mean_normalized_counts",
            technical_partition_independent_cultures=False)
        return prepare_roles(Path(output)/"prepared", x, y, graph["complete_pairs"], panel_record["genes"], qualification,
            partitions=swaps, stratum_records=strata, budget=budget, metric_positives=metric_positives, program=program)
