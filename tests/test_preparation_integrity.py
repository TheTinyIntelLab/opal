import itertools
import numpy as np
import pytest
from scipy import sparse
from opal.artifacts import read_json, write_json, load_npz, sha256
from opal.data.prepare import prepare
from opal.data.synthetic import generate
from opal.training import fit
from opal.inference import export_members, Predictor
from opal.training import predict
from opal.training.benchmark import development, final
from opal.data.synthetic import save_prepared


def test_prepare_response_intervention_preserves_reference_preprocessing(tmp_path):
    targets = [f"T{i}" for i in range(8)]
    conditions = ["ctrl"] + targets + [a + "+" + b for a, b in itertools.combinations(targets, 2)]
    labels = np.repeat(conditions, 6)
    rows = np.tile([12, 18, 10, 0], (len(labels), 1))
    source = tmp_path / "counts.npz"
    sparse.save_npz(source, sparse.csr_matrix(rows))
    metadata = tmp_path / "metadata.json"
    write_json(metadata, {"genes": ["G1", "G2", "G3", "excluded"], "conditions": labels.tolist(),
                          "cells": [f"C{i}" for i in range(len(labels))], "strata": ["batch"] * len(labels)})
    mapping = {c: {"components": [] if c == "ctrl" else c.split("+"), "context": "fixture", "guide_ids": [c]} for c in conditions}
    config = {"input": str(source), "metadata": str(metadata), "sha256": sha256(source),
              "output": str(tmp_path / "before"), "context": "fixture", "condition_map": mapping,
              "count_provenance": "synthetic raw integer counts", "reuse_terms": "generated fixture",
              "qc": {"minimum_library": 1, "maximum_library": 100, "minimum_cells": 4}}
    prepare(config)
    before = load_npz(tmp_path / "before/profiles.npz")
    groups = read_json(tmp_path / "before/qc_report.json")["pair_groups"]
    held_labels = {label for i in before["query_indices"] for label in groups[i]}
    changed = np.isin(labels, list(held_labels))
    rows[changed] = [12, 8, 10, 10]  # Same library size, cell identity, context and QC.
    sparse.save_npz(source, sparse.csr_matrix(rows))
    prepare({**config, "output": str(tmp_path / "after"), "sha256": sha256(source)})
    after = load_npz(tmp_path / "after/profiles.npz")
    for key in ["singles", "control", "genes", "doubles", "pairs", "query_indices"]:
        assert np.array_equal(before[key], after[key])
    assert read_json(tmp_path / "before/preprocessing.json") == read_json(tmp_path / "after/preprocessing.json")
    assert read_json(tmp_path / "before/splits.json") == read_json(tmp_path / "after/splits.json")
    assert not np.array_equal(load_npz(tmp_path / "before/evaluator/profiles.npz")["doubles"],
                              load_npz(tmp_path / "after/evaluator/profiles.npz")["doubles"])


@pytest.mark.parametrize("mutation", ["weights", "gene_order", "profile_bank"])
def test_checkpoint_rejects_tampering(tmp_path, mutation):
    data = generate(targets=8, genes=20, latent=4)
    member = fit({"family": "additive"}, data["singles"], data["pairs"], data["doubles"])
    path = tmp_path / "checkpoint"
    export_members([member], data, path, {"context": "synthetic", "scale": "log2"})
    manifest = read_json(path / "manifest.json")
    if mutation == "weights":
        with (path / "member_0.npz").open("ab") as f:
            f.write(b"changed")
    elif mutation == "gene_order":
        manifest["gene_order"] = manifest["gene_order"][::-1]
    else:
        reference = load_npz(path / "references.npz")
        reference["singles"][0, 0] += 1
        np.savez_compressed(path / "references.npz", **reference)
        # A valid transport hash must not bypass the independent bank identity.
        manifest["files"]["references.npz"] = sha256(path / "references.npz")
    write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        Predictor(path)


@pytest.mark.parametrize("training_pairs", [1, 2, 3])
def test_fixed_output_basis_caps_rank_and_roundtrips(tmp_path, training_pairs):
    data = generate(targets=8, genes=20, latent=4)
    member = fit({"family": "bilinear", "fixed_basis": True, "rank": 8, "epochs": 2},
                 data["singles"], data["pairs"][:training_pairs], data["doubles"][:training_pairs])
    assert member["model"].rank == training_pairs
    reference = {**data, "pairs": data["pairs"][:training_pairs]}
    export_members([member], reference, tmp_path / "checkpoint", {"context": "synthetic"})
    loaded = Predictor(tmp_path / "checkpoint")
    a, b = data["targets"][data["pairs"][5]]
    assert np.allclose(loaded.predict(a, b)["effect"], predict(member, data["pairs"][[5]])[0])


def test_changed_evaluator_refused_before_final_fitting(tmp_path):
    data = generate(targets=8, genes=20, latent=4)
    bank, run = tmp_path / "bank", tmp_path / "run"
    save_prepared(bank, data, {"synthetic": True})
    development(bank, run, {"epochs": 1, "ranks": [2], "ridge_alpha": [1], "ablations": False,
                            "screening_seeds": [1], "finalist_seeds": [1], "ensemble_seeds": [1]})
    evaluator = load_npz(bank / "evaluator/profiles.npz")
    evaluator["doubles"][0, 0] += 1
    np.savez_compressed(bank / "evaluator/profiles.npz", **evaluator)
    with pytest.raises(ValueError, match="Evaluator"):
        final(bank, run)
    assert not (run / "predictions.npz").exists()
    assert read_json(run / "final_status.json")["status"] == "failed"


def test_evaluator_order_guard_independent_of_transport_hash(tmp_path):
    data = generate(targets=8, genes=20, latent=4)
    bank, run = tmp_path / "bank", tmp_path / "run"
    save_prepared(bank, data, {"synthetic": True})
    development(bank, run, {"epochs": 1, "ranks": [2], "ridge_alpha": [1], "ablations": False,
                            "screening_seeds": [1], "finalist_seeds": [1], "ensemble_seeds": [1]})
    evaluator = load_npz(bank / "evaluator/profiles.npz")
    evaluator["pair_indices"] = evaluator["pair_indices"][::-1]
    evaluator["doubles"] = evaluator["doubles"][::-1]
    np.savez_compressed(bank / "evaluator/profiles.npz", **evaluator)
    selection = read_json(run / "selection.json")
    selection["evaluator_sha256"] = sha256(bank / "evaluator/profiles.npz")
    write_json(run / "selection.json", selection)
    with pytest.raises(ValueError, match="pair order"):
        final(bank, run)
    assert not (run / "test_results.json").exists()


def test_exported_query_support_uses_identities_without_test_responses(tmp_path):
    arrays = generate(targets=8, genes=20, latent=4)
    bank = tmp_path / "bank"
    save_prepared(bank, arrays, {"synthetic": True})
    data = load_npz(bank / "profiles.npz")
    member = fit({"family": "additive"}, data["singles"], data["pairs"], data["doubles"])
    path = tmp_path / "checkpoint"
    export_members([member], data, path, {"context": "synthetic"})
    model = Predictor(path)
    train = data["targets"][data["pairs"][0]]
    held = data["targets"][data["query_pairs"][0]]
    assert model.predict(*train)["combination_support"] == "development_combination"
    assert model.predict(*held[::-1])["combination_support"] == "registered_test_combination"
    assert set(load_npz(path / "references.npz")) == {"singles", "targets", "genes", "control"}
    # Legacy manifests lack query identities, so the API must not infer them.
    manifest = read_json(path / "manifest.json")
    del manifest["benchmark_pairs"]
    write_json(path / "manifest.json", manifest)
    assert Predictor(path).predict(*held)["combination_support"] == "other_pair_in_supported_context"
