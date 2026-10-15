"""Integrity of the frozen information contract and its numerical refusal rules."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from opal.frontier_bridge.artifacts import load_state, save_state
from opal.frontier_bridge.contracts import (ContractError, assert_roles, canonical_pair,
                                           load_config, query_bundle, read_json, sha256, write_json)
from opal.frontier_bridge.evaluation import bootstrap_weights, metric_controls, retrieval
from opal.frontier_bridge.model import (Unsupported, adapt, adapted_predict, coordinates,
                                       fit_encoder, fit_source, fit_source_free, mode_features, nuisance_predict,
                                       solve_target)
from opal.frontier_bridge.preparation import synthetic_banks
from opal.frontier_bridge.resources import Budget
from opal.frontier_bridge.splits import (bridge_permutation, bridges, folds, manifest,
                                        source_permutation)
from opal.frontier_bridge.workflow import predict_checkpoint

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/experiments/opal_frontier_bridge_v1.json"


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


def test_frozen_contract_cannot_relax_support_or_change_budget(tmp_path):
    cfg = load_config(CONFIG)
    for key, value in [("minimum_common_genes", 207), ("bridge_count", 8), ("target_penalty", .1), ("bootstrap_draws", 100)]:
        altered = copy.deepcopy(cfg)
        altered[key] = value
        path = tmp_path / "config.json"
        path.write_text(json.dumps(altered))
        with pytest.raises(ContractError, match="Frozen contract mismatch"):
            load_config(path)


def test_alias_and_reversal_global_exclusion_and_dose_tokens():
    assert canonical_pair('["B","old-A"]', {"old-A": "A"}) == '["A","B"]'
    with pytest.raises(ContractError, match="Self pairs"):
        canonical_pair('["A","old-A"]', {"old-A": "A"})
    assert canonical_pair('["A@1nM","B"]') != canonical_pair('["A@2nM","B"]')
    with pytest.raises(ContractError, match="Global query exclusion"):
        assert_roles(['["B","A"]'], ['["C","D"]'], ['["A","B"]'])
    with pytest.raises(ContractError):
        assert_roles(['["A","B"]'], ['["C","D"]'], ['["E","F"]'])


def test_metadata_folds_bridges_shuffle_shared_across_strata():
    banks, _ = synthetic_banks("transfer", genes=32)
    ids = banks["TNFA+"]["pair_ids"]
    split = manifest(ids, load_config(CONFIG))
    assert sorted(p for f in split for p in f["query"]) == sorted(ids)
    for f in split:
        assert_roles(f["source"], f["bridges"], f["query"])
        assert len(f["bridges"]) == 4
        assert sum(f["component_overlap_counts"].values()) == len(f["query"])
    assert folds(ids) == folds(ids[::-1])
    assert bridges(ids) == bridges(ids[::-1])
    mapping = source_permutation(ids, 20261021)
    assert mapping == source_permutation(np.tile(ids, 4), 20261021)
    assert sorted(mapping.values()) == sorted(ids)
    cyclic = bridge_permutation(split[0]["bridges"], 20261021)
    assert all(p != q for p, q in cyclic.items())


def test_query_schema_and_swap_symmetry():
    banks, genes = synthetic_banks("transfer", genes=32)
    bank = banks["TNFA+"]
    encoder = fit_encoder(bank["singles"])
    np.testing.assert_allclose(coordinates(bank["singles"], encoder), coordinates(bank["singles"][:, ::-1], encoder), atol=1e-12)
    x = bank["singles"][:1]
    ids = [json.dumps(json.loads(bank["pair_ids"][0])[::-1])]
    q = query_bundle(x[:, ::-1], ids, genes, "effect", genes, "effect")
    np.testing.assert_array_equal(q["singles"], x)
    for candidate_genes, scale in [(genes[::-1], "effect"), (genes, "counts")]:
        with pytest.raises(ContractError, match="Mismatched"):
            query_bundle(x, ids, candidate_genes, scale, genes, "effect")


def test_normalized_target_objective_is_invariant_to_gene_replication():
    rng = np.random.default_rng(33)
    features = rng.normal(size=(4, 2, 20))
    eta = np.asarray([.8, -.4])
    outcome = np.einsum("nlg,l->ng", features, eta) + rng.normal(size=20)
    fitted = solve_target(features, outcome)
    repeated = solve_target(np.tile(features, (1, 1, 5)), np.tile(outcome, (1, 5)))
    np.testing.assert_allclose(fitted["eta"], repeated["eta"], atol=1e-14)
    centered = features - features.mean(0)
    design = centered.transpose(0, 2, 1).reshape(-1, 2)
    expected = np.linalg.solve(design.T @ design / outcome.size + np.eye(2),
                               design.T @ (outcome - outcome.mean(0)).reshape(-1) / outcome.size)
    np.testing.assert_allclose(fitted["eta"], expected, atol=1e-14)


def test_rank_deficient_bridges_refuse_complete_design():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(4, 2, 20))
    features[:, 1] = features[:, 0]
    with pytest.raises(Unsupported, match="numerical rank"):
        solve_target(features, rng.normal(size=(4, 20)))
    banks, _ = synthetic_banks("rank_deficient", genes=32)
    with pytest.raises(Unsupported):
        fit_encoder(banks["TNFA+"]["singles"])


def test_null_and_affine_distortion_do_not_create_source_modes():
    for case in ("null", "affine"):
        banks, _ = synthetic_banks(case, genes=32)
        bank = banks["TNFA+"]
        if case == "null":
            with pytest.raises(Unsupported):
                fit_source(bank["singles"], bank["responses"], bank["pair_ids"], 1)
        else:
            # Ridge towards one deliberately leaves affine distortion in
            # cross-fitted residuals; the matched affine comparator explains it.
            from opal.frontier_bridge.model import fit_nuisance
            reference = fit_nuisance(bank["singles"], bank["responses"], "affine", .1)
            pred = bank["singles"].sum(1) + nuisance_predict(reference, bank["singles"])
            assert np.mean((pred - bank["responses"]) ** 2) < np.mean((bank["singles"].sum(1) - bank["responses"]) ** 2)


def test_global_query_mutation_cannot_change_source_or_prediction():
    banks, _ = synthetic_banks("transfer", genes=32)
    cfg = load_config(CONFIG)
    f = manifest(banks["TNFA+"]["pair_ids"], cfg)[0]
    bank = banks["TNFA+"]
    mask = np.isin(bank["pair_ids"], f["source"])
    before = fit_source(bank["singles"][mask], bank["responses"][mask], bank["pair_ids"][mask], .1)
    mutated = copy.deepcopy(banks)
    for b in mutated.values():
        b["responses"][np.isin(b["pair_ids"], f["query"])] += 1e6
    after = fit_source(mutated["TNFA+"]["singles"][mask], mutated["TNFA+"]["responses"][mask], bank["pair_ids"][mask], .1)
    assert manifest(mutated["TNFA+"]["pair_ids"], cfg) == manifest(bank["pair_ids"], cfg)
    for state_key in ("basis", "scales"):
        np.testing.assert_array_equal(before[state_key], after[state_key])
    np.testing.assert_array_equal(mode_features(before, banks["TNFA-"]["singles"]), mode_features(after, mutated["TNFA-"]["singles"]))
    # Planted source structure is learnable without query pair labels.
    q = ~mask
    pred = bank["singles"][q].sum(1) + nuisance_predict(before["nuisance"], bank["singles"][q])
    pred += np.einsum("nlg,l->ng", mode_features(before, bank["singles"][q]), before["scales"])
    assert np.mean((pred - bank["responses"][q]) ** 2) < np.mean((bank["singles"][q].sum(1) - bank["responses"][q]) ** 2)


def test_nontransferable_modes_receive_zero_coefficient():
    rng = np.random.default_rng(61)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(32, 4)))
    z = rng.normal(size=(4, 2))
    features = z[:, :, None] * orthogonal[:, :2].T[None, :, :]
    outcome = z @ orthogonal[:, 2:].T
    target = solve_target(features, outcome)
    np.testing.assert_allclose(target["eta"], 0., atol=1e-15)


def test_source_free_uses_source_scale_and_exact_gene_averaged_penalty():
    banks, _ = synthetic_banks("transfer", genes=32)
    source, target = banks["TNFA+"], banks["TNFA-"]
    encoder = fit_encoder(source["singles"])
    state = fit_source_free(source["singles"], target["singles"][:4], target["responses"][:4], encoder)
    source_phi = coordinates(source["singles"], encoder)
    np.testing.assert_allclose(state["scale"], np.maximum(source_phi.std(0), 1e-6))
    phi = (coordinates(target["singles"][:4], encoder) - state["center"]) / state["scale"]
    y = target["responses"][:4] - target["singles"][:4].sum(1)
    xc, yc = phi - phi.mean(0), y - y.mean(0)
    expected = np.linalg.solve(xc.T @ xc / y.size + np.eye(10), xc.T @ yc / y.size)
    np.testing.assert_allclose(state["coefficients"], expected)


def test_source_mode_predictions_use_source_representation_for_query_singles():
    banks, _ = synthetic_banks("transfer", genes=32)
    source = banks["TNFA+"]
    model = fit_source(source["singles"], source["responses"], source["pair_ids"], 1)
    before = copy.deepcopy(model["encoder"])
    query = banks["TNFA-"]["singles"].copy() * 100
    _ = mode_features(model, query)
    for key in before:
        np.testing.assert_array_equal(model["encoder"][key], before[key])


def test_retrieval_qualifies_exact_and_preserves_undefined_denominator():
    rng = np.random.default_rng(7)
    truth = rng.normal(size=(10, 20))
    controls = metric_controls(truth)
    assert controls["qualified"]
    assert controls["exact"]["score"] == pytest.approx(1.)
    assert controls["constant"]["score"] is None
    pred = truth.copy()
    pred[0] = pred[1:].mean(0)
    result = retrieval(pred, truth)
    assert result["score"] is None and result["undefined_rows"] == [0]
    assert result["candidate_count"] == 10


def test_product_draws_share_pair_blocks_and_zero_draws():
    ids = ['["A","B"]', '["B","C"]', '["A","B"]']
    weights, targets, multiplicity = bootstrap_weights(ids, draws=2000)
    np.testing.assert_array_equal(weights[:, 0], weights[:, 2])
    assert np.any(weights.sum(1) == 0)
    np.testing.assert_array_equal(multiplicity.sum(1), len(targets))


def test_safe_inference_replay_in_separate_process_without_evaluator(tmp_path):
    rng = np.random.default_rng(5)
    state = dict(query=dict(singles=rng.normal(size=(8, 2, 20)), pair_ids=np.repeat('["A","B"]', 8)),
                 model=dict(kind="reference", reference=dict(family="bridge_offset", alpha=1, intercept=rng.normal(size=20))))
    expected = predict_checkpoint(state)
    record = save_state(tmp_path / "checkpoint", state)
    np.testing.assert_array_equal(predict_checkpoint(load_state(record)), expected)
    receipt_path = tmp_path / "record.json"
    receipt_path.write_text(json.dumps(record))
    result_path = tmp_path / "result.npz"
    script = "from opal.frontier_bridge.artifacts import load_state; from opal.frontier_bridge.contracts import read_json; from opal.frontier_bridge.workflow import predict_checkpoint; import numpy as np,sys; np.savez(sys.argv[2], prediction=predict_checkpoint(load_state(read_json(sys.argv[1]))))"
    subprocess.run([sys.executable, "-c", script, str(receipt_path), str(result_path)], check=True)
    with np.load(result_path, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["prediction"], expected, atol=1e-10, rtol=0)
    (tmp_path / "checkpoint/state.json").write_text("{}")
    with pytest.raises(ContractError, match="tampered"):
        load_state(record)


def test_live_resource_guard_leaves_failure_receipt_space(tmp_path):
    cfg = load_config(CONFIG)
    inherited = dict(artifact_bytes=cfg["resources"]["artifact_cap_bytes"] - 100)
    budget = Budget(ROOT, cfg, tmp_path, inherited=inherited)
    with pytest.raises(ContractError, match="artifact cap"):
        budget.check()
    cfg = copy.deepcopy(cfg)
    cfg["resources"]["new_retained_bytes"] = 100
    budget = Budget(ROOT, cfg, tmp_path, inherited=dict(artifact_bytes=0))
    with pytest.raises(ContractError, match="retained-byte"):
        budget.check()


def test_strict_cli_replay_binds_hashes_and_refuses_overwrite(tmp_path):
    state = dict(query=dict(singles=np.ones((4, 2, 20), dtype=np.float64), pair_ids=np.repeat('["A","B"]', 4)),
                 model=dict(kind="reference", reference=dict(family="additive", alpha=1)))
    checkpoint = save_state(tmp_path / "checkpoint", state)
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(predictions, prediction=predict_checkpoint(state))
    code = ROOT / "src/frontier_bridge/workflow.py"
    manifest_path = tmp_path / "run_manifest.json"
    write_json(manifest_path, dict(code_hashes={str(code): sha256(code)},
                                   predictions=dict(path=str(predictions), sha256=sha256(predictions)),
                                   checkpoints=[dict(**checkpoint, prediction_key="prediction")]))
    write_json(tmp_path / "prediction_freeze.json", dict(run_manifest_sha256=sha256(manifest_path)))
    output = tmp_path / "replay"
    command = [sys.executable, str(ROOT / "tools/run_frontier_bridge.py"), "--replay", str(manifest_path), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    receipt = read_json(output / "replay.json")
    assert receipt["status"] == "passed" and receipt["max_absolute_error"] == 0
    assert receipt["exact_frozen_code"] and not receipt["evaluator_expression_reopened"]
    digest = sha256(output / "replay.json")
    repeated = subprocess.run(command, capture_output=True)
    assert repeated.returncode != 0
    assert sha256(output / "replay.json") == digest


def test_public_qualification_stops_before_pair_expression_access():
    receipt = ROOT / "results/opal_frontier_bridge_v1/qualification-20261012-a1/qualification.json"
    if not receipt.exists():
        pytest.skip("Local exposed-source qualification receipt unavailable")
    q = read_json(receipt)
    assert not q["qualified"]
    assert q["common_gene_count"] == 207
    assert q["common_pair_count"] == 51
    assert not q["query_expression_accessed"]


def test_preparation_canonicalizes_aliases_and_reorders_measured_singles(tmp_path):
    from opal.frontier_bridge.preparation import prepare_banks, load_npz
    cfg = load_config(CONFIG)
    banks, genes = synthetic_banks("transfer")
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    before = prepare_banks(banks, genes, cfg, baseline_dir, {"qualified": False}, synthetic_case="transfer")
    altered = copy.deepcopy(banks)
    cfg["aliases"] = {"alias-T00": "T00"}
    for bank in altered.values():
        bank["pair_ids"] = np.asarray([json.dumps(["alias-T00" if c == "T00" else c for c in json.loads(p)[::-1]]) for p in bank["pair_ids"]])
        bank["singles"] = bank["singles"][:, ::-1].copy()
    altered_dir = tmp_path / "altered"
    altered_dir.mkdir()
    after = prepare_banks(altered, genes, cfg, altered_dir, {"qualified": False}, synthetic_case="transfer")
    assert after["folds"] == before["folds"]
    for context in cfg["contexts"]:
        for left, right in zip(before["contexts"][context]["folds"], after["contexts"][context]["folds"]):
            for role in ("source", "references", "bridges", "evaluator"):
                original, canonical = load_npz(left[role]), load_npz(right[role])
                for key in original:
                    np.testing.assert_array_equal(original[key], canonical[key])


def test_preparation_rejects_duplicate_alias_and_nonfinite_outcomes_before_writes(tmp_path):
    from opal.frontier_bridge.preparation import prepare_banks
    cfg = load_config(CONFIG)
    banks, genes = synthetic_banks("transfer")
    banks["TNFA+"]["responses"][0, 0] = np.nan
    with pytest.raises(ContractError, match="Finite matched"):
        prepare_banks(banks, genes, cfg, tmp_path, {}, synthetic_case="transfer")
    assert not list(tmp_path.iterdir())
    banks, genes = synthetic_banks("transfer")
    banks["TNFA+"]["pair_ids"][1] = banks["TNFA+"]["pair_ids"][0]
    with pytest.raises(ContractError, match="Duplicate canonical"):
        prepare_banks(banks, genes, cfg, tmp_path, {}, synthetic_case="transfer")
    assert not list(tmp_path.iterdir())


def test_resource_amendment_resolves_from_config_tree_not_installed_module(tmp_path, monkeypatch):
    from opal.frontier_bridge import contracts
    monkeypatch.setattr(contracts, "__file__", str(tmp_path / "site-packages/opal/frontier_bridge/contracts.py"))
    cfg = contracts.load_config(CONFIG)
    assert cfg["resources"]["amendment"]["path"] == "docs/frontier/resource_amendment_20261012.json"
