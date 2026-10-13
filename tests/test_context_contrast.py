"""Small-fixture software checks, never biological scoring or experiment training."""
import copy
import itertools
import json
from pathlib import Path
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from opal.artifacts import fingerprint, read_json, sha256, write_json
from opal.context_contrast.artifacts import load_npz, load_state, save_state
from opal.context_contrast.contracts import (PROTOCOL, ContractError, Unsupported,
    canonical_pair, folds, shared_panel, qualify_graph, load_config)
from opal.context_contrast.model import (features, encoder, head_fit, head_predict,
    fit, predict, base_menu, rank_check, menu, select)
from opal.context_contrast.preparation import prepare_roles, columns, aggregate
from opal.context_contrast.resources import authorize, inventory
from opal.context_contrast.evaluation import (retrieval, retrieval_null,
    evaluate, metric_qualification, five_gates)
from opal.context_contrast.readiness import blank_package, validate_readiness
from opal.context_contrast.workflow import run_predictions, evaluate_run, verify_freeze


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


def fixture(n=64, genes=16):
    rng = np.random.default_rng(412)
    components = [f"i{i}" for i in range(16)]
    pairs = list(itertools.combinations(components, 2))[:n]
    ids = np.asarray([canonical_pair(p) for p in pairs])
    basis, _ = np.linalg.qr(rng.normal(size=(genes, 4)))
    single = rng.normal(size=(16, 2, 4))@basis.T
    x = np.asarray([[single[components.index(a), c], single[components.index(b), c]] for a, b in pairs for c in range(2)]).reshape(n, 2, 2, genes)
    return x, ids, rng


def test_protocol_frozen_and_old_refusal_preserved(tmp_path):
    cfg = Path(__file__).parents[1]/"configs/experiments/opal_context_contrast_v1.json"
    assert load_config(cfg) == PROTOCOL
    altered = dict(PROTOCOL, minimum_common_genes=207)
    write_json(tmp_path/"bad.json", altered)
    with pytest.raises(ContractError):
        load_config(tmp_path/"bad.json")
    old_path = Path(__file__).parents[1]/"configs/experiments/opal_frontier_bridge_v1.json"
    # The inherited dirty bridge checkout is deliberately not committed by
    # this successor. Check it when present without making clean checkout
    # validation depend on somebody else's untracked work.
    if old_path.is_file():
        old = read_json(old_path)
        assert old["minimum_common_genes"] == 256 and old["fold_seed"] == 20261011


def test_project_cli_routes_to_successor_without_numerical_work(capsys):
    from opal.cli import main
    assert main(["context-contrast", "protocol"]) == 0
    assert json.loads(capsys.readouterr().out) == PROTOCOL


def test_panel_reference_only_lexical_ties_and_207_refusal():
    genes = [f"g{i:03}" for i in range(300)][::-1]
    effects = {c: np.tile(np.arange(3)[:, None], (1, 300)) for c in PROTOCOL["contexts"]}
    selected, record = shared_panel(genes, effects)
    assert record["genes"] == sorted(genes)
    assert record["genes_hash"] == fingerprint(sorted(genes))
    with pytest.raises(Unsupported, match="256"):
        shared_panel(genes[:207], {c: v[:, :207] for c, v in effects.items()})


def test_global_alias_reversal_and_folds_outcome_blind():
    _, ids, _ = fixture()
    assert folds(ids) == folds(ids[::-1])
    assert canonical_pair('["B","old-A"]', {"old-A": "A"}) == canonical_pair(["A", "B"])
    assert canonical_pair(["A@1nM", "B"]) != canonical_pair(["A@2nM", "B"])
    with pytest.raises(ContractError):
        canonical_pair(["A", "B"], {"A": "B"})
    with pytest.raises(ContractError, match="cycle"):
        canonical_pair(["A", "C"], {"A": "B", "B": "A"})


def test_literal_graph_missing_rights_and_assignments_cannot_pass(tmp_path):
    _, ids, _ = fixture(36)
    rows = [dict(pair_id=p, context=c, stratum="s", control_cells=10,
        single_a_cells=10, single_b_cells=10, pair_cells=10) for p in ids for c in PROTOCOL["contexts"]]
    result = qualify_graph(rows, {}, tmp_path)
    assert not result["qualified"] and "rights_unresolved" in result["reasons"]
    rows.append(dict(rows[0], pair_id=json.dumps(json.loads(rows[0]["pair_id"])[::-1])))
    with pytest.raises(ContractError, match="Duplicate"):
        qualify_graph(rows, {}, tmp_path)


def test_exact_features_symmetry_and_four_coordinates():
    x, _, _ = fixture()
    enc = encoder(x)
    z = x@enc["P"]
    d, bar = z[:, 1]-z[:, 0], (z[:, 1]+z[:, 0])/2
    raw = bar[:, 0, :, None]*d[:, 1, None, :]+bar[:, 1, :, None]*d[:, 0, None, :]
    sym = (raw+raw.transpose(0, 2, 1))/2
    expected = np.column_stack([sym[:, i, j]*(1 if i == j else np.sqrt(2)) for i in range(4) for j in range(i, 4)])
    phi = features(x, enc)
    assert phi.shape == (64, 18) and features(x, enc, "phi8").shape == (64, 8)
    np.testing.assert_allclose(phi[:, 8:], expected)
    np.testing.assert_allclose(phi, features(x[:, :, ::-1], enc), atol=1e-14)


def test_rank_two_gene_rms_and_exact_ridge_objective():
    x, _, rng = fixture()
    phi = features(x, encoder(x))
    b, _ = np.linalg.qr(rng.normal(size=(16, 2)))
    target = phi@rng.normal(size=(18, 2))@b.T
    fitted = head_fit(phi, target, .1)
    np.testing.assert_allclose(fitted["B"].T@fitted["B"]/16, np.eye(2), atol=1e-14)
    scaled = (phi-fitted["center"])/fitted["scale"]
    expected = np.linalg.solve(scaled.T@scaled/64+.1*np.eye(18), scaled.T@(target@fitted["B"]/16)/64)
    np.testing.assert_allclose(fitted["W"].T, expected, atol=1e-14)
    repeated = head_fit(phi, np.tile(target, (1, 3)), .1)
    np.testing.assert_allclose(head_predict(repeated, phi), np.tile(head_predict(fitted, phi), (1, 3)), atol=1e-12)


def test_realizable_positive_symmetric_response_reconstruction_and_replay(tmp_path):
    x, ids, rng = fixture()
    enc = encoder(x)
    phi = features(x, enc)
    b, _ = np.linalg.qr(rng.normal(size=(16, 2)))
    b *= 4
    w = rng.normal(size=(2, 18))
    head = dict(center=np.zeros(18), scale=np.ones(18), W=w, B=b)
    from opal.context_contrast.model import kernel_fit
    pooled = kernel_fit(x.mean(1), np.full((len(x), 16), .3), 1)
    state = dict(spec=dict(family="phi18", alpha=1), encoder=enc, head=head, pooled=pooled, h=np.full(16, .2))
    t = .2+phi@w.T@b.T
    expected = x.sum(2)+np.stack((.3-t/2, .3+t/2), axis=1)
    np.testing.assert_allclose(predict(state, x, ids), expected, atol=1e-13)
    save_state(tmp_path/"checkpoint", state)
    np.testing.assert_array_equal(predict(load_state(tmp_path/"checkpoint"), x, ids), predict(state, x, ids))
    asset = next((tmp_path/"arrays").iterdir())
    asset.write_bytes(b"tampered")
    with pytest.raises(ContractError, match="mismatch"):
        load_state(tmp_path/"checkpoint")


def test_isolated_source_output_and_feature_rank_refusals():
    x, _, rng = fixture()
    with pytest.raises(Unsupported, match="single_PCA"):
        encoder(np.ones_like(x))
    phi = features(x, encoder(x))
    with pytest.raises(Unsupported, match="contrast_output"):
        head_fit(phi, np.ones((64, 16)), 1)
    with pytest.raises(Unsupported, match="feature_design"):
        head_fit(np.ones((64, 18)), rng.normal(size=(64, 16)), 1)
    with pytest.raises(Unsupported):
        rank_check(np.diag([1., 1e-5]), 2, "ill_conditioned")


@pytest.mark.parametrize("family", ["no_change", "additive", "offset", "affine", "incidence", "pooled_kernel", "per_context_profile", "per_context_product", "joint_context"])
def test_references_calibration_and_swap_symmetry(family):
    x, ids, _ = fixture(24)
    y = .7*x.sum(2)+.2
    state = fit(dict(family=family, alpha=1), x, y, ids)
    prediction = predict(state, x, ids)
    assert prediction.shape == y.shape and np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction, predict(state, x[:, :, ::-1], ids), atol=1e-12)


def test_base_refits_finite_contract_on_permitted_labels(tmp_path):
    x, ids, _ = fixture(18)
    y = .7*x.sum(2)+.2
    state = fit(dict(family="base", alpha=1), x, y, ids)
    assert len(base_menu()) == 24
    assert len(state["base"]["selection"][0]) == 24
    assert state["fitting_ids"].tolist() == ids.tolist()
    assert np.isfinite(predict(state, x, ids)).all()
    save_state(tmp_path/"base", state)
    np.testing.assert_array_equal(predict(load_state(tmp_path/"base"), x, ids), predict(state, x, ids))


def test_head_fit_crossfits_nuisance_and_retains_all_attempts():
    x, ids, rng = fixture()
    phi = features(x, encoder(x))
    b, _ = np.linalg.qr(rng.normal(size=(16, 2)))
    t = phi@rng.normal(size=(18, 2))@b.T
    y = x.sum(2)+np.stack((-.5*t, .5*t), axis=1)
    attempts = []
    state = fit(dict(family="phi18", alpha=.1), x, y, ids, lambda row, model: attempts.append(row))
    assert sum(r["stage"] == "pooled_inner" for r in attempts) == 36
    assert sum(r["stage"] == "contrast_offset_oof" for r in attempts) == 3
    assert state["head"]["head_coefficients"] == 36
    assert all(set(r["fitting_ids"]).issubset(ids) for r in attempts)


def test_full_finite_selection_on_small_fixture_retains_unsupported_models():
    x, ids, rng = fixture(36, genes=8)
    y = .7*x.sum(2)+rng.normal(size=(36, 2, 8))*.1
    attempts = []
    states, selection = select(x, y, ids, lambda row, state: attempts.append(row))
    assert len(selection["comparisons"]) == 55
    assert len(states) == 21
    assert states[selection["selector"]["spec"]["family"]] is not None
    assert all(set(row["fitting_ids"]).issubset(ids) for row in attempts)
    assert any(not row["success"] for row in attempts)
    assert all("validation_ids" in row["spec"] for row in attempts if row["stage"] == "arm_inner")


def test_full_library_normalization_precedes_gene_restriction(tmp_path):
    import h5py
    from scipy import sparse
    # Forty cells, two references; the excluded gene carries most library counts.
    values = np.tile([1., 9., 90.], (40, 1))
    keys = [("TNFA+", "s", ())]*20+[("TNFA+", "s", ("A",))]*20
    meta = dict(keys=keys, genes=["a", "b", "excluded"], counts={keys[0]: 20, keys[-1]: 20})
    with h5py.File(tmp_path/"tiny.h5", "w") as file:
        matrix = file.create_group("counts")
        data = sparse.csc_matrix(values)
        matrix.attrs["encoding-type"] = "csc_matrix"
        matrix.create_dataset("data", data=data.data)
        matrix.create_dataset("indices", data=data.indices)
        matrix.create_dataset("indptr", data=data.indptr)
        result, counts = aggregate(matrix, meta, np.asarray([0]), True, FixtureBudget(tmp_path))
    np.testing.assert_allclose(result["primary"][keys[0]], np.log2(1+100))
    assert counts["a"][str(keys[0])] == 10 and counts["b"][str(keys[0])] == 10


def test_metric_null_ties_undefined_and_conditional_accounting():
    x, ids, _ = fixture(24)
    truth = x.sum(2)
    assert retrieval(truth[:, 0], truth[:, 0])["score"] == 1
    assert retrieval(np.zeros_like(truth[:, 0]), truth[:, 0])["score"] is None
    null = retrieval_null(truth, truth)
    assert len(null["distribution"]) == 99 and len(null["permutations"]) == 99
    controls = metric_qualification(truth, truth, truth.mean(0))
    assert controls["qualified"] and not controls["biological_confirmation"]
    predictions = dict(selector=truth+.2, phi18=truth)
    fallback = {name: np.zeros(len(ids), bool) for name in predictions}
    result = evaluate(predictions, truth, x, ids, fallback, np.arange(len(ids))%3)
    assert result["metrics"]["phi18"]["contexts"][0]["mse"] == 0
    assert result["uncertainty"]["phi18"]["relative_gain"] == [1, 1]
    assert result["sensitivity"]["requested_draws"] == 2000
    assert not five_gates({}, {}, {"passed": True}, [{"passed": True}]*2)["development_nominated"]


def test_readiness_template_cannot_invent_confirmation(tmp_path):
    result = validate_readiness(blank_package(), tmp_path)
    assert result["status"] == "measurement_not_ready"
    assert not result["independent_biological_confirmation"] and not result["frontier_claim"]


def test_resource_inventory_and_missing_authorization(tmp_path):
    plan = inventory(51, 512, 24)
    assert plan["logical_fits"] == 53280 and plan["raw_new_bytes"] == 0
    assert plan["uncompressed_checkpoint_upper_estimate_bytes"] > 25_000_000
    write_json(tmp_path/"acceptance.json", {})
    with pytest.raises(ContractError, match="owner"):
        authorize(tmp_path/"acceptance.json", "predict", tmp_path/"out", "input", "inventory")


class FixtureBudget:
    """Tests only: hard 8 MB retained ceiling, no production bypass."""
    def __init__(self, root):
        self.root = root
    def check(self, extra=0):
        size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        assert size+extra < 8_000_000
    def write_json(self, path, value):
        self.check(len(json.dumps(value).encode()))
        write_json(path, value)


def test_separated_roles_mutation_fallback_and_evaluator_freeze(tmp_path, monkeypatch):
    from opal.context_contrast import workflow
    x, ids, _ = fixture(36, genes=256)
    y = .7*x.sum(2)+.2
    genes = [f"g{i}" for i in range(256)]
    prepare_roles(tmp_path/"prepared", x, y, ids, genes, dict(qualified=True))
    manifest = read_json(tmp_path/"prepared/prepared_manifest.json")
    assert "stratum_records" not in manifest
    calls = []
    def software_selector(x, y, ids, recorder):
        calls.append((x.copy(), y.copy(), ids.copy()))
        additive = fit(dict(family="additive", alpha=1), x, y, ids)
        return dict(additive=additive, phi18=None), dict(selector=dict(spec=dict(family="additive", alpha=1)), comparisons=[])
    monkeypatch.setattr(workflow, "select", software_selector)
    for run in ("run1", "run2"):
        (tmp_path/run).mkdir()
        if run == "run2":
            # Mutate only fold-0 evaluator; rehash it as a valid new evaluator
            # record. No permitted input is changed.
            record = manifest["records"][0]["evaluator"]
            path = tmp_path/"prepared"/record["path"]
            data = load_npz(tmp_path/"prepared", record)
            np.savez_compressed(path, responses=data["responses"]+100, pair_ids=data["pair_ids"])
            record["sha256"] = sha256(path)
            write_json(tmp_path/"prepared/prepared_manifest.json", manifest)
        workflow.run_predictions(tmp_path/"prepared", tmp_path/run, FixtureBudget(tmp_path))
    seal1, seal2 = verify_freeze(tmp_path/"run1"), verify_freeze(tmp_path/"run2")
    for r1, r2 in zip(seal1["records"], seal2["records"]):
        p1 = load_npz(tmp_path/"run1", r1["predictions"])
        p2 = load_npz(tmp_path/"run2", r2["predictions"])
        np.testing.assert_array_equal(p1["phi18"], p2["phi18"])
        assert p1["phi18__fallback"].all() and len(p1["phi18"]) == 12
    for a, b in zip(calls[:3], calls[3:]):
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
    (tmp_path/"evaluation").mkdir()
    decision = evaluate_run(tmp_path/"run2", tmp_path/"evaluation", FixtureBudget(tmp_path))
    result = read_json(tmp_path/"evaluation/metrics.json")["primary"]["phi18"]
    assert result["original_query_pairs"] == 36 and result["scored_query_pairs"] == 36
    assert result["fallback_pairs"] == 36
    assert not decision["development_nominated"] and not decision["independent_biological_confirmation"]
    assert len(decision["gates"]) == 5 and not decision["frontier_claim"]
