from pathlib import Path
import numpy as np
import pytest

from opal.artifacts import load_npz, read_json, sha256, write_json
from opal.contextual import ContextPredictor, ContextualKernel, ReferenceBank, prepare_dataset, load_dataset
from opal.contextual.benchmark import run_benchmark, select
from opal.contextual.inference import export_checkpoint
from opal.contextual.reliability import empirical_radius, selective_risk
from opal.contextual.synthetic import generate


@pytest.fixture
def fixture():
    return generate(targets=6, genes=12, contexts=4)


def prepare(path, fixture, responses=None):
    bank, pairs, contexts, y = fixture
    return prepare_dataset(path, bank, pairs, contexts, y if responses is None else responses,
                           held_contexts=[str(bank.arrays["contexts"][-1])], calibration_fraction=.2, seed=11)


def test_reference_availability_is_not_inferred_from_zero_effect(fixture):
    bank, pairs, contexts, y = fixture
    arrays = {k: v.copy() for k, v in bank.arrays.items()}
    arrays["singles"][0, 0] = 0
    # A measured null single can still have an interaction.
    measured = ReferenceBank(arrays, bank.metadata)
    measured.validate_rows(np.array([[0, 1]]), np.array([0]))
    arrays["available"][0, 0] = False
    unavailable = ReferenceBank(arrays, bank.metadata)
    with pytest.raises(ValueError, match="Both measured singles"):
        unavailable.validate_rows(np.array([[0, 1]]), np.array([0]))


def test_references_reject_unregistered_modalities_assays_and_response_features(fixture):
    import copy
    bank = fixture[0]
    for field in ["modality", "assay"]:
        metadata = copy.deepcopy(bank.metadata)
        metadata["context_records"][str(bank.arrays["contexts"][0])][field] = "different"
        with pytest.raises(ValueError, match="pooling"):
            ReferenceBank(bank.arrays, metadata)
    metadata = copy.deepcopy(bank.metadata)
    metadata["context_features_source"] = "combination_outcomes"
    with pytest.raises(ValueError, match="combination outcomes"):
        ReferenceBank(bank.arrays, metadata)
    with pytest.raises(ValueError, match="exactly match"):
        ReferenceBank({**bank.arrays, "test_responses": np.zeros((2, 12))}, bank.metadata)


@pytest.mark.parametrize("kind", ["shared_offset", "global", "pooled_pair", "contextual_pair", "pooled_sum", "contextual_sum"])
def test_contextual_kernel_psd_symmetry_intercept_and_roundtrip(tmp_path, fixture, kind):
    bank, pairs, contexts, y = fixture
    train, query = np.flatnonzero(contexts != 3), np.flatnonzero(contexts == 3)
    model = ContextualKernel(bank, {"kind": kind, "alpha": .1}).fit(pairs[train], contexts[train], y[train])
    gram = model.kernel(pairs, contexts, pairs, contexts)
    assert np.linalg.eigvalsh(gram).min() > -1e-8
    predictions = model.predict(pairs[query], contexts[query])
    np.testing.assert_allclose(predictions, model.predict(pairs[query, ::-1], contexts[query]), atol=1e-11)
    weights = model.weights(pairs[query], contexts[query])
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-10)
    residual = y[train] - bank.additive(pairs[train], contexts[train])
    np.testing.assert_allclose(predictions, bank.additive(pairs[query], contexts[query]) + weights @ residual, atol=1e-10)
    export_checkpoint(model, tmp_path / "checkpoint")
    predictor = ContextPredictor(tmp_path / "checkpoint")
    pair = bank.arrays["targets"][pairs[query[0]]]
    result = predictor.predict(*pair, context="synthetic_state_3")
    np.testing.assert_allclose(result["effect"], predictions[0], atol=1e-10)
    np.testing.assert_allclose(result["effect"], result["additive"] + result["residual"])
    assert result["support"]["same_context_training_conditions"] == 0
    assert result["route"] == "new_context_with_measured_references"
    assert result["reliability_status"] == "synthetic_engineering_only"
    assert predictor.predict(context="synthetic_state_3")["route"] == "control_reference"
    with pytest.raises(ValueError, match="Repeated-target"):
        predictor.predict(pair[0], pair[0], context="synthetic_state_3")
    with pytest.raises(ValueError, match="Unsupported context"):
        predictor.predict(*pair, context="unregistered")
    with pytest.raises(ValueError, match="Gene order"):
        predictor.predict(*pair, context="synthetic_state_3", genes=bank.arrays["genes"][::-1])
    with (tmp_path / "checkpoint/fitted.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        ContextPredictor(tmp_path / "checkpoint")


def test_context_dataset_pair_group_calibration_and_identity_isolation(tmp_path, fixture):
    path = prepare(tmp_path / "dataset", fixture)
    bank, fitting, query, protocol = load_dataset(path)
    calibration = load_npz(path / "calibration/profiles.npz")
    assert not set(map(tuple, fitting["pairs"])) & set(map(tuple, calibration["pairs"]))
    assert 3 not in fitting["contexts"] and set(query["contexts"]) == {3}
    assert set(query) == {"pairs", "contexts", "identities"}
    with pytest.raises(FileExistsError):
        prepare(path, fixture)
    protocol["fitting_identities"][0] = protocol["evaluation_identities"][0]
    write_json(path / "protocol.json", protocol)
    with pytest.raises(ValueError, match="overlap"):
        load_dataset(path)


def test_held_context_and_calibration_labels_cannot_select_or_fit(tmp_path, fixture):
    bank, pairs, contexts, y = fixture
    original = prepare(tmp_path / "original", fixture)
    _, _, _, protocol = load_dataset(original)
    changed_y = y.copy()
    changed_y[np.array(protocol["evaluation_identities"])] += 100
    changed_y[np.array(protocol["calibration_identities"])] -= 100
    changed = prepare(tmp_path / "changed", fixture, changed_y)
    configs = [{"kind": "global", "alpha": .1}, {"kind": "contextual_pair", "alpha": 1.}]
    before = run_benchmark(original, tmp_path / "run_original", configs)
    after = run_benchmark(changed, tmp_path / "run_changed", configs)
    a, b = (read_json(tmp_path / name / "selection.json") for name in ["run_original", "run_changed"])
    assert a == b
    for field in ["selected", "global", "contextual_pair"]:
        np.testing.assert_array_equal(load_npz(tmp_path / "run_original/predictions.npz")[field], load_npz(tmp_path / "run_changed/predictions.npz")[field])
    ca, cb = (load_npz(tmp_path / name / "checkpoint/fitted.npz") for name in ["run_original", "run_changed"])
    for key in ca:
        np.testing.assert_array_equal(ca[key], cb[key])
    assert before["mse"]["selected"] != after["mse"]["selected"]
    assert before["capability_claim"] == "none; synthetic engineering"


def test_prediction_and_access_record_exist_before_opening_evaluator(tmp_path, fixture, monkeypatch):
    import opal.contextual.benchmark as benchmark
    path = prepare(tmp_path / "dataset", fixture)
    real_load = benchmark.load_npz
    run = tmp_path / "run"
    access = []

    def checked_load(filename):
        filename = Path(filename)
        if filename == path / "evaluator/profiles.npz":
            record = read_json(run / "test_access.json")
            assert record["predictions_sha256"] == sha256(run / "predictions.npz")
            assert (run / "selection.json").exists()
            assert (run / "checkpoint/manifest.json").exists()
            access.append(filename)
        return real_load(filename)

    monkeypatch.setattr(benchmark, "load_npz", checked_load)
    benchmark.run_benchmark(path, run, [{"kind": "global", "alpha": 1.}])
    assert len(access) == 1
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(path, run)
    assert len(access) == 1


def test_grouped_radius_finite_rank_and_no_formal_guarantee():
    predictions, responses = np.zeros((7, 3)), np.ones((7, 3))
    pairs = np.array([(i, i + 10) for i in range(7)])
    result = empirical_radius(predictions, responses, pairs)
    assert result["radius"] is None and result["ordered_rank"] == 8
    predictions = np.zeros((9, 3))
    pairs = np.array([(i, i + 10) for i in range(9)])
    result = empirical_radius(predictions, np.ones_like(predictions), pairs)
    assert result["radius"] == 1 and result["formal_guarantee"] is False
    repeated = empirical_radius(np.tile(predictions, (2, 1)), np.ones((18, 3)), np.tile(pairs, (2, 1)))
    assert repeated["calibration_groups"] == 9


def test_selective_risk_includes_equal_support_ties():
    results = selective_risk(np.arange(5), np.ones(5))
    assert all(r["accepted_conditions"] == 5 for r in results)
    assert all(r["accepted_fraction"] == 1 for r in results)


def test_invalid_checkpoint_state_and_parameters(fixture):
    bank, pairs, contexts, y = fixture
    with pytest.raises(ValueError, match="nonnegative"):
        ContextualKernel(bank, {"context_weight": -1})
    model = ContextualKernel(bank).fit(pairs, contexts, y)
    state = model.state()
    state["context_scale"] = np.zeros_like(state["context_scale"])
    with pytest.raises(ValueError, match="scales"):
        ContextualKernel(bank).restore(state)
