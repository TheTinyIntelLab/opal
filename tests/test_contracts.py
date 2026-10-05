import numpy as np
import pytest
import torch
from scipy import sparse
from opal.profiles import validate_counts, library_normalize, effect, eligible_genes
from opal.splits import canonical_pair, random_pair_split, partner_splits
from opal.data.synthetic import generate, save_prepared
from opal.training import fit, predict
from opal.models import ResidualNetwork
from opal.inference import export_members, Predictor
from opal.evaluation import metrics, cosine, joint_reference_bootstrap
from opal.artifacts import read_json, write_json, load_npz
from opal.training.benchmark import development, final


def test_counts_provenance_and_validity():
    x = np.array([[1, 2], [2, 0]])
    validate_counts(x, "source UMI matrix")
    for bad, provenance in [(x, ""), (x - 4, "UMIs"), (x + 0.1, "UMIs"), (np.full((2, 2), np.nan), "UMIs")]:
        with pytest.raises(ValueError):
            validate_counts(bad, provenance)


def test_sparse_dense_and_source_total_normalization():
    x = np.array([[1, 3, 6], [2, 2, 16]])
    expected = library_normalize(x, 10)
    assert np.allclose(library_normalize(sparse.csr_matrix(x), 10).toarray(), expected)
    assert np.allclose(expected.sum(1), 10)
    assert not np.allclose(library_normalize(x[:, :2], 10), expected[:, :2])


def test_distinct_count_and_log_additive_nulls():
    control = np.array([2., 10.])
    a, b = np.array([3., 11.]), np.array([4., 12.])
    ab = a + b - control
    assert np.allclose(effect(ab, control, "count"), effect(a, control, "count") + effect(b, control, "count"))
    assert not np.allclose(effect(ab, control), effect(a, control) + effect(b, control))


def test_gene_selection_controls_singles_only():
    x = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 500]])
    keep = eligible_genes(x[:2], np.array(["control", "A"]))
    assert keep.tolist() == [True, True, False]


def test_pair_groups_and_partner_holdout():
    pairs = [(f"T{i}", f"T{j}") for i in range(8) for j in range(i + 1, 8)]
    pairs.append(tuple(reversed(pairs[0])))
    split = random_pair_split(pairs)
    assert (0 in split["test"]) == (len(pairs) - 1 in split["test"])
    assert split == random_pair_split(pairs)
    for fold in partner_splits(pairs, minimum_development=1, minimum_test=1):
        held = set(fold["held_targets"])
        assert all(not held.intersection(pairs[i]) for i in fold["development"])
        assert all(held.intersection(pairs[i]) for i in fold["test"])
    with pytest.raises(ValueError):
        canonical_pair("A", "A")


@pytest.mark.parametrize("family", ["coordinate", "bilinear", "independent", "network", "sum_network"])
def test_additive_initialization_symmetry_and_first_gradient(family):
    torch.manual_seed(1)
    model = ResidualNetwork(np.random.default_rng(3).normal(size=(8, 5)), 13, {"family": family, "rank": 3, "features": 7})
    p = torch.tensor([[0, 1], [2, 4], [5, 7]])
    assert torch.equal(model(p), torch.zeros((3, 13), dtype=torch.float64))
    loss = (model(p) - torch.randn((3, 13), dtype=torch.float64)).square().mean()
    loss.backward()
    assert model.coefficient.weight.grad.norm() > 0
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    opt.step()
    assert torch.equal(model(p), model(p.flip(1)))
    assert model.basis.shape == (13, 3)


def test_mse_identity_zero_denominators():
    rng = np.random.default_rng(2)
    a, y, p = rng.normal(size=(3, 5)), rng.normal(size=(3, 5)), rng.normal(size=(3, 5))
    assert np.allclose((p - y) ** 2, ((p - a) - (y - a)) ** 2)
    summary, _ = metrics(a, a, a)
    assert summary["skill_additive"] is None
    assert summary["cosine"] is None


@pytest.mark.parametrize("family", ["additive", "ridge", "reduced_rank", "global", "nearest", "bilinear"])
def test_safe_export_roundtrip_and_routes(tmp_path, family):
    a = generate(targets=8, genes=24, latent=4)
    model = fit({"family": family, "epochs": 5, "rank": 3}, a["singles"], a["pairs"][:18], a["doubles"][:18])
    data = {**a, "pairs": a["pairs"][:18]}
    export_members([model], data, tmp_path / "checkpoint", {"context": "synthetic", "scale": "log2"})
    loaded = Predictor(tmp_path / "checkpoint")
    row = loaded.predict(*a["targets"][a["pairs"][19]])
    assert np.allclose(row["effect"], predict(model, a["pairs"][[19]])[0], atol=1e-10)
    assert np.allclose(row["effect"], row["additive"] + row["residual"])
    assert np.allclose(loaded.predict("T000")["effect"], a["singles"][0])
    assert np.allclose(loaded.predict()["effect"], 0)
    for kwargs in [{"a": "missing", "b": "T001"}, {"a": "T000", "b": "T000"}, {"context": "wrong"}, {"genes": a["genes"][::-1]}]:
        with pytest.raises(ValueError):
            loaded.predict(**kwargs)


def test_shared_reference_resampling_reuses_single():
    rng = np.random.default_rng(7)
    control = rng.normal(size=(20, 4))
    singles = [rng.normal(size=(20, 4)) for _ in range(3)]
    doubles = [np.zeros((20, 4)), np.zeros((20, 4))]
    pairs = np.array([[0, 1], [0, 2]])
    residual, s, c = joint_reference_bootstrap(control, singles, doubles, pairs)
    assert np.allclose(residual[0] - residual[1], s[2] - s[1])


def test_evaluator_mutation_does_not_change_fit(tmp_path):
    a = generate(targets=10, genes=20, latent=4)
    save_prepared(tmp_path / "dataset", a, {"synthetic": True, "context": "synthetic"})
    data = load_npz(tmp_path / "dataset/profiles.npz")
    cfg = {"family": "bilinear", "rank": 3, "epochs": 5, "fixed_basis": True}
    before = fit(cfg, data["singles"], data["pairs"], data["doubles"], 5)
    np.savez_compressed(tmp_path / "dataset/evaluator/profiles.npz", doubles=np.full((len(data["query_pairs"]), 20), 1e9))
    after = fit(cfg, data["singles"], data["pairs"], data["doubles"], 5)
    assert np.array_equal(predict(before, data["pairs"]), predict(after, data["pairs"]))
    for k, v in before["model"].state_dict().items():
        assert torch.equal(v, after["model"].state_dict()[k])


def test_end_to_end_and_stale_frozen_manifest(tmp_path):
    a = generate(targets=8, genes=20, latent=4)
    dataset, run = tmp_path / "dataset", tmp_path / "run"
    save_prepared(dataset, a, {"synthetic": True, "context": "synthetic", "scale": "log2"})
    protocol = {"epochs": 2, "ranks": [2], "ridge_alpha": [1], "ablations": False,
                "screening_seeds": [1], "finalist_seeds": [1], "ensemble_seeds": [1]}
    development(dataset, run, protocol)
    result = final(dataset, run)
    assert result["models"][result["selected_id"]]["pairs"] == len(load_npz(dataset / "profiles.npz")["query_pairs"])
    Predictor(run / "checkpoint").predict("T000", "T001")
    with pytest.raises(FileExistsError):
        final(dataset, run)
    write_json(run / "selection.json", {**read_json(run / "selection.json"), "data_fingerprint": "stale"})
    # First-access check takes precedence; a separate stale-run scenario is checked below.
    other = tmp_path / "other"
    other.mkdir()
    write_json(other / "selection.json", read_json(run / "selection.json"))
    with pytest.raises(ValueError, match="changed"):
        final(dataset, other)
