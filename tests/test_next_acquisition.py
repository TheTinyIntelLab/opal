import numpy as np
from opal.artifacts import load_npz
from opal.data.synthetic import generate
from tools.study_acquisition import study


def test_unacquired_outcomes_cannot_change_tuning_or_acquisition(tmp_path):
    data = generate(targets=8, genes=12)
    seed = 37
    order = np.random.default_rng(seed).permutation(len(data["pairs"]))
    val = order[:int(np.ceil(len(order) * .25))]
    initial = order[len(val):len(val) + 5]
    changed = data["doubles"].copy()
    changed[~np.isin(np.arange(len(changed)), initial)] += 1000
    config = {"seeds": [seed], "initial_labels": 5, "budgets": [5, 6], "validation_fraction": .25,
              "policies": ["uniform", "target_balance", "diversity", "kernel_variance"], "scope": "synthetic engineering"}
    records = []
    for label, outcomes in [("original", data["doubles"]), ("changed", changed)]:
        dataset = tmp_path / label
        dataset.mkdir()
        np.savez_compressed(dataset / "profiles.npz", singles=data["singles"], pairs=data["pairs"], doubles=outcomes,
                            pair_indices=np.arange(len(outcomes)))
        records.append(study(dataset, tmp_path / (label + "_run"), config)["records"])
    for before, after in zip(*records):
        assert before["model_config"] == after["model_config"]
        assert before["acquired"] == after["acquired"]
        assert not set(before["acquired"]) & set(before["validation"])


def test_infeasible_acquisition_budgets_are_not_silently_capped(tmp_path):
    data = generate(targets=8, genes=12)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    np.savez_compressed(dataset / "profiles.npz", singles=data["singles"], pairs=data["pairs"], doubles=data["doubles"],
                        pair_indices=np.arange(len(data["pairs"])))
    config = {"seeds": [37], "initial_labels": 5, "budgets": [5, 100], "validation_fraction": .25,
              "policies": ["uniform"], "scope": "synthetic engineering"}
    result = study(dataset, tmp_path / "run", config)
    assert result["records"][-1]["status"] == "infeasible"
    assert result["records"][-1]["budget"] == 100
    assert len(result["curves"]) == 1
