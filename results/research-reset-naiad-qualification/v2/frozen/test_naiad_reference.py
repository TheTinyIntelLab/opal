"""Native-comparator prerequisites use synthetic values only."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from opal.naiad_reference import NativeConfig, NativeNAIAD, fit_native, load_official

SOURCE = Path(os.environ.get("OPAL_NAIAD_SOURCE", "C:/Users/USER/Desktop/opal-research-next-20261020/NAIAD"))


def fixture():
    genes = np.array([f"g{i:02}" for i in range(8)])
    pairs = np.array([(i, j) for i in range(8) for j in range(i + 1, 8)])
    singles = np.linspace(-.8, .2, 8)
    targets = singles[pairs].sum(1) + .05 * np.sin(pairs.sum(1))
    split = np.array([0] * 17 + [1] * 6 + [2] * 5)
    return genes, pairs, singles, targets, split


def test_native_contract_rejects_missingness_duplicates_and_no_support():
    genes, pairs, singles, y, split = fixture()
    for field in ["single", "train", "support", "duplicate"]:
        args = [x.copy() for x in (genes, pairs, singles, y, split)]
        if field == "single":
            args[2][0] = np.nan
        elif field == "train":
            args[3][0] = np.nan
        elif field == "support":
            args[4][np.any(args[1] == 0, axis=1)] = 2
        else:
            args[1][1] = args[1][0]
        with pytest.raises(ValueError):
            fit_native(SOURCE, *args, NativeConfig(epochs=2))


@pytest.mark.skipif(not (SOURCE / "naiad/naiad.py").is_file(), reason="Pinned optional native source unavailable")
def test_native_no_resplit_test_sentinels_replay_and_numerical_serialization(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    native = load_official(SOURCE)
    def forbidden(*args, **kwargs):
        raise AssertionError("Native hidden random resplit called")
    monkeypatch.setattr(native.NAIAD, "prepare_data", forbidden)
    genes, pairs, singles, y, split = fixture()
    config = NativeConfig(epochs=2)
    calls = []
    a = fit_native(SOURCE, genes, pairs, singles, y, split, config, check=lambda: calls.append(1))
    y[split == 2] = np.nan
    b = fit_native(SOURCE, genes, pairs, singles, y, split, config)
    np.testing.assert_array_equal(a.predict(pairs), b.predict(pairs))
    assert a.training_metrics == b.training_metrics
    assert calls and set(a.training_metrics) == {"train_loss", "val_loss"}
    assert a.fit_rows == {"train": np.flatnonzero(split == 0).tolist(), "val": np.flatnonzero(split == 1).tolist()}
    assert a.best_epoch == np.argmin(a.training_metrics["val_loss"]) + 1
    a.save(tmp_path / "native")
    loaded = NativeNAIAD.load(SOURCE, tmp_path / "native")
    np.testing.assert_array_equal(a.predict(pairs), loaded.predict(pairs))
    with np.load(tmp_path / "native/state.npz", allow_pickle=False) as arrays:
        assert all(not arrays[k].dtype.hasobject for k in arrays.files)
    state = tmp_path / "native/state.npz"
    state.write_bytes(state.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="signature"):
        NativeNAIAD.load(SOURCE, tmp_path / "native")


@pytest.mark.skipif(not (SOURCE / "naiad/naiad.py").is_file(), reason="Pinned optional native source unavailable")
def test_native_batch_guard_interrupts():
    torch.set_num_threads(1)
    calls = []
    def check():
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("synthetic resource stop")
    with pytest.raises(RuntimeError, match="resource stop"):
        fit_native(SOURCE, *fixture(), NativeConfig(epochs=2), check=check)
    assert len(calls) == 3
