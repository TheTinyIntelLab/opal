"""Technical strata retain independent bridge budgets and equal source weight."""
import json

import numpy as np
import pytest

from opal.frontier_bridge import references
from opal.frontier_bridge import model
from opal.frontier_bridge.model import (Unsupported, fit_encoder, fit_nuisance,
                                       grouped_masks, source_weights)
from opal.frontier_bridge.preparation import synthetic_banks


def test_source_weights_share_repeated_pairs_and_balance_unequal_strata():
    ids = np.asarray(['["A","B"]', '["A","B"]', '["A","C"]', '["A","B"]'])
    strata = np.asarray(["one", "one", "one", "two"])
    weights = source_weights(ids, strata)
    np.testing.assert_allclose(weights, [.125, .125, .25, .5])
    x = np.zeros((4, 2, 3))
    y = np.asarray([[0.] * 3, [0.] * 3, [4.] * 3, [10.] * 3])
    fitted = fit_nuisance(x, y, "mean", weights=weights)
    np.testing.assert_allclose(fitted["intercept"], 6.)


def test_encoder_does_not_overweight_repeated_source_stratum():
    rng = np.random.default_rng(9)
    first, second = rng.normal(size=(12, 2, 8)), rng.normal(size=(12, 2, 8)) + 3
    x = np.concatenate([first, second])
    st = np.asarray(["one"] * 12 + ["two"] * 12)
    repeated = np.concatenate([first, first, first, second])
    rst = np.asarray(["one"] * 36 + ["two"] * 12)
    before, after = fit_encoder(x, st), fit_encoder(repeated, rst)
    for key in ("mean", "scale", "basis", "singular_values"):
        np.testing.assert_allclose(before[key], after[key], atol=1e-12)


def test_inner_support_is_required_in_each_retained_stratum():
    banks, _ = synthetic_banks("transfer", genes=8)
    ids = banks["TNFA+"]["pair_ids"][:36]
    partial = ids[:10]
    pooled = np.concatenate([ids, partial])
    strata = np.asarray(["supported"] * len(ids) + ["under_supported"] * len(partial))
    with pytest.raises(Unsupported, match="source-inner"):
        list(grouped_masks(pooled, strata=strata))


def test_reference_selection_simulates_four_bridges_per_stratum(monkeypatch):
    banks, _ = synthetic_banks("transfer", genes=8)
    ids = banks["TNFA+"]["pair_ids"][:36]
    ids = np.concatenate([ids, ids])
    strata = np.asarray(["one"] * 36 + ["two"] * 36)
    x = np.zeros((72, 2, 8))
    y = np.asarray([[2.] * 8] * 36 + [[10.] * 8] * 36)
    calls = []
    original_fit = references.fit_reference

    def recording_fit(family, alpha, sx, sy, sid, bx, by, bid, base, source_strata=None):
        calls.append((bid.copy(), by.copy(), source_strata.copy()))
        assert len(bid) == len(set(bid)) == 4
        assert len(np.unique(by[:, 0])) == 1
        return original_fit(family, alpha, sx, sy, sid, bx, by, bid, base, source_strata)

    monkeypatch.setattr(references, "FAMILIES", ("bridge_offset",))
    monkeypatch.setattr(references, "source_base", lambda *args: ({}, []))
    monkeypatch.setattr(references, "fit_reference", recording_fit)
    receipts = []
    _, selector, _ = references.select_references(x, y, ids, receipts.append, strata=strata)
    assert selector["mse"] == pytest.approx(0.)
    assert len(calls) == 6
    assert all(set(c[2]) == {"one", "two"} for c in calls)
    for first, second in zip(calls[::2], calls[1::2]):
        assert set(first[0]) == set(second[0])
    assert {r["stratum"] for r in receipts} == {"one", "two"}


def test_reference_selection_g1_missing_bridge_in_one_stratum(monkeypatch):
    banks, _ = synthetic_banks("transfer", genes=8)
    unique = banks["TNFA+"]["pair_ids"][:36]
    # Keep enough training support but leave fewer than eight common held
    # identities in every inner block. G1 must fail rather than redraw.
    ids = np.concatenate([unique, unique[:20]])
    strata = np.asarray(["one"] * 36 + ["two"] * 20)
    x, y = np.zeros((56, 2, 8)), np.zeros((56, 8))
    monkeypatch.setattr(references, "FAMILIES", ("bridge_offset",))
    monkeypatch.setattr(references, "source_base", lambda *args: ({}, []))
    with pytest.raises(Unsupported, match="G1|source-inner"):
        references.select_references(x, y, ids, strata=strata)


def test_source_shuffle_preserves_residual_blocks_and_reference_anchors(monkeypatch):
    banks, _ = synthetic_banks("transfer", genes=8)
    unique = banks["TNFA+"]["pair_ids"][:36]
    ids = np.concatenate([unique, unique])
    strata = np.asarray(["one"] * 36 + ["two"] * 36)
    rng = np.random.default_rng(18)
    x, residual = rng.normal(size=(72, 2, 8)), rng.normal(size=(72, 8))
    residual[36:] = residual[:36] * 3 + 2
    nuisance = dict(kind="mean", gamma=1., intercept=np.ones(8), alpha=1.)
    y = x.sum(1) + nuisance["intercept"] + residual
    monkeypatch.setattr(model, "crossfit_nuisance", lambda *args: (residual.copy(), [{"anchor": "source_only"}]))
    monkeypatch.setattr(model, "select_nuisance", lambda *args: (nuisance.copy(), [{"kind": "mean", "alpha": 1.}]))
    original = model.fit_source(x, y, ids, 1., strata)
    shuffled = model.fit_source(x, y, ids, 1., strata, shuffle_seed=20261021)
    np.testing.assert_array_equal(original["nuisance"]["intercept"], shuffled["nuisance"]["intercept"])
    for key in original["encoder"]:
        np.testing.assert_array_equal(original["encoder"][key], shuffled["encoder"][key])
    permuted, receipt = model.shuffled_residuals(residual, ids, strata, 20261021)
    np.testing.assert_allclose(permuted[36:], permuted[:36] * 3 + 2)
    for st in ("one", "two"):
        mask = strata == st
        assert sorted(map(tuple, residual[mask])) == sorted(map(tuple, permuted[mask]))
    assert receipt["mapping"] == shuffled["shuffle_metadata"]["mapping"]
    assert shuffled["shuffle_metadata"]["nuisance_selection"][0]["kind"] == "mean"
    np.testing.assert_allclose(x.sum(1), (y - residual - nuisance["intercept"]), atol=1e-14)
    assert not np.allclose(original["mapping"]["coefficients"], shuffled["mapping"]["coefficients"])


def test_source_shuffle_restricted_mapping_cannot_acquire_validation_labels():
    banks, _ = synthetic_banks("transfer", genes=8)
    ids = banks["TNFA+"]["pair_ids"][:36]
    train, val = next(model.grouped_masks(ids))
    residual = np.arange(train.sum() * 8, dtype=float).reshape(train.sum(), 8)
    _, receipt = model.shuffled_residuals(residual, ids[train], np.repeat("source", train.sum()), 20261021)
    assert set(receipt["mapping"]) == set(ids[train])
    assert set(receipt["mapping"].values()) == set(ids[train])
    assert not set(receipt["mapping"].values()) & set(ids[val])
