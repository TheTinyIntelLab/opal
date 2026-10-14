"""Regression checks for cumulative and unavailable-artifact accounting."""
import json
from pathlib import Path

import pytest

from opal.frontier_bridge.contracts import ContractError
from opal.frontier_bridge.resources import Budget, exclusive_paths, inventory, prior_cpu_receipts


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def setup_budget(tmp_path):
    cfg = dict(resources=dict(inherited_ledger="ledger.json", new_retained_bytes=10000000,
                              artifact_cap_bytes=20000000, reserve_bytes=0, cpu_core_hours=24))
    dump(tmp_path / "ledger.json", dict(physical_roots={}, artifact_bytes=0,
                                      raw_asset_classification="pre-existing cached raw inputs"))
    output = tmp_path / "results/opal_frontier_bridge_v1/run-new"
    output.mkdir(parents=True)
    return Budget(tmp_path, cfg, output), cfg


def test_missing_nonempty_historical_roots_remain_charged(tmp_path):
    present = tmp_path / "old-present"
    present.mkdir()
    (present / "data").write_bytes(b"12345")
    absent = tmp_path / "old-absent"
    dump(tmp_path / "ledger.json", dict(physical_roots={str(present): 4, str(absent): 17},
                                      artifact_bytes=21, raw_asset_classification="cached"))
    result = inventory(tmp_path, dict(resources=dict(inherited_ledger="ledger.json")))
    assert result["artifact_bytes"] == 22
    assert result["available_artifact_bytes"] == 5
    assert result["unavailable_physical_roots"] == {str(absent.resolve()): 17}


def test_nested_and_alias_roots_are_counted_once(tmp_path):
    root = tmp_path / "task"
    nested = root / "nested"
    nested.mkdir(parents=True)
    assert exclusive_paths([root, nested, root / "."]) == [root.resolve()]


def test_registered_temporary_bytes_enforce_limit(tmp_path):
    budget, cfg = setup_budget(tmp_path)
    temporary = tmp_path / "outside-results-temporary"
    temporary.mkdir()
    budget.register_temporary_root(temporary)
    before = budget.new_bytes()
    (temporary / "intermediate").write_bytes(b"x" * 200)
    assert budget.new_bytes() == before + 200
    budget.register_temporary_root(temporary / ".")
    assert budget.new_bytes() == before + 200
    cfg["resources"]["new_retained_bytes"] = budget.new_bytes() + budget.reserved_receipt_bytes
    with pytest.raises(ContractError, match="retained-byte"):
        budget.check()


def test_prior_cpu_sums_observations_not_cumulative_totals(tmp_path):
    frontier = tmp_path / "results"
    dump(frontier / "old-a/resource_ledger.json", dict(cpu_seconds=2, cumulative_observed_cpu_seconds=200))
    dump(frontier / "old-b/resource_ledger.json", dict(cpu_seconds=3, cumulative_observed_cpu_seconds=203))
    dump(frontier / "current/resource_ledger.json", dict(cpu_seconds=100))
    dump(frontier / "copied/resource_ledger.json", dict(cpu_seconds=2, command_id=str((frontier / "old-a").resolve())))
    dump(frontier / "test-temporaries/fixture/resource_ledger.json", dict(cpu_seconds=9000))
    receipts, unknown = prior_cpu_receipts(frontier, frontier / "current")
    assert sum(receipts.values()) == 5
    assert unknown == []


def test_guarded_json_refuses_write_before_limit(tmp_path):
    budget, cfg = setup_budget(tmp_path)
    destination = budget.output / "too-large.json"
    cfg["resources"]["new_retained_bytes"] = budget.new_bytes() + budget.reserved_receipt_bytes + 20
    with pytest.raises(ContractError, match="retained-byte"):
        budget.write_json(destination, dict(value="x" * 100))
    assert not destination.exists()


def test_periodic_inventory_observes_historical_growth(tmp_path):
    budget, cfg = setup_budget(tmp_path)
    historical = tmp_path / "historical"
    historical.mkdir()
    dump(tmp_path / "ledger.json", dict(physical_roots={str(historical): 0}, artifact_bytes=0,
                                      raw_asset_classification="cached"))
    (historical / "new-data").write_bytes(b"x" * 900)
    budget.last_inventory_wall -= 6
    budget.check()
    assert budget.current_inventory["available_artifact_bytes"] == 900
    assert budget.evidence()["cpu_accounting_complete"] is False


def test_live_tree_inventory_matches_original_walk_and_file_sizes(tmp_path):
    from opal.frontier_bridge.resources import tree_bytes
    for directory in (tmp_path / "a", tmp_path / "a/b", tmp_path / "c"):
        directory.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        (tmp_path / "a/b" / str(i)).write_bytes(bytes(i + 1))
    (tmp_path / "c/.hidden").write_bytes(b"hidden")
    expected = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert tree_bytes(tmp_path) == expected
    assert tree_bytes(tmp_path / "c/.hidden") == 6
