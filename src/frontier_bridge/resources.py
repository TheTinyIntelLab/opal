"""Conservative cumulative resource accounting for frontier engineering attempts."""
import json
import os
import shutil
import time
from pathlib import Path

from .contracts import ContractError, read_json


def exclusive_paths(paths):
    """Resolve aliases before dropping duplicate and nested accounting roots."""
    resolved = sorted({Path(p).resolve() for p in paths}, key=lambda p: (len(p.parts), str(p)))
    result = []
    for path in resolved:
        if not any(path.is_relative_to(parent) for parent in result):
            result.append(path)
    return result


def tree_bytes(path):
    path = Path(path)
    if not path.exists():
        raise ContractError(f"Artifact ledger root is unavailable: {path}")
    if path.is_file():
        return path.stat().st_size
    # DirEntry carries Windows file attributes from enumeration, avoiding
    # a separate filesystem round-trip per file on every live guard check.
    # Retain the original no-directory-symlink traversal and followed file
    # symlink size semantics; missing roots remain explicit errors.
    total, pending = 0, [os.fspath(path)]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                elif entry.is_file():
                    total += entry.stat().st_size
    return total


def inventory(root, cfg):
    ledger = read_json(Path(root) / cfg["resources"]["inherited_ledger"])
    original = {Path(p).resolve(): count for p, count in ledger["physical_roots"].items()}
    roots, available, missing = {}, {}, {}
    for path in exclusive_paths(original):
        if path.exists():
            count = tree_bytes(path)
            available[str(path)] = count
        else:
            count = original[path]
            if count:
                missing[str(path)] = count
        roots[str(path)] = count
    return dict(physical_roots=roots, artifact_bytes=sum(roots.values()),
                available_artifact_bytes=sum(available.values()),
                available_physical_roots=available, unavailable_physical_roots=missing,
                unavailable_last_known_bytes=sum(missing.values()),
                accounting_policy="Unavailable roots retain last-known charges; available roots are measured live",
                prior_recorded_bytes=ledger["artifact_bytes"],
                free_bytes=shutil.disk_usage(root).free,
                raw_asset_classification=ledger["raw_asset_classification"],
                historical_cpu_complete=None, historical_gpu_seconds=None)


def prior_cpu_receipts(frontier, current_output):
    """Sum per-command observations, never the cumulative totals they report."""
    receipts, unknown = {}, []
    if not frontier.exists():
        return receipts, unknown
    # Only final command directories qualify. Test fixtures and code snapshots
    # can contain realistic ledgers nested below them and are not executions.
    for path in frontier.glob("*/resource_ledger.json"):
        if path.is_relative_to(current_output):
            continue
        value = read_json(path)
        seconds = value.get("cpu_seconds")
        if not isinstance(seconds, (int, float)) or seconds < 0:
            unknown.append(str(path))
            continue
        # A preserved receipt can retain its explicit command identity. Legacy
        # receipts have one final ledger per unique command output directory.
        key = value.get("command_id", str(path.parent.resolve()))
        receipts[key] = max(receipts.get(key, 0), float(seconds))
    return receipts, unknown


class Budget:
    def __init__(self, root, cfg, output, inherited=None, initial_new_bytes=0):
        self.root, self.cfg, self.output = Path(root).resolve(), cfg, Path(output).resolve()
        self.frontier = self.root / "results/opal_frontier_bridge_v1"
        self.before = inventory(self.root, cfg) if inherited is None else inherited
        self.current_inventory = self.before
        self.refresh_inventory = inherited is None
        self.last_inventory_wall = time.monotonic()
        self.initial_new_bytes = initial_new_bytes
        self.start_cpu, self.start_wall = time.process_time(), time.monotonic()
        self.prior_cpu, self.unknown_cpu_receipts = prior_cpu_receipts(self.frontier, self.output)
        self.temporary_roots = set()
        self.peak_retained_bytes = 0
        self.reserved_receipt_bytes = 65536

    def register_temporary_root(self, path):
        path = Path(path).resolve()
        self.temporary_roots.add(path)
        registry = self.frontier / "verification_temporary_roots.json"
        existing = self._registered_temporary_roots()
        payload = json.dumps({"roots": sorted(map(str, existing | self.temporary_roots))}, indent=2) + "\n"
        self.check(len(payload.encode("utf-8")))
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(payload, encoding="utf-8")
        self.check()

    def _registered_temporary_roots(self):
        registry = self.frontier / "verification_temporary_roots.json"
        if not registry.exists():
            return set()
        record = read_json(registry)
        paths = record.get("roots", record.get("verification_temporary_roots", record)) if isinstance(record, dict) else record
        if isinstance(paths, dict):
            paths = paths.keys()
        return {Path(p).resolve() for p in paths}

    def check(self, pending_bytes=0):
        if pending_bytes < 0:
            raise ContractError("Pending byte reservation cannot be negative")
        if self.refresh_inventory and time.monotonic() - self.last_inventory_wall >= 5:
            self.current_inventory = inventory(self.root, self.cfg)
            self.last_inventory_wall = time.monotonic()
        limits = self.cfg["resources"]
        used = self.new_bytes()
        self.peak_retained_bytes = max(self.peak_retained_bytes, used)
        planned = used + pending_bytes + self.reserved_receipt_bytes
        if planned >= limits["new_retained_bytes"]:
            raise ContractError("New retained-byte allowance exhausted")
        if self.current_inventory["artifact_bytes"] + planned >= limits["artifact_cap_bytes"]:
            raise ContractError("Inherited artifact cap cannot support this attempt")
        if shutil.disk_usage(self.output).free - pending_bytes - self.reserved_receipt_bytes < limits["reserve_bytes"]:
            raise ContractError("Writable-volume free-space reserve reached")
        if (sum(self.prior_cpu.values()) + time.process_time()) / 3600 >= limits["cpu_core_hours"]:
            raise ContractError("Cumulative CPU core-hour allowance exhausted")

    def write_json(self, path, value):
        """Guard complete UTF-8 JSON serialization before touching the destination."""
        payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        self.check(len(payload.encode("utf-8")))
        Path(path).write_text(payload, encoding="utf-8")
        self.check()

    def copy_file(self, source, destination):
        self.check(Path(source).stat().st_size)
        shutil.copy2(source, destination)
        self.check()

    def evidence(self):
        if self.refresh_inventory:
            self.current_inventory = inventory(self.root, self.cfg)
        cpu = time.process_time()
        return dict(inventory_before=self.before, inventory_current=self.current_inventory,
                    command_id=str(self.output), cpu_seconds=cpu,
                    cpu_scope="full current single-thread process including startup; prior final command receipts charged separately",
                    timed_budget_cpu_seconds=cpu - self.start_cpu,
                    cumulative_observed_cpu_seconds=sum(self.prior_cpu.values()) + cpu,
                    prior_command_cpu_seconds=self.prior_cpu,
                    cpu_accounting_complete=False, unknown_cpu_receipts=self.unknown_cpu_receipts,
                    unknown_cpu_scope="Unreceipted shell/test/build/tool and earlier subprocess overhead remains unknown",
                    wall_seconds=time.monotonic() - self.start_wall,
                    new_retained_bytes=self.new_bytes(), peak_retained_bytes=self.peak_retained_bytes,
                    temporary_roots=sorted(map(str, self._registered_temporary_roots() | self.temporary_roots)),
                    gpu_seconds=None, gpu_measurement="unmeasured; no GPU work requested",
                    historical_cpu_complete=None, external_tool_cpu_seconds=None,
                    limits=self.cfg["resources"], receipt_reserve_bytes=self.reserved_receipt_bytes)

    def new_bytes(self):
        roots = [self.frontier, self.output, self.root / "src/frontier_bridge", self.root / "docs/frontier",
                 self.root / "PLAN.md", self.root / "docs/next_cycle/model_series_v2_plan.md",
                 self.root / "configs/experiments/opal_frontier_bridge_v1.json"]
        roots += list(self._registered_temporary_roots() | self.temporary_roots)
        roots += list((self.root / "tools").glob("*frontier*.py"))
        roots += list((self.root / "tests").glob("test_frontier*.py"))
        # Missing registered temporaries remain visible in evidence; absent
        # files contribute no live bytes, unlike missing historical assets.
        return sum(tree_bytes(p) for p in exclusive_paths(roots) if p.exists()) + self.initial_new_bytes
