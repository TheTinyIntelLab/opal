"""Per-run acceptance, exact logical procedure inventory and live stop limits."""
import os
import json
import shutil
import time
from pathlib import Path
from ..artifacts import fingerprint, read_json, sha256, write_json
from .contracts import ContractError, PROTOCOL


def inventory(pair_count, gene_count, base_count):
    from .model import menu
    # Enumerates every scheduled top-level and dependent fit. No successful-fit
    # discount is used for unsupported designs or cached identical requests.
    plan = []
    for swap in ("primary", "a_to_b", "b_to_a"):
        for outer in range(3):
            specs = menu()
            for spec in specs:
                for inner in range(3):
                    plan.extend(fit_plan(spec, f"{swap}/outer{outer}/inner{inner}"))
            # Exactly one selected penalty refit per family/seed. Its identity
            # depends only on inner labels; all branches have the same count.
            refits = {}
            for spec in specs:
                refits.setdefault((spec["family"], spec.get("seed")), spec)
            for spec in refits.values():
                plan.extend(fit_plan(spec, f"{swap}/outer{outer}/selected_refit"))
    base = [p for p in plan if p["operation"] == "base_menu"]
    for p in base:
        p["fits"] = base_count * 3 * 2 + 2
    fits = sum(p["fits"] for p in plan)
    arms = len({(spec["family"], spec.get("seed")) for spec in menu()})+1
    predictions = 3 * pair_count * 2 * gene_count * 8 * arms
    return dict(experiment_id=PROTOCOL["experiment_id"], pairs=pair_count, genes=gene_count,
        base_configurations=base_count, scheduled_operations=plan, logical_fits=fits,
        retained_state_policy="every scheduled attempt including nested fits; content-addressed arrays",
        raw_existing_bytes=PROTOCOL["raw_bytes"], raw_new_bytes=0, downloads=0,
        prediction_bytes_per_phase=predictions, prediction_bytes=predictions*2,
        prediction_arm_count=arms, prediction_copies=2,
        uncompressed_checkpoint_upper_estimate_bytes=fits * (8 * (pair_count * 4 * gene_count + pair_count**2 + 8 * gene_count + 1000)),
        proposed_cpu_core_hours=24, initial_cpu_hours_estimate=[0.5, 4], estimate_profiled=False,
        aggregation_ram_target_bytes=8_000_000_000, fitting_ram_target_bytes=4_000_000_000,
        retained_pilot_proposal_bytes=25_000_000, failure_reserve_bytes=5_000_000,
        scratch_estimate_bytes=8 * 13901 * 1000,
        note="25 MB is conditional; accept the complete inventory, never head size or unmeasured compression")


def fit_plan(spec, prefix):
    rows = [dict(id=prefix + "/" + spec["id"], operation="fit", fits=1)]
    if spec["family"] in ("phi18", "phi8", "mean18", "context18", "mean_response18", "alignment_response", "alignment_contrast"):
        # pooled kernel: 3 penalties * 3 validations + refit, repeated
        # for each nuisance OOF fold and final pooled refit, plus 4 h estimates.
        rows.append(dict(id=prefix + "/" + spec["id"] + "/nuisance", operation="pooled_and_h_encoder_head", fits=46))
    if spec["family"] == "base":
        rows.append(dict(id=prefix + "/base", operation="base_menu", fits=0))
    if spec["family"] in ("incidence", "per_context_profile", "per_context_product", "pooled_kernel"):
        rows.append(dict(id=prefix + "/" + spec["id"] + "/context_dependencies", operation="reference_dependencies", fits=2))
    if spec["family"] == "joint_context":
        rows.append(dict(id=prefix + "/joint_dependency", operation="reference_dependencies", fits=1))
    return rows


def authorize(path, operation, output, input_hash, inventory_hash):
    acceptance = read_json(path)
    required = dict(experiment_id=PROTOCOL["experiment_id"], operation=operation,
        output=str(Path(output).resolve()), input_hash=input_hash,
        inventory_hash=inventory_hash, protocol_hash=fingerprint(PROTOCOL), threads=1, workers=1)
    if acceptance.get("approved") is not True or not acceptance.get("owner") or not acceptance.get("approved_at"):
        raise ContractError("Separate owner numerical acceptance required")
    for key, value in required.items():
        if acceptance.get(key) != value:
            raise ContractError("Authorization mismatch: " + key)
    for key in ("cpu_seconds", "ram_bytes", "retained_bytes", "scratch_bytes", "failure_reserve_bytes", "reserve_bytes"):
        value = acceptance.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ContractError("Explicit positive numerical limit required: " + key)
    if acceptance["reserve_bytes"] < 1073741824 or acceptance["cpu_seconds"] > 86400:
        raise ContractError("Reserve/CPU proposal exceeded")
    if any(acceptance.get(k) is not False for k in ("installs", "downloads", "gpu")):
        raise ContractError("Installation/download/GPU not permitted")
    return acceptance


class Budget:
    def __init__(self, output, acceptance):
        self.output = Path(output).resolve()
        if self.output.exists():
            raise FileExistsError(self.output)
        self.acceptance = acceptance
        self.start = time.process_time()
        self.wall = time.monotonic()
        self.peak_bytes = 0
        self.peak_rss = 0
        self.failure_receipt_bytes = 0
        self.check()
        self.output.mkdir(parents=True)
        write_json(self.output / "acceptance.json", acceptance)

    def check(self, additional=0):
        import psutil
        process = psutil.Process(os.getpid())
        if process.children(recursive=True):
            raise ContractError("Numerical subprocess forbidden")
        rss = process.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        size = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file()) if self.output.exists() else 0
        self.peak_bytes = max(self.peak_bytes, size)
        parent = self.output if self.output.exists() else self.output.parent
        while not parent.exists():
            parent = parent.parent
        if size + additional + self.acceptance["failure_reserve_bytes"] > self.acceptance["retained_bytes"]:
            raise ContractError("Retained asset/failure reserve limit")
        if shutil.disk_usage(parent).free - additional < self.acceptance["reserve_bytes"]:
            raise ContractError("Writable volume reserve")
        if time.process_time() - self.start > self.acceptance["cpu_seconds"] or rss > self.acceptance["ram_bytes"]:
            raise ContractError("CPU/RAM stop limit")

    def receipt(self, status):
        write_json(self.output / "resource_ledger.json", dict(status=status,
            cpu_seconds=time.process_time()-self.start, wall_seconds=time.monotonic()-self.wall,
            cpu_accounting_complete=True, scope="this one-thread process only; historical accounting unchanged",
            retained_bytes=sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file()),
            peak_retained_bytes=self.peak_bytes, peak_rss_bytes=self.peak_rss,
            gpu_seconds=0, downloads=0, installs=0, acceptance_hash=fingerprint(self.acceptance)))

    def write_json(self, path, value):
        self.check(len(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))+8192)
        write_json(path, value)
        self.check()

    def write_failure_json(self, path, value):
        """Use the already reserved receipt allowance after a stop limit fires."""
        encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
        needed = len(encoded)+8192
        if self.failure_receipt_bytes+needed > self.acceptance["failure_reserve_bytes"]:
            raise ContractError("Failure receipt reserve exhausted")
        target = Path(path).resolve()
        if not target.is_relative_to(self.output):
            raise ContractError("Failure receipt must stay inside accepted output")
        write_json(target, value)
        self.failure_receipt_bytes += needed
