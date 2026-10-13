"""Predict first, seal every inference dependency, then evaluate separately."""
import json
from pathlib import Path
import numpy as np
from ..artifacts import array_hash, fingerprint, read_json, sha256, write_json
from ..replicated.models import ProfileKernel, ComponentAdditive
from .artifacts import load_npz, load_state, save_npz, save_state
from .contracts import PROTOCOL, ContractError
from .model import select, predict
from .evaluation import evaluate, metric_qualification, development_checks, five_gates


def safe_state(value):
    if isinstance(value, (ProfileKernel, ComponentAdditive)):
        return dict(class_name=type(value).__name__, fitted=value.__dict__.copy())
    if isinstance(value, dict):
        return {k: safe_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_state(v) for v in value]
    return value


class Recorder:
    def __init__(self, output, budget):
        self.output, self.budget = Path(output), budget
        self.count = 0
        self.rows = []

    def __call__(self, row, state):
        self.budget.check()
        record = dict(row, attempt=self.count)
        if state is not None:
            checkpoint = self.output / "attempts" / f"{self.count:06d}"
            save_state(checkpoint, safe_state(state), self.budget)
            record["checkpoint"] = str(checkpoint.relative_to(self.output))
        with (self.output / "fit_attempts.jsonl").open("a", encoding="utf-8") as stream:
            data = json.dumps(record, sort_keys=True, allow_nan=False)+"\n"
            self.budget.check(len(data.encode("utf-8")))
            stream.write(data)
        self.rows.append(record)
        self.count += 1


def validate_manifest(manifest):
    if manifest.get("experiment_id") != PROTOCOL["experiment_id"] or manifest.get("protocol_hash") != fingerprint(PROTOCOL):
        raise ContractError("Prepared successor protocol mismatch")
    if fingerprint(manifest["genes"]) != manifest["genes_hash"] or len(set(manifest["genes"])) != len(manifest["genes"]):
        raise ContractError("Literal gene order/hash mismatch")
    expected = set(manifest["folds"])
    for role in {r["role"] for r in manifest["records"]}:
        rows = [r for r in manifest["records"] if r["role"] == role]
        queries = [p for r in rows for p in r["query_ids"]]
        if len(rows) != 3 or set(queries) != expected or len(queries) != len(expected):
            raise ContractError("Complete exclusive global query population required")
        for r in rows:
            if set(r["source_ids"]) & set(r["query_ids"]) or set(r["source_ids"]) | set(r["query_ids"]) != expected:
                raise ContractError("Source/query global exclusion failed")


def run_predictions(prepared, output, budget):
    prepared, output = Path(prepared), Path(output)
    write_json = budget.write_json
    manifest = read_json(prepared / "prepared_manifest.json")
    validate_manifest(manifest)
    write_json(output / "protocol.json", PROTOCOL)
    write_json(output / "input_manifest.json", manifest)
    recorder = Recorder(output, budget)
    records, metric_controls, supported = [], {}, []
    for record in manifest["records"]:
        source = load_npz(prepared, record["sourcefit"])
        query = load_npz(prepared, record["references"])
        if set(query) != {"singles", "pair_ids"}:
            raise ContractError("Query reference role must contain no outcomes or derived targets")
        if set(source)-{"singles", "responses", "pair_ids", "metric_positive_a", "metric_positive_b"}:
            raise ContractError("Unexpected source-fit inputs")
        if source["pair_ids"].tolist() != record["source_ids"] or query["pair_ids"].tolist() != record["query_ids"]:
            raise ContractError("Role identity ordering mismatch")
        states, selection = select(source["singles"], source["responses"], source["pair_ids"], recorder)
        role_fold = record["role"]+"-"+str(record["fold"])
        write_json(output / (role_fold+"-selection.json"), selection)
        # Qualify the metric with source-only halves before access to query outcomes.
        a, b = source.get("metric_positive_a"), source.get("metric_positive_b")
        metric_controls[role_fold] = metric_qualification(a, b, source["responses"].mean(0))
        selector_name = selection["selector"]["spec"]["family"]
        selector_state = states[selector_name]
        predictions, fallbacks, replay = {}, {}, {}
        for arm, state in states.items():
            fallback = state is None
            used_state = selector_state if fallback else state
            checkpoint = output / "inference" / (role_fold+"-"+arm)
            save_state(checkpoint, dict(model=used_state, query=query,
                genes_hash=manifest["genes_hash"], contexts=PROTOCOL["contexts"],
                fallback=fallback, locked_selector=selection["selector"]), budget)
            pred = predict(used_state, query["singles"], query["pair_ids"])
            restored = load_state(checkpoint)
            replayed = predict(restored["model"], restored["query"]["singles"], restored["query"]["pair_ids"])
            if not np.array_equal(pred, replayed):
                raise ContractError("Saved-state replay mismatch")
            predictions[arm] = pred
            fallbacks[arm] = np.full(len(pred), fallback)
            replay[arm] = dict(exact=True, prediction_hash=array_hash(pred))
        predictions["selector"] = predictions[selector_name].copy()
        fallbacks["selector"] = np.zeros(len(query["pair_ids"]), bool)
        saved = save_npz(output / (role_fold+"-predictions.npz"), budget,
            pair_ids=query["pair_ids"], **predictions,
            **{name+"__fallback": v for name, v in fallbacks.items()})
        records.append(dict(role=record["role"], fold=record["fold"], predictions=saved,
            evaluator=record["evaluator"], references=record["references"], replay=replay,
            complete_support=all(state is not None for state in states.values())))
        supported.append(records[-1]["complete_support"])
        budget.check()
    write_json(output / "metric_qualification.json", metric_controls)
    write_json(output / "rank_support.json", [r for r in recorder.rows if r.get("diagnostic") is not None or r["stage"] in ("single_encoder", "residual_head")])
    write_json(output / "replay.json", dict(records=records, exact=True))
    # Hash every file, including dependency arrays and failed attempt receipts.
    seal = dict(prepared_root=str(prepared.resolve()), records=records,
        files={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "resource_ledger.json"},
        full_population=True, query_outcomes_accessed=False, complete_support=all(supported))
    write_json(output / "prediction_freeze.json", seal)
    return seal


def verify_freeze(run):
    run = Path(run)
    seal = read_json(run / "prediction_freeze.json")
    for name, digest in seal["files"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or sha256(path) != digest:
            raise ContractError("Prediction freeze tampered: " + name)
    return seal


def evaluate_run(run, output, budget, software_integrity=None):
    run, output = Path(run), Path(output)
    write_json = budget.write_json
    seal = verify_freeze(run)
    manifest = read_json(run / "input_manifest.json")
    prepared = Path(seal["prepared_root"])
    controls = read_json(run / "metric_qualification.json")
    results, gate_checks = {}, {}
    stratum_diagnostics = {}
    for role in sorted({r["role"] for r in seal["records"]}):
        ids, truth, singles, fold_ids, predictions, fallback = [], [], [], [], {}, {}
        records = [r for r in seal["records"] if r["role"] == role]
        for r in records:
            p = load_npz(run, r["predictions"])
            evaluator = load_npz(prepared, r["evaluator"])
            refs = load_npz(prepared, r["references"])
            if not np.array_equal(p["pair_ids"], evaluator["pair_ids"]) or not np.array_equal(p["pair_ids"], refs["pair_ids"]):
                raise ContractError("Evaluation identity alignment mismatch")
            ids.extend(p["pair_ids"]); truth.append(evaluator["responses"]); singles.append(refs["singles"])
            fold_ids.extend([r["fold"]]*len(p["pair_ids"]))
            for name, values in p.items():
                if name == "pair_ids":
                    continue
                target = fallback if name.endswith("__fallback") else predictions
                key = name.removesuffix("__fallback")
                target.setdefault(key, []).append(values)
        order = np.argsort(ids)
        ids = np.asarray(ids)[order]
        truth, singles, fold_ids = np.concatenate(truth)[order], np.concatenate(singles)[order], np.asarray(fold_ids)[order]
        predictions = {k: np.concatenate(v)[order] for k, v in predictions.items()}
        fallback = {k: np.concatenate(v)[order] for k, v in fallback.items()}
        results[role] = evaluate(predictions, truth, singles, ids, fallback, fold_ids, program=manifest.get("program"))
        if role == "primary" and manifest.get("stratum_evaluator"):
            record = manifest["stratum_evaluator"]
            path = (prepared/record["path"]).resolve()
            if not path.is_relative_to(prepared.resolve()) or sha256(path) != record["sha256"]:
                raise ContractError("Stratum evaluator hash/path mismatch")
            rows = read_json(path)
            index = {p: i for i, p in enumerate(ids)}
            for arm, prediction in predictions.items():
                grouped = {}
                for row in rows:
                    if row["pair_id"] not in index or row["context"] not in PROTOCOL["contexts"]:
                        raise ContractError("Unknown stratum query identity/context")
                    c = PROTOCOL["contexts"].index(row["context"])
                    key = row["context"]+"/"+row["stratum"]
                    loss = np.mean((prediction[index[row["pair_id"]], c]-np.asarray(row["response"]))**2)
                    grouped.setdefault(key, []).append(float(loss))
                stratum_diagnostics[arm] = {key: dict(mse=float(np.mean(values)), rows=len(values),
                    scope="aggregate prediction against original matched-stratum outcomes; secondary only") for key, values in grouped.items()}
        save_npz(output / (role+"-predictions.npz"), budget, pair_ids=ids, **predictions,
            **{name+"__fallback": v for name, v in fallback.items()})
        gate_checks[role] = development_checks(results[role], all(r["complete_support"] for r in records),
            all(controls[role+"-"+str(r["fold"])]["qualified"] for r in records),
            manifest["reference_swaps_available"])
        budget.check()
    original = manifest["qualification"]
    source_identity = original.get("source_identity", {})
    qualification = dict(original, numerical_acceptance_verified=True,
        qualified=bool(original.get("qualified") is True and source_identity.get("qualified") is True and
            source_identity.get("sha256") == PROTOCOL["raw_sha256"] and
            len(manifest["genes"]) >= 256 and len(manifest["folds"]) >= 36))
    # A supplied claim flag is insufficient: tests must be hash-reviewed by CLI.
    integrity = dict(qualified=bool(software_integrity and software_integrity.get("qualified") is True and
        seal["complete_support"] and manifest["reference_swaps_available"] and all(v["qualified"] for v in controls.values())),
        software_checks=software_integrity, query_mutation_protection="query API omits pair responses; fixture regression required",
        replay_exact=True, independent_cultures=False)
    decision = five_gates(qualification, integrity, gate_checks["primary"],
        [gate_checks[r] for r in ("a_to_b", "b_to_a") if r in gate_checks])
    write_json(output / "metrics.json", {k: v["metrics"] for k, v in results.items()})
    write_json(output / "uncertainty.json", {k: v["uncertainty"] for k, v in results.items()})
    write_json(output / "controls.json", {k: v["retrieval_null"] for k, v in results.items()})
    write_json(output / "sensitivity.json", {k: v["sensitivity"] for k, v in results.items()})
    write_json(output / "stratum_metrics.json", stratum_diagnostics)
    write_json(output / "qualification.json", qualification)
    write_json(output / "integrity.json", integrity)
    write_json(output / "decision.json", decision)
    (output / "report.md").write_text("# Context-contrast development\n\n"+decision["status"]+
        ". Complete canonical-pair aggregate scores and all fallbacks are retained.\n\n"
        "Both TNF contexts are observed training information. Conditional technical sensitivity supplies no independent biological confirmation or frontier claim.\n", encoding="utf-8")
    return decision
