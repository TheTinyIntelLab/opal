"""Freeze inference artifacts before evaluator access; replay in a new process."""
from pathlib import Path
import json
import hashlib
import shutil

import numpy as np

from .artifacts import load_state, save_state
from .contracts import ContractError, assert_roles, fingerprint, read_json, sha256, write_json
from .evaluation import direction_gate, evaluate
from .model import (Unsupported, adapt, adapted_predict, coordinates, fit_encoder,
                    fit_mean_modes, fit_source_free, mode_features, nuisance_predict, ridge,
                    ridge_predict, select_source)
from .preparation import load_npz, save_npz
from .references import (FAMILIES, fit_reference, predict_reference,
                         select_references, source_base)
from .splits import bridge_permutation, source_permutation


REQUIRED_JSON = ("qualification", "input_manifest", "exposure_ledger", "label_ledger",
                 "folds", "bridges", "source_selection", "rank_and_stability",
                 "metrics", "controls", "uncertainty", "replay")


def failure_receipt(output, cfg, reason, budget=None, qualification=None):
    output = Path(output)
    write_json(output / "protocol.json", cfg)
    for name in REQUIRED_JSON:
        path = output / f"{name}.json"
        if not path.exists():
            write_json(path, qualification if name == "qualification" and qualification else dict(status="not_started", reason=reason))
    if not (output / "fit_attempts.jsonl").exists():
        (output / "fit_attempts.jsonl").write_text("", encoding="utf-8")
    if not (output / "predictions.npz").exists():
        np.savez_compressed(output / "predictions.npz", predictions=np.empty((0, 0), dtype=np.float64))
    decision = dict(engineering_complete=False, source_qualified_for_public_prototype=False,
                    adaptation_supported=False, development_gate_passed=False,
                    independent_biological_confirmation=False, frontier_claim=False,
                    status="infeasible_public_prototype" if qualification and not qualification.get("qualified") else "engineering_failure", reason=reason,
                    next_action="retain synthetic verification; qualified independent measurements and reviewed power required",
                    unknown_measurements=["independent culture repeatability", "procedure uncertainty", "biological useful margin", "historical complete CPU", "GPU seconds"])
    write_json(output / "decision.json", decision)
    write_json(output / "resource_ledger.json", budget.evidence() if budget else dict(measurement_unavailable=True, reason=reason))
    (output / "report.md").write_text(f"# Frontier bridge qualification\n\nStatus: incomplete attempt; see decision and saved fit index.\n\n{reason}\n\nThis attempt did not complete. Existing fit checkpoints and earlier receipts are retained; no public advancement is inferred. Independent biological confirmation and frontier claims remain false.\n", encoding="utf-8")
    return decision


def predict_checkpoint(state):
    x, ids, model = state["query"]["singles"], state["query"]["pair_ids"], state["model"]
    if model["kind"] == "reference":
        result = predict_reference(model["reference"], x, ids)
    elif model["kind"] == "adapted":
        result = adapted_predict(model["source"], model["adapter"], x)
    elif model["kind"] == "source_free":
        result = x.sum(1) + ridge_predict(model["mapping"], coordinates(x, model["encoder"]))
    elif model["kind"] == "zero_bridge":
        source = model["source"]
        result = x.sum(1) + nuisance_predict(source["nuisance"], x)
        result += np.einsum("nlg,l->ng", mode_features(source, x), source["scales"])
    else:
        raise ContractError("Unknown safe inference checkpoint")
    if result.dtype != np.float64 or not np.isfinite(result).all():
        raise Unsupported("Nonfinite float64 prediction")
    return result


def code_hashes(root):
    paths = sorted((Path(root) / "src/frontier_bridge").glob("*.py"))
    paths += [Path(root) / p for p in ("src/replicated/models.py", "src/series.py", "src/series_bilinear.py", "configs/experiments/replicated_fixed_state_v1.json", "tools/run_frontier_bridge.py", "tools/prepare_frontier_bridge.py")]
    return {str(p.resolve()): sha256(p) for p in paths}


def run(prepared, cfg, root, output, budget):
    output = Path(output)
    if prepared["configuration_hash"] != fingerprint(cfg):
        raise ContractError("Prepared configuration mismatch")
    if prepared["role"] != "synthetic_engineering":
        # The currently qualified public source fails the frozen gene minimum.
        # Do not treat a hand-edited prepared manifest as owner authorization
        # to bypass qualification or substitute a new dataset.
        from .contracts import qualify
        current = qualify(root, cfg)
        if not current["qualified"] or current != prepared["source_qualification"]:
            raise ContractError("Public source is not qualified for this prepared manifest")
    budget.write_json(output / "protocol.json", cfg)
    budget.write_json(output / "qualification.json", prepared["source_qualification"])
    budget.write_json(output / "folds.json", prepared["folds"])
    union_bridges = sorted({p for f in prepared["folds"] for p in f["bridges"]})
    budget.write_json(output / "bridges.json", dict(per_fold=[dict(fold=f["fold"], pair_ids=f["bridges"]) for f in prepared["folds"]],
                                               distinct_acquisition_union=union_bridges,
                                               union_size=len(union_bridges), per_adaptation_budget=4))
    budget.write_json(output / "exposure_ledger.json", dict(role=prepared["role"], prior_exposure="synthetic only" if prepared["synthetic_case"] else "historically exposed", independent_confirmation=False))
    hashes = code_hashes(root)
    snapshot = output / "code_snapshot"
    for path in hashes:
        original = Path(path)
        destination = snapshot / original.relative_to(root)
        budget.check(original.stat().st_size + 8192)
        destination.parent.mkdir(parents=True, exist_ok=True)
        budget.copy_file(original, destination)
    budget.write_json(output / "input_manifest.json", dict(prepared=prepared, code_hashes=hashes,
                                                    configuration_hash=fingerprint(cfg)))
    attempts_path = output / "fit_attempts.jsonl"
    attempts_path.write_text("", encoding="utf-8")
    sequence = 0

    def attempt(row):
        nonlocal sequence
        payload = json.dumps(dict(sequence=sequence, **row), sort_keys=True, allow_nan=False) + "\n"
        budget.check(len(payload.encode("utf-8")))
        with attempts_path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
        sequence += 1

    selection_records, stability, checkpoints, acquisitions = [], [], [], []
    arm_chunks, metadata, evaluator_records, support, control_support = {}, {}, {}, {}, {}
    for source_context, target_context in [("TNFA+", "TNFA-"), ("TNFA-", "TNFA+")]:
        direction = source_context + "_to_" + target_context
        arm_chunks[direction], metadata[direction], evaluator_records[direction], support[direction], control_support[direction] = {}, [], [], [], []
        for fold in prepared["folds"]:
            k = fold["fold"]
            source_record = prepared["contexts"][source_context]["folds"][k]
            target_record = prepared["contexts"][target_context]["folds"][k]
            source_bank = load_npz(source_record["source"])
            refs = load_npz(target_record["references"])
            sx, sy, sid = [source_bank[key] for key in ("singles", "responses", "pair_ids")]
            if set(sid) != set(fold["source"]) or set(refs["pair_ids"]) != set(fold["query"]):
                raise ContractError("Prepared global role membership changed")
            assert_roles(sid, fold["bridges"], refs["pair_ids"])
            if len(set(sid)) < cfg["minimum_source_pairs"]:
                raise Unsupported("Insufficient outer source fitting pairs")
            selected, selector, reference_attempts = select_references(sx, sy, sid, attempt, strata=source_bank["strata"])
            base, base_attempts = source_base(sx, sy, sid, strata=source_bank["strata"])
            source_model, source_failure, source_attempts = None, None, []
            try:
                source_model, source_attempts = select_source(sx, sy, sid, attempt, strata=source_bank["strata"])
            except Unsupported as error:
                source_failure = str(error)
            shuffled_sources, shuffle_maps = {}, {}
            for seed in cfg["shuffle_seeds"]:
                mapping = source_permutation(sid, seed)
                shuffle_maps[str(seed)] = mapping
                index = {(str(st), str(p)): i for i, (st, p) in enumerate(zip(source_bank["strata"], sid))}
                order = [index[(str(st), mapping[str(p)])] for st, p in zip(source_bank["strata"], sid)]
                try:
                    shuffled_sources[seed], _ = select_source(sx, sy, sid, attempt, strata=source_bank["strata"], shuffle_seed=seed)
                except Unsupported as error:
                    shuffled_sources[seed] = None
                    attempt(dict(stage="source_shuffle", seed=seed, success=False, reason=str(error)))
            frozen = dict(direction=direction, fold=k, selected_references=selected, selector=selector,
                          reference_attempts=reference_attempts, base_attempts=base_attempts,
                          source_attempts=source_attempts, source_failure=source_failure,
                          source_shuffle_mappings=shuffle_maps,
                          source_label_ids=sorted(set(sid)), target_outcomes_accessed=False)
            selection_records.append(frozen)
            # This receipt and fitted source states exist before target bridge
            # outcomes are opened. Query evaluator files remain unopened.
            source_state = save_state(output / f"source-{len(selection_records)}", dict(source=source_model, shuffled=shuffled_sources, base=base), budget)
            frozen["source_checkpoint"] = source_state
            budget.write_json(output / "source_selection.json", selection_records)
            bridge_bank = load_npz(target_record["bridges"])
            if set(bridge_bank["pair_ids"]) != set(fold["bridges"]):
                raise ContractError("Prepared bridge membership changed")
            for st in sorted(set(refs["strata"])):
                qmask, bmask = refs["strata"] == st, bridge_bank["strata"] == st
                qx, qid = refs["singles"][qmask], refs["pair_ids"][qmask]
                bx, by, bid = [bridge_bank[key][bmask] for key in ("singles", "responses", "pair_ids")]
                if len(bid) != 4 or len(set(bid)) != 4:
                    raise ContractError("Four distinct bridge identities per target stratum required")
                acquisitions.append(dict(direction=direction, fold=k, stratum=str(st), source_pair_ids=sorted(set(sid)),
                                         bridge_pair_ids=bid.tolist(), measured_target_outcomes=4,
                                         query_controls_singles_allowed=True, source_fitting_uses="nuisance/crossfit/modes/reference/selection/shuffles"))
                arms = {}
                for family in FAMILIES:
                    reference = fit_reference(family, selected[family]["alpha"], sx, sy, sid, bx, by, bid, base, source_strata=source_bank["strata"])
                    arms[family] = dict(kind="reference", reference=reference)
                    attempt(dict(stage="target_reference", direction=direction, fold=k, stratum=str(st), family=family, success=True))
                arms["selector"] = arms[selector["family"]]
                fallback = arms["selector"]
                candidates = {"candidate": source_model}
                for seed in cfg["shuffle_seeds"]:
                    candidates[f"source_shuffle_{seed}"] = shuffled_sources[seed]
                    candidates[f"bridge_shuffle_{seed}"] = source_model
                try:
                    candidates["mean_feature"] = fit_mean_modes(source_model, sx, sid, source_bank["strata"]) if source_model else None
                except Unsupported:
                    candidates["mean_feature"] = None
                candidate_supported = False
                for arm, source in candidates.items():
                    try:
                        if source is None:
                            raise Unsupported("Source mode fit unavailable")
                        mapping = None
                        if arm.startswith("bridge_shuffle_"):
                            bm = bridge_permutation(bid, int(arm.rsplit("_", 1)[1]))
                            mapping = [bid.tolist().index(bm[str(p)]) for p in bid]
                        adapter = adapt(source, bx, by, qx, mapping)
                        arms[arm] = dict(kind="adapted", source=source, adapter=adapter)
                        diagnostic = dict(direction=direction, fold=k, stratum=str(st), arm=arm,
                                          supported=True, rank=adapter["diagnostic"], stability=adapter["stability"],
                                          full_mode_rms=adapter["full_mode_rms"], zero_mode_branch=adapter["zero_mode_branch"],
                                          bridge_permutation=mapping, target_numeric_values=len(prepared["genes"]) + 2)
                        if arm == "candidate":
                            candidate_supported = True
                    except Unsupported as error:
                        arms[arm] = fallback
                        diagnostic = dict(direction=direction, fold=k, stratum=str(st), arm=arm, supported=False, fallback="locked_selector_complete_population", reason=str(error))
                    stability.append(diagnostic)
                    attempt(dict(stage="target_adaptation", **diagnostic, success=diagnostic["supported"]))
                support[direction].append(candidate_supported)
                control_support[direction].append(all(r["supported"] for r in stability if r["direction"] == direction and r["fold"] == k and r["stratum"] == str(st) and r["arm"] != "candidate"))
                try:
                    encoder = source_model["encoder"] if source_model else fit_encoder(sx, source_bank["strata"])
                    arms["source_free"] = dict(kind="source_free", encoder=encoder,
                                                mapping=fit_source_free(sx, bx, by, encoder, sid, source_bank["strata"]))
                except Unsupported as error:
                    control_support[direction][-1] = False
                    arms["source_free"] = fallback
                    attempt(dict(stage="source_free", success=False, reason=str(error)))
                attempt(dict(stage="source_free", direction=direction, fold=k, stratum=str(st), success=arms["source_free"]["kind"] == "source_free"))
                arms["zero_bridge"] = dict(kind="zero_bridge", source=source_model) if source_model else dict(kind="reference", reference=fit_reference("frozen_base", 1, sx, sy, sid, bx, by, bid, base, source_strata=source_bank["strata"]))
                # The zero-bridge diagnostic never falls back to a bridge fit.
                for arm, model in arms.items():
                    inference = dict(frozen_prediction=None, query=dict(singles=qx, pair_ids=qid), model=model,
                                     genes=prepared["genes"], scale=prepared["scale"],
                                     direction=direction, fold=k, stratum=str(st), arm=arm)
                    prediction = predict_checkpoint(inference)
                    inference["frozen_prediction"] = prediction
                    record = save_state(output / f"checkpoint-{len(checkpoints):04d}", inference, budget)
                    key = f"prediction_{len(checkpoints):04d}"
                    checkpoints.append(dict(**record, direction=direction, fold=k, stratum=str(st), arm=arm, prediction_key=key))
                    budget.write_json(output / "fit_checkpoint_index.json", checkpoints)
                    arm_chunks[direction].setdefault(arm, []).append(prediction)
                metadata[direction].append(dict(pair_ids=qid, strata=np.repeat(st, len(qid)), fold_ids=np.repeat(k, len(qid)), singles=qx))
            evaluator_records[direction].append(target_record["evaluator"])
    prediction_arrays = {}
    prediction_keys = {}
    offsets = {direction: {arm: 0 for arm in chunks} for direction, chunks in arm_chunks.items()}
    for checkpoint in checkpoints:
        d, arm = checkpoint["direction"], checkpoint["arm"]
        index = offsets[d][arm]
        array = arm_chunks[d][arm][index]
        digest = hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()
        if digest in prediction_keys:
            checkpoint["prediction_key"] = prediction_keys[digest]
        else:
            prediction_keys[digest] = checkpoint["prediction_key"]
            prediction_arrays[checkpoint["prediction_key"]] = array
        offsets[d][arm] += 1
    prediction_record = save_npz(output / "predictions.npz", budget, **prediction_arrays)
    frozen_manifest = dict(configuration_hash=fingerprint(cfg), code_hashes=hashes,
                           checkpoints=checkpoints, predictions=prediction_record,
                           evaluator_access_started=False, role=prepared["role"])
    budget.write_json(output / "run_manifest.json", frozen_manifest)
    freeze_hash = sha256(output / "run_manifest.json")
    budget.write_json(output / "prediction_freeze.json", dict(run_manifest_sha256=freeze_hash,
                                                       predictions_sha256=prediction_record["sha256"], evaluator_access_started=False))
    budget.write_json(output / "label_ledger.json", dict(acquisitions=acquisitions, bridge_identity_union=union_bridges,
                                                 target_pair_stratum_union=sorted({(a["direction"], a["stratum"], p) for a in acquisitions for p in a["bridge_pair_ids"]}),
                                                 repeated_measurement_events=sum(a["measured_target_outcomes"] for a in acquisitions),
                                                 interpretation="four bridges per adaptation; campaign costs count union and repeated strata"))
    budget.write_json(output / "rank_and_stability.json", stability)
    metrics, controls, uncertainty, gates = {}, {}, {}, {}
    for direction, chunks in arm_chunks.items():
        meta = {key: np.concatenate([m[key] for m in metadata[direction]]) for key in metadata[direction][0]}
        truth_lookup = {}
        for record in evaluator_records[direction]:
            evaluator = load_npz(record)  # First study evaluator expression access.
            truth_lookup.update({(str(p), str(s)): y for p, s, y in zip(evaluator["pair_ids"], evaluator["strata"], evaluator["responses"])})
        truth = np.asarray([truth_lookup[(str(p), str(s))] for p, s in zip(meta["pair_ids"], meta["strata"])])
        predictions = {a: np.concatenate(values) for a, values in chunks.items()}
        metrics[direction], controls[direction], uncertainty[direction] = evaluate(predictions, truth, **meta, cfg=cfg)
        gates[direction] = direction_gate(metrics[direction], controls[direction], uncertainty[direction], all(support[direction]), cfg, controls_supported=all(control_support[direction]))
    budget.write_json(output / "metrics.json", metrics)
    budget.write_json(output / "controls.json", dict(metric_controls=controls, direction_gates=gates,
                                              shuffle_seeds=cfg["shuffle_seeds"], source_mappings="source_selection.json",
                                              bridge_mappings="rank_and_stability.json"))
    budget.write_json(output / "uncertainty.json", uncertainty)
    synthetic = prepared["role"] == "synthetic_engineering"
    decision = dict(engineering_complete=True, source_qualified_for_public_prototype=not synthetic,
                    adaptation_supported=all(all(v) for v in support.values()),
                    development_gate_passed=not synthetic and all(g["passed"] for g in gates.values()),
                    synthetic_gate_checks={d: g["passed"] for d, g in gates.items()},
                    independent_biological_confirmation=False, frontier_claim=False,
                    reason="synthetic engineering cannot pass biological or public development gates" if synthetic else "fixed public gate evaluated",
                    next_action="retain engineering evidence; public source qualification/independent measurement required" if synthetic else "independent measurement pilot" if all(g["passed"] for g in gates.values()) else "stop this operator family on the public contract",
                    unknown_measurements=["independent-reference attribution", "procedure uncertainty", "biological useful margin", "historical complete CPU", "GPU seconds"])
    budget.write_json(output / "decision.json", decision)
    budget.write_json(output / "resource_ledger.json", budget.evidence())
    budget.write_json(output / "replay.json", dict(status="pending_independent_process", tolerance=1e-10))
    (output / "report.md").write_text(report(decision, metrics, gates), encoding="utf-8")
    budget.check()
    return decision


def report(decision, metrics, gates):
    lines = ["# Frontier bridge evidence", "", decision["reason"], "",
             "| Direction | Candidate MSE | Locked selector MSE | Supported | Synthetic gate checks |",
             "| --- | ---: | ---: | --- | --- |"]
    for d, m in metrics.items():
        lines.append(f"| {d} | {m['candidate']['mse']:.10g} | {m['selector']['mse']:.10g} | {gates[d]['checks']['adaptation_supported']} | {gates[d]['passed']} |")
    lines += ["", "All arms and attempted fits are retained. Intervals condition on fixed labels, reference estimates, source selection and bridge fitting. They are target-incidence sensitivity ranges, not biological confidence intervals.", "", "No source-derived scientific program or independent reference partition was frozen; utility and independent-reference attribution are unavailable. Synthetic evidence cannot advance the public or biological gate.", ""]
    return "\n".join(lines)


def replay(manifest_path, output, budget, verify_compatible_code=False):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    freeze = read_json(manifest_path.parent / "prediction_freeze.json")
    if sha256(manifest_path) != freeze["run_manifest_sha256"]:
        raise ContractError("Run manifest changed after prediction freeze")
    changed_code = {}
    for path, digest in manifest["code_hashes"].items():
        if sha256(path) != digest:
            changed_code[path] = dict(frozen=digest, current=sha256(path))
    if changed_code and not verify_compatible_code:
        raise ContractError("Inference code changed after freeze; compatible-code verification must be explicit")
    predictions = load_npz(manifest["predictions"])
    errors = []
    for checkpoint in manifest["checkpoints"]:
        budget.check()
        state = load_state(checkpoint)
        actual = predict_checkpoint(state)
        expected = predictions[checkpoint["prediction_key"]]
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ContractError("Replay prediction shape/nonfinite mismatch")
        error = float(np.max(np.abs(actual - expected)))
        if error > 1e-10:
            raise ContractError(f"Independent replay exceeded tolerance: {error}")
        errors.append(error)
    # Restoration verifies every arm and fit identity, while the immutable
    # shared predictions remain referenced rather than copied a second time.
    budget.write_json(Path(output) / "predictions_manifest.json", manifest["predictions"])
    receipt = dict(status="passed", max_absolute_error=max(errors, default=0.), tolerance=1e-10,
                   checkpoints=len(errors), evaluator_expression_reopened=False,
                   run_manifest_sha256=sha256(manifest_path), separate_process=True,
                   compatible_code_verification=verify_compatible_code,
                   changed_code=changed_code, exact_frozen_code=not changed_code)
    budget.write_json(Path(output) / "replay.json", receipt)
    budget.write_json(Path(output) / "resource_ledger.json", budget.evidence())
    return receipt
