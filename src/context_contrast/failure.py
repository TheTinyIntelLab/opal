"""Complete, honest receipts for refused source qualification and partial attempts."""
from pathlib import Path
from ..artifacts import read_json, write_json, fingerprint
from .contracts import PROTOCOL, Unsupported
from .evaluation import five_gates


def record_failure(output, operation, accepted, error, budget):
    output = Path(output)
    path = output/"qualification.json"
    qualification = read_json(path) if path.exists() else dict(qualified=False, status="not_qualified")
    refusal = path.exists() and isinstance(error, Unsupported) and qualification.get("qualified") is False
    write = getattr(budget, "write_failure_json", budget.write_json)
    reason = str(error)
    # Authorization and scientific qualification are separate facts. Owner
    # acceptance passed before the operation; source evidence may still fail.
    decision = five_gates(dict(qualification, numerical_acceptance_verified=True), {}, {}, [])
    decision.update(status="source_ineligible" if refusal else "failed_partial_attempt",
        owner_execution_authorized=True, acceptance_hash=fingerprint(accepted),
        operation=operation, reason=reason, source_status=qualification.get("status"),
        fitting_performed=False if operation == "prepare" else None,
        biological_scoring_performed=False if operation in ("prepare", "predict") else None,
        panel_selected=(output/"panel.json").exists(),
        qualified_pair_count=len(qualification.get("complete_pairs", [])),
        partial_assets_retained=True, scientific_negative_established=False)
    write(output/"protocol.json", PROTOCOL)
    write(output/"failure.json", dict(operation=operation, reason=reason,
        classification=decision["status"], experiment_id=PROTOCOL["experiment_id"],
        owner_execution_authorized=True, partial_assets_retained=True, experiment_completed=False))
    for name in ("folds", "source_selection", "rank_support", "metric_qualification", "metrics", "controls", "uncertainty", "replay", "predictions"):
        if not (output/(name+".json")).exists():
            write(output/(name+".json"), dict(status="not_run", reason=reason,
                population_scored=0, owner_execution_authorized=True))
    write(output/"decision.json", decision)
    stop = "The scientific qualification gate stopped this attempt. " if refusal else "The operation stopped with a partial attempt. "
    report = ("# Authorized context-contrast execution\n\nStatus: "+decision["status"]+
        ". Owner execution authorization was accepted.\n\n"+reason+
        "\n\n"+stop+"Existing source files and every partial artifact remain preserved. "
        "No model success, biological failure, independent confirmation or frontier superiority is established.\n")
    (output/"report.md").write_text(report, encoding="utf-8")
    budget.receipt(decision["status"])
    return decision
