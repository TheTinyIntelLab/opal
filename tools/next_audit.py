"""Read-only evidence inventory, ceiling checks, data gates and native probes."""
from pathlib import Path
import subprocess
import sys
import zipfile
from itertools import combinations
import numpy as np
from opal.artifacts import environment, load_npz, read_json, sha256, write_json


def archive_evidence(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected = []
    for folder in ["results/runs", "results/synthetic", "results/robustness", "results/improvements", "results/architecture"]:
        selected.extend(p for p in (root / folder).rglob("*") if p.is_file() and
                        (p.name in {"protocol.json", "selection.json", "final_status.json", "test_access.json", "test_results.json", "source_snapshot.zip"} or
                         p.parent == root / folder and p.suffix == ".json"))
    for folder in ["configs", "docs", "paper"]:
        selected.extend(p for p in (root / folder).rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".tex", ".bib"}
                        and p.name != "opal_next_master_plan.md")
    for dataset in (root / "data/prepared").iterdir():
        if dataset.is_dir():
            selected.extend(p for p in dataset.glob("*.json"))
            if (dataset / "profiles.npz").exists():
                selected.append(dataset / "profiles.npz")
    selected = sorted(set(selected))
    files = {str(p.relative_to(root).as_posix()): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in selected}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=True).stdout
    with zipfile.ZipFile(output / "metadata_archive.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in selected:
            if p.suffix != ".npz":
                archive.write(p, p.relative_to(root).as_posix())
    manifest = {"git_revision": revision, "worktree_status_at_inventory": status, "files": files,
                "archive_sha256": sha256(output / "metadata_archive.zip"), "historical_evaluator_responses_opened": False,
                "scope": "retrospective metadata/source archive and allowed-profile fingerprints; not a claim of a prospective historical freeze",
                "large_array_policy": "existing arrays retained locally; inventory hashes are not an independent backup"}
    write_json(output / "manifest.json", manifest)
    return {"inventoried_files": len(files), "archive_sha256": manifest["archive_sha256"], "status": "complete"}


def audit_ceiling(root):
    from tools.report_ceiling import verify
    records = []
    for dataset in ["norman-log2", "wessels-log2", "wessels-expanded-protected-log2"]:
        for version in ["joint-uncertainty-v1", "joint-uncertainty-v2", "joint-uncertainty-v3", "retuned-curves-v2"]:
            folder = Path(root) / "results/architecture" / version / dataset
            if not folder.exists():
                records.append({"dataset": dataset, "study": version, "status": "unavailable"})
                continue
            try:
                result = verify(folder, dataset, {})
                checks = 0
                for kind in ["random", "partner"]:
                    for record in result[kind]["records"]:
                        if "train" in record and set(record["train"]) & set(record["validation"]):
                            raise ValueError("Fitting/validation overlap")
                        if record.get("status") == "complete" and record.get("fraction") == 1:
                            if not np.isclose(record["mse"]["frozen_full_size_kernel"], record["mse"]["retuned_unweighted"], rtol=1e-10, atol=1e-12):
                                raise ValueError("Full-size curve controls differ")
                        checks += 1
                    for record in result[kind].get("error_radius_diagnostics", []):
                        if record["status"] != "complete":
                            continue
                        fit, cal, val = (set(record[k]) for k in ["fit", "calibration", "outer_validation"])
                        if fit & cal or fit & val or cal & val or record["selection_uses_calibration_responses"]:
                            raise ValueError("Calibration isolation failed")
                        for radius in record["radii"]:
                            if radius["finite_radius"] and (radius["radius"] is None or not np.isfinite(radius["radius"])):
                                raise ValueError("Invalid finite radius")
                if "export_verification" in result and result["export_verification"]["maximum_absolute_difference"] > 1e-6:
                    raise ValueError("Saved covariance checkpoint disagrees with fitting model")
                records.append({"dataset": dataset, "study": version, "status": "passed", "fold_records_checked": checks})
            except Exception as error:
                records.append({"dataset": dataset, "study": version, "status": "failed", "error": f"{type(error).__name__}: {error}"})
    return {"records": records, "historical_evaluator_responses_opened": False,
            "scope": "input/source hashes, saved split invariants, full-size controls and existing export agreement; numerical regression tests reported separately"}


def data_gate(root):
    """Inspect allowed development references and source metadata, not test labels."""
    records, canonical = [], []
    for name in ["norman-log2", "wessels-log2", "wessels-expanded-protected-log2", "chemical-exploratory"]:
        folder = Path(root) / "data/prepared" / name
        if not (folder / "profiles.npz").exists():
            continue
        data, source = load_npz(folder / "profiles.npz"), read_json(folder / "source_manifest.json")
        condition_map = source.get("condition_map", {})
        contexts = sorted({r.get("context", "unspecified") for r in condition_map.values()})
        if name.startswith("norman"):
            state, modality, study = "K562", "CRISPRa", "Norman2019"
        elif name.startswith("wessels"):
            state, modality, study = "THP1", "Cas13_RNA_targeting", "Wessels2023"
        else:
            state, modality, study = "A549", "chemical_1uM_24h", "ComboSciPlex"
        record = {"dataset": name, "state": state, "modality": modality, "study": study, "contexts": contexts,
                  "observed_singles": len(data["targets"]), "fitting_pairs": len(data["pairs"]), "query_identities": len(data["query_pairs"]),
                  "genes": len(data["genes"]), "source_manifest_sha256": sha256(folder / "source_manifest.json"),
                  "derived_track_not_independent": "expanded" in name, "test_outcomes_accessed": False,
                  "targets": data["targets"].tolist(), "development_pair_identities": data["targets"][data["pairs"]].tolist()}
        records.append(record)
        if not record["derived_track_not_independent"]:
            canonical.append(record)
    overlaps = []
    for a, b in combinations(canonical, 2):
        pa, pb = ({tuple(sorted(p)) for p in r["development_pair_identities"]} for r in [a, b])
        overlaps.append({"datasets": [a["dataset"], b["dataset"]], "shared_targets": len(set(a["targets"]) & set(b["targets"])),
                         "shared_development_pairs": len(pa & pb), "same_modality": a["modality"] == b["modality"]})
    eligible_groups = {}
    for r in canonical:
        eligible_groups.setdefault(r["modality"], set()).add(r["state"])
    accepted = any(len(contexts) >= 3 for contexts in eligible_groups.values()) and any(r["same_modality"] and r["shared_development_pairs"] for r in overlaps)
    pilots = []
    folder = Path(root) / 'data/prepared/emeraldbay-pilot-v1'
    if (folder / 'protocol.json').exists():
        from opal.contextual.contracts import load_dataset
        bank, fitting, query, protocol = load_dataset(folder)
        pilots.append({'dataset': 'emeraldbay-pilot-v1', 'modality': 'chemical',
                       'contexts': bank.arrays['contexts'].tolist(), 'held_contexts': protocol['held_contexts'],
                       'fitting_conditions': len(fitting['pairs']), 'query_conditions': len(query['pairs']),
                       'protocol_sha256': sha256(folder / 'protocol.json'), 'synthetic': bank.metadata['synthetic'],
                       'status': 'eligible_exploratory_observed_single_transfer', 'independent_confirmation': False,
                       'test_outcomes_accessed': False})
    return {"status": "exploratory_chemical_transfer_available" if pilots else ("requires_source_review" if accepted else "blocked_no_qualified_crossed_context_combination_corpus"),
            "accepted_for_context_transfer": False, "sources": records, "overlap": overlaps,
            "exploratory_context_transfer_pilots": pilots,
            "reasons": ["Current independent sources change modality with biological context.",
                        "Expanded Wessels tracks are additional coverage of the same experiment, not new contexts.",
                        "Chemical source has only seven matched-single pairs and one context.",
                        "Cross-context combination outcomes and biological replication remain required."],
            "fallback": "controlled shared-modality multi-context screen specification; single-screen transfer remains a separate task"}


def native_probe(root):
    """Fresh import/runtime evidence; never report a native score from an adaptation."""
    registry = read_json(Path(root) / "data/public_method_registry.json")
    records = {}
    for name, module in [("GEARS", "gears"), ("CPA", "cpa")]:
        directory = Path(root) / "tools/external" / name
        code = f"import sys; sys.path.insert(0, {str(directory)!r}); import {module}"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        records[name] = {"revision": registry[name]["revision"], "import_exit_code": result.returncode,
                         "diagnostic": result.stderr[-4000:], "native_scores": None,
                         "status": "importable_adapter_not_executed" if result.returncode == 0 else "runtime_blocked"}
    import torch
    records["PertAdapt"] = {"revision": registry["PertAdapt"]["revision"], "cuda_available": torch.cuda.is_available(),
                            "status": "native_cuda_execution_unavailable" if not torch.cuda.is_available() else "adapter_not_executed",
                            "native_scores": None, "resources": "supplied checkpoint/GO/gene-order resources; see existing resource audit"}
    return {"environment": environment(), "methods": records, "none_of_these_are_native_benchmark_scores": True}
