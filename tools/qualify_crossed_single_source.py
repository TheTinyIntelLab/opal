"""Outcome-blind, byte-bounded qualification of the PLAN's Tahoe single-response lead.

Fetches only the pinned card, sample catalog and cell-line catalog. It never
opens expression, differential expression, observation cells or model weights.
Catalog context labels do not certify a measured crossed condition graph.
"""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

DATASET = "tahoebio/Tahoe-100M"
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 4_000_000


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def assignments(value):
    """Parse literal author assignments without evaluating source code."""
    try:
        rows = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(rows, (list, tuple)) or len(rows) != 1:
            return None
        drug, dose, unit = rows[0]
        if not isinstance(drug, str) or not drug or not isinstance(unit, str) or not unit:
            return None
        if isinstance(dose, bool) or not isinstance(dose, (int, float)) or not math.isfinite(dose) or dose < 0:
            return None
        return dict(intervention_id=drug, dose=float(dose), dose_unit=unit)
    except (ValueError, TypeError, SyntaxError, OverflowError):
        return None


def summarize(rows, context_ids):
    seen, duplicate_ids, invalid, controls, conditions = set(), set(), [], {}, []
    blocks = defaultdict(set)
    for row in rows:
        sample = str(row.get("sample") or "")
        plate_value = row.get("plate")
        plate = "" if plate_value is None else str(plate_value)
        if sample in seen:
            duplicate_ids.add(sample)
        seen.add(sample)
        if row.get("drug") == "DMSO_TF" and sample and plate:
            controls.setdefault(plate, []).append(sample)
            continue
        literal = assignments(row.get("drugname_drugconc"))
        if not sample or not plate or literal is None or literal["dose"] <= 0:
            invalid.append(sample)
            continue
        signature = json.dumps(literal, sort_keys=True)
        blocks[signature].add(plate)
        conditions.append(dict(sample_id=sample, plate_id=plate, **literal))
    for row in conditions:
        row.update(control_sample_ids=sorted(controls.get(row["plate_id"], [])),
                   treatment_duration=None, duration_unit=None, initiation_id=None,
                   culture_id=None, measured_context_ids=None,
                   exposure="public catalog; no expression accessed in this audit")
    reasons = ["measured_context_condition_edges_unverified", "duration_unverified",
               "independent_initiation_unverified", "condition_cell_support_unverified",
               "exact_vehicle_equivalence_unreviewed", "byte_qualified_expression_subset_absent"]
    if duplicate_ids:
        reasons.append("duplicate_sample_ids")
    if invalid:
        reasons.append("invalid_literal_assignments")
    if any(not r["control_sample_ids"] for r in conditions):
        reasons.append("missing_plate_vehicle")
    return dict(schema="crossed_single_source_qualification_v1", task="single_response_by_context",
        qualified=False, status="metadata_only_unqualified", reasons=reasons,
        sample_catalog_rows=len(rows), literal_single_sample_rows=len(conditions),
        unique_literal_single_assignments=len(blocks), catalog_context_ids=sorted(set(context_ids)),
        catalog_context_count=len(set(context_ids)), measured_crossed_edge_count=None,
        independently_initiated_culture_count=None, duplicate_sample_ids=sorted(duplicate_ids),
        invalid_sample_ids=sorted(invalid), vehicle_sample_ids_by_plate=controls,
        repeated_assignment_plate_groups={k: sorted(v) for k,v in blocks.items() if len(v)>1},
        conditions=conditions, expression_accessed=False, source_scored=False,
        biological_confirmation=False, pair_composition_claim=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.monotonic(), time.process_time()
    import requests
    import pyarrow.parquet as pq
    import psutil
    fetched, used = [], 0
    def fetch(url, target, expected_bytes=None, expected_sha256=None):
        nonlocal used
        response = requests.get(url, stream=True, timeout=(15, 30))
        response.raise_for_status()
        size = 0
        with target.open("xb") as file:
            for chunk in response.iter_content(65536):
                size += len(chunk)
                used += len(chunk)
                if size > MAX_FILE_BYTES or used > MAX_TOTAL_BYTES:
                    raise ValueError("Metadata byte stop limit; partial file retained")
                file.write(chunk)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if expected_bytes is not None and size != expected_bytes:
            raise ValueError("Pinned metadata size mismatch")
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("Pinned metadata hash mismatch")
        fetched.append(dict(url=url, path=target.name, bytes=size, sha256=digest))
        return target
    try:
        api = "https://huggingface.co/api/datasets/"+DATASET
        revision = json.loads(fetch(api+"/revision/main",args.output/"revision.json").read_text())["sha"]
        tree = json.loads(fetch(api+"/tree/"+revision+"/metadata", args.output/"metadata_tree.json").read_text())
        catalog = {x["path"]: x for x in tree if x["type"] == "file"}
        for name in ("sample_metadata", "cell_line_metadata"):
            entry = catalog["metadata/"+name+".parquet"]
            if entry["size"] > MAX_FILE_BYTES:
                raise ValueError("Catalog exceeds declared metadata ceiling")
            fetch("https://huggingface.co/datasets/"+DATASET+"/resolve/"+revision+"/"+entry["path"],
                  args.output/(name+".parquet"), entry["size"], entry.get("lfs",{}).get("oid"))
        card = fetch("https://huggingface.co/datasets/"+DATASET+"/raw/"+revision+"/README.md", args.output/"source_card.md")
        rights = "license: cc0-1.0" in card.read_text(encoding="utf-8").split("---",2)[1]
        samples = pq.read_table(args.output/"sample_metadata.parquet", columns=["sample","plate","drug","drugname_drugconc"]).to_pylist()
        contexts = pq.read_table(args.output/"cell_line_metadata.parquet", columns=["Cell_ID_Cellosaur"]).column(0).to_pylist()
        result = summarize(samples, contexts)
        result.update(dataset=DATASET, pinned_revision=revision, rights_verified=rights,
            rights_scope="CC0 author-posted data catalog; exact expression subset remains absent",
            asset_sizes={k:v["size"] for k,v in catalog.items()})
        if not rights:
            result["reasons"].append("rights_unresolved")
        write(args.output/"qualification.json", result)
        write(args.output/"decision.json", {k:v for k,v in result.items() if k not in ("conditions", "repeated_assignment_plate_groups", "vehicle_sample_ids_by_plate")})
        write(args.output/"source_receipts.json", dict(fetched=fetched, revision=revision,
             excluded_assets=["expression_data","obs_metadata","pseudobulk_differential_expression","weights"]))
        print(json.dumps(dict(status=result["status"],samples=len(samples),
            catalog_contexts=result["catalog_context_count"],bytes_fetched=used)))
    except Exception as error:
        write(args.output/"failure.json",dict(status="qualification_failed",reason=str(error),partial_assets_retained=True))
        raise
    finally:
        write(args.output/"resource_ledger.json",dict(cpu_seconds=time.process_time()-cpu,
            wall_seconds=time.monotonic()-start, current_rss_bytes=psutil.Process().memory_info().rss,
            metadata_download_bytes=used, metadata_download_limit_bytes=MAX_TOTAL_BYTES,
            retained_bytes=sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()),
            retained_bytes_excludes_this_ledger=True,
            scope="metadata audit process only; no scientific fitting", installs=0,gpu_seconds=0,
            raw_expression_download_bytes=0,numerical_threads=1))


if __name__ == "__main__":
    main()
