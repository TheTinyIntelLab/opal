"""Reproduce a metadata-only audit; never load expression or infer initiation."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS = ROOT / "data/source_audits/replicated_fixed_state_v1/source_snapshots/public-search-20261009-v1"
SOURCE = SNAPSHOTS / "sciPlex_4_hash_sample_sheet.txt.gz"


def audit():
    retrieval = json.loads((Path(__file__).parent / "sciplex4_design.json").read_text())
    raw = SOURCE.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != retrieval["sha256"]:
        raise ValueError("Hash sample sheet differs from recorded retrieval")
    # Author Notebook_1_cds_cleanup assigns field 2 to trametinib and field 4
    # to the named compound; GEO's prose describes the opposite ordering.
    rows = []
    for line in gzip.decompress(raw).decode("utf-8").splitlines():
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) != 3:
            raise ValueError("Expected label, oligonucleotide, and axis count")
        fields = columns[0].split("_")
        if len(fields) != 5:
            raise ValueError("Unexpected treatment label format")
        plate, trametinib, compound, dose, replicate = fields
        rows.append((plate, float(trametinib), compound, float(dose), replicate, columns[1]))
    if len({r[5] for r in rows}) != len(rows):
        raise ValueError("Hash oligonucleotide is not unique")
    replicates = sorted({r[4] for r in rows})
    support = []
    for rep in replicates:
        subset = [r for r in rows if r[4] == rep]
        controls = [r for r in subset if r[2] == "vehicle" and r[1] == 0 and r[3] == 0]
        compound_singles = {(r[2], r[3]) for r in subset if r[2] != "vehicle" and r[1] == 0 and r[3] > 0}
        trametinib_singles = {r[1] for r in subset if r[2] == "vehicle" and r[1] > 0 and r[3] == 0}
        pairs = {(r[2], r[3], r[1]) for r in subset if r[2] != "vehicle" and r[1] > 0 and r[3] > 0}
        missing = sorted(p for p in pairs if (p[0], p[1]) not in compound_singles or p[2] not in trametinib_singles or not controls)
        support.append({"replicate_label": rep, "barcode_rows": len(subset), "control_barcodes": len(controls), "compound_single_conditions": len(compound_singles), "trametinib_single_doses": sorted(trametinib_singles), "pair_conditions": len(pairs), "pairs_missing_reference_labels": missing})
    out = {
        "source_sha256": actual,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expression_matrix_accessed": False,
        "biological_initiation_established": False,
        "semantics": "Authors' Notebook_1_cds_cleanup: plate, trametinib dose, named compound, compound dose, replicate label",
        "geo_description_conflicts_with_author_code": True,
        "rows": len(rows),
        "replicate_labels": replicates,
        "plate_by_replicate_label": dict(collections.Counter(f"{r[0]}/{r[4]}" for r in rows)),
        "support": support,
        "primary_confirmation_qualified": False,
        "boundary": "Two replicate labels do not establish six independent culture blocks. Barcode support is design support, not evidence of measured cells, independent initiation, precision, or power. No labels are repaired.",
    }
    (Path(__file__).parent / "sciplex4_support_audit.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    audit()
