"""Reproduce prebiology final software receipt and metadata-only budget proposal."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from opal.naiad_reference import NativeConfig, REVISION, SOURCE_HASHES, file_hash, load_official

OUT = Path(__file__).resolve().parent
SOURCE = Path("C:/Users/USER/Desktop/opal-research-next-20261020/NAIAD")
METADATA = Path("C:/Users/USER/Desktop/opal-research-next-20261020/block-v1/protocol.json")
CODE = ["src/naiad_reference.py", "tools/qualify_naiad.py", "tests/test_naiad_reference.py",
        "docs/research_reset/naiad_protocol.md",
        "results/research-reset-naiad-qualification/freeze_final.py"]


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    load_official(SOURCE)
    profile = json.loads((OUT / "v2/result.json").read_text())
    metadata = json.loads(METADATA.read_text())
    folds, total_rows, total_batches = {}, 0, 0
    for task, panel in metadata["panels"].items():
        folds[task] = []
        for fold in panel["folds"]:
            train, val = fold["train_pairs"], fold["validation_pairs"]
            rows = train + val
            batches = math.ceil(train / 1024) + math.ceil(val / 1024)
            total_rows += rows
            total_batches += batches
            folds[task].append(dict(train=train, validation=val, test=fold["test_pairs"],
                                    rows=rows, batches_per_epoch=batches))
    profile_rows, profile_batches = 115680 + 21690, math.ceil(115680 / 1024) + math.ceil(21690 / 1024)
    row_estimate = profile["fit_seconds"] * total_rows / profile_rows
    batch_estimate = profile["fit_seconds"] * total_batches / profile_batches
    budget_estimate = 1.25 * max(row_estimate, batch_estimate) + 30 * 2
    amendment = dict(id="naiad-native-resource-v3-prebiology", status="proposal for parent registration; no biological fits",
                     written_utc=datetime.now(timezone.utc).isoformat(),
                     source_metadata_sha256=file_hash(METADATA), source_metadata_path=str(METADATA),
                     profile_sha256=file_hash(OUT / "v2/result.json"),
                     folds=folds, total_train_validation_rows=total_rows,
                     total_batches_per_epoch_across_fits=total_batches,
                     estimates_seconds=dict(row_scaled=row_estimate, batch_scaled=batch_estimate,
                                            conservative_margin_and_setup=budget_estimate,
                                            thirty_largest=profile["fit_seconds"] * 30),
                     method="Scale measured max-profile fit by total rows and minibatches; take larger estimate, add25% and2s setup perfit. Not a measured complete run.",
                     proposed_native_only_wall_seconds=7200, rss_bytes=2 * 1024**3, threads=1,
                     fits_within_proposed_cap=budget_estimate <= 7200,
                     unchanged_method=dict(epochs=100, seed=20261022, batch_size=1024,
                                           native_defaults=True, no_biological_capacity_selection=True),
                     stop="Stop and record incomplete full roster at cap; no epoch/width reduction or outcome-based subset selection.")
    write("resource_amendment.json", amendment)
    command = [sys.executable, "-m", "pytest", "tests/test_naiad_reference.py", "-q"]
    test = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (OUT / "final_prerequisite_tests.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
    packages = json.loads((OUT / "v2/registration.json").read_text())["dependencies"]
    receipt = dict(id="naiad-native-final-prebiology-v3", written_utc=datetime.now(timezone.utc).isoformat(),
                   scope="synthetic prerequisite tests and metadata-only resource amendment; zero biological fits",
                   revision=REVISION, canonical_lf_source_hashes=SOURCE_HASHES,
                   actual_checkout_hashes={name: file_hash(SOURCE / name) for name in SOURCE_HASHES},
                   code_hashes={name: file_hash(ROOT / name) for name in CODE},
                   dependencies={name: importlib.metadata.version(name) for name in packages},
                   python=sys.version, tests_command=command, tests_returncode=test.returncode,
                   tests_output_sha256=file_hash(OUT / "final_prerequisite_tests.txt"),
                   profile_scope="Measured v2 predates final train-row permutation/queried-single validation/newline portability changes; exact old bytes preserved v2/frozen. Native architecture/trainer/epochs/shapes unchanged; timing remains an estimate for final wrapper.",
                   resource_amendment_sha256=file_hash(OUT / "resource_amendment.json"))
    write("final_registration.json", receipt)
    if test.returncode:
        raise RuntimeError("Final native prerequisites failed")
    print(json.dumps(amendment["estimates_seconds"], indent=2))


if __name__ == "__main__":
    main()
