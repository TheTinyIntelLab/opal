"""Freeze and profile the native scalar comparator using synthetic data only."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.naiad_reference import NativeConfig, NativeNAIAD, REVISION, SOURCE_HASHES, file_hash, fit_native

CODE = ["src/naiad_reference.py", "tools/qualify_naiad.py", "tests/test_naiad_reference.py",
        "docs/research_reset/naiad_protocol.md"]
DEPENDENCIES = ["torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
                "seaborn", "tqdm", "openpyxl", "psutil", "threadpoolctl"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--genes", type=int, default=600)
    p.add_argument("--pairs", type=int, default=144601)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = NativeConfig()
    n_train, n_val = int(args.pairs * .8), int(args.pairs * .15)
    n_test = args.pairs - n_train - n_val
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
                        scope="synthetic only; no biological targets accessed",
                        config=asdict(config), revision=REVISION, source_hashes=SOURCE_HASHES,
                        code_hashes={name: file_hash(ROOT / name) for name in CODE},
                        dependencies={name: importlib.metadata.version(name) for name in DEPENDENCIES},
                        python=sys.version, platform=platform.platform(),
                        profile=dict(genes=args.genes, pairs=args.pairs, train=n_train, val=n_val, test=n_test,
                                     max_wall_seconds=300, max_rss_bytes=2 * 1024**3,
                                     threads=1, seed=20261022, epochs=config.epochs))
    (args.output / "registration.json").write_text(json.dumps(registration, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, "-m", "pytest", "tests/test_naiad_reference.py", "-q"]
    receipt = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (args.output / "prerequisite_tests.txt").write_text(receipt.stdout + receipt.stderr, encoding="utf-8")
    if receipt.returncode:
        raise RuntimeError("Native prerequisites failed; synthetic profile not run")
    torch.set_num_threads(1)
    process, start, peak = psutil.Process(), time.perf_counter(), [0]
    def check():
        rss = process.memory_info().rss
        peak[0] = max(peak[0], rss)
        if time.perf_counter() - start > 300 or rss > 2 * 1024**3:
            raise RuntimeError("Synthetic NAIAD resource cap exceeded")
    genes = np.array([f"synthetic_{i:03}" for i in range(args.genes)])
    rng = np.random.default_rng(20261022)
    all_pairs = np.array([(a, b) for a in range(args.genes) for b in range(a + 1, args.genes)])
    pairs = all_pairs[rng.choice(len(all_pairs), args.pairs, replace=False)]
    singles = rng.normal(0, .2, args.genes)
    targets = singles[pairs].sum(1) + .1 * np.sin(pairs.sum(1))
    split = np.repeat([0, 1, 2], [n_train, n_val, n_test])
    targets[split == 2] = np.nan
    with threadpool_limits(limits=1):
        model = fit_native(args.source, genes, pairs, singles, targets, split, config, check=check)
        fit_seconds = time.perf_counter() - start
        prediction = model.predict(pairs[split == 2])
        model.save(args.output / "synthetic_model")
        loaded = NativeNAIAD.load(args.source, args.output / "synthetic_model")
        replay = loaded.predict(pairs[split == 2])
        np.testing.assert_array_equal(prediction, replay)
        check()
    result = dict(scope="synthetic software/resource qualification, not biological performance",
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  fit_seconds=fit_seconds, total_seconds=time.perf_counter() - start,
                  peak_rss_bytes=peak[0], parameter_count=model.parameter_count,
                  model_args=model.model_args, config=asdict(config),
                  best_epoch=model.best_epoch, exact_numerical_replay=True,
                  metrics_keys=list(model.training_metrics),
                  estimated_30_fit_seconds=30 * fit_seconds,
                  passes_30_fit_one_hour_estimate=30 * fit_seconds <= 3600,
                  source_worktree_bytes=sum(f.stat().st_size for f in args.source.rglob("*") if f.is_file()),
                  prerequisite_returncode=receipt.returncode,
                  code_hashes=registration["code_hashes"])
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
