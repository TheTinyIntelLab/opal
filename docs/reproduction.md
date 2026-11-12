# Reproducibility

A public checkout supports code tests, synthetic engineering checks and aggregate evidence replay. Biological replay also requires exact source data or preserved local bundles, which are not included.

The active tree keeps current ledgers and runtime-required contracts. Before reproducing an older workflow, restore its original documents and result records from the [scientific archive](archive/README.md) into a separate source copy. This preserves the old paths and hashes without repopulating the active checkout with historical clutter.

The full historical collection is distributed with the GitHub research source/evidence, rather than the lightweight Python wheel or source distribution. Use the repository archive for historical verification, and the Python packages for installation.

## Source-free verification

Install [the dependencies](installation.md), then run:

```bash
python -m pytest -q
python tools/report_trained_growth_benchmark.py --source results/frontier-execution-20261026/growth-v3/results.json --output tmp/benchmark-replay.json
python -m build
```

Use a fresh replay output path. The report verifies the frozen ledger and projects existing aggregate results without fitting or rescoring. Tests use small synthetic fixtures, not new biological experiments. Optional pinned NAIAD source is not bundled and its tests may skip explicitly. The frozen v1 gzip defect remains an expected failure with successor regression coverage.

Several tests enforce at least a 1 GiB disk reserve. Put pytest temporary output on a sufficiently large volume, for example `python -m pytest -q --basetemp PATH_ON_LARGE_VOLUME`. Keep registered reserves unchanged, and use a fresh path rather than a scientific result directory.

## Biological prediction replay

The retained growth bundle can be checked without training or opening test targets:

```bash
python tools/verify_trained_growth_delivery.py --bundle PATH_TO_TRAINED_GROWTH --source-run PATH_TO_ORIGINAL_GROWTH_V3 --output tmp/growth-api-replay.json --expected-manifest-sha256 fa8490abe83e5a5d0d0f8456639d11df16a2a6ca3ed440fb40bfd74459488e72
```

This receipt identifies the October 27 bundle. The original run tree supplies saved predictions, selections and hashes. All six checkpoints and 17,176 predictions replayed exactly during v1 preparation, with no new fits or scores. Replay does not establish biological replication or transfer.

[Source preparation](model_development_20261020.md), [growth registration](frontier_execution/growth_registration.md), [delivery registration](model_delivery/registration.md) and [historical training reproduction](model_delivery/training_reproduction.md) retain full prerequisites. Their training commands are historical evidence, not release checks.

## Paper and packaging

Use an existing Tectonic installation:

```bash
python tools/build_paper.py --tectonic PATH_TO_TECTONIC --cache-directory PATH_TO_CACHE
python tools/build_arxiv_submission.py --config configs/releases/v1.0.0.json --tectonic PATH_TO_TECTONIC --cache-directory PATH_TO_CACHE --output releases/arxiv/opal-1.0.0-arxiv-source.zip
```

These commands submit nothing. The [v1 recipe](../configs/releases/v1.0.0.json) excludes weights. Build packages from an explicit source snapshot or clean checkout because setuptools otherwise collects untracked Python modules. The release packager reads Git-index membership; `--allow-dirty` marks review assets.

The [validation receipt](../results/release-v1.0.0/validation.json) records actual checks. Historical receipts retain their original cutoffs; the old guide is in [the archive](archive/README.md).
