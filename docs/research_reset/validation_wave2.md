# Wave-two delivery validation

The current full suite passes with 588 tests and four warnings in 142.31 seconds. Package build and installed prediction replays also pass at the preflight stage. Report-aware rebuilding, all-branch installed replay and the 132-document build pass with zero unavailable local links. Remote synchronization and hosted CI are checked after the final push and reported separately. [validation.json](../../results/research-wave2-validation/validation.json) binds the actual local artifacts, logs and receipts by SHA256.

## Tests and preserved failure

The first complete suite produced 585 passes, two failures and four warnings in 153.66 seconds. Both failures were sidecar synthetic lifecycle tests whose real-process peak-memory measurement inherited earlier ML-test allocations. Their small standalone scientific process was not being represented correctly inside the large pytest process. The successor isolates memory measurement in those synthetic fixtures and adds an explicit test that production memory, disk and artifact guards still reject excess usage. The standalone sidecar's scientific code and resource limits are unchanged, and its new registered run/replay reproduces its original scientific artifacts exactly.

The subsequent full suite produced 588 passes, zero failures, zero skips and four warnings. Its log reports 142.31 seconds; JUnit records 142.289 seconds. The warnings are the preserved official NAIAD warmup-step integer coercion warning, not new model failures. Both original and successful JUnit/log pairs remain at `C:/Users/USER/Desktop/opal-wave2-20261021/`, named `full-tests.*` and `full-tests-final.*`. Their exact hashes and original failure counts are retained in the validation receipt rather than replacing the negative engineering evidence.

## Preflight build and archive inspection

The preflight build produced `opal_research-0.1.3.dev0-py3-none-any.whl`, 273,622 bytes with 105 members, and `opal_research-0.1.3.dev0.tar.gz`, 1,315,329 bytes with 631 members. The saved build log contains the successful wheel/sdist marker. Setuptools license-table deprecation and expected unmatched MANIFEST/exclusion warnings remain visible. These artifacts precede final report updates and are not being represented as the final documentation-complete distribution.

Archive-member inspection found no root data, results, released artifacts, external checkout, numerical bank, checkpoint/weight or outreach-register assets. The wheel's `opal/data/*.py` files and sdist's `src/data/*.py` and `configs/releases/*.json` are implementation/configuration, not biological data or released weights. Member-name manifest hashes and archive SHA256 values are recorded in the receipt. This is a concrete packaging inclusion check; a final working-tree credential/content review is still required before delivery.

## Installed and synthetic replay

The wheel was installed to a separate target directory without dependencies, and replay receipts confirm imports from that installed package. The host dependency closure was reused, so these checks establish package-only API reproducibility rather than a clean dependency installation.

| Check | Measured outcome | Scope |
| --- | --- | --- |
| Historical reset replay | Passed; frozen external and development composite prediction differences zero; original test differences at most 5.55e-17 | Historical evidence and artifact/prerequisite order remain reproducible |
| Block/native replay | Exact arrays for all six tasks | Existing Q2/Q3 predictions reproduce from installed package |
| Wave-two replay | Q1b nine prepared arrays, control-partner results and all six Q2b tasks exact | This preflight receipt excludes the not-yet-completed Q3b budget branch |
| Synthetic assay rehearsal | All six commands return zero; lineage, query blindness, reference transforms and restoration pass | Engineering test with original integer synthetic counts, not independent biological confirmation |
| Installed synthetic series v1 and v2 | All three model CLI train/predict/evaluate cycles pass; API/CLI and source-refit prediction differences zero | Installed package behavior on synthetic fixtures, not biological model promotion |

The wave-two installed replay took 76.24482859999989 wall seconds and 76.03125 CPU seconds, with 403,521,536 peak bytes. These delivery checks are distinct from each experiment's preregistered scientific-run resource ledger. The assay and series receipts explicitly mark synthetic data and do not claim scientific confirmation. SPIDR responses remain unopened.

## Reproduce

Use fresh output paths to preserve prior receipts. The original output root is `C:/Users/USER/Desktop/opal-wave2-20261021`; replace `-new` names below with a new run identifier. Source and numerical run paths are hash-bound in their registrations.

```powershell
.venv/Scripts/python.exe -m pytest -q --basetemp C:/Users/USER/Desktop/opal-wave2-20261021/full-tests-new --junitxml C:/Users/USER/Desktop/opal-wave2-20261021/full-tests-new.xml
.venv/Scripts/python.exe -m build --outdir C:/Users/USER/Desktop/opal-wave2-20261021/dist-new
.venv/Scripts/python.exe -m pip install --no-deps --target C:/Users/USER/Desktop/opal-wave2-20261021/installed-new C:/Users/USER/Desktop/opal-wave2-20261021/dist-new/opal_research-0.1.3.dev0-py3-none-any.whl
.venv/Scripts/python.exe tools/verify_reset_delivery.py --installed C:/Users/USER/Desktop/opal-wave2-20261021/installed-new --output C:/Users/USER/Desktop/opal-wave2-20261021/reset-replay-new.json
.venv/Scripts/python.exe tools/verify_block_delivery.py --installed C:/Users/USER/Desktop/opal-wave2-20261021/installed-new --block-run C:/Users/USER/Desktop/opal-research-next-20261020/block-v2 --native-run E:/opal-research-next-20261021/native-block-v6 --output C:/Users/USER/Desktop/opal-wave2-20261021/block-replay-new.json
.venv/Scripts/python.exe tools/verify_wave2_delivery.py --installed C:/Users/USER/Desktop/opal-wave2-20261021/installed-new --q1b E:/opal-wave2-20261021/q1b-v2 --partners E:/opal-wave2-20261021/q1b-partners-v2 --flexible E:/opal-wave2-20261021/q2b-v1 --output C:/Users/USER/Desktop/opal-wave2-20261021/wave2-replay-new.json
.venv/Scripts/python.exe tools/rehearse_replicated_assay.py --output C:/Users/USER/Desktop/opal-wave2-20261021/assay-new
$env:PYTHONPATH='C:/Users/USER/Desktop/opal-wave2-20261021/installed-new'
.venv/Scripts/python.exe tools/check_installed_series.py --synthetic --require-installed --output C:/Users/USER/Desktop/opal-wave2-20261021/series-v1-new
.venv/Scripts/python.exe tools/check_installed_series.py --synthetic --require-installed --config configs/experiments/opal_model_series_v2.json --output C:/Users/USER/Desktop/opal-wave2-20261021/series-v2-new
```

The completed all-branch installed replay adds `--budget C:/Users/USER/Desktop/opal-wave2-20261021/q3b-v3` and verifies all twelve budget/task prediction archives exactly. It passed in 45.2486817 wall seconds and 44.203125 CPU seconds with 573,050,880 peak bytes. The final wheel has byte-identical Python modules to the independently installed preflight wheel used for these replays. The report-aware build passes; its archives and documentation receipts are bound in validation.json. Final diff/package inspection and origin synchronization accompany the delivered commit, with hosted Windows/Linux Python 3.11/3.14 CI checked after pushing.

Q3b fit, score and exact scientific replay completed within the unchanged cumulative cap; its independent read-only audit also passed. Neither label budget clears the all-task scientific gate, so successful delivery checks do not promote a predictor. The [budget results](label_budget_results.md), [guide results](reagent_gamma_results.md), [flexible-reference results](flexible_singles_results.md) and [source qualification](wave2_source_qualification.md) remain the scientific records.
