# Reset delivery validation

This note records the first wave on 20 October 2026. The [21 October continuation checks](validation_next.md) supersede its software/native-execution status.

The complete suite passes **436 tests in 47.35 seconds** on Windows/Python 3.14, with temporary files on `E:/opal-research-reset-20261020-validation/full-tests-final`. An initial run passed 433 tests and failed three existing one-GiB disk-reserve guards after C: filled; the first relocated attempt lacked its parent directory and produced fixture setup errors. Creating the temporary parent and rerunning the full unchanged suite resolved those environmental failures. No guard or test was weakened.

Wheel and sdist builds pass, with the existing setuptools license-table deprecation warning. Package-only installation and prediction replay pass outside the checkout using host dependencies; this is not a fresh dependency installation. Four new development composites replay exactly, all four historical external prediction arrays replay exactly, and original 16,294-pair predictions differ by at most 5.56e-17. The verifier separately binds development code, data, assignment, checkpoint, selection and prerequisite ordering hashes.

The replicated-assay rehearsal passes, and installed-wheel synthetic model-series v1 and v2 train/predict/evaluate checks both verify outside the checkout. These are engineering fixtures, not biological confirmations. Documentation renders and local-link checks run after the final notes are written. Distribution inspection excludes raw/derived data banks, trained weights and private correspondence; source modules under `src/data` and public release configuration JSON are intentionally allowed.

The prior plan is archived and old frontier programs remain historical. The concurrent paper update `af79135` was preserved. Numerical arrays/checkpoints stay ignored or outside Git; only aggregate evidence, metadata qualification, code, tests and reports are staged. Source-specific licenses and unresolved mirror/raw-SRA qualifications remain explicit.

Exact results are in [the compact validation receipt](../../results/research-reset-validation/validation.json). Local logs, XML, installed package, builds and generated fixtures remain under `C:/Users/USER/Desktop/opal-research-reset-20261020/`. Hosted CI and remote synchronization are checked after the normal push and reported at delivery; this local receipt does not pretend that a not-yet-pushed commit has passed hosted CI.

```powershell
.venv/Scripts/python.exe -m pytest -q --basetemp E:/opal-research-reset-20261020-validation/full-tests-final --junitxml C:/Users/USER/Desktop/opal-research-reset-20261020/validation/tests-final.xml
.venv/Scripts/python.exe -m build --outdir C:/Users/USER/Desktop/opal-research-reset-20261020/dist-final
.venv/Scripts/python.exe -m pip install --no-deps --target C:/Users/USER/Desktop/opal-research-reset-20261020/installed C:/Users/USER/Desktop/opal-research-reset-20261020/dist-final/opal_research-0.1.3.dev0-py3-none-any.whl
.venv/Scripts/python.exe tools/verify_reset_delivery.py --installed C:/Users/USER/Desktop/opal-research-reset-20261020/installed --output C:/Users/USER/Desktop/opal-research-reset-20261020/validation/installed-replay.json
.venv/Scripts/python.exe tools/rehearse_replicated_assay.py --output C:/Users/USER/Desktop/opal-research-reset-20261020/assay-validation
$env:PYTHONPATH='C:/Users/USER/Desktop/opal-research-reset-20261020/installed'
.venv/Scripts/python.exe tools/check_installed_series.py --require-installed --synthetic --output C:/Users/USER/Desktop/opal-research-reset-20261020/series-validation
.venv/Scripts/python.exe tools/check_installed_series.py --require-installed --synthetic --config configs/experiments/opal_model_series_v2.json --output C:/Users/USER/Desktop/opal-research-reset-20261020/series-bilinear-validation
.venv/Scripts/python.exe tools/build_docs.py --output C:/Users/USER/Desktop/opal-research-reset-20261020/docs-final
```

Commands that create evidence directories refuse reuse. Change output paths for a new check, and acquire the licensed hash-matched inputs for biological replay. A fresh clean-dependency environment was not created locally because disk space was constrained; the existing hosted CI workflow exercises clean dependency installs on Windows/Linux and Python 3.11/3.14. No native competitor fit, new-source response scoring, prospective assay or calibrated biological interval was completed in this delivery.
