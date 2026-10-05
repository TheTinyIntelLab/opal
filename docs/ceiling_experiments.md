# Joint uncertainty and retuned learning curves

These are late exploratory development studies; their split seeds and shared biological profiles are dependent. Historical query responses were not read. The correlated and permuted searches have 469 candidates, diagonal has 201, and unweighted has 67. Ranges below are extrema over three seeds, not confidence intervals.

| Dataset | Split | Diagonal gain (%) | Correlated gain (%) | Permuted gain (%) |
|---|---|---:|---:|---:|
| Norman | random | 1.26 to 1.48 | 0.97 to 1.47 | -0.20 to 0.00 |
| Norman | partner | 0.86 to 1.29 | 0.77 to 1.30 | 0.00 to 0.15 |
| Wessels original | random | 0.36 to 1.30 | 0.36 to 1.06 | -0.03 to 0.00 |
| Wessels original | partner | -0.42 to -0.07 | -0.42 to -0.07 | -0.03 to 0.02 |
| Wessels protected | random | -0.06 to 0.90 | 0.29 to 0.90 | -0.04 to 0.02 |
| Wessels protected | partner | -0.39 to -0.16 | -0.96 to -0.28 | -0.00 to 0.00 |

Each subset retunes its model using fitting-only inner folds. Three nested subset orders share each outer validation set; full size is fitted once. The v2 full-size fixed and retuned kernel controls use identical pair order and folds. The original v1 used a reordered full-size partition and is retained as exploratory evidence.

Technical sensitivity refits the kernel and coefficients under each of 16 joint cell-bootstrap draws, preserving shared single/control draws. It conditions on the selected hyperparameters, covariance, QC and library normalizer. Its percentiles measure sensitivity to technical fitting/reference noise against the point validation response; they are not confidence intervals for biological generalization.

Error radii use a separate fitting/calibration split: calibration responses cannot select hyperparameters or fit the model. The radius is the ceil((n+1)*coverage)-th ordered calibration pair RMSE; unavailable finite ranks are recorded instead of silently clipped. Outer coverage is empirical because these biological pair examples are dependent. The diagnostic does not support a deployed abstention guarantee. See [Shafer and Vovk (2008)](https://jmlr.org/papers/v9/shafer08a.html) for the assumptions behind conformal confidence statements.

Run `.venv/Scripts/python.exe main.py ceiling` to verify inputs, reproduce missing stages, test the code, regenerate tables and build the paper. Full numerical records, technical sensitivity values, calibration errors, fit/calibration/validation identities and source hashes are in `results/architecture/ceiling_report.json` and the per-dataset study folders.
