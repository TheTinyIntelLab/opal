# Local model-series inference and evidence

The local review version is 0.1.3.dev0; published 0.1.2 assets are preserved. `OpalBase`, `OpalInteraction`, `OpalGuard` and the three-entry registry keep their public names. The internal `symmetric-bilinear-nested-memory` architecture selects main effects, adds full symmetric bilinear residuals, and optionally learns a nested three-expert mixture. Historical safe checkpoints load through the same API.

## Source-specific weights

Evaluation manifests are `results/opal_model_series_v2/primary/<source>/conditions.json` and `retrospective/<source>/fold-<outer>.json`. They point to actual local checkpoints under the preserved D artifact root, the C spill root, or the separate D continuation root. Canonical source inputs and their hashes stay local. Retrospective budget completion reuses these records and retains new safe numeric-row stores on C. The final [deployment manifest](../../results/opal_model_series_v2/deployment-final/manifest.json) records 21 actual later all-development refits, three per source, under `C:/Users/USER/Desktop/opal-model-series-20261010-v2/deployment-final`. Their weights do not inherit an evaluation score. The earlier deployment attempt is preserved separately because additional reference comparisons subsequently completed.

Use Base for the three primary sources under the measured-single contract. Interaction gains less than 0.1% over Base on Combi-seq and TNF-positive. Guard loses to constant mixing there. Neither passes the prespecified 2% promotion rule. Norman's retrospective Guard gain is source-qualified and log/count scales are not independent confirmations. No universal weights, biological interaction attribution, exact-guide equivalence, calibrated uncertainty or prospective label savings are established.

## Installed batch inference

Choose a source checkpoint from its manifest and provide a query NPZ containing only `singles`, `pair_ids` and `culture_ids`, with the checkpoint's exact ordered genes and units in the JSON contract. Pair-supervised fitting is unavailable at zero budget. Source/assay/context/gene/unit/reference mismatches are refused. Query outcomes belong only in the sealed evaluator.

```python
import opal
model = opal.load_model(checkpoint_path)
result = model.predict(query_bundle, contract)
# result contains predictions, provenance, support/fallback and expert attribution.
```

```powershell
opal series-predict --checkpoint $checkpoint --inputs $query --contract $contract --output $freeze
opal series-evaluate --freeze $freeze --inputs $query --contract $contract --outcomes $outcomes --output $scores
```

Checkpoints use hash-checked JSON and safe NPZ arrays, with no executable deserialization. Full-gene checkpoints may share a content-addressed numeric-row store. Preserve its relative directory layout when copying a complete artifact bundle, or set `opal.load_model(checkpoint_path, array_store=local_array_directory)` and CLI `--array-store`. Every row is verified by file and numerical hashes. Missing or modified rows cause refusal. The store is part of the checkpoint, so its size must be included in storage accounting.

Strict support is the default. Explicit same-context sample-label transport requires an unchanged source, assay, ordered genes, units, reference provenance and context contract. Unsupported interaction/retrieval experts fall back to the fitted Base; a missing essential input is refused. Guard returns expert weights and acquired neighbor identities, which are attribution records, not correctness probabilities.

## Comparisons and verification

The unchanged primary grids completed 858 conditions, including 0/25/50/100% prefixes, two orders, three acquisition seeds and fixed full-label training-seed diagnostics. Fifty-one pooled, same-context transport and target-group sensitivities and evaluator-only technical/metric controls are retained. Pinned upstream PertSpectra, GEARS and CPA completed 24 adapted full-budget folds each; extra cell, single-supervision and GO inputs are disclosed separately. None is the complete author configuration. Public-only TxPert and native PRESAGE retain exact dependency/output-contract failures, so frontier competitiveness is unestablished.

The 1,296 historical budget and 72 transport checkpoints replayed exactly. The final full local suite passed 268 tests in 38.85 seconds after a preserved disk-full attempt. Final installed Windows Python 3.11/3.14 actual-source training/prediction/scoring/replay passed; the focused Python 3.11 suite passed 34 tests in 9.01 seconds. The Windows/Linux Python 3.11/3.14 integrity matrix is extended; new remote CI remains pending because pushing is unauthorized. Models, matrices and per-profile arrays are not committed or uploaded.

## Final local verification

All 858 primary and 312 retrospective budget conditions, twelve retrospective ablation folds, 51 sensitivities with completed references, evaluator-only technical/normalization diagnostics and 72 upstream adaptation folds are saved. Native safe-state replay passes within the declared float32 tolerance; Opal replay is exact. The full local suite passed 268 tests. Windows Python 3.11/3.14 installed actual-source train/predict/score/API replay passed for all three models. The four-environment remote integrity matrix is extended but pending because no push is authorized.

Deployment `series-predict` accepts inference queries under the exact source contract, including already acquired pairs for serialization checks. `series-evaluate` rejects these predictions, even if the display role marker is removed, because contract and provenance also identify deployment. This prevents assigning in-sample deployment scores to the frozen evaluation table. `--array-store PATH` and `opal.load_model(checkpoint, array_store=PATH)` support explicit shared-store relocation. Preserve every hashed store row referenced by a checkpoint.

The matched extra-single linear reference beats Base on TNF-negative, at 0.070020 versus 0.071218 MSE. Native adaptations are weaker on these particular runs; missing canonical configurations and native training RAM prevent frontier and complete efficiency claims. Default nonlinear promotion fails. The exact retained-artifact and process-cost ledger is in `results/opal_model_series_v2/verification/final_resources.json`; failed-attempt counters that cannot be recovered are explicitly missing.
