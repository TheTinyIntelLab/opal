# Exposed-cohort assay diagnosis

The frozen original model's failure is robust to the three registered construct-count thresholds. Assay-matched minimum-single effects predict much better, including when singles and pair targets come from opposite infections. Residuals above minimum singles repeat across cohorts within a study, but those residuals still include gene effects and shared measurement artifacts. These results support testing assay-conditioned prediction; they do not validate an adapted model or identify mechanistic interactions.

All four cohorts were previously scored. This entire analysis is exploratory, including its newly frozen comparisons. The [protocol](diagnostic_protocol.md), [code](../../tools/reset_diagnostics.py) and seven synthetic tests were committed at `b7aff5e` before the new diagnostic run. The earlier external protocol, model assets, predictions and scores were verified against their hashes and preserved unchanged.

## Main effects and infection dependence

The table reports exactly the historical bank-novel pair population, with no missing-single exclusions. Opposite-infection MSE averages the two directional losses in the JSON receipt. It scores individual infection targets, so it is not numerically interchangeable with mean-target MSE.

| Cohort | Pairs / genes | Frozen completion mean-target MSE | Local minimum mean-target MSE | Opposite-infection minimum MSE, directional average | Pair-minus-local-minimum infection correlation |
| --- | --- | --- | --- | --- | --- |
| Environmental 1 | 130 / 17 | 0.07106017 | 0.00583726 | 0.00699940 | 0.68534 |
| Environmental 2 | 130 / 17 | 0.05040156 | 0.00914504 | 0.01026329 | 0.80209 |
| PARPi 2022 | 423 / 30 | 0.10609866 | 0.00252622 | 0.00352013 | 0.80711 |
| PARPi 2023 | 411 / 30 | 0.10671178 | 0.00234228 | 0.00256538 | 0.77789 |

No local-single predictor here uses external pair labels to fit a coefficient, but it has extra assay-specific information unavailable to the original model. The large difference is evidence about information value, not an equal-input model victory. Cross-infection results reduce shared-infection noise without removing shared reagents, upstream processing, control normalization or correlated single-effect error.

Existing original-versus-external single correlations are 0.161, 0.308, -0.030 and -0.038 in `results/fitness-external-20261020/diagnostics.json`. A pure positive scalar rescaling cannot repair those correlations. This rejects scaling as the sole explanation; it does not isolate guide efficacy, growth window, cell state, censoring or mixed normalization effects. The new executable records target distributions and prediction correlations, rather than fitting an oracle affine correction.

## Coverage sensitivity

At fixed minimum construct counts 2, 4 and 8, remaining pair counts are 130/124/107, 130/128/112, 423/408/305 and 411/364/215 for the four cohorts. Completion has higher point-estimate MSE than frozen main effects at all 12 comparisons. At threshold 8 the completion/main-effect MSEs are 0.06892403/0.06791721, 0.04936657/0.04856428, 0.09072442/0.08954445 and 0.06990519/0.06967616. Local minimum remains much stronger, at 0.00549134, 0.00927660, 0.00289254 and 0.00312195.

Every conditional gene-resampling interval for completion minus main effects still includes zero. These are nested panels and dependent sensitivity comparisons; they neither prove harmful completion across a population of studies nor repair the historical failed superiority gate. Increasing coverage changes which pairs are represented and cannot resolve missing-not-at-random depletion. The audit separately establishes that all 509 distinct bank-novel pairs occur in the original raw screen but fail its eligibility rule.

## Observed residual transfer

This diagnostic uses a previously measured source-cohort pair residual to predict the same pair in another cohort, after giving the target its own measured minimum singles. It tests repeatability, not prediction of an unmeasured pair and not transfer of the fitted Opal correction.

Environmental experiments share 130 bank-novel pairs. Raw pair correlation is 0.90566 and local-minimum residual correlation is 0.72200. Adding the source's observed residual with fixed unit weight lowers target local-minimum MSE by 80.74% from experiment 1 to 2 and 69.83% in reverse. PARPi cohorts share 411 pairs, with raw correlation 0.94400 and residual correlation 0.84544; the corresponding reductions are 76.29% and 78.09%. All four within-study paired gene-resampling upper bounds are below zero. Twenty fixed shuffled-edge controls have higher average MSE than aligned residual transfer in both studies.

Cross-study overlap is only 44 pairs for PARPi 2022 and 38 for PARPi 2023. Residual correlations range from 0.54507 to 0.64957. Fixed residual-transfer gains are asymmetric, from 7.16% to 70.35%; all four environmental-to-PARPi conditional intervals include zero. These small, overlapping panels cannot establish general study transfer. Edge shuffling also destroys gene effects, and minimum singles leave substantial main-effect miscalibration, so alignment is not sufficient evidence of a pair-specific biological mechanism.

The independently fitted node-reference residuals in the [cross-infection experiment](development_results.md) correlate only 0.38405 in K562 and 0.24048 in Jurkat. These values use a different source and nuisance adjustment, so they are not directly comparable to the external coefficients. They demonstrate why raw or minimum-residual repeatability needs stronger main-effect controls.

## Reproduction and limits

Compact protocol, scores, resources and replay receipts are in [results/research-reset-diagnostics](../../results/research-reset-diagnostics/). The actual run used one numerical thread, 0.95703 seconds and 102,916,096 bytes process peak working set. Recomputed scientific JSON matched exactly. Only aggregate evidence was copied into Git; source banks and the original frozen predictions remain local.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_reset_diagnostics.py
.venv/Scripts/python.exe tools/reset_diagnostics.py freeze --original C:/Users/USER/Desktop/opal-next-data-20261020 --prepared C:/Users/USER/Desktop/opal-external-validation-20261020/prepared-v2 --frozen C:/Users/USER/Desktop/opal-external-validation-20261020/evaluation-v1 --output C:/Users/USER/Desktop/opal-research-reset-20261020/diagnostics-reproduction
.venv/Scripts/python.exe tools/reset_diagnostics.py run --output C:/Users/USER/Desktop/opal-research-reset-20261020/diagnostics-reproduction
.venv/Scripts/python.exe tools/reset_diagnostics.py replay --output C:/Users/USER/Desktop/opal-research-reset-20261020/diagnostics-reproduction
```

Use a new output directory. Freeze and run refuse drift or overwritten results, but the runner does not itself enforce the prefit pytest invocation; the tool transcript and pre-result registration commit establish that order in this run. Question 1's directional average above is derived from the two stored directional MSEs. A fully empty complete-single panel would fail explicitly instead of producing an unavailable panel, a protocol/implementation limitation that did not occur here. Empty coverage-threshold subsets are handled. Fix either limitation only in a new registered version, preserving these exact replay bytes.
