# Research findings

The retained evidence supports assay-local prediction of missing pairs among training-supported genes. It does not establish reliable transfer to a new biological study, causal interaction recovery or a frontier-level ranking.

## Six-task growth delivery

The [benchmark ledger](../results/model-delivery-20261027/benchmark.json) preserves all comparators, infection targets, coverage and conditional gene sensitivities. The [source result](../results/frontier-execution-20261026/growth-v3/results.json) is hash-bound by the benchmark projector.

| Task | Held pairs across five blocks | Selected simple MSE | Completion MSE | Reduction |
| --- | ---: | ---: | ---: | ---: |
| K562 | 16,210 | 0.00190091929 | 0.00179069074 | 5.80% |
| Jurkat | 16,210 | 0.000507630148 | 0.000486775245 | 4.11% |
| Environmental 1 | 9,833 | 0.000551581712 | 0.000412898098 | 25.14% |
| Environmental 2 | 9,833 | 0.000727352832 | 0.000518265563 | 28.75% |
| PARPi 2022 | 29,016 | 0.000308097243 | 0.000254004302 | 17.56% |
| PARPi 2023 | 28,651 | 0.000302940513 | 0.000260679297 | 13.95% |

Coverage is 100% on these registered populations. Each procedure uses 16,000 training and 4,000 validation pair labels, measured source-local singles, and training-supported genes. The all-task criterion requires at least 5% gain in every task; Jurkat fails at 4.11%, so no model is promoted. The result is exposed-source development. Conditional gene resampling is not independent-study uncertainty, and pooled results do not evaluate the single exported fold-0 checkpoint.

## Frozen external transfer failed

An earlier K562 predictor was frozen before external scoring. On bank-novel pairs, completion MSE was 1.08–2.15% higher than frozen main effects across four environmental/PARPi cohorts, and original-single minimum performed better in every cohort. [Original external results](../results/fitness-external-20261020/results.json) and [source-bound manuscript evidence](../paper/evidence/model_development_20261020.json) retain the scores and definitions.

Those four panels contain 509 unique pairs, not four independent pair populations. All 509 occur in the raw original table but were excluded by the eligible-bank coverage rule. Assay shift and graph support shift are confounded. Later local retraining and same-pair response-transfer diagnostics reuse exposed sources under different information contracts; they do not rescue frozen unseen-pair transfer.

## Negative and limiting experiments

- [Expression composition](model_development_20261020.md) fails all four Norman/Wessels pair and partner comparisons; gains over additivity do not imply gains over selected simple references.
- [Original growth evidence](../results/fitness-20261020/results.json) retains the failed shared/private-context promotion rule despite independent-context gains.
- [Local block completion](research_reset/block_results.md) and [native comparison](research_reset/native_results.md) report local gains and mixed comparator outcomes, rather than a general method ranking.
- [Guide matching](research_reset/reagent_gamma_results.md) and [nonlinear-single controls](research_reset/flexible_singles_results.md) narrow some explanations without identifying molecular interactions.
- [Pair-label budgets](research_reset/label_budget_results.md) nominate neither 5,000 nor 20,000 labels per procedure; the latter still fails the required gain in original K562 and Jurkat.
- [Numerical closeout](research_reset/wave5_closeout.md) and [completion audit](research_reset/wave5_completion/closeout.md) preserve failed legacy-compatibility checks and distinguish deterministic arithmetic from accurate arithmetic. The strongest-reference scientific comparison remains unanswered.
- [Historical model-series findings](next_cycle/opal_model_series_v2.md) preserve failed complexity/promotion outcomes. The [archived study index](archive/README.md) retains state, dose, context-transfer, source-qualification and measurement findings.

The [paper](../paper/main.pdf) presents endpoints, methods and limitations. No new biological experiment or training run was performed during release preparation.
