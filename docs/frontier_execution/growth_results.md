# Exposed-growth architecture tournament results

The registered all-task promotion gate failed. Jurkat gained 4.11% MSE against its validation-selected simple reference, below the required 5%. The other five tasks passed their task gates. No threshold, selection, model or label budget was changed after scoring. These are already exposed historical development cohorts with new source-local training, not untouched-source or frozen-weight transfer confirmation.

## Frozen comparison and executed workload

All six tasks used five original disjoint within-gene blocks, 16,000 TRAIN plus 4,000 VALIDATION pair labels per block, the same measured singles and identities, and the frozen 16-reference/five-interaction configuration menu. Three configuration-matched residual shuffles were included. All 30 procedures completed and sealed their selections/checkpoints/predictions before scoring. There were 1,080 candidate instances including fixed references, and 840 saved checkpoints. Exact replay from the external installed wheel passed all 840 with zero refits. The complete metric ledger includes mean and both infection targets, common support and full populations, MAE, bias, correlation, all fixed references, shuffled policies and conditional gene sensitivity.

Every fold selected completion, I0, by VALIDATION. The kernel and composition network therefore remain evaluated alternatives; neither is promoted. The TEST-best diagnostic is explicitly not a nominee.

## Primary registered result

Positive gain means lower MSE than the validation-selected simple reference. Coverage was 100% for every task, so common support equals the full test population. The interval is a conditional gene-resampling sensitivity interval for paired policy-minus-simple squared error, not a biological-replication confidence interval.

| Task | Test pairs | Genes | Simple MSE | Policy MSE | Gain | Conditional 95% difference interval | Gate |
|---|---:|---:|---:|---:|---:|---|---|
| horlbeck_K562 | 16210 | 427 | 0.00190091929 | 0.00179069074 | 5.80% | [-0.00022318, -1.41589e-05] | pass |
| horlbeck_Jurkat | 16210 | 427 | 0.000507630148 | 0.000486775245 | 4.11% | [-3.45453e-05, -8.3413e-06] | FAIL: gain <5% |
| environment1 | 9833 | 313 | 0.000551581712 | 0.000412898098 | 25.14% | [-0.000182177, -0.00010274] | pass |
| environment2 | 9833 | 313 | 0.000727352832 | 0.000518265563 | 28.75% | [-0.00027143, -0.000156782] | pass |
| parp2022 | 29016 | 539 | 0.000308097243 | 0.000254004302 | 17.56% | [-6.78166e-05, -4.23211e-05] | pass |
| parp2023 | 28651 | 537 | 0.000302940513 | 0.000260679297 | 13.95% | [-5.26247e-05, -3.30965e-05] | pass |

The policy beat all 16 fixed simple references and all three validation-selected shuffled policies on each task. Every leave-one-gene-out mean paired difference stayed negative. Both infection-target point comparisons improved in every task; these are checks on the same experiment, not new independent initiations. Jurkat still fails the unchanged effect-size gate.

## Aligned architecture comparison

Each family uses its own VALIDATION-selected configuration per fold. The following descriptive gains use the same selected-simple denominator. They do not change the registered full-policy selection.

| Task | I0 completion | I1 kernel | I2 composition network |
|---|---:|---:|---:|
| horlbeck_K562 | 5.80% | -0.06% | -10.54% |
| horlbeck_Jurkat | 4.11% | 0.04% | -15.08% |
| environment1 | 25.14% | 0.31% | 14.51% |
| environment2 | 28.75% | 0.09% | 15.61% |
| parp2022 | 17.56% | 0.49% | -4.75% |
| parp2023 | 13.95% | 0.05% | -1.82% |

## Every registered simple reference

These are pooled mean-target test MSE values. They are reported for comparison, not for TEST-based model selection. Full MAE/bias/correlation and infection metrics are in the immutable result JSON.

| Reference | horlbeck_K562 | horlbeck_Jurkat | environment1 | environment2 | parp2022 | parp2023 |
|---|---:|---:|---:|---:|---:|---:|
| B0--zero | 0.2596612 | 0.0628492684 | 0.0504620541 | 0.0559787104 | 0.0294338984 | 0.0277299683 |
| B0--mean | 0.0149600251 | 0.0155971357 | 0.0111401796 | 0.0162547166 | 0.00873673494 | 0.00870400598 |
| B0--sum | 0.0462216206 | 0.00477697676 | 0.00471633484 | 0.00429897149 | 0.00330519286 | 0.00308470195 |
| B0--minimum | 0.0100189934 | 0.00191389134 | 0.00196993193 | 0.00236444875 | 0.000799999415 | 0.000728726053 |
| B1--alpha1 | 0.00289612415 | 0.000661776949 | 0.000861347752 | 0.00110836093 | 0.000460962667 | 0.000421766463 |
| B1--alpha10 | 0.00289611399 | 0.000661756034 | 0.000861335529 | 0.00110833138 | 0.000460943127 | 0.000421749791 |
| B1--alpha100 | 0.00289623595 | 0.000661620721 | 0.000861258211 | 0.00110816357 | 0.000460807087 | 0.00042162948 |
| B1--alpha1000 | 0.00290447355 | 0.000664758917 | 0.000863598148 | 0.0011113016 | 0.000462848162 | 0.00042361961 |
| B2--alpha1 | 0.00190114472 | 0.000509942876 | 0.000554470677 | 0.000735546161 | 0.000319586778 | 0.000311504273 |
| B2--alpha10 | 0.00190091929 | 0.000509925587 | 0.00055446325 | 0.00073552571 | 0.000319556641 | 0.000311476342 |
| B2--alpha100 | 0.00190363939 | 0.000509983863 | 0.00055451232 | 0.000735378831 | 0.000319298135 | 0.000311252746 |
| B2--alpha1000 | 0.00192410852 | 0.000514207059 | 0.000557819615 | 0.00073760397 | 0.000320175596 | 0.00031255591 |
| B3--nodes0-alpha10 | 0.00288365166 | 0.000655259711 | 0.000839687512 | 0.00105243664 | 0.000433212096 | 0.000397144483 |
| B3--nodes0-alpha100 | 0.0028842483 | 0.000656751511 | 0.000841500089 | 0.00105586093 | 0.000434137977 | 0.00039764525 |
| B3--nodes1-alpha10 | 0.00190001871 | 0.000507630148 | 0.000551581712 | 0.000727352832 | 0.000308097243 | 0.000302500656 |
| B3--nodes1-alpha100 | 0.00190096329 | 0.000508558938 | 0.000551605254 | 0.000727429514 | 0.000308551513 | 0.000302940513 |

## Resources and reproducibility

The external watchdog measured fit/score/installed replay at 1492.040/9.011/36.391 seconds. Fit peak process-tree memory was 754,958,336 bytes. Internal cumulative active fit/score/replay was 1527.290 seconds and 1520.328 CPU seconds, using one thread. The final retained program footprint was 203,544,810 bytes under the 512 MiB cap. The separately charged profile and packing costs remain recorded.

The first source-scale storage forecast failed at 598,749,898.5 bytes. Upper-triangle packing then missed the combined 64 MiB profile cap by 256,915 bytes. Both failures and original artifacts remain preserved. A new reversible byte-plane shuffle reduced the passing forecast to 393,199,332.5 bytes and combined profile to 65,824,295 bytes, with all 112 profile predictions exactly replayed. No quantization, model arithmetic or scientific gate changed.

The wheel was installed outside the checkout with host dependencies reused, not in a clean virtual environment. Its original 103 bound runtime files were byte-identical; the new codec is separately versioned. Biological checkpoints and query/response arrays stay local and are not redistributed. The committed receipt directory contains only protocols, hashes, resource receipts and aggregate scientific metrics.

## Interpretation and stopping rule

The evidence favors completion over this finite kernel/neural menu for these assay-local tasks and this local label budget. It also supports pair-aligned predictive structure beyond the registered simple references. It does not identify molecular interactions: reusable construct/technical pair effects can produce the same improvement, as the observationally identical synthetic worlds demonstrate. The network recovered the larger-budget synthetic partner-curve world but did not turn that capacity into superiority here.

The program has no all-task positive result, no independently confirmed new biological source, no cold-gene or frontier-performance result and no reversal of the historical frozen-weight external failure. These exposed test populations cannot become untouched by registering new analyses. The conditional intervals do not capture experiment-to-experiment uncertainty.

The finite registered menu and stopping rule are exhausted. Further outcome-driven tuning on these tasks would be development work requiring a new question, not rescue of this failed gate. The next highest-value evidence is a primary COLO1 companion establishing both-arm guide sequence/target/control mapping, annotation lineage and sample/time/culture/initiation keys, followed by source-specific qualification and registration. The registered mapping archive route returned HTTP 403 with zero body reads; it is closed to speculative retries. COLO1 responses remain sealed. Missing old E: custody paths separately block some historical package replays.

See [registration](growth_registration.md), [packing amendment](growth_packing_registration.md), [byte-shuffle amendment](growth_shuffle_registration.md), [synthetic results](synthetic_results.md), and [aggregate ledger](../../results/frontier-execution-20261026/growth-v3/results.json).
