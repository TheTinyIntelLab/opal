# Six-task missing-block development results

The fixed scalar rank-eight correction passed the complete preregistered six-task gate. This supports assay-local prediction with measured singles and local pair supervision. All six source-context populations had already been exposed to evaluation or diagnosis; these are exploratory development results, not independent source confirmation, cold-gene prediction or transport of frozen weights.

Registration `block-v2` and passing prerequisite tests preceded fitting. Commit `fe9537b` preserved the scientific implementation before fits; native recipe and its test-only portability correction were also frozen before Q2 test scoring. The superseded first registration had no fits or scores. Its checkpoint-storage and replay defects were corrected before outcomes, with exact prior code retained. No rank, penalty, seed or reference menu was changed after results.

| Task | Disjoint test pairs | Selected-simple MSE | Rank-eight MSE | MSE reduction | Conditional gene loss-difference interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Horlbeck K562 | 16,210 | 0.0018656418356246718 | 0.0013418626513990082 | 28.0750% | [-0.000665778, -0.000422282] |
| Horlbeck Jurkat | 16,210 | 0.0005009162998320582 | 0.0004341567404863752 | 13.3275% | [-0.0000832033, -0.0000520223] |
| Environmental 1 | 9,833 | 0.0005450429006015583 | 0.0003655341517065346 | 32.9348% | [-0.000230666, -0.000138797] |
| Environmental 2 | 9,833 | 0.0007248761005929065 | 0.00046475313111452585 | 35.8852% | [-0.000333150, -0.000200013] |
| PARPi 2022 | 29,016 | 0.0003095503277684733 | 0.00021159725552219704 | 31.6437% | [-0.000117223, -0.0000816079] |
| PARPi 2023 | 28,651 | 0.00030161626937119646 | 0.00022298166618172846 | 26.0711% | [-0.0000932155, -0.0000653116] |

Each row clears the 5% mean-target MSE margin, has an upper conditional gene bound below zero, improves on both infection targets, beats all three stratified shuffled-residual controls, and has no point-estimate loss against any of the twelve fixed simple references. Validation chooses the simple policy separately for each fold; the all-reference sensitivity is a veto, not a test-selected replacement. The protocol defines equal pair weights and complete within-gene-fold test blocks. Every query gene has training support outside its held block.

The negative controls preserve the graph and coarse degree, absolute-single and construct-count strata. They test whether aligned residual labels add predictive information under this nuisance control. They do not prove a molecular interaction mechanism, exclude all unmodeled main effects, or create independent biological replication. Both infection comparisons use a predictor fitted to other pairs' mean responses. They are replicate-target sensitivity checks, not held-infection training experiments. No pooled cross-source metric is defined.

All 30 folds, 360 simple-reference fits and 120 completion/control fits were retained. Fit time was 118.2621777 seconds, CPU time 115.640625 seconds and process peak working set 1,053,954,048 bytes. The fit-plus-score receipt records 123.8373867 active wall seconds and 121.171875 CPU seconds. Local artifacts occupied 61,420,673 bytes at independent review, below 128 MiB. The replay checked exact stored-versus-recomputed arrays for every comparator and exact scientific JSON equality. Replay shares the resource guard but its receipt does not separately persist elapsed resources. Windows peak working set is available; Linux checks sample RSS. Guards run at numerical boundaries rather than enforcing OS limits.

Native NAIAD is a separate preregistered comparator. Its result cannot be inferred from these scores, and this report makes no superiority claim against it. Passing Q2 permits continued source qualification and a tightly bounded successor investigation; it does not promote a new predictor or justify a larger architecture.

Exact machine-readable results and compact selected-model records are in [the evidence directory](../../results/research-reset-block/block-v2/). Numerical inputs, assignments, checkpoints and predictions remain outside Git at `C:/Users/USER/Desktop/opal-research-next-20261020/block-v2`, bound by hashes. The [protocol](block_protocol.md) specifies eligibility, splits, controls, gates and uncertainty.

```powershell
.venv/Scripts/python.exe tools/study_block_completion.py fit --output C:/Users/USER/Desktop/opal-research-next-20261020/block-v2
.venv/Scripts/python.exe tools/study_block_completion.py score --output C:/Users/USER/Desktop/opal-research-next-20261020/block-v2
.venv/Scripts/python.exe tools/study_block_completion.py replay --output C:/Users/USER/Desktop/opal-research-next-20261020/block-v2
```

Those lifecycle commands refuse an existing fit or scoring attempt. Reproduction requires a new output directory, the hash-matched locally licensed banks, `freeze`, then `verify` before fitting. Preserved paths and timestamps identify the actual completed run, rather than commands that would overwrite it.
