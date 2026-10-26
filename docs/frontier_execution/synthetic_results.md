# Frozen synthetic architecture tournament results

The complete96-cell, two-budget tournament evaluated6,912 candidate instances and saved5,376 checkpoints before TEST scoring. That count includes fixed/null references: there were1,536 neural optimization runs,768 completion runs,3,840 ridge fits,192 fitted means and576 fixed predictors. Installed-package replay reproduced every saved prediction exactly and independently recomputed32,256 MSE values. All96 construct-aggregation checks passed. Twelve low-rank/persistent-technical cell pairs had identical input arrays and predictions, so successful recovery cannot identify a biological cause.

Different architectures work on different registered structures. At the1,239-label budget, completion reduced mean second-initiation observed MSE by82.20% against the validation-selected simple policy in the noisy complete low-rank slice. The small network reduced it by65.06% on structured partner curves and was selected on VALIDATION in all three seeds there. The kernel had smaller gains on partner curves and nonlinear singles. These are finite simulator results, not biological validation or a global architecture promotion.

## Complete noisy-slice comparison

The following descriptive slice uses noise0.1, complete observation and all three registered seeds. Every row pools equal-sized531-pair tests; the full96-cell roster, both initiations, all masks/noise levels, every fixed reference, latent diagnostics, conditional sensitivities, interval diagnostics and per-gene errors remain in the machine-readable results. A positive percentage means lower second-initiation observed MSE than the validation-selected simple policy. Negative values are losses. The selected policy was fixed before TEST scoring; these rows do not select another deployed model.

| World | Charged TRAIN+VALIDATION labels | Selected policy gain | Completion gain | Pair kernel gain | Small network gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Additive |309|-1.16%|-11.26%|-3.31%|-201.04%|
| Additive |1239|0.00%|-26.19%|-1.83%|-73.28%|
| Persistent low rank |309|1.40%|13.75%|-8.33%|-203.58%|
| Persistent low rank |1239|82.20%|82.20%|-4.55%|35.49%|
| Nonlinear singles |309|0.45%|-6.08%|2.93%|-160.60%|
| Nonlinear singles |1239|2.45%|-0.91%|3.56%|-45.11%|
| Gene-specific partner curves |309|13.18%|11.64%|11.76%|-397.88%|
| Gene-specific partner curves |1239|65.06%|56.47%|16.90%|65.06%|
| Independent sparse edges |309|-0.60%|-8.20%|1.44%|-283.69%|
| Independent sparse edges |1239|0.00%|-32.21%|-0.48%|-109.86%|
| Guide/batch nuisance |309|0.23%|-2.80%|-0.22%|-67.63%|
| Guide/batch nuisance |1239|0.00%|-10.81%|-0.18%|-32.44%|
| Persistent technical alias |309|1.40%|13.75%|-8.33%|-203.58%|
| Persistent technical alias |1239|82.20%|82.20%|-4.55%|35.49%|
| Redrawn technical structure |309|-7.74%|-14.74%|-6.49%|-182.08%|
| Redrawn technical structure |1239|-48.70%|-48.70%|-1.23%|-64.05%|

The network performs poorly at the smaller label budget. Completion also loses on additive and sparse worlds, and the validation selector sometimes chooses an interaction correction that loses on TEST. The redrawn technical world produces a48.70% selected-policy MSE increase at the larger budget despite strong same-structure recovery. These failures rule out a universal winner and show why source-local validation cannot establish robustness to a hidden technical shift.

## Interpretation and uncertainty

The partner-curve result supports testing the distinct composition architecture rather than privileging completion. It does not establish a biological mechanism: the generator deliberately supplies reusable structure through allowed identities/singles, and arbitrary unobserved sparse edges remain unpredictable. The persistent technical alias gives exactly the same apparent low-rank success. Correlated constructs, repeated masks, simulation seeds and the two causal labels do not create independent biological evidence.

The90% VALIDATION-residual wrapper is a diagnostic radius, not a distribution-free or OOD guarantee. Each procedure also retains2,000 endpoint-multiplicity gene resamples and leave-one-gene sensitivity for the selected policy versus selected simple. Those intervals condition on the generated graph and data; they do not estimate between-study uncertainty. The historical5% all-task biological promotion gate remains separate and unchanged.

All methods saw the same selected initiation-one TRAIN/VALIDATION labels, measured singles and identities. No true latent factors, second-initiation labels, TEST targets or guide-offset oracle entered fitting. The guide/batch world is explicitly an unmodeled technical control; construct aggregation was tested, but no identifiable guide/biological decomposition is claimed. Low-quality genes trigger the same fixed-sum fallback, and full test denominators remain intact.

## Resources and reproduction

External monitored fit/score/installed-replay commands used693.499s,10.297s and9.587s wall respectively. Their total is713.383s, with maximum sampled process-tree peak321,462,272 bytes. Numerical work used one thread. The frozen7200-second/2GiB/256MiB limits were not raised. The final artifact size and internal cumulative CPU/wall ledger are in the resource receipts; sampled peaks are not OS memory reservations.

Pilot v1 is preserved as an unscored, unqualified fit because root began it before independent fixture delivery. Pilot v2 followed the reviewed20-test suite and installed replay; both pilot attempts remain charged/disclosed. One independent test had a self-referential seed-scan defect, and the first root run exposed a Windows manifest-slash bug. Root corrected those before evaluation generation, retaining the failed receipt and original committed lifecycle. Neither changed a model or scientific threshold.

The exact [registration](experiment_matrix.md), [source-bound protocol](../../results/frontier-execution-20261026/tournament-v1/protocol.json), [code binding](../../results/frontier-execution-20261026/tournament-v1/code-binding.json), [full compact results](../../results/frontier-execution-20261026/tournament-v1/results.compact.json), [summary](../../results/frontier-execution-20261026/tournament-v1/summary.json) and [installed replay](../../results/frontier-execution-20261026/tournament-v1/replay.json) provide the evidence. Compact JSON is lossless at the parsed-data level; the summary records its SHA256 and the original unmodified scoring-file SHA256. Arrays/checkpoints remain local under ignored `tmp/frontier-execution-20261026/tournament-v1`, bound by the complete prediction manifest. Reproduction requires the pinned stack and a new output path, not reuse of a closed run directory.

No biological outcome was scored in this tournament. The separately registered exposed-growth comparison and COLO1 qualification remain separate decisions.
