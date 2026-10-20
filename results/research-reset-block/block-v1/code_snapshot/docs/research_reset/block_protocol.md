# Registered six-task missing-block experiment

This is exploratory development on previously exposed outcomes. It tests complete unobserved within-gene-fold blocks, not cold genes, new cultures, source transfer or confirmation. The historical random-pair assignments, assets and results remain unchanged. The source roster, minimum support, hash splits and prospective gates come from root `plan.md`.

## Population and frozen splits

Use `scalar_screen.load_screen_roster` to verify the historical banks and load exactly Horlbeck K562/Jurkat, environmental 1/2 and PARPi 2022/2023. The later cohorts use full source-local eligible banks, not the small original-supported transfer panels. Keep both local singles and at least two constructs per infection, and finite responses; preserve exclusion counts. All pair variants and infections share the canonical pair's split.

Hash each gene with SHA256 of UTF-8 `opal-reset-block-v1|gene`, first eight hexadecimal digits modulo five. In fold k, every pair with both genes assigned k is TEST. Among the other pairs, SHA256 of `opal-reset-v2|accession|geneA|geneB`, first eight hex digits modulo five, gives validation for zero and training for one through four. Every query gene must have training-pair support. The five test blocks are disjoint; folds, guides and contexts are not independent biological replications. Freeze all assignments and panel/control metadata before computing new outcome losses.

## Fixed references and completion

Fit zero, training mean, single sum, single minimum, symmetric single calibration and gene-main-effect ridge. Calibration and node penalties are 1, 10, 100 and 1000, selected by validation only. Explicit training mean is an additional null reference, declared before outcomes. Ridge uses the historical symmetric `[sum, absolute difference, product]` design and optional gene incidence. Stream training sufficient statistics in 4,096-row chunks rather than allocating a full pair-by-gene matrix. Test equivalence to the old dense reference on small fixtures. No test response enters training moments or selection.

Each task/fold fits an independent symmetric rank-eight correction, penalty .01, Adam .03, at most 500 steps, seed 20261022 and validation every 25 steps. Use a training-only target standard deviation. The scalar implementation uses half its normalized data MSE plus the factor penalty, preserving the historical two-context independent objective's per-context error-to-penalty ratio. It selects a checkpoint separately for each task, rather than fabricating a second context or averaging validation across contexts. This is a registered scalar implementation of the existing algorithm, not a claim of bitwise equivalence to old fits or a new architecture.

Freeze three residual-label negative controls, seeds 20261023/24/25. Keep the graph fixed. Stratify TRAIN edges by unordered endpoint training-degree tertiles, unordered endpoint absolute-single tertiles and minimum construct counts 2-3/4-7/at least8. Quantile ties enter the higher bin. Original K562/Jurkat use shared across-context single summaries and minimum counts so they receive identical pair permutations. Shuffle within strata; singletons remain unchanged. Require at least80% of training edges in nonsingleton strata for an informative control. Compare aligned completion against every control, without choosing a favorable seed. Shuffling preserves only coarse nuisance strata and cannot isolate biological mechanisms.

The native NAIAD comparator is separately qualified and registered using these identical assignments and data access. Do not replace it with a reimplementation or imply it ran when profiling/budget qualification failed. Q2 scoring is valid independently, but no superiority to NAIAD or Q4 promotion follows without Q3.

## Metrics and gate

Fit every task/fold and persist the complete reference/checkpoint choices before TEST metrics. Aggregate canonical test pairs equally across the five disjoint blocks. Report each block's candidate/reference MSE, mean-target MSE/MAE/bias/centered Pearson/variance, each infection separately, all twelve reference policies and all controls. The selected-simple policy is chosen independently on each fold's validation set. All-reference aggregation is a post hoc robustness sensitivity and never a test-selected deployed reference.

Primary paired differences are aligned candidate squared loss minus selected-simple squared loss on identical pairs. Use2,000 endpoint-multiplicity gene-resampling draws, seed20261022, with discarded empty draws and leave-one-gene-out influence. These are conditional sensitivity bounds, not population/culture intervals. Dependence across folds, cohorts, guides and common controls remains.

The six-task gate requires every task to have at least50 aggregate test pairs/ten genes, informative controls, at least5% lower mean-target MSE than the validation-selected simple policy, negative upper gene bound, no MSE loss on either infection, no point-estimate loss to any fixed simple reference and lower MSE than every shuffled control. One failed task fails the gate; missing eligibility/control support is insufficient. Report losses without exclusions, rescue tuning or a wider menu. No model is promoted from these exposed data alone.

## Tests, ordering and resources

Before any biological fit, test schema/splits, reference arithmetic, streaming equivalence, outcome exclusion, symmetry, deterministic replay, safe serialization, callback guards, metric/gate rules and tamper/overwrite refusal. `freeze` hashes code, tests, protocol, banks and assignment bytes. `verify` writes passing tests with UTC ordering; `fit` requires it, writes a start receipt, and refuses completed/partial-run overwrite. It saves every model and all validation choices before `score`. `replay` requires exact scientific JSON equality and verifies checkpoints/prediction bytes. Numeric arrays/checkpoints remain local and ignored.

One CPU numerical thread; one-hour CPU/wall cap,2GiB process peak,128MiB new artifacts and1GiB free-volume reserve. Guard checks occur during streamed reference blocks and every completion optimizer step; native comparison uses its own qualified budget. A call executing in numerical code can return before the next check, so this is not a kernel-enforced hard limit. Stop on drift/cap/failure and preserve a failure receipt. No other numerical worker may compete with a resource profile or this fit.

Commands: `python tools/study_block_completion.py freeze --original C:/Users/USER/Desktop/opal-next-data-20261020 --prepared C:/Users/USER/Desktop/opal-external-validation-20261020/prepared-v2 --output C:/Users/USER/Desktop/opal-research-next-20261020/block-v1`, then `verify`, `fit`, `score`, `replay` with the same output. Commit the tested registration before fitting. Use new output IDs for any new protocol, never overwrite the historical banks or failed attempts.
