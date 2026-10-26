# Grok growth critique and implementation proposal

Independent note for root review, written before any biological fit or score. This adapter does not reopen the stopped historical gate, repair it, qualify COLO1, or read source banks. A reported new source route returning HTTP 403 with an empty body is not used here. A reported synthetic pilot fit of 144 configurations is outside this contract and is not scored by this worker.

## What is being compared

The historical six-task, five-block, 20,000-label union stays the population. Block membership stays `scalar_screen.block_assignments`. The charged rows stay the full-SHA256 prefix of `opal-budget-v1|accession|geneA|geneB` inside each original train and validation partition: 16,000 train and 4,000 validation. Leftover rows are not pulled in to replace a charged row. The original test block is stored in full and is not a budget prefix.

The predictor is the existing tournament menu, called unchanged: 16 simple configurations, five aligned interaction configurations, and three matched residual shuffles. Checkpoints remain on contract `opal-frontier-synthetic-v1`. The experiment protocol is separately named `opal-frontier-growth-v1`. Historical completion seed 20261022, historical shuffle seeds 20261023–25, and historical checkpoints are not rerun and not overwritten. Growth fits use the tournament fit seed 261401. The tournament shuffles are the four-stratum median splits inside `fit_models`, seeds 261201–261203, on that screen's own measured singles. They are not the historical tertile shared-summary permutations.

## Unused genes and nonfinite singles

`validate_screen` requires finite singles only at `singles[pairs]`. The gene array may still contain genes that occur in no eligible pair, and those entries may be NaN. `fit_models` requires every entry of its singles vector to be finite.

The prospective rule is response-free. From gene names, pair endpoints, and singles only:

- Keep a gene only when it occurs in the eligible pair roster.
- Preserve original relative order, names, and every pair row.
- Reindex pair endpoints onto that collapsed gene list.
- Pass the collapsed finite singles vector to `fit_models`.

No missing single is imputed. No test pair is deleted. A nonfinite single on a gene that occurs in any eligible pair, including a test-only gene, fails qualification before fitting. Integer indices change when an unused gene sits between used genes; the persisted identity is the gene-name pair. Block and budget hashes are computed on the original names and original pair rows before reindexing, then the same row indices are reused. Collapsing unused genes also shrinks the gene-incidence width to genes that can actually be partners. That is a consequence of the roster, not a label repair.

This worker did not open banks, so the number of unused genes on the six tasks is unmeasured.

## Selection, seal, and gate

Each fold fits the equal-infection mean only. Both infection columns and the mean, on the full original test block, are written under `sealed-test/` before any checkpoint. `fit_models` reads `fit-args.npz` only. Validation selection uses every charged finite validation row, with the tournament's fixed single-sum fallback below two distinct training partners. The deployed policy is the validation minimum among unshuffled families, menu order on exact ties. Persisted artifacts are every fixed simple model and the validation-selected aligned and shuffled interaction families: 28 checkpoints when shuffles are on.

Test metrics start only after all 30 procedures have a sealed checkpoint and prediction manifest. The pass bit is unchanged: pooled common-support pairs, equal-pair weight, at least 90% two-partner support, 50 pairs, 10 genes, informative shuffles, at least 5% mean-target MSE reduction versus the validation-selected simple, upper 2,000-draw conditional gene bound below zero (seed 20261022 via `paired_gene_uncertainty`, including its leave-one-gene map), no infection loss versus that simple, no point loss to any of the 16 fixed simples, and a strict win over each shuffle's validation-selected policy. One failed task fails the program.

The results JSON also stores MSE, MAE, bias, and correlation for every one of those baselines, not only the veto names. Each block is the mean target and both infections, on common support and on the full test population. The pooled series are the validation-selected full policy, the validation-selected simple, aligned I0, I1, and I2 using each fold's own validation-chosen configuration, and the three validation-selected shuffled policies. A lowest TEST mean-MSE label is stored as `test_best_diagnostic` with `nominee` false. It does not replace the validation-selected policy and it does not change the pass bit. No new fit or selection is added.

`fit_models` raises on an uninformative shuffle instead of returning a partial menu. Because that function is frozen, the stage stops and keeps the failure receipt. There is no seed, stratum, rank, or task rescue.

## Profile and forecast

Profile is synthetic, before any roster load. It uses `make_world` additive and low-rank labels at 467 and 600 genes, noise 0.1, complete mask, world seed 261501. That seed is not 261101–261103 and not pilot seed 261001. Each world charges a 16,000/4,000 hash prefix of its own train and validation partitions, using `frontier-budget-<role>`. Unavailable prefix rows stay charged and are not replaced. The complete mask makes used equal charged.

The generator's 600-gene holdout is `179700 - (179700//2 + 179700//5) = 53910` pairs. That is larger than a within-gene-fold biological test. Profile predictions use a fixed 10,000-query prefix of that holdout, ordered by the full SHA256 of `frontier-budget-test|geneA|geneB` on the integer endpoints. The prefix does not read labels. It is not lengthened if a biological fold is larger, and it is not shortened to manufacture a passing forecast. Both 467 and 600 supply at least 10,000 holdout rows (32,644 and 53,910). The profile persists 28 models per procedure and replays those 10,000 queries with no refit.

Before any biological `fit_models` call, every original block-test count is compared with 10,000. A larger fold fails qualification. That fold's test pairs are not truncated, and the 10,000 profile budget is not raised. Folds at or under 10,000 keep every test row.

## Storage amendment

This changes the artifact allowance only. The six-task gate is unchanged: 5% gain, 90% support, 50 pairs, 10 genes, informative shuffles, negative upper conditional bound, no infection loss, no fixed-simple point loss, and strict shuffle superiority.

Each procedure keeps four dense float64 completion matrices (aligned I0 and three shuffles). At the 600-gene profile bound:

`4 * 600 * 600 * 8 * 30 = 345,600,000` raw bytes.

`256 MiB = 268,435,456`, which is less than those matrices alone, before simple checkpoints, kernels, neural weights, predictions, manifests, or the profile tree. Root's pre-outcome allowance is therefore `512 MiB = 536,870,912`. The raw matrices use 345,600,000 of that, leaving 191,270,912 bytes for everything else. Compression may be smaller; the forecast still has to come in under 512 MiB of measured files. This amendment does not declare the forecast qualified.

Profile caps stay 600 active wall seconds, 600 CPU seconds, 2 GiB process peak, and 64 MiB artifacts. Four compressed procedures plus 10,000 queries are the workload that cap is meant to hold. The program cap is 7,200 wall seconds, 7,200 CPU seconds, 2 GiB, 512 MiB artifacts, and 1 GiB free disk plus 32 MiB working allowance. Forecast is `30 * max(measured procedure) * 1.25` for wall and CPU, and profile bytes plus `30 * max(procedure bytes) * 1.25` for artifacts. A failed forecast writes the blocker and `fit` refuses.

## Custody

`freeze` hashes this adapter, its tests, this note, the tournament model and tool, the transitive model modules, and the block, budget, and growth protocols. It records the bank hashes already named in `scalar_screen` and the original/prepared paths parsed from the single freeze command in `docs/research_reset/block_protocol.md`. It does not stat or open those paths. `load_screen_roster` is called only from `fit`, and only after a qualified forecast. Existing output directories, partial stages, marker files, binding drift, and a changed sealed-test snapshot are refused. Replay reloads checkpoints with `Predictor.load` and the expected digest, recomputes predictions, and does not call `fit_models`. An installed replay root must sit outside this checkout, and the imported `opal.frontier_tournament` file must sit under that root. The tool does not put `src/` on `sys.path`. Helpers come from checkout `tools/frontier_tournament.py`.

Commands, for a new output directory, after root review:

```powershell
python tools/frontier_growth.py freeze --output <new-output>
python tools/frontier_growth.py profile --output <new-output>
python tools/frontier_growth.py fit --output <new-output>
python tools/frontier_growth.py score --output <new-output>
python tools/frontier_growth.py replay --output <new-output> --installed-root <installed-tree>
```

## Known limitations

- No biological result is claimed. Fixture tests use an independent seed, `controls=False`, and 5 optimizer steps. They are not the registered profile.
- This session does not run profile, fit, score, or the fixture suite. Root owns numerical execution.
- `fit_models` has no per-step guard callback, so wall, CPU, and memory can overshoot between procedures. The memory figure is this process's peak working set, not a kernel reservation.
- The dense gene-incidence design inside the frozen fitter is the main 600-gene memory risk. The 512 MiB allowance covers the raw completion matrices and still fails closed if measured artifacts exceed it. Families are not dropped to force a pass.
- Profile query cost is 10,000 synthetic pairs, not the 53,910-pair 600-gene holdout. A biological fold above 10,000 test pairs stops the fit.
- K562 and Jurkat do not share the historical shuffle permutation, because the frozen fitter shuffles each screen's own singles.
- Installed replay must be a process that already imports `opal` from the installed tree. This module does not rewrite `sys.path` to swap an editable checkout import for a wheel.
- Native NAIAD is not fit. Its published budget results stay context.
- Passing the six-task gate would still be exposed-source development, not confirmation and not a COLO1 qualification.
