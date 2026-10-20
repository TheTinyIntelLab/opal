# Scalar-screen adapter and split prerequisites

The Q2/Q3 adapter loads six previously exposed source-context tasks without restricting later screens to the original gene vocabulary or historical external panels. It fits no predictor and calculates no new biological loss. Its metadata checks passed before the parent experiment registration. This gene-level adapter does not qualify SPIDR's exact-guide aggregation, controls, units or response access.

## Populations and identity checks

`opal.scalar_screen.load_screen_roster(original_root, external_prepared)` accepts the original preparation directory or its parent and the existing external `prepared-v2` directory. It verifies all five bank SHA256 identities against historical receipts, the external preparation receipt SHA256, and the original preparation receipt's declared source/bank identities. When original raw archives remain at their historical relative paths, it also verifies their bytes against the recorded source hashes. A missing archive is recorded as `raw_source_hash_verified=false`, without claiming raw verification. Every local archive was present and passed on this machine. No raw archives or arrays are added to Git.

| Task key | Stored genes | Genes in eligible pairs | Canonical pairs | Five block-test sizes |
| --- | ---: | ---: | ---: | --- |
| `horlbeck_K562` | 467 | 427 | 81,466 | 2,970 / 2,970 / 3,110 / 3,948 / 3,212 |
| `horlbeck_Jurkat` | 467 | 427 | 81,466 | 2,970 / 2,970 / 3,110 / 3,948 / 3,212 |
| `environment1` | 313 | 313 | 48,826 | 1,711 / 1,540 / 2,278 / 1,378 / 2,926 |
| `environment2` | 313 | 313 | 48,828 | 1,711 / 1,540 / 2,278 / 1,378 / 2,926 |
| `parp2022` | 543 | 539 | 144,601 | 3,916 / 7,343 / 5,766 / 5,348 / 6,643 |
| `parp2023` | 543 | 537 | 142,873 | 3,898 / 7,251 / 5,715 / 5,284 / 6,503 |

The Horlbeck prepared population remains unchanged. For every later task, the adapter requires finite pair targets, two surviving constructs in each infection, and finite local singles supported by two constructs in each infection for both genes. No additional pair exclusions arose on these banks. Stored but unqueried genes can lack singles; querying a missing single fails validation. Each of the 30 blocks has at least 50 pairs and ten test genes, with training-edge support for every validation/test query gene. These are coverage facts, not evidence of prediction quality or biological replication.

Each returned mapping contains Unicode `genes`, canonical integer `pairs`, gene-vector `singles`, gene-by-two-infection `single_replicates` and `single_counts`, pair-by-two-infection `replicates` and `counts`, and pair-vector `responses` equal to the infection mean. `source`, `accession`, `context`, `units`, `exclusions`, bank/source hashes and exposure labels retain provenance. Singles average the two infection measurements equally. The primary adapter target is gamma; LFC cannot enter this contract silently. Names remain case-sensitive and no biological synonym mapping is inferred. Whitespace and pair orientation can be normalized with `canonical_pair`; banks themselves must already have unique canonical names/pairs.

## Split and negative-control contracts

`block_assignments(screen)` returns five `int8` vectors with train/validation/test codes 0/1/2. A gene's block is the first eight SHA256 hex digits of UTF-8 `opal-reset-block-v1|gene` modulo five. Fold k tests every pair with both endpoints in k. Remaining edges are validation when the first eight SHA256 hex digits of `opal-reset-v2|source_accession|geneA|geneB` modulo five equal zero, and training otherwise. Shared source accession and canonical names keep paired contexts, orientations and infections together. `pair_assignments` implements the separate 60/20/20 contract with modulo ten and cutoffs six/eight. A block holds an interaction submatrix, not a cold gene.

`split_report` preserves every registered panel and reports unsupported query genes rather than dropping them. `training_statistics` reads only the training graph, permitted singles and training counts. It derives degree and absolute-single tertile cutpoints from active training genes with NumPy's linear quantile rule and assigns values on a cutpoint to the upper bin. Edge strata are the unordered endpoint-degree bins, unordered endpoint-single bins, and minimum construct-count bin 2-3 / 4-7 / at least 8. It never reads pair outcomes.

`residual_permutations` returns three controls with seeds 20261023, 20261024 and 20261025. Every result includes absolute training-row `permutation`, `local_permutation` for a training-only residual vector, `train_indices`, `strata`, cutpoints, moved fraction and permutable fraction. Permutable means the stratum has at least two edges; it differs from the moved fraction because a random permutation can retain a fixed point. Singleton strata remain fixed. The `sufficient` flag requires at least 80% of training edges in permutable groups. The graph and split never change. No seed can be selected from scored results.

The original two contexts share their graph and must share residual-label permutations. The loader therefore supplies identical `control_singles_summary`, averaging original singles over both contexts and infections, and `control_counts`, taking each infection's minimum count across contexts. These freeze conservative common count strata without using pair outcomes. Both original contexts then return identical permutations. `singles_summary` can explicitly supply a predeclared joint summary for another compatible joint task. Any new joint task also needs a shared count summary and graph; independently computing task-specific strata does not preserve context covariance.

## Metrics and conditional uncertainty

`metrics(target, prediction)` requires finite aligned nonempty vectors and reports equal-pair MSE, MAE, prediction-minus-target bias, centered Pearson correlation, target variance and pair count. Constant vectors return unavailable Pearson, never an invented zero. No metric pools different endpoints or units.

`paired_gene_uncertainty(loss_difference, pairs, n_genes=None, draws=2000, seed=20261022)` accepts paired candidate-minus-reference losses on identical canonical pairs. Each draw samples the active endpoint genes with replacement and weights each pair by the product of its two endpoint multiplicities. Empty draws are counted and discarded. The result reports the 2.5/97.5 percentiles, requested/nonempty/empty draw counts, active-gene count, and each leave-one-gene-out mean difference. Inactive stored genes do not alter the resampling population. These are conditional gene sensitivity bounds on measured outcomes, without an independent-culture or source-level coverage claim.

## Executable verification

Run the synthetic prerequisites before any new fit:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_scalar_screen.py -q
```

Nine tests cover canonical identity and duplicate/orientation rejection, missing values and integer counts, units and infection aggregation, the exact deterministic hash rule, shared-context splits, training support and insufficient panels, training-only strata, all three residual permutations and context covariance, common original-context strata, hand-calculated metrics, endpoint-multiplicity draws and leave-one-gene-out effects, historical hash refusal before deserialization, and source-local genes absent from Horlbeck. Synthetic loader integration replaces only expected fingerprints and does not claim a synthetic biological benchmark. No response-scoring result follows from these tests.
