# Model development decisions, 20 October 2026

The original expression experiment below remains unchanged. The appended execution records a new measured-growth endpoint, newly acquired public data, and a separate locked test; it is not a rerun or a revised interpretation of the expression results.

Outcome: **FAILED HYPOTHESIS**. The implemented hierarchical gene-local composition model improves substantially over literal additivity, but loses to simpler matched calibration or profile kernels. Close this candidate. Its failure is evidence against this particular response law, not proof that biological interaction prediction is impossible.

## What the research established

Three specialist reviews inspected official papers and source, with pinned revisions and explicit licensing/availability limits: [architecture evidence](research_20261020_architectures.md), [data/evaluation evidence](research_20261020_data.md), and [first-principles assessment](research_20261020_identifiability.md). No external implementation code or pretrained weights entered the new model.

TxPert does evaluate observed-single Norman doubles, but its strongest model uses proprietary PxMap/TxMap graphs and reports only a slight gain over additive Pearson delta. Public released entry points cover inference rather than a complete double-model training recipe. [Official TxPert source](https://github.com/valence-labs/TxPert/tree/08d82eea86746b044cf7531f4ec8c5f60e1cb73f) has noncommercial terms. [GEARS](https://github.com/snap-stanford/GEARS/tree/f374e43e197b295016d80395d7a54ddb81cc6769), MIT licensed, learns GO/coexpression representations but explicitly needs combination supervision; it trains with fourth-power errors and selects on DE-gene MSE. [CPA](https://github.com/theislab/cpa/tree/fbd7c0250edc23eff003a10c99655579c53afd63), BSD licensed, supports count likelihoods and perturbation/covariate embeddings, but its Norman tutorial holds out only two pairs and validates on random cells from known conditions. Those information and selection contracts cannot supply matched Opal scores.

The [2025 critical benchmark](https://www.nature.com/articles/s41592-025-02772-6) makes strong simple baselines essential. The [October 2026 calibrated-metric study](https://www.nature.com/articles/s41587-026-03307-w) reports PRESAGE beating additive on 15/18 Wessels metrics, with different normalization, population and endpoints. This supports a response-sensitive secondary metric, not importing those scores or changing the primary endpoint after a loss. scPerturBench's broad method/dataset count likewise does not establish superiority on a matched observed-single task. The full newer benchmark recommends five 24 GB GPUs, 384 GB RAM and 2 TB storage; that is a whole-framework budget, not a minimum for each model.

More general response-evidence approaches are scientifically interesting. PIE trains evidence-conditioned gene-query heads for DE and fold change, with an eight-GPU cross-dataset recipe and noncommercial share-alike terms. State uses cell-set representations at much greater data scale. Tahoe supplies 100.6 million cells across 50 cell lines and approximately 1,100 drugs, but its raw release is about 337.6 GB and current DE export spans 1,026 large shards. DE summary statistics are not interchangeable with matched expression means. Neither a pretrained embedding nor a larger single-perturbation corpus identifies arbitrary double interactions.

## Direction choice from first principles

Write a pair effect as a sum of single effects plus an interaction. For any observed singles there are infinitely many interaction functions that agree on those singles, so pair evidence or justified restrictions must supply the missing information. Under independent arm-mean errors, measured additive-feature error and residual-label error have covariance -Var(single A error) - Var(single B error) - 2 Var(control error). A fitted attenuation law can therefore correct measurement error without recovering causal antagonism. Shared singles and controls also make pair errors dependent.

Improving the observed-single task has narrow but immediate value: two coherent local fitting banks and exact provenance already exist. Their 88 and 110 pair labels are small compared with ten thousand genes, making a shared low-dimensional composition rule testable. A general identity-only predictor adds main-effect and context extrapolation to this missing interaction information, while compatible pretraining extraction is presently unfunded/unprofiled. Cell-distribution prediction requires actual cell-level targets and a different evaluation contract. Cross-context rank-16, bilinear and noise-aware alternatives already have unfavorable local results. These facts selected a bounded composition test rather than making the old architecture compulsory.

## Implemented hypothesis

For gene g, features are the sum s=a+b, same-sign signed overlap sign(s) min(abs(a),abs(b)) when ab>0, and opposite-sign signed overlap when ab<0. The prediction is a gene intercept plus a gene-specific linear combination of these features. Train-centered gene covariance fits are shrunk toward pooled slopes through a positive diagonal regularizer estimated from training features. A tiny positive floor handles constant features. Exact opposing effects have zero overlap, and signed opposite overlap jumps when the sum crosses zero. This restricted law can be inappropriate near cancellation and cannot express arbitrary equal-and-opposite interactions.

The model is independently implemented in src/models/composition.py with fit/predict/save/load and safe numeric checkpoints. Gene count is validated; callers must enforce gene order and assay identity using the saved schema. This is predictive response calibration, not causal discovery. Matched affine and pooled-overlap models remove the nonlinear features and gene-specific slopes respectively, testing the two proposed advantages.

## Data and locked evaluation

Only data/prepared/norman-log2/profiles.npz and wessels-expanded-protected-log2/profiles.npz were read. They contain 88 pairs/105 singles/10,623 genes and 110 pairs/28 singles/10,590 genes. The complete preparations have 126 and 158 eligible pairs; separate historical evaluator responses were excluded. Both local sources have prior development exposure. Gene panels, original QC, log2 scales and source contracts were unchanged, and no raw expression or weights are committed.

The protocol was frozen and the initial implementation committed at 538423e before scoring. Five outer and three inner folds use seed 20261020 under pair and partner holdouts. All held pair labels are globally excluded from fitting and selection. The finite menu includes additivity, no change, shared residual mean, 67 profile-kernel configurations, nine reduced-rank residual ridge configurations, three affine penalties, three composition penalties and one pooled-overlap fit. Each fold chooses its reference inside training data. This is not an exact official GEARS/CPA/TxPert reproduction; prior official adaptation results remain separate historical evidence.

Primary MSE averages genes and pairs equally. Partner folds overlap: a pair can be held out once or twice, and its leak-free predictions are averaged before computing primary squared error. This is a cross-fold ensemble estimand, not the average single-fit partner-fold loss. Each constituent fit excludes all pairs incident to its held target group; the ensemble does not establish prediction when both intervention identities lack pair evidence everywhere. The secondary uses outer-training response-variance gene weights and averages held-out occurrences rather than ensemble predictions, so its aggregation differs from the primary. Conditional paired bootstrap uses 2,000 draws. Its intervals condition on shared references and deposited pairs and are not biological prediction intervals. Promotion required a 2% gain across both datasets and protocols plus a below-zero random-split loss-difference upper bound. No criterion changed after outcomes were inspected.

## Quantitative result and failure analysis

| Source / holdout | Additive | Kernel | Affine ablation | Pooled overlap | Composition | Selected baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Norman / pair | .00827860 | .00620155 | .00608876 | .00634816 | .00621337 | .00608876 |
| Norman / partner | .00827860 | .00673369 | .00632210 | .00644660 | .00634312 | .00632210 |
| Wessels / pair | .02469173 | .00711547 | .00844173 | .00970153 | .00850956 | .00711547 |
| Wessels / partner | .02469173 | .01370019 | .00919508 | .01034119 | .00928111 | .00919508 |

Composition is worse than the locked reference selector by 2.05%, 0.33%, 19.59% and 0.94% respectively. The random-pair 95% conditional loss-difference intervals are [0.0000517, 0.0002180] for Norman and [0.0008875, 0.0018585] for Wessels, both favoring the reference. These are within-source conditional intervals only. All fold losses, every baseline family, selection histories, coverage and secondary scores are retained in [aggregate results](../results/composition-20261020/results.json).

The important ablation is affine calibration: it beats composition in all four settings. Hierarchy helps against pooled overlap, but the overlap terms do not earn their added freedom. Wessels' kernel is strong on random pairs and deteriorates on partner holdouts, indicating an extrapolation/support limitation. Composition does not solve it. Closed-form fitting reaches the stated regularized solution, synthetic recovery passes, symmetry is exact, and checkpoint replay is exact; there is no evidence that longer optimization would rescue this hypothesis. Limited pair supervision and noise remain plausible restrictions, but this experiment cannot separate their biological contributions without repeated measurements.

The evidence-backed decision is to close the overlap law. Keep affine calibration as the relevant reference on these Norman and partner-held Wessels splits, and the kernel on random-held Wessels. Do not select a universal default from exposed outer outcomes. A next candidate must bring compatible additional pair-response information or an explicitly justified new endpoint and data extraction, rather than a larger residual network.

## Compute, operational failure and reproducibility

The successful compact run took 33.491 seconds with one numerical worker/BLAS thread. Windows OS peak working set was 494,342,144 bytes, about 471 MiB; the retained-artifact estimate was 66,131,599 bytes, about 63 MiB. The registered limits were one hour, 2 GiB RSS and 128 MiB outputs. Resource checks occur at fold/finalization boundaries rather than enforcing an instantaneous process kill. Actual resource use stayed well below those limits.

The first attempt finished all fits but failed its final storage check, retaining 246 MB of baseline prediction arrays. It remains preserved. The only replay change retained gene-level arrays for composition and the selected reference, while preserving every baseline scalar and tuning record. Every fold record and score reproduced exactly. [The protocol](../results/composition-20261020/protocol.json) pins the compact runner; [the artifact manifest](../results/composition-20261020/artifact_manifest.json) hashes checkpoints, schemas and predictions. Final all-development refits contain 42,492 and 42,360 fitted gene coefficients/intercepts, have no evaluation score and reproduce after load with zero maximum absolute difference. Source-specific weights stay local under existing source terms.

The aggregate's final FAILED HYPOTHESIS adjudication replaces the raw runner's conservative default INSUFFICIENT EVIDENCE label; neither raw scores nor the immutable local summary was edited. Independent evidence remains insufficient even though this particular development hypothesis failed.

## Repository reset and delivery checks

plan.md is the sole forward plan. Six individually inspected obsolete plan/dispatch/handoff files were removed. The former README is a closed dated engineering record; ignored private context/roadmap snapshots were preserved verbatim outside the active tree, since the roadmap contained unique historical scientific specifications. Historical experiment contracts, failures, source provenance, licenses, measurement records, tests and private correspondence protections remain. [The cleanup audit](../results/composition-20261020/planning_audit.json) records exact dispositions.

Validation and remote/CI receipts are recorded separately in results/composition-20261020/validation.json after execution. The first full local test invocation passed 395 tests and hit three existing free-volume guards on D:. All 398 tests then passed on E: in 58.77 seconds without changing those guards. Wheel/sdist builds, installed-wheel fresh-refit replay and package private-data exclusions passed; the documentation build rendered 93 documents with zero missing local links. The shared host environment's pip check reports pre-existing datasets/fsspec and platformio/click/starlette conflicts; no unrelated dependency versions were changed. A fresh heavyweight dependency installation is inappropriate under the storage constraint, so installed-wheel import/replay and hosted clean CI provide the package checks.


## Executed follow-up: measured-growth pair completion

The plan audit found one material forward gap. Its completed composition experiment had a concrete objective, architecture, data, baselines, splits, compute and decision criteria, but its continuation requirement named neither an acquisition route nor an executable experiment. plan.md now specifies the executed functional task below. No duplicate roadmap or new research report was created.

### Data actually obtained and qualified

The official GSE116198 pair-phenotype gzip was downloaded successfully, 112,096,990 bytes in about 80 seconds. The source is Horlbeck et al., [Mapping the Genetic Landscape of Human Cells](https://doi.org/10.1016/j.cell.2018.06.010), with the [author dataset](https://data.mendeley.com/datasets/rdzk59n6j4/1) explicitly CC BY 4.0, corroborated by DataCite. The Mendeley file API returned 403, so acquisition used the official GEO deposition. Exact file-byte equivalence with Mendeley remains unverified, and GEO has no separately verified license. This qualification is retained rather than silently relabeling the mirror; local scientific analysis uses attribution, with raw data and trained data-derived assets excluded from git.

The public NCI-ALMANAC compact archive was also downloaded as a practical fallback. It was not mixed with genetic growth measurements. The preferred source succeeded, so no drug-model or alternate-endpoint search was run.

Preparation streams 922,455 ordered guide-pair rows and uses only the triple-sequencing gamma columns. Gamma measures control-centered growth normalized by replicate doublings. Matched singles come from gene/negative-control pairs. Published GI scores and GI correlations are excluded because they incorporate global pair outcomes during construction. Eligibility depends only on observation counts: at least 20 single observations and four ordered guide-pair observations in each context and infection replicate. This yields 81,466 canonical pairs among 427 supported genes in K562 and Jurkat, from 467 parsed genes. Equal guide/orientation weights are used, not inverse-variance weighting. The shared-coverage panel is enriched for growth-related genes and cannot represent the whole genome.

The complete [acquisition and preparation receipts](../results/fitness-20261020/) retain source URLs, SHA256 hashes, license evidence, counts and exclusions. Preparation took 9.83 seconds. No historical expression evaluator files were accessed.

### Locked task and actual implementation

The target is the measured mean double-CRISPRi growth effect, given matched single growth and other measured gene-pair effects in the same two contexts. This is a functional endpoint change, not a claim of improved expression prediction or general cell simulation. It replaces approximately one hundred expression pair labels per source with over eighty thousand compatible pairs and supplies measured infection replicates without a large atlas extraction.

The candidate fits an initial calibrated gene-main-effect reference, then a symmetric indefinite low-rank residual matrix with common factors and context-private factors. References include zero, measured-single additivity, minimum single effect, scalar calibration, calibrated gene-main-effect ridge, independent-context completion and pooled-only completion. Joint, independent and pooled factorizations have equal parameter budgets, 4 * 467 * rank. Common rank r plus private rank r/2 per context tests whether sharing helps at equal size.

The protocol and implementation were committed at fa517d8 before fitting. Seed 20261021 fixes 48,879 training, 16,293 validation and 16,294 test canonical pairs, identically in both contexts. Every guide variant and orientation remains with its pair. All query genes have substantial training-pair support. No held pair can supply a label through the other context.

Reference scalers and coefficients use training data only. Validation chooses the reference per context, completion mode/rank/penalty and checkpoint every 25 steps. The fixed grid is ranks 8/16 and penalties 0.001/0.01, one initialization seed, Adam learning rate 0.03 and 500 steps maximum. All selections were persisted before computing test losses. Candidate selection uses context-balanced validation MSE normalized by training-only target variances; the strongest non-joint reference is selected independently per context by validation. No settings changed after test scoring.

### Actual result and decision

| Context | Additive MSE | Calibrated main-effect MSE | Selected independent completion MSE | Joint candidate MSE |
| --- | ---: | ---: | ---: | ---: |
| K562 | 0.04631078 | 0.00187049 | 0.00137593 | 0.00136626 |
| Jurkat | 0.00486848 | 0.00050049 | 0.00042630 | 0.00045576 |

Independent completion improves over the validation-selected calibrated main-effect reference by 26.44% and 14.82%. Those gains support retaining a fitted specialized growth predictor. They do not establish novelty over matrix completion, improvement on expression, or superiority to published functional-screen predictors under matched evaluation.

The cross-context sharing hypothesis is **FAILED HYPOTHESIS**. The joint candidate gains only 0.70% in K562 and loses 6.91% in Jurkat against the strongest validation-selected reference. Its mean normalized MSE is 0.06060042 versus 0.05997171. K562's conditional loss-difference interval is [-0.00002059, 0.00000117], and Jurkat's is [0.00002373, 0.00003533]. Both-context 2% superiority and below-zero upper bounds were required, so promotion fails. These pair-bootstrap intervals condition on this deposited screen and shared references; they do not cover unseen genes, independent cultures or another laboratory.

The joint winner is rank 8, penalty 0.01, step 125. Validation selects independent rank 8/penalty 0.01/step 300 for K562 and rank 16/penalty 0.01/step 200 for Jurkat. These choices were not changed to the best test-performing configurations. Full histories and all reference scores are retained in [results.json](../results/fitness-20261020/results.json).

The practical outcome is a **BASELINE-LEVEL specialized predictor relative to the strong completion reference**, with a measured improvement over single/main-effect calibration on this task. Retain the independent-context models and close the shared/private menu. Pooled-only completion is also weaker, consistent with harmful sharing of some context-specific structure. Increasing optimization steps or sweeping ranks is not justified by this test.

Infection-replicate disagreement MSE is 0.00575468 in K562 and 0.00210994 in Jurkat. These are descriptive discrepancies, not a biological noise ceiling: the target averages replicates and guides, and shared normalization/control errors remain. High full-growth correlations, 0.953 and 0.986 for the selected reference, partly reflect main effects; they do not establish mechanistic interaction recovery.

### Usable models, cost and verification

The finite experiment took 18.72 wall seconds and 18.48 CPU seconds with one numerical/BLAS/PyTorch thread. Peak Windows working set was 960,188,416 bytes, about 916 MiB. It retained 41,914,030 bytes before the final receipt, under the 64 MiB experiment cap. Every first-fit/resource check passed; no failed numerical attempt or rescue grid occurred. All completion checkpoints replay exactly after numeric save/load.

The named-gene FitnessPredictor API combines the correct observed-single/main-effect reference with each validation-selected independent completion checkpoint. It accepts new pairs of supported gene names in K562 or Jurkat and refuses unknown genes, unsupported training genes, identical-gene pairs, new contexts and modified assets. It does not return calibrated intervals. The complete assets remain local at C:/Users/USER/Desktop/opal-next-data-20261020/fitness-reference-model. See the README for use.

The original runner stored residual checkpoints and reference predictions, but not reference coefficients for arbitrary supported pair queries. Export therefore repeated only the exact closed-form training-reference fit to serialize coefficients, with no factor retraining, no new selection and no test labels in fitting. Composite predictions reproduce the locked test reference to at most 5.6e-17. The export receipt and manifest document this specific replay reason.

The next evidence needed is an independent compatible growth screen, preferably with repeat-level measurements and overlapping genes, to test whether the within-source completion gain survives. New-gene prediction would require separately sourced gene features and a node-held split. Neither objective is silently added to this experiment, and the failed expression candidate remains closed.

Validation for this follow-up passes all 401 tests in 54.10 seconds, wheel/sdist builds, installed-wheel composite replay, original frozen source hashes, data hashes and 93 rendered documents with zero missing local links. Hosted CI and final remote head are checked at delivery; the exact verification receipt is in results/fitness-20261020/validation.json.
