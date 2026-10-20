# Independent arithmetic and artifact review of block-v2

The six registered block-held development tasks pass the specified gate. A separate agent checked the frozen artifacts and recomputed the losses, reference mathematics, checkpoint selection and conditional uncertainty without fitting another model or selecting another configuration. No material arithmetic, split or control-alignment error was found. This validates the reported locally supervised development result, not zero-shot transfer or untouched-source confirmation.

## What was verified

The audit read `C:/Users/USER/Desktop/opal-research-next-20261020/block-v2`. Code, source-bank, assignment, per-fold selection, checkpoint and prediction SHA256 values agree with the frozen registration and result chain. The protocol SHA256 is `4d903a39ff8f46c4626c072593f6e221193f5580caeeb53a76fbdceff71817de`; selection SHA256 is `69fdb30c8cf489dc6c99be4fab9f2f134871c05a33a19fa57a149533f43b77d8`; scientific results SHA256 is `f34ec5adc9e97802667dd43e41c2bbb5abc4056913110a73da4cd58c915cabe4`. The copied public receipts are under [results/research-reset-block/block-v2](../../results/research-reset-block/block-v2/).

The registration preceded 20 passing prerequisites, which preceded fitting. UTC timestamps are registration 16:11:21.174171, tests completed 16:11:26.386805, fitting started 16:15:24.343644, all selections completed 16:17:22.608219, and exact replay completed 16:18:23.665716 on 20 October 2026. Both historical banks and current code remained hash-identical during this audit.

All 30 assignments were independently compared with the SHA256 block recipe. Each test block contains every eligible edge whose endpoints share that gene fold. Validation and training use the separate registered pair hash, with no test edge in either. Every validation/test gene has training-edge support. All five test blocks within each task are disjoint. The runner passes only that fold's training outcomes to reference statistics and residual fitting; validation outcomes select references/checkpoints; test outcomes enter scoring after all selections are saved. Other cross-validation folds reuse these outcomes for their own training, so the experiment is cross-fitted development rather than a globally sealed outcome bank.

All 360 saved reference validation losses were recomputed through an independent direct formula, agreeing within `2.168404344971009e-19` MSE. Every selected reference was a gene-main-effect model. For all 30, training feature centers/scales agree with direct calculations, and the saved coefficients satisfy `Z.T @ (y - prediction) = alpha * coefficient`; the largest absolute normal-equation residual is `1.8068213591959648e-11`. This checks the streamed sufficient-statistic ridge objective against the actual training labels without solving another fit.

The aligned and three control fits use residuals from the same validation-selected reference. Each permutation maps training indices to training indices within its frozen nuisance stratum. All five original K562/Jurkat control records, including permutation hashes, agree exactly, preserving shared-context pair alignment. The code changes the residual-label assignment rather than the training graph. Permutations and their eligible/moved fractions were recomputed and checked against the prefit registration. These coarse strata do not preserve every gene-specific nuisance component and do not establish biological mechanism.

For all 120 completion checkpoints, the recorded best step is the minimum recorded validation loss and the stored scale equals the training-target standard deviation. Independent reconstruction of the symmetric packed operator and constituent reference reproduces saved test predictions to maximum absolute difference `5.551115123125783e-16`; the frozen runner's own replay is exact. Recomputed checkpoint validation losses differ from recorded pre-storage histories by at most `9.893613706069004e-12`, because float32 symmetrization occurs when the selected operator is stored. This roundoff is far below the observed margins and does not alter the recorded checkpoint choice or any gate.

## Independently recomputed results

The primary metric gives equal weight to each held canonical pair. The interval is candidate-minus-selected-reference squared loss, using 2,000 endpoint-multiplicity draws with seed 20261022. A separate implementation reproduced all six interval endpoints and every leave-one-gene-out mean exactly. There were zero empty draws, and every leave-one-gene-out mean difference remains negative.

| Task | Held pairs | Candidate MSE | Selected-simple MSE | Reduction | Conditional 95% difference interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Horlbeck K562 | 16,210 | .0013418626513990082 | .0018656418356246718 | 28.0750128039% | [-.0006657777573958883, -.0004222821262717539] |
| Horlbeck Jurkat | 16,210 | .0004341567404863752 | .0005009162998320582 | 13.3274879193% | [-.0000832032930257144, -.000052022296971933526] |
| Environmental 1 | 9,833 | .0003655341517065346 | .0005450429006015583 | 32.9347926002% | [-.00023066612655072257, -.00013879717823451096] |
| Environmental 2 | 9,833 | .00046475313111452585 | .0007248761005929065 | 35.8851628941% | [-.0003331499274281014, -.00020001251848345236] |
| PARPi 2022 | 29,016 | .00021159725552219704 | .0003095503277684733 | 31.6436661374% | [-.00011722309079475957, -.00008160793367582865] |
| PARPi 2023 | 28,651 | .00022298166618172846 | .00030161626937119646 | 26.0710747976% | [-.00009321545440590561, -.00006531156220378251] |

All six gains exceed 5%, all upper conditional bounds are negative, and both individual-infection losses decrease in every task. The same saved prediction is evaluated against each infection; these are not models independently trained on one infection and evaluated on the other.

| Task | Infection 1 reduction | Infection 2 reduction | Best fixed-simple sensitivity | Reduction against that simple |
| --- | ---: | ---: | --- | ---: |
| Horlbeck K562 | 14.4712892958% | 17.7868968498% | node/10 | 28.0554604343% |
| Horlbeck Jurkat | 6.5268487425% | 6.2171403386% | node/1 | 13.3274879193% |
| Environmental 1 | 18.9681962254% | 17.3995335377% | node/1 | 32.9305307778% |
| Environmental 2 | 26.5088402697% | 26.6239763886% | node/100 | 35.7227897206% |
| PARPi 2022 | 15.6399443590% | 14.8204056454% | node/100 | 31.6417388509% |
| PARPi 2023 | 19.2815036220% | 18.7923476744% | node/100 | 26.0701654692% |

The all-reference check recomputes all twelve policies on identical pairs and confirms no candidate loss to any of them. The best-simple column is a post hoc robustness sensitivity, not a replacement selected on test outcomes. Every aligned candidate also beats all three shuffled controls:

| Task | Shuffled 20261023 MSE | Shuffled 20261024 MSE | Shuffled 20261025 MSE |
| --- | ---: | ---: | ---: |
| Horlbeck K562 | .0019398315482482603 | .0019294581601625898 | .001926510136141657 |
| Horlbeck Jurkat | .0005010491922324894 | .0005011489521394496 | .000501207492523213 |
| Environmental 1 | .0005516759662992939 | .0005509109981368739 | .0005502871919789376 |
| Environmental 2 | .0007304275252445958 | .0007306869872054508 | .0007328735935707119 |
| PARPi 2022 | .00030821616722921555 | .00030788945371707036 | .0003078634660238047 |
| PARPi 2023 | .0003003413124267093 | .0003001518568007434 | .0003004124840200874 |

Some shuffled controls modestly improve the selected-simple reference, especially PARPi. The gate correctly compares aligned completion directly with every control, so this does not invalidate its pass. It does caution against interpreting all residual gains as unique biological interactions.

## Scope and remaining limits

Block-v2 stored 61,420,673 bytes when inspected, below its 128 MiB cap. The fit receipt reports 118.2621777 seconds active wall time, 115.640625 seconds CPU and 1,053,954,048 bytes Windows process peak. The fit-plus-score receipt reports 123.8373867 seconds wall and 121.171875 seconds CPU. These receipts concern the original registered run; this separate audit does not replace them. Software callbacks provide measured/checked resource guards rather than kernel-enforced limits.

The result supports useful source-local pair alignment after strong local calibration, including held interaction blocks among supported genes. It does not reverse the failed unchanged-weight external experiment: later tasks now use their own measured singles, many local pair-training labels, validation selection and full source-local populations. They also differ from the historical small bank-novel panels. Shared source lineage, overlapping panels and common controls preclude treating six tasks as six independent biological replications. Conditional gene bounds are not culture-level uncertainty, and shared measurement/reference artifacts remain possible.

No new fit, architecture search, outcome exclusion, SPIDR response access or confirmatory claim was introduced by this audit. Comparable native-method results and any reserved-source evaluation require their separately registered evidence. The [main result report](block_results.md) records the experimental interpretation and reproduction commands. With the original local assets, run `python tools/study_block_completion.py replay --output C:/Users/USER/Desktop/opal-research-next-20261020/block-v2` after other guarded numerical jobs finish to reproduce the frozen prediction/scientific-JSON check.
