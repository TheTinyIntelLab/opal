# Cross-infection development results

The fixed low-rank diagnostic retains some predictive value beyond fitted gene main effects within the Horlbeck source, but it does not establish cross-study transfer. Its registered development gate passed against the validation-selected node baseline in both infection directions and contexts. A calibrated-single baseline nevertheless slightly beats the candidate on one Jurkat direction, so this is not consistent superiority to every simple comparator.

## What ran

The program reused all 81,466 eligible canonical pairs, the original 48,879/16,293/16,294 train/validation/test assignment, 427 eligible single-perturbation genes and two contexts. The schema retains 467 gene names; genes without eligible singles do not enter the pair population. Both infection directions used only source-infection singles and source-infection train/validation pair outcomes during fitting. The opposite infection's pair outcomes entered only final test scoring. Ten fixed reference fits and four neural fits ran with one CPU worker in 9.7668163 seconds. There was no rank, seed or penalty sweep, external retraining, post-result exclusion or rescue fit.

Each neural fit used the existing symmetric independent-context completion, rank 8, penalty .01, 500 steps and seed 20261022. Source-validation MSE selected node references in both contexts and directions. Normalized validation across the two contexts selected step 50 for infection 1 aligned residuals and step 75 for infection 2 aligned residuals; the shuffled controls selected step 25. The two directions share gene pairs, guides and study procedures and are not independent replications.

## Exact primary results

All numbers below score the same 16,294 held-out pairs on the opposite infection. Intervals are 2,000-draw paired gene-node multiplier sensitivity bounds for candidate minus baseline squared loss, not study-level confidence bounds.

- K562, infection 1 to infection 2: node MSE 0.004520430580963455, aligned completion MSE 0.004330892889946074, a 4.192912325998% reduction. The paired difference interval is [-0.00031342117397468186, -0.00005651624002883616]. Shuffled completion MSE is 0.004850925212199444, 7.311131656966% worse than node.
- Jurkat, infection 1 to infection 2: node MSE 0.0017150364889717867, aligned MSE 0.0016525404751805432, a 3.644004905616% reduction. The difference interval is [-0.00008822793012879588, -0.00003850818069527034]. Shuffled MSE is 0.0017308842134913813, 0.924045909314% worse than node. The calibrated-single reference reaches 0.0016497471036728134 and beats the aligned model by about 0.17%; this comparison was reported after scoring, without selecting or refitting a replacement.
- K562, infection 2 to infection 1: node MSE 0.0057359791210650525, aligned MSE 0.005269932970850844, an 8.124962458505% reduction. The difference interval is [-0.0006248305400282618, -0.00031773349914243523]. Shuffled MSE is 0.0059197827698176195, 3.204398845832% worse than node.
- Jurkat, infection 2 to infection 1: node MSE 0.001744245720273182, aligned MSE 0.0016890879576804416, a 3.162270198037% reduction. The difference interval is [-0.00007678263109822224, -0.00003527858449269994]. Shuffled MSE is 0.0017640401970603708, 1.134844509413% worse than node.

The frozen 2% gate and negative-control comparison pass in all four dependent comparisons. This authorizes a later untouched-source test of a fixed model and comparator policy. It does not promote a new architecture or reverse the recorded external failure.

## Secondary diagnosis

Replicate residual Pearson correlations after separately fitted node references are 0.38405057191907305 in K562 and 0.24047729446323093 in Jurkat, versus raw response correlations 0.8792044535647301 and 0.9382663312266531. These lower residual correlations show why raw response repeatability cannot establish pair-specific biology. Their positive values and aligned-versus-shuffled gains support useful within-study pair alignment after this particular main-effect adjustment, but shared guide effects, protocol artifacts and incomplete main-effect calibration remain possible explanations. The slightly stronger calibrated reference in one Jurkat direction reinforces that limitation.

Using target-infection singles as an explicit extra-information diagnostic gives minimum-single MSEs 0.009574427768501195 and 0.0023320278120007596 for K562/Jurkat in the first direction, and 0.012576061246177906 and 0.002198505969529191 in the reverse direction. These are worse than the source-fitted node references. Minimum-single alone is therefore an inadequate sole development comparator on this source, even though it can be a strong external comparator. Additive and zero predictors are also reported in the exact JSON receipt.

Same-infection candidate/node MSEs are 0.0029965961205532803/0.003323466493451854 in K562 and 0.0007443028977357639/0.0008197156849363185 in Jurkat for source infection 1. For infection 2 they are 0.0017578452982640485/0.0021144510145607674 and 0.0007381557772092951/0.0007941221600938662. Cross-infection error is larger in every case, so the source's internal gains coexist with meaningful infection shift.

## Ordering, tests and replay

The executable protocol and code were frozen at 2026-10-20 11:17:29.855182 UTC. Six synthetic tests passed after freezing, beginning at 11:17:31.489196 and ending at 11:17:35.266969 UTC. They cover data integrity, missingness, canonical duplicates and train support, baseline calculations, test-response exclusion from reference selection, deterministic negative controls and node uncertainty, pair symmetry, serialization, query rejection, pair leakage and the mandatory prefit receipt. Biological fitting began at 11:17:43.583564 UTC. Selection froze before scoring at 11:17:52.853601 UTC, and results completed at 11:17:53.352307 UTC. There were no recorded deviations.

The protocol SHA-256 is `308fe0f8652b6e024f89151abd7a07258ffa5c8e9dc09fa8ada5dc12299d5193`; prepared-bank SHA-256 is `7a0a6a4c7f79e79983071deb59b18d52493f4333133dfa89c100d6720b12cb94`; the unchanged assignment SHA-256 is `fe1bc4656ddecf73600ab3304ee04b497be34dbcc37e0c8ddd5babad27a6550f`. The receipts under [results/research-reset-development](../../results/research-reset-development/) include code hashes, package versions, exact per-reference metrics, histories and prerequisite ordering. Four checkpoint/composite prediction replays across all 16,294 test pairs have maximum absolute error 0.0. Raw bank arrays, checkpoints and predictions stay local behind that directory's `.gitignore`; no source measurements are redistributed.

The runner records wall time and checks its cap between fits; it does not instrument peak process RSS, enforce a disk budget, or interrupt a single stalled fit at the cap. No measured fit-memory claim follows from this run. The replay checks frozen protocol, prediction arrays and individual checkpoint hashes, but does not independently bind the selection receipt hash or current code/data to the result receipt. Those are audit limitations, not reasons to rerun outcome-dependent fits. The node multiplier intervals have no formal population coverage guarantee and remain conditional descriptive sensitivity bounds.

Reproduce using the original locally acquired data, not a fabricated fixture:

```powershell
.venv/Scripts/python.exe tools/reset_development.py freeze --data C:/Users/USER/Desktop/opal-next-data-20261020/prepared --prior-assignment C:/Users/USER/Desktop/opal-next-data-20261020/fitness-v1/assignment.npz --output results/research-reset-development-replay
.venv/Scripts/python.exe tools/reset_development.py verify --output results/research-reset-development-replay
.venv/Scripts/python.exe tools/reset_development.py run --output results/research-reset-development-replay
.venv/Scripts/python.exe tools/reset_development.py replay --output results/research-reset-development-replay
```

The original receipt directory is immutable; use a fresh output directory. A fresh numerical run is a deterministic replay of exposed development evidence, not another scientific replication. A machine without the prepared bank needs the exact GSE116198 source acquisition and `tools/prepare_fitness.py` preparation first; provenance and licensing limitations remain those in the original acquisition/preparation receipts, including the unverified identity between the GEO mirror and the separately licensed author deposit. The next study must test a frozen endpoint, comparator policy and shift handling on an actually untouched compatible source. This experiment leaves that source and its access resolution to the authoritative root plan.
