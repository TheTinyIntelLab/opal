# Completed state-response build

Opal now has an executable baseline-only main-effect benchmark, a compact
state-conditioned prototype and symmetric staged/joint residual heads. The new
scientific result is negative: baseline state has not added material held-context
value over exact-intervention source averaging in either qualified development
source, and the learned models are worse than the matched simple references.

## Main-effect results

| Source and fixed population | Target mean MSE | Selected simple MSE | Learned MSE | Simple gain over mean |
| --- | ---: | ---: | ---: | ---: |
| EmeraldBay, 136 supported condition means, 8 held line groups | 0.064758 | 0.064680 | 0.073678 | 0.121% |
| Corrected sci-Plex 3, 144 condition means, 3 held line groups | 0.074913 | 0.074906 | 0.091868 | 0.010% |

The 5% main-state and representation gates fail on both sources. Learned errors
are 13.91% and 22.65% higher than selected simple errors. Shuffled-input and
removed-state controls use the same supervision and source-average output
anchor. The sci-Plex single-dose descriptor is constant, so shuffling it is an
uninformative negative control, not evidence about chemical structure.

The corrected source was downloaded and publisher-MD5 verified before use.
Qualification excludes unassigned cells and the separate 72-hour A549 block.
There are 24 exact 1,000-nM/24-hour interventions, 24 line/plate reference strata
and 512 control-selected human genes. Plate-matched baselines prevent duplicated
or mismatched controls; whole-line grouping keeps both replicate labels together.
Nominal replicate-difference MSE is 0.063736, much larger than the 0.00000718
selected-simple improvement. This contrast includes batch/plate and biological
variation and cannot identify a measurement-only noise floor.

## Combination results

Only 54 existing EmeraldBay source-development pair rows enter new outer
context folds. Original pair calibration and evaluator outcomes are not opened.

| Estimated-single procedure | Pair MSE |
| --- | ---: |
| Additive | 0.189780 |
| Source-selected matched residual kernel | 0.094605 |
| Staged state residual, sum/difference | 0.112389 |
| Joint with frozen encoders | 0.101631 |
| Joint with fine-tuned encoders | 0.101485 |
| Joint without neural single pretraining | 0.095165 |

State conditioning improves pooled neural residual heads by about 4%, and joint
fine-tuning improves the staged neural pair score. Every learned procedure still
loses to the matched kernel. Per-fold held-single diagnostics and all four terms
of the total/main/residual/cross-term error identity remain recorded. Large
negative cross terms prevent claiming that favorable pair errors establish main
transfer or recovered biological interaction. No outer winner is promoted after
viewing these comparisons. Intermediate information-mismatched runs remain
preserved with explicit [amendments](../results/state/amendments.json).

## Reliability, resources and reproducibility

Separate risk-fit, calibration and evaluation groups produce development
diagnostics. Finite nominal-90% group radii are unavailable. EmeraldBay's frozen
risk threshold accepts 51.4% of evaluation conditions but misses its registered
error target; sci-Plex accepts only 10.4%, failing usefulness. Accept-none and
accept-all controls are included. No new interval or calibrated abstention is
exposed, and the earlier D-SPIN 34.3% coverage failure remains unchanged.

Subset-retuned context budgets reduce MSE on a fixed exposed query population
from roughly 0.085--0.093 with two source contexts to 0.0682 with six. This
supports better source coverage for the tested simple procedure, without proving
representation value or prospective acquisition utility. Orders share labels;
condition-assay cost units are hypothetical. New risk-based acquisition is not
extended after the failed gates.

GPU tensor peaks are about 18.4 MB for the bounded sci-Plex state experiment and
21.4 MB for the final combination experiment, excluding CPU and driver memory.
The separately named 256-token PertAdapt probe passes three precisions at batches
one and two, but full-gene native execution remains blocked by memory, NCCL and
unverified foundation alignment. Existing native aggregate-input CPA and GEARS
results are preserved, with their information and assay limits.

The suite passes 115 tests. Two-seed NumPy exports identify source reference
contexts and residual training conditions without evaluator labels. Fresh
NumPy-only wheel installations outside the checkout validate Python and CLI
routes. The scored pair fold restores within 4.3e-7 of CUDA predictions; installed
NumPy restoration agrees exactly. All checkpoints, source data, prediction banks
and optimizer state remain local. The private website and master plan remain
excluded. Generated tables/figures use saved compact records, and the paper is
compiled and visually inspected.

An attempted CPU refit did not reproduce the scored CUDA pair fold, with a
maximum prediction difference of 0.238. That failed reconstruction is preserved;
it does not affect NumPy restoration of fitted weights. Retraining is scoped to
the recorded numerical environment and device. Resume checks include data,
protocol, implementation and dependency fingerprints, so a changed environment
cannot silently reuse an optimizer checkpoint.

## Remaining evidence

The [capability ledger](../results/state/capability_ledger.json) separates
implementation, source qualification, development, confirmation and reliability.
No positive independent confirmation is added. Larger representation training,
routing, unknown interventions, unseen doses, higher-order outcomes and temporal
operators remain unsupported. The public sci-Plex release has additional dose
structure, but it is outside this bounded registered state experiment.

A result that changes this conclusion needs independently repeated shared exact
interventions across new biological states, with matched controls, a frozen
procedure beating the strongest source-selected simple reference by the
registered margin, and separate useful risk confirmation. The
[experimental design](state_confirmation_design.md) specifies randomized batches,
independent cultures, controls, singles, genuine pairs, a sealed confirmation
block and precision-based sample planning. No prospective experiment or paid
compute was commissioned.
