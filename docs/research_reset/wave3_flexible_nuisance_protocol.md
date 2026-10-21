# Strong-reference reagent discriminator

21 October 2026. Prospective successor to the already executed
[nuisance v1](wave3_nuisance_protocol.md), whose outcome is exposed. V1's
polynomial guide reference omitted the established Q2b 64-center nonlinear
single reference. Its passing gate is preserved, but cannot answer the requested
strongest-reference discriminator. This successor corrects that comparison;
it is not a rescue of the failed label-budget gates. No rank, architecture,
budget, seed or feature search follows. All original-source outcomes are exposed
development. SPIDR and SCHEMATIC responses and headers remain prohibited.

## Hypotheses and information contract

Compare the strongest established single-effect reference, Q2b's nonlinear map
plus additive gene incidence, with precisely that reference augmented by the
fifteen frozen v1 guide/arm/control-derived summaries, then with augmented
reference plus unchanged rank-eight completion. The augmented family is always
the completion baseline, even if validation favors the unaugmented family.
All methods have the same permitted control singles, pair identities, train and
validation labels. The unaugmented comparator deliberately ignores the extra
reagent inputs. No method receives query double-response values as features.

H1: completion retains at least 5% lower held-block MSE than the augmented
reference in both contexts. H0 for this screen: the specified nuisance reference
removes that advantage. Failure to pass H1 is not proof of causal absence.
If the augmented predictor matches completion, prefer simplification and close
biological-interaction interpretation. If the gain survives, conclude only that
this finite nuisance representation did not explain it. Arbitrary pair-technical
effects remain observationally indistinguishable from molecular interactions.

## Frozen splits and preprocessing

Use all 81,466 eligible canonical pairs in each original context, unchanged
GSE116198 five-block assignments and ordered union of 16,210 test pairs.
Constructs, orientations and infections stay with their canonical pair. The
processed-gamma units, exact arm-matched control singles, opposite-arm singles,
equal-construct/equal-infection feature aggregation and finite eligibility mask
are exactly v1's. Double outcomes are used only for inherited finite membership
and exact target-equivalence verification. Neither their magnitudes nor fitted
GI/activity fields enter the covariates. All extracted feature bytes and raw input
hashes are frozen; target, construct-count or split drift stops the run.

Retain the fixed 64-center symmetric Nystrom map, seed 20261022, training-only
coordinate moments, centers, nonzero median bandwidth and eigensystem. Retain
the original sum/absolute-difference/product features and additive gene incidence.
The offset remains the sum of the original measured gene singles. For each
family, center and scale its full design using TRAIN only, with unpenalized
intercept and ridge penalties exactly 1, 10, 100, 1000. First minimum validation
MSE chooses the penalty. No cross-family test selection occurs. Require numerical
agreement within 1e-12 with historical Q2b node predictions on validation and
test, independently of whether nuisance helps. Train support remains mandatory.

The extra features are those registered in v1, including construct-first nonlinear
singles, physical-arm difference, same-guide arm shifts, across-construct
single variance, count and orientation imbalance. Constant control-partner
membership and constant orientation cannot identify corresponding effects;
report them rather than inventing control covariates. The fixed library aliases
guide identity to gene identity, and no independent pair-technical measurement
exists. Infection repeats do not break this confounding.

## Fitting, evaluation and decisions

Train the existing rank-eight ScalarCompletion on augmented-reference TRAIN
residuals, penalty .01, 500 Adam steps, learning rate .03, seed 20261022, train
target standard deviation, validation checkpoint every 25 steps. Keep all three
original stratified training-residual permutations, seeds 20261023/24/25; require
at least 80% permutable rows per fold. Genuine validation residuals select every
checkpoint. Freeze all ten validation records and checkpoint hashes before any
test scoring. Report per-fold train/validation label counts and their union;
there is no low-label or measured-assay-savings claim.

Report MSE, MAE, bias, Pearson, target variance and both infection losses for the
two nested references, aligned completion, all three shuffled controls, frozen
historical completion/main effects, historical minimum, guide minimum and zero.
Report paired squared-loss differences for aligned versus each reference/control
and nuisance versus unaugmented, with 2,000 endpoint-multiplicity gene bootstrap
draws, seed 20261022, empty draws and leave-one-gene-out influence. These are
conditional sensitivity intervals, not independent biological confidence.
The synthetic coverage audit's poor cohort/global coverage precludes using them
as study-level uncertainty.

The unchanged gate requires at least 5% lower MSE versus augmented reference in
BOTH contexts, upper conditional paired bound below zero, no worse MSE on either
infection, informative controls and lower MSE than all three controls. Veto any
point-estimate loss versus either nested reference, historical calibrated main,
historical or guide minimum or zero. Frozen old completion is a descriptive
comparator. Report every failure without a successor sweep. Passing cannot
promote a biological-interaction claim or open reserved outcomes.

## Lifecycle and resource stop

Freeze code, tests, protocol, dependencies, prior v1 evidence, full Q2b states,
raw inputs and splits in a timestamped hash registration before postfreeze tests
or new fits. Commit this registration before execution. Refuse overwrite, drift,
incomplete validation roster, missing prerequisites and any prior failure marker.
Use the independently tested strict completion loader for new replay, keeping
historical loader and checkpoints unchanged. Exact prediction-array and scientific
JSON replay is mandatory. It does not repeat optimization.

One numerical CPU thread; fit, score and one replay together at most 1,800 active
wall/CPU seconds, 1 GiB measured process peak working set, 100 MiB outputs and
1 GiB disk reserve. Checks run at raw batches, ridge chunks and optimizer steps;
they are cooperative, not interrupting OS guarantees. Freeze/hash and prerequisite
tests are administrative overhead. Preserve a failure and stop without raising
caps. Numerical states and source assets remain on E:, outside Git. Compact
registrations, selections, results and receipts may be committed. No delegation.

Commands use `.venv/Scripts/python.exe tools/wave3_flexible_nuisance.py`:

```powershell
freeze --prior E:/opal-wave3-20261021/nuisance-v1 --flexible E:/opal-wave2-20261021/q2b-v1 --output E:/opal-wave3-20261021/flexible-nuisance-v1
verify --output E:/opal-wave3-20261021/flexible-nuisance-v1
fit --output E:/opal-wave3-20261021/flexible-nuisance-v1
score --output E:/opal-wave3-20261021/flexible-nuisance-v1
replay --output E:/opal-wave3-20261021/flexible-nuisance-v1
```
