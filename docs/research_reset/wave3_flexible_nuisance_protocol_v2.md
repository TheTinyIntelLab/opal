# Strong-reference discriminator arithmetic amendment

This prospectively supersedes the implementation in
[v1](wave3_flexible_nuisance_protocol.md), with exactly the same scientific
features, splits, ridge penalties, completion recipe, resource caps and gate.
V1 stopped in K562 fold zero before any completion fit or test loss. Its
strong-reference validation predictions differed from Q2b by at most
1.2463305998355878e-08, above the prespecified 1e-12 equivalence requirement.
Both selected alpha 10. Validation MSEs were .0018767438143871195 and
.0018767438143707626. No threshold is relaxed and no new scientific menu is added.

The Nystrom centers, whitener and singles agreed exactly. V1 appended gene
incidence after the nonlinear coordinates and centered residuals before
cross-product accumulation, while Q2b keeps incidence before nonlinear
coordinates and centers after accumulating uncentered products. Those are
mathematically equivalent but numerically distinguishable for nearly constant
coordinates. V2 preserves Q2b's exact feature order and accumulation arithmetic
for both nested references. The unaugmented reference still must agree within
1e-12 with historical validation and test predictions, without inspecting test
loss to alter the implementation. New tests reconstruct this equivalence and
the augmented dense normal equations. Preserve all v1 bytes and failure receipts.

Use `tools/wave3_flexible_nuisance_v2.py` with fresh output
`E:/opal-wave3-20261021/flexible-nuisance-v2`. Freeze takes the original
`--prior` and `--flexible` arguments plus
`--failed E:/opal-wave3-20261021/flexible-nuisance-v1`.
The registration binds the failed attempt and both implementations before
postfreeze prerequisites, a prefit commit and fitting. A further failure stops
the branch. Fit/score/replay retain v1's cumulative 1,800-second scientific cap
for this attempt; the failed attempt's 41.643018 wall seconds and 41.593750 CPU
seconds are disclosed separately as additional program cost. No reserved source
access or model promotion follows.
