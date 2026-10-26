# Executed method inventory

This closes the method-inventory checklist for the actual shortlist. The source and numerical recipes remain frozen in the [experiment matrix](experiment_matrix.md), [growth registration](growth_registration.md) and committed code bindings. Excluded expression/graph/meta-learning methods remain literature context in [B's review](B_literature.md), not executed entrants with invented runtimes.

| Procedure | Exact implementation identity | Inputs and TRAIN preprocessing | Selection and unsupported queries |
|---|---|---|---|
| B0, four fixed references | `src/frontier_tournament.py`, SHA256 `a1c54b4668de6fc64de358cfaac5048c282ab8b88576c7ee1f59db6330686b98` | Zero, TRAIN mean, measured-single sum or minimum. No global pair summaries or external metadata. | Menu-order ties; all unsupported queries retain fixed single-sum fallback. |
| B1, four calibrated references | Same pinned module | Sum, absolute difference and product of singles; TRAIN moments; ridge1/10/100/1000 with unpenalized intercept. | VALIDATION MSE, no TRAIN-plus-VALIDATION refit. |
| B2, four gene-main references | Same pinned module | B1 plus gene incidence. Identities are supplied consistently to every eligible family. | Same charged VALIDATION and fallback rule. |
| B3, four flexible-single references | Same pinned module; distinct from historical FlexibleSingles | Sorted measured singles, at most64 TRAIN-only hash-selected RBF centers, with/without gene incidence; ridge10/100. | Same VALIDATION/fallback; no compatibility claim for the historical flexible predictor. |
| I0, completion reference | Same pinned wrapper plus unchanged `src/block_completion.py`, SHA256 `e4b19ba22831d08dc9792d0c9c3792fdf29220bf4d1477665cfc86e86d8d0480` | Residuals from the saved selected-simple policy, TRAIN residual scale, rank8 U/V factors giving symmetric rank at most16, penalty0.01. | Up to500 Adam updates; VALIDATION checkpoint every25. This extra checkpoint choice differs from fixed-step I2 and makes validation ranking optimistic. |
| I1, two pair-kernel settings | Pinned tournament module | Half identity plus half measured-single RBF gene kernel; symmetric tensor pair kernel;64 TRAIN landmarks and TRAIN-only eigenspace/scaling; ridge10/100. | Family configuration chosen on VALIDATION, no TEST fitting or landmarks. |
| I2, two composition settings | Pinned tournament module; NAIAD-like, not native NAIAD | Learned4d gene embeddings, shared encoder, symmetric sum/product head; TRAIN-only residual scaling. Fixed300 Adam updates; L2 settings0.001/0.01. | VALIDATION chooses regularization, not neural update count. |

All interaction procedures receive the same selected-simple residual reference. Three TRAIN-only stratified residual shuffles repeat all five interaction configurations. Stored validation-selected family policies and all sixteen fixed simple references are reported; a TEST-best diagnostic cannot become the nominee. Every family requires two distinct TRAIN partners at each endpoint. Missing or nonfinite required singles are rejected. The prospective design-contract API adds explicit provenance/missingness validation for future source use; it does not change these executed inputs.

## Endpoint, partition and information budgets

Synthetic primary endpoints are equal-construct observed pair means from the registered worlds. Sixty genes and1770 unordered pairs are split885 TRAIN/354 VALIDATION/531 TEST. Budgets charge309 or1239 pair labels, including consulted validation labels and unavailable selected rows. Only simulated initiation one fits/selects; initiation two is a separate diagnostic. Missingness rules and full denominators are frozen, not selected from outcomes.

The exposed-growth endpoint is the historical eligible processed-bank mean with both infection columns checked separately. Six tasks each use five original within-gene blocks and16000 TRAIN plus4000 VALIDATION pair labels per fold. Singles are measured inputs, not held-double derivatives. Tasks and folds are dependent. Full-population and common-support metrics are reported; all test rows happened to be supported. No new graph, expression, partner-fitness, interaction-score or pretrained feature enters any model.

Uncertainty is2000 paired gene-multiplicity resamples plus leave-one-gene sensitivity, conditional on the observed graph. Synthetic validation-error radii are diagnostics, not biological or OOD coverage guarantees. No cold-gene result was obtained; the new held-gene fixtures check exclusion logic only.

## Actual compute and external method provenance

The full synthetic fit used693.499 externally measured seconds, score10.297 and installed replay9.587, with321,462,272-byte fit peak process-tree memory. The growth fit used1492.040 seconds, score9.011 and replay36.391, with754,958,336-byte fit peak. These are measured whole-procedure workloads on one CPU thread, not per-family profiles. Per-family wall/CPU was not separately instrumented and is unknown. Profile failures, packing costs and retained artifacts are preserved in the [result report](growth_results.md). No GPU was used.

The primary external inspiration is [NAIAD, ICML2025](https://proceedings.mlr.press/v267/qin25g.html). The separately implemented historical native adapter pins revision `9207e9e855e865160a196a6989392f6db622b0d6` and hashes five upstream Python files in `src/naiad_reference.py`; it was not rerun in this shortlist. Its published benchmark/resources cannot be assigned to I2. I0 is a local scalar completion procedure, not a repurposed two-context joint model. I1 and B3 are local implementations with the exact equations and pins above, not claimed reproductions of an external paper.
