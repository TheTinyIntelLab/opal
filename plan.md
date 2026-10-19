# Opal model development

Authoritative forward plan, 20 October 2026. This supersedes former planning and dispatch documents. Historical protocols, source terms, private-data restrictions and results remain scientific records. The engineering workstream is closed. Current authorization covers research, fitting, cleanup, tests, commits and normal push, not external communication, purchases or publication.

## Objective and evidence

Build a competitive specialized predictor of matched-context population-average double-perturbation effects, given measured constituent singles and other measured pairs. Test whether a hierarchical gene-local composition law generalizes better than full-profile residual models. This is a new hypothesis rather than another expansion of the former neural architecture.

The evidence is in [architectures](docs/research_20261020_architectures.md), [data and evaluation](docs/research_20261020_data.md), and [identifiability](docs/research_20261020_identifiability.md). TxPert's strongest doubles use proprietary graphs; GEARS requires pair supervision; CPA's example cell-validation split does not establish held-pair model selection. Recent calibrated-metric research supports response-sensitive secondary endpoints, but cannot rescue a failed primary endpoint after scoring. Prior Opal bilinear, kernel, noise-aware and transfer results argue against another capacity sweep.

Singles do not identify arbitrary interaction effects. Shared single/control errors can produce apparent antagonism, so composition is predictive calibration rather than causal mechanism recovery. An identity-only general predictor needs compatible intervention-response pretraining. Cross-context prediction needs compatible contextual evidence, while cell-distribution targets need raw cells and separate distributional evaluation. Neither follows from these population means or current resources.

## Candidate and data

For each gene, fit an intercept and slopes for the sum of measured single effects, signed same-direction overlap, and signed opposite-direction overlap. Shrink gene slopes toward a pooled coefficient vector using training-only covariance. Exact cancellation gives zero signed overlap, a deliberate limitation. The independent NumPy implementation imports no upstream model code or weights.

Use unchanged norman-log2 and wessels-expanded-protected-log2 fitting banks: 88 pairs/105 singles/10,623 genes and 110 pairs/28 singles/10,590 genes. Full preparations contain 126 and 158 eligible pairs, but separate historical evaluator responses remain unread. Keep QC, scale, source hashes and gene ordering fixed. All local sources have prior development exposure; the protected filename is not independent confirmation. Source terms remain in force; raw/derived expression matrices and checkpoints remain local.

## Locked evaluation

configs/experiments/composition_20261020.json is the executable registration. tools/study_composition.py freeze records source/code hashes and all nested splits before run can score. Five outer and three inner folds use seed 20261020 under canonical-pair and stricter partner holdouts. Pair identity is excluded globally; no outer outcomes select penalties, representations, baselines, endpoints or seeds. Evaluator response files are outside the input contract.

Baselines are additivity, no change, shared residual mean, the existing finite profile-kernel menu, reduced-rank residual ridge, pooled overlap and gene-wise affine calibration. Each outer fold selects its strongest reference by inner MSE. Matched affine and pooled ablations test overlap features and hierarchy separately. Penalties are 0.1, 1 and 10; no post-score rescue search. Kernel reference normalization can access all measured singles, matching the declared observed-single information contract.

Primary endpoint is equal-pair all-gene MSE; overlapping partner-fold predictions are averaged per pair before scoring, as frozen in the runner. Secondary MSE uses training-only response-variance gene weights. Conditional pair-bootstrap intervals describe deposited-pair sampling, not independent cultures or calibrated prediction intervals. Shared singles/controls and overlapping partner folds limit uncertainty interpretation. Preserve fold results, coverage, all baseline scores, replay, checkpoint hashes, runtime, memory and storage. Final development refits have no evaluation score.

Promotion requires at least 2% lower primary MSE than the inner-selected strongest baseline in both sources and both protocols, plus a below-zero conditional random-split bootstrap upper bound. A loss rejects the hypothesis; a partial/small gain remains baseline-level or insufficient evidence. Published metrics cannot populate matched comparison tables or establish state of the art here.

## Budget and milestones

One numerical worker, one BLAS thread, at most one hour wall time, 2 GiB process RSS and 128 MiB new experiment artifacts. Active environment is CPU-only. Inspect first-fold resource use before continuing. C: began full; inspected temporary documentation/runtime folders were preserved on D: to allow small edits. Outputs use D:/opal-research-20261020/. Do not delete scientific artifacts for space.

1. Primary-source/code/license investigation and prior-negative audit: complete.
2. Implement candidate, ablations, safe checkpoints and frozen registration; review/test before scoring: complete.
3. Commit protocol, profile, train/evaluate unchanged, retain results and make one continuation decision: complete. FAILED HYPOTHESIS; close overlap law. The compact retention-only replay matched every fold exactly.
4. Remove individually audited obsolete planning dispatches; preserve scientific specifications, provenance, custody and negative evidence. Update README: complete; audit in results/composition-20261020/planning_audit.json.
5. Run targeted/full tests, package/data checks and replay within available storage; commit lowercase batches, push origin/main, verify remote head and CI: local checks and push complete. All 398 tests pass; wheel/sdist, installed replay and all documentation links pass. Hosted clean-install CI is checked at delivery; its actual status is reported in the final handoff.

## Continuation and termination

A successful development result earns a prospective external-data experiment after verifying source terms, intervention compatibility and independent measurement lineage. It does not earn another local hyperparameter sweep. A failed overlap/hierarchy hypothesis closes this candidate and retains the strongest simple reference. Before the next model, require relevant additional pair evidence or a separately justified response-evidence endpoint with a feasible compatible extraction. Large atlas downloads, foundation training and more graph capacity are not the default next step.

Independent biological confirmation remains absent. Existing outreach and qualification records are historical evidence, not active monitoring assignments. No new split, synthetic fixture or edited plan supplies independent cultures, causal identification or calibrated biological uncertainty.

## Executed decision

Composition MSE loses to the nested reference selector on every source/protocol: Norman pair 0.006213 versus 0.006089, Norman partner 0.006343 versus 0.006322, Wessels pair 0.008510 versus 0.007115, Wessels partner 0.009281 versus 0.009195. The affine ablation wins throughout. Do not expand this architecture. See [the decision report](docs/model_development_20261020.md) for conditional intervals, exact replay, the first storage-cap failure and measured compute. A new compatible pair-evidence source or separately justified endpoint is required before the next candidate.
