# Identifiability, information and candidate choice

Research assessment, 20 October 2026. This is an evidence record, not a forward plan or a result from the new candidate. The authoritative development decision belongs in root `plan.md` after the research synthesis.

## What the observations identify

Let `mu_0`, `mu_a`, `mu_b` and `mu_ab` be population expression means under the matched control, two singles and the pair, on one explicitly fixed normalization scale. Define `d_a = mu_a - mu_0` and similarly for `d_b`. The decomposition

```text
mu_ab - mu_0 = d_a + d_b + h_ab
h_ab = mu_ab - mu_a - mu_b + mu_0
```

is an identity, not evidence for additivity or a causal model. For every arbitrary vector `v`, a response function can match the observed control and both singles while setting `h_ab = v`. Thus singles alone do not identify arbitrary interactions. A graph or pretrained embedding changes assumptions about which `v` is plausible; it does not create measurements of the missing pair. Pair supervision identifies transferable structure only to the extent that the supervised pairs cover the mechanisms of the query.

Even in a hypothetical smooth response surface, singles identify values on the two intervention axes, while interactions depend on mixed curvature away from those axes. Monotonicity or a specified mechanistic response equation could constrain that curvature, but generic gene graphs cannot establish its sign. CRISPRa, CRISPRi, knockout and drugs also define different interventions. Pooling them without intervention-type and context conditioning does not increase effective sample size for one common response function.

The outcome scale changes the additive null. Additivity of log-transformed population means differs from additivity of molecule counts, and neither equals the average of cell-level log effects. Preserve the existing count/log banks as separate endpoints. Do not select whichever scale gives the candidate its largest gain.

## Measurement error can resemble interaction learning

Suppose independent arm-mean errors are `e_0`, `e_a`, `e_b` and `e_ab`. The estimated residual is `h_ab + e_ab - e_a - e_b + e_0`. Its covariance is the sum of the arm covariances only under independent arm errors. Two residuals sharing single `a` and the same control have covariance contributions `Var(e_a) + Var(e_0)` even if their biological interactions are independent. Plate effects and shared normalization add further dependence.

The feature error in measured additive input is `e_a + e_b - 2 e_0`, while residual error is `e_ab - e_a - e_b + e_0`. Under independent arms their covariance is `-Var(e_a) - Var(e_b) - 2 Var(e_0)`. A residual regression can therefore learn a negative correction to noisy singles without discovering antagonism. Predicting a better population mean is still useful, but the mechanism cannot be inferred from that gain.

Read noise-aware fitting as a measurement model, not as calibrated uncertainty. Cell bootstraps estimate conditional sampling variation within the sampled culture; they cannot estimate independent-culture variation or unknown guide efficiency. Shared controls and reference profiles must remain coupled in resampling. Pairs that share components are not fully independent uncertainty units either.

The local measurement audit already shows the problem concretely. In [the GxE technical pilot](frontier/gxe_measurement_results.md), residual-half MSE exceeds the zero-residual reference in every origin: A172 `0.17053105` versus `0.16909918`, T98G `0.17143288` versus `0.12594109`, and U87MG `1.01566160` versus `0.59480819`. Technical halves are not cultures, and this does not prove all interactions are noise. It does argue against treating residual fitting as easier than direct response prediction.

## What current evidence changes

[Ahlmann-Eltze et al.](https://www.nature.com/articles/s41592-025-02772-6) compare deep methods with simple additive and linear references. Their result supports retaining strong same-information controls; it does not prove every future architecture is incapable of useful prediction. Their additive double baseline uses both measured singles and controls, so a method without those measurements addresses a harder information contract.

[Systema](https://www.nature.com/articles/s41587-025-02777-8) shows that common treatment shifts can inflate control-referenced correlation and proposes evaluating perturbation-specific effects against a perturbed reference. Its matching-mean baseline averages the two single centroids rather than adding both effects. Include this inexpensive alternative because saturation or broad shared stress can make it competitive even when simple additivity fails.

[Miller et al., published 1 October 2026](https://www.nature.com/articles/s41587-026-03307-w), finds useful model signal under effect-weighted and retrieval metrics, while retaining direct error as a guardrail. Their combination training coverage differs substantially between Norman and Wessels, reported as 0.63% and 10.3%. That supports studying supervision coverage before model size. Their DEG weights use outcome information in scoring; held-out weights must never become inference inputs. A changed endpoint must be declared before candidate scoring, and cannot retroactively turn a failed historical MSE experiment into success.

The [official metric-benchmark code](https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines) recommends five GPUs with at least 24 GB each, 384 GB RAM and 2 TB storage for full reproduction. This is a whole-benchmark recommendation, not the minimum for every model. The framework is MIT, but its PRESAGE and scLambda components retain separate noncommercial and GPL terms. This repository's workstation cannot reproduce that whole benchmark unchanged.

[GEARS' official implementation](https://github.com/snap-stanford/GEARS) explicitly warns that singles-only training does not reliably predict combinations. Its [paper](https://www.nature.com/articles/s41587-023-01905-6) also limits reliable use to matched cell types or experimental conditions. These are substantive boundaries on broad transfer claims. [TxPert's author methods](https://www.valencelabs.com/publications/txpert-leveraging-biochemicalrelationships-for-out-of-distributiontranscriptomic-perturbation-prediction/) identify PxMap and TxMap as proprietary. A public STRING/GO implementation is a valid separate candidate, but cannot claim the full information access of the proprietary multi-graph configuration.

## The repository has already tested many obvious successors

The current README goes beyond the initial historical neural summary. V2 bilinear interaction gains over Base are only 0.078% on Combi-seq and 0.084% on TNF-positive; Guard loses to constant mixing. The matched main-effect Combi-seq ablation is better than the bilinear head. These are exposed development results, not fresh confirmation.

`src/models/advanced.py` already implements reliability shrinkage, jointly sampled single-reference augmentation, structured gene/pathway output bases, weighted signal loss and source fusion. `src/models/noise_kernel.py` and `src/models/covariance_kernel.py` implement heteroscedastic and shared-reference covariance regularizers. `src/replicated/models.py` includes component-incidence residual effects. Reimplementing these under a new name would not be a new hypothesis.

The [crossed-single rank-16 study](frontier/crossed_single_feasibility_results.md) also completed 288 compound-by-target predictions. Gains against its locked selector were 2.600%, 6.737% and 2.371% on A549, K562 and MCF7; all spending filters failed, and difference ridge beats the candidate on MCF7. The fit used about 521 CPU seconds. This limits enthusiasm for another low-rank cross-context operator without new identifying information.

## Direction comparison and recommendation

Improving observed-single pair prediction has the best immediate experimental feasibility because the local source banks, manifests and evaluators exist. It also makes the narrowest assumptions. Expanded protected Wessels contains 158 eligible pairs according to its QC report, compared with 40 in the earlier primary bank; use its existing exclusions and provenance rather than silently pooling incompatible conditions. Historical exposure means a new split supplies honest held-out fitting, but not independent confirmation of the research program.

An identity-only or unseen-intervention predictor has wider scientific value but must infer main effects as well as interactions. More single-target labels and legal public priors are relevant there; they do not directly multiply the combination supervision available here. Given nearly exhausted C storage and only roughly 1 GB free on each other local drive reported in this session, a new large pretraining corpus is not presently feasible. Cross-context transfer adds another unidentified operator and lacks qualified independent pairs locally. Cell-distribution prediction adds unpaired coupling ambiguity and cannot be justified by improvements in population MSE alone.

I recommend a bounded hierarchical gene-local composition candidate on the observed-single task. The hypothesis is that saturation and cancellation depend partly on the two measured effects on each output gene, and that a shared composition law can be estimated more reliably than a high-dimensional profile interaction. For each gene, use the sum, separate positive/negative overlap, opposing-sign overlap and an intercept; shrink gene-specific slopes toward pooled slopes, with the additive law available as a nested limiting case. Preserve exchange symmetry and ensure overlap terms vanish when either single is zero. Separate up/up and down/down overlap avoids assuming activation and repression saturate identically. Opposite-sign overlap should use a fixed continuous signed convention, or an explicitly unsigned magnitude with a learned coefficient, before fitting.

This is a statistical composition model, not a molecular simulator. Saturation may result from the assay transform or noisy features rather than biology. Its advantage is an explicit, low-variance sharing assumption testable with existing pair labels. A source inspection found no corresponding sign/overlap composition law in existing model implementations, although prior kernels and bilinear products can approximate related behavior.

## A fair finite experiment

Freeze the source hashes, pair populations, normalization, gene order, outer splits, overlap definitions, finite penalty menu, seed and compute limits before final scoring. Fit all scalers and pooled coefficients on training pairs only. Query singles remain permitted and this must be explicit. Keep unordered pairs together; retain partner-held folds as a distinct extrapolation diagnostic. Never use a held pair's response to construct features, basis vectors or selection weights.

Use the same nested selection budget for additive/no-change, training pair mean, matching mean, global affine, gene-wise affine, strongest existing simple residual reference, pooled overlap and hierarchical overlap. Gene-wise affine separates local calibration from the nonlinear hypothesis; pooled overlap separates hierarchical adaptation from the proposed shared law. At most one targeted ablation should remove overlap after matching all other terms and tuning effort.

Keep raw full-gene response MSE primary for comparability. Prespecify complementary response-strength weighting, training-reference-centered correlation and perturbation retrieval before evaluation. If outcome-derived DEG weights are used, confine them to the evaluator and disclose selection noise. Report every source and failure rather than averaging incompatible scales into one score. Conditional pair-bootstrap intervals describe the exposed source and fixed references, not future biological cultures. Component-held diagnostics and source disagreement should limit claims even if random-pair MSE improves.

Profile one CPU fit first, cap the entire finite experiment, save numeric state and exact predictions, replay them in another process, and stop after the registered comparison. A local 8 GB GPU is unnecessary for this convex small-feature candidate. If overlap loses to matched affine calibration, reject this composition hypothesis on the tested sources; do not respond with extra ranks, endpoints or hidden feature searches. The next justified improvement would be more informative pair supervision or independently measured reference reliability, depending on which diagnostic actually fails.
