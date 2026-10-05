# Opal architecture research — 5 October 2026

The strongest next direction is a noise-aware residual model that fuses measured-single responses with biological priors and predicts corrections through several structured output modules. The evidence supports testing these changes; it does not establish that a larger neural model will win. Our plausible frontier target is accuracy and useful biological signal per measured combination and per unit of compute, within the seen-2/2 population-mean task.

This note separates published evidence, findings in our repository, and proposed adaptations. No proposed architecture was trained during this research. Historical preparation, models, and evaluation results remain unchanged.

## What is limiting the present model

The current neural implementation maps single-response embeddings through symmetric coordinate products, shared bilinear projections, or an MLP. Its usual output is an offset plus one global rank-k basis. Selected historical ranks were two for Norman and eight for Wessels. The current kernel candidate adds a shared correction, a reference-scaled linear correction, and symmetric response kernels. These are sensible small-data designs, but they have little external biological information and no explicit model of uncertainty in their single-response inputs.

Our existing evidence argues against increasing depth first. Simpler corrections beat the historical neural selections across all ten Wessels splits. The new kernel search improves nested development MSE against a separately tuned expanded linear search by roughly 1–4%, with dependent partner folds and an aggregation-sensitive comparison. The frozen foundation positional-table experiment also failed its shuffled-gene control. None of those results tests a contextual foundation encoder or proves that biological priors cannot help. See [the improvement study](improvement_study.md).

### An actionable coverage issue

The 50-cell rule retains 16 Wessels single targets and 58 pairs. Twelve additional observed singles have only 15–49 retained cells, which excludes many otherwise measured combinations. Recounting existing metadata under alternative thresholds gives:

| Minimum retained cells | Potential observed singles | Potential pairs with both singles |
| --- | ---: | ---: |
| 50, current | 16 | 58 |
| 25 | 27 | 142 |
| 15 | 28 | 158 |

These are metadata-only eligibility counts, not a completed preparation. Exact stratum support, controls, guide orientation and construct confidence must still pass matching checks. The [machine-readable audit](research_sources/architecture-20261005/wessels_coverage_audit.json) records the source hashes and exclusions. No excluded expression profiles were opened to obtain these counts.

The October 1 benchmark reports 28 singles and 157 Wessels pairs and finds PRESAGE ahead of additivity on 15 of 18 metrics. Its broader coverage and different training split prevent a direct score comparison with our current preparation. The same paper shows why sparse perturbation signals require calibrated metrics. [Miller et al., Nature Biotechnology, 2026](https://www.nature.com/articles/s41587-026-03307-w).

A separate broader-coverage track is worth registering, but reducing the cutoff alone would turn uncertain singles into noisier model inputs. The first architectural change should address that measurement problem.

## Ranked changes worth testing

### 1. Treat measured singles as uncertain inputs

**Proposal.** Estimate uncertainty for each available single response by jointly resampling its cells and matched controls through the complete normalization and effect-estimation pipeline. Feed a shrunken single representation and its reliability into the pair model. Perturb the input estimates during training using those joint draws, so the correction cannot rely on small fluctuations in a sparsely measured single.

Control errors and errors from a reused single correlate across combinations. Resampling each derived residual independently would miss that dependence. Preserve shared reference draws across pairs. With few cells, shrink gene-wise variance estimates toward a pooled or pathway-level estimate; do not estimate a dense covariance matrix from fifteen observations.

Use training-double uncertainty only in the training likelihood or bounded loss weights. At prediction time, the gate can use the measured singles, controls, and distance from training pairs. It cannot use the cell count, variance, or DEG list of the unmeasured combination.

Keep the declared output decomposition exact. If an estimator replaces observed single effects d_A and d_B by denoised estimates eta_A and eta_B internally, expose its prediction as:

    predicted_delta_AB = d_A + d_B
                         + (eta_A - d_A) + (eta_B - d_B)
                         + predicted_interaction

The exported residual then includes single-estimation corrections as well as a learned interaction departure. Report those contributions separately rather than interpreting all of the residual as interaction biology.

**Decisive test.** Compare the same model with ordinary inputs, input-noise augmentation, and uncertainty-aware shrinkage. On a separately registered broader-coverage track, score both the common original pairs and newly eligible pairs, with the same gene panel on the common-pair comparison. This separates architecture gains from gains caused by more training conditions or a changed evaluation population. Cell resampling measures technical sampling uncertainty, not biological replication.

**Stop condition.** If the uncertainty features only encode sequencing depth or shrink useful rare effects away, retain the simpler estimator. Prediction intervals need an independent calibration assessment before we call them calibrated.

### 2. Fuse biological priors with the measured response

**Published evidence.** PRESAGE combines several biological knowledge sources, and its authors report that source choice matters strongly. Its implementation projects source embeddings and pools them using mean, learned, or graph-attention pooling. That attention operates across knowledge/pathway inputs rather than requiring transcriptome-wide gene self-attention. [Author repository](https://github.com/Genentech/PRESAGE); [inspected implementation at pinned revision](https://github.com/Genentech/PRESAGE/blob/2c7b231c60cf110c4aab0acf8ecd2c2b3268fffb/src/presage.py).

TxPert supplies complementary evidence for graph priors. Its double-perturbation configuration uses GAT-MLG; Exphormer-MG is the selected single-perturbation configuration. Some of its strongest graph combinations include proprietary resources, so a public-graph adaptation must be reported separately. [TxPert paper](https://www.nature.com/articles/s41587-026-03113-4); [author implementation](https://github.com/valence-labs/TxPert).

**Proposal.** Give each target several representations: its measured-single expression embedding, GO/pathway membership, a STRING representation, and one semantic/protein or independently measured functional signature. Project each into a small shared space, initially 64–128 dimensions, and use source attention with explicit missing-source masks. Freeze external encoders first and train only the small projections and source weights. Feed single reliability to the fusion layer so it can reduce reliance on an uncertain expression estimate.

Keep the pair representation exactly symmetric, for example with the sum, absolute difference, and elementwise product of the two fused target embeddings. Prior-source attention and pair composition are separate choices; test them separately. Public graph topology or target annotations can be permitted inputs even for held-out pair identities, but held-out combination expression cannot enter an external feature cache.

**Decisive test.** Compare each source alone, concatenation, and attention against the measured-single kernel. Include shuffled target identities, random embeddings, missing-source controls, and a ridge model using identical fused features. Audit external resources for overlap with benchmark combination responses. An attention model must beat a simpler model supplied with the same information before attributing the gain to attention.

**Stop condition.** Reject a source if its true target mapping does not beat a shuffled mapping, or if its benefit comes from benchmark-response contamination. Our prior positional-table result makes this control especially important.

### 3. Replace one global output basis with a structured decoder

**Proposal.** Keep a strongly regularized shared correction, then add several pathway/module-specific bases and a small gene-local remainder. The symmetric pair representation predicts module coefficients; a shared gene decoder uses output-gene annotations or reference-expression features to determine which genes receive those corrections. This permits a pair to change a small biological program without needing every such response to share the same two or eight global directions.

Use overlapping pathway memberships with normalization, a penalty on module activation, and stronger shrinkage on the gene-local remainder. Avoid a separate free neural head for every gene. Module definitions must come from external annotations or allowed controls/singles; any basis learned from combination responses must be refitted inside each training fold. Output-specific annotations are distinct from target-gene annotations.

A schematic candidate is:

    h_A = source_fusion(single_features_A, priors_A, reliability_A)
    h_B = source_fusion(single_features_B, priors_B, reliability_B)
    h_AB = symmetric_features(h_A, h_B)
    residual = shared_correction(d_A + d_B)
               + sum(module_gate_m(h_AB) * basis_m * coefficients_m(h_AB))
               + shrunken_gene_decoder(h_AB, output_gene_features)
    predicted_delta_AB = d_A + d_B + residual

This is an original proposed adaptation, not a validated model or a claim of methodological novelty. Start with the shared correction and one structured branch. Add other branches only when their ablations justify them.

**Decisive test.** Compare a larger global rank, an unrestricted decoder, and the structured decoder with matched feature inputs and comparable tuning budgets. Inspect reproducible sparse effects and pathway recovery alongside equal-pair all-gene error. The existing rank ablation does not establish that output rank is currently the bottleneck.

**Stop condition.** If a larger regularized global basis works equally well, prefer it. Reject apparent pathway gains that follow only from choosing the evaluation genes after seeing predictions.

### 4. Train for reliable perturbation signals, with an all-gene guardrail

**Published evidence.** PertAdapt uses a condition-sensitive adapter, gene-similarity masking, and an adaptive loss for perturbation-sensitive genes. Its full backbone was not evaluated by our frozen positional-table experiment. [PertAdapt author repository](https://github.com/BaiDing1234/PertAdapt).

**Proposal.** Combine ordinary response MSE with a bounded perturbation-sensitive term and, if useful, a module-level term. Estimate signal weights from training responses and reference noise, then cap them so a tiny number of apparently extreme genes cannot dominate. Signal magnitude and measurement reliability should both matter: magnitude alone can emphasize sampling noise, while inverse variance alone can favor abundant, weakly affected genes.

Do not copy an adaptive loss and assume it solves our objective. Evaluate residual directions, suppression of false positive changes, and retrieval of the correct condition. Include the same loss on a linear decoder to learn whether a gain comes from optimization rather than architectural capacity.

**Decisive test.** Select the mixture coefficient inside development folds, freeze it before any final comparison, and retain equal-pair all-gene MSE as a guardrail. Calibrate secondary metrics using positive and negative controls. Held-out DEG lists and technical duplicates may support evaluator-only scoring; they cannot supply training weights or select the model.

**Stop condition.** If weighted loss improves one sparse metric by creating large false positive responses or worsening overall response error beyond a preregistered tolerance, reject that tradeoff.

### 5. Transfer representations from larger single-perturbation screens

**Proposal.** Pretrain the target encoder and structured expression decoder on independent single-perturbation screens, then freeze most parameters while fitting the small pair correction. Use explicit cell-context and intervention-modality labels; CRISPR activation, inhibition, and Cas13 knockdown should not be pooled as if they have the same effect semantics.

This can teach useful response directions and target relationships. Single-perturbation data alone cannot identify the pair-interaction function. The pair branch still needs measured combinations, and any gain must survive comparison with frozen features plus ridge or a kernel. Start with public functional signatures before attempting foundation-backbone fine-tuning on the present CPU environment.

State provides evidence for learning responses across cell sets and diverse contexts, but its population/distribution modeling addresses a broader task. Borrow compact control-state features only when our data contain independently varying measured states. In one fixed context, an unconstrained context operator applied to a constant state is not an identifiable mechanistic advance. [State, Cell, 2026](https://doi.org/10.1016/j.cell.2026.07.052); [author repository](https://github.com/ArcInstitute/state).

**Decisive test.** Compare no transfer, public frozen signatures, and task-specific transfer with the same small pair decoder. Separate public resources from unavailable private training data. Audit duplicate cells, conditions, and combination labels across resources.

**Stop condition.** Avoid expensive backbone adaptation unless the frozen contextual representation beats both the positional-table negative control and information-matched classical models.

### 6. Keep a strong small-data expert and allow the model to abstain from complex corrections

**Proposal.** Retain the shared/global correction and symmetric kernel as competing experts. A structured neural branch can add a shrunken deviation from that prediction. Fit a small gate or a single validation-selected mixture weight first; a large per-query gating network would be difficult to justify with forty Wessels development pairs. Uncertain or unsupported pairs should move toward the validated simple expert rather than invent a confident interaction.

A multi-output Gaussian process over the same symmetric features could supply a structured prior and an uncertainty hypothesis. Its posterior variance is not automatically calibrated in the presence of noisy shared singles, estimated kernels, and selected hyperparameters. A retrieval branch is another later option, but it must retrieve training-pair responses only and exclude guide-equivalent copies of a validation pair.

A useful additional challenger is a frozen tabular foundation predictor on compact symmetric features and training-fold output PCs. The general TabPFN work motivates a small-data comparison; it does not establish performance on our combination task. Fit the response PCA within each fold, predict its components, invert the projection, and restore the observed additive term. [TabPFN, Nature, 2025](https://www.nature.com/articles/s41586-024-08328-6).

**Decisive test.** Compare every ensemble with its best individual expert under the same nested split and a declared search budget. Include the zero learned-interaction option. Measure end-to-end latency and preprocessing cost, not just the final decoder.

## Implementation order and the claim we could earn

| Stage | Minimum implementation | Question it resolves |
| --- | --- | --- |
| First | Joint reference resampling, reliability features, shrinkage ablation, separate coverage protocol | Are uncertain singles and discarded conditions limiting the result? |
| Second | One public prior source plus measured responses, then a small multi-source fusion | Does additional biological information generalize beyond response geometry? |
| Third | One module decoder, compared with larger global rank | Does local output structure preserve useful effects without overfitting? |
| Fourth | Bounded signal-sensitive loss on both classical and neural models | Does the objective recover reliable sparse effects? |
| Later | External single-screen transfer, contextual backbone, optional GP/retrieval expert | Does additional compute or external training data buy a reproducible gain? |

Keep the present historical track intact. Register any expanded coverage, changed loss, and external information policy before fitting that track. Our historical test findings have already informed model design, so new random seeds on the same pairs remain exploratory. Newly eligible pairs are potential new evidence only after checking all prior analysis and external-resource exposure; an independent dataset is stronger confirmation.

The decisive comparison includes adapted PRESAGE under the same available-single information, the best global/ridge/kernel controls, and executable native published comparators where their environments permit. Match normalization, gene panels, cell/reference access, splitting, evaluation aggregation, external resources, and tuning budget. Current CPU restrictions and unexecuted native methods remain practical comparison gaps; they cannot be replaced by a claim that our small adaptations beat the published models.

A defensible frontier claim would name its dimension: seen-2/2 response accuracy, recovery of independently reproducible sparse effects, broader reliable measured-single coverage, or accuracy per unit of compute. It needs a measured accuracy/compute comparison and independent confirmation. Architectural complexity, a foundation checkpoint, and a favorable metric alone do not establish that claim.

## Source and evidence provenance

All literature links above are primary papers or author repositories, inspected on 5 October 2026. The October 1 article was published four days before this audit. Published evidence is not a prediction of Opal's future scores.

PRESAGE source was inspected at revision `2c7b231c60cf110c4aab0acf8ecd2c2b3268fffb`. Cached `presage.py`, `model_harness.py`, and the upstream `LICENSE.txt` retain verified Git blob hashes in [the source manifest](research_sources/architecture-20261005/source_manifest.json). They are research evidence, not integrated code. Review upstream reuse terms before incorporating implementation or distributing derived resources.

Local evidence comes from the current model implementation, preparation rules, saved QC reports, and already completed development studies. The coverage count reads metadata and QC retention counts only. No new training, test scoring, or manuscript claims were added by this research note.
