# Opal v1: research and implementation roadmap

**Working research title:** Structured residual learning for combinatorial perturbation prediction.

**Status:** Research protocol and implementation specification, revised 3 October 2026. No model performance is established by this document.

**First public model version:** Opal v1. The goal is a complete, competitive, reproducible first release with enough capacity for its task. Model size is selected from evidence, not a branding constraint.

## 1. The result we are trying to produce

Opal v1 predicts the mean expression response to an unmeasured pair of simultaneous interventions when both single interventions have been measured in the same experimental context. It uses the observed singles as an explicit additive prediction and learns a symmetric correction.

The research result must explain whether this structure helps, which part helps, how much capacity is justified, and where the method fails. The engineering result must let another person reconstruct the benchmark, train the selected model, reproduce its evaluation, and run inference locally or through Hugging Face.

The lab's first release includes independent genetic validation, chemical validation, strong baselines, reliability assessment, and tested deployment artifacts. These are v1 work, not improvements reserved for v2. Positive findings are empirical outcomes, not acceptance criteria that can be achieved by changing the test.

## 2. Claims and their tests

| Claim | Required evidence | Finding that limits or contradicts it |
| --- | --- | --- |
| Explicit additive prediction helps | Residual versus direct-response models with the same features and comparable search budgets | Direct prediction performs equally well or better |
| Pair structure helps | Symmetric multiplicative model versus symmetric sum-only, regularized, and neural alternatives | A shared correction or simpler symmetric predictor explains the gain |
| Low output rank helps | Rank sweep, unrestricted-output comparison, reliability-adjusted spectral analysis | Rank restriction harms held-out performance or useful rank approaches the available training rank |
| Single-profile representations help | PCA, learned single encoder, and identity-feature comparisons under the same information access | Gains depend on pair memorization or external embeddings rather than measured singles |
| Freezing helps | Frozen versus jointly fine-tuned representations | Fine-tuning is more accurate and stable |
| The method is sample-efficient | Nested training-pair subsets on fixed test conditions | Its advantage only appears near full pair coverage |
| The method generalizes beyond random pair holdout | Prespecified combination-partner holdout and independent datasets | Gains disappear when pair neighborhoods or experimental systems change |
| Capacity is used efficiently | Validation and held-out accuracy versus total parameters, storage, training time, and inference cost | Larger versions add cost without reproducible improvement |

Freezing and a particular rank are hypotheses, not mandatory properties of the final model. If a more expressive candidate wins on development data, it can be Opal v1. If the selected model fails a structural comparison on test data, report that failure without substituting another architecture after seeing the result.

Novelty must be established against existing compositional, bilinear, residual, and combination-learning work. Residual subtraction, symmetry, PCA, and low-rank prediction are established tools. The contribution must lie in the precise method, controlled evidence, or a reproducible scientific finding.

## 3. Dataset suite and source verification

### Norman 2019: development and benchmark continuity

Use the K562 CRISPRa combinatorial screen as the first implementation dataset. It measures activation rather than knockout. The frequently used GEARS variant has 105 single and 131 double conditions, but those are reference counts for a particular processed release, not assertions for every source file. Controls and exclusions must be counted explicitly. [Norman et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/), [GEARS study](https://www.nature.com/articles/s41587-023-01905-6)

Keep two named evaluation tracks: Opal's provenance-checked count-derived protocol, and a compatibility track using a public benchmark's documented preprocessing and splits. Scores from the two tracks must not be merged.

### Wessels CaRPool-seq: independent genetic benchmark

Prepare the Cas13 RNA-targeting combination experiment from its original or verified full release, including single-target and control conditions. Audit guide assignments and target labels before constructing effects. The source is linked to GEO GSE213957. [Wessels et al.](https://pubmed.ncbi.nlm.nih.gov/36550277/), [source accession](https://www.ncbi.nlm.nih.gov/bioproject/883380)

Some hosted evaluation versions exclude single conditions. Such a file cannot support the observed-single additive task without recovering those data. Verify this before training; do not reconstruct missing singles from held-out doubles. [Benchmark preparation documentation](https://scperteval.readthedocs.io/en/latest/user-guide/datasets.html)

This second genetic benchmark is required because Norman alone has limited value for establishing non-additive predictive gains. A recent calibration study recommends additional combination datasets and evaluates Wessels as a denser alternative. It also motivates checking whether metrics discriminate reproducible signal from uninformative prediction. [Miller et al., 2026](https://www.nature.com/articles/s41587-026-03307-w)

### ComboSciPlex: chemical method validation

Use the verified combination experiment associated with CPA. Treat compound, dose, exposure time, cell line, vehicle, plate, and replicate as part of a condition where relevant. A matched-dose single profile is required for each component. The single dose must equal that component's dose in the combination; a total mixture dose is insufficient metadata. [CPA study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10258562/), [official combination tutorial](https://cpa-tools.readthedocs.io/en/latest/tutorials/combosciplex.html)

Fit this dataset separately after freezing the architecture-selection procedure and search space on genetic development data. Dataset-specific normalization and regularization may be selected using chemical development data within the frozen procedure. Report this as method transfer with retraining, not zero-shot transfer of genetic weights.

### Dataset acceptance

For every dataset, produce a source manifest containing download URL or accession, checksum, license or reuse terms, source version, count location, gene identifiers, metadata mapping, guide definitions, and context. Reconcile expected and observed counts. Never invent a count layer or silently infer that integer-valued data must be raw UMIs.

If a planned dataset cannot support matched singles, reliable controls, or the registered task, document the reason and select a replacement by those criteria before examining its model test results. Publish the amendment. The release must retain an independent genetic benchmark; a chemical result cannot substitute for it.

## 4. Task and inference contract

The primary unit is an unordered pair in a fixed context:

$$
p=(\{A,B\},c).
$$

A context c includes the cell line or type, intervention modality, relevant dose and time, and supported control definition. Guide-level identity is retained in metadata even when evaluation aggregates to target-level identity.

Primary inference receives A, B, the supported context identifier, observed single-effect profiles, and the saved preprocessing contract. Neither the combined profile nor combined-profile features are inputs.

Use canonical target IDs rather than relying on string order. Normalize labels such as A+ctrl into single A through an explicit mapping. Canonicalize A+B and B+A to one pair. Do not silently merge two guides targeting the same gene, aliases, repeated-target constructs, multiple dosages, or higher-order combinations.

The built-in inference mode resolves profiles from a released reference bank. An advanced profile-input API may accept new singles only when the assay, modality, context, expression scale, and gene schema are compatible. Its broader generalization is a separate claim and must be tested before presenting it as validated.

Control and single queries return their reference profiles through explicit inference routes. Same-target pairs and combinations of more than two interventions are rejected unless their condition definition and evaluation have been registered.

Opal v1 predicts a population mean. It does not synthesize a cell distribution or infer individual trajectories merely because its source data are single-cell measurements.

Return the effect prediction, additive effect, and effective residual in their registered units. If absolute expression is exported, reconstruct it from the matching control reference and document the inverse transform. Log-scale additive predictions can imply invalid negative log expression because of the pseudocount. Any positivity projection is a separately registered prediction transform, evaluated consistently across methods; do not silently clip outputs or label unconstrained values as nonnegative expression.

## 5. Data preparation and QC

The input adapter identifies an explicit raw-count location, condition metadata, cell identifiers, gene identifiers, controls, and available batch or replicate fields. Check nonnegativity, finiteness, dimensions, declared count provenance, and consistency with source metadata. Numerical heuristics can reject suspicious matrices; they cannot certify count provenance.

Preserve sparse storage and avoid materializing the entire count matrix in memory.

For Norman, begin with the existing proposed library-size interval of 500 to 20,000 and a minimum of 50 retained cells per condition. These are initial configuration values, not established universal rules. Before locking the benchmark, audit their retention using QC metadata, controls, and singles. Inspect detected genes, mitochondrial fraction where annotatable, guide-assignment confidence, and source doublet annotations. Freeze a source-appropriate rule rather than copying Norman cutoffs to the other assays.

Audit perturbation efficacy and guide construction using allowed singles and source metadata. In genetic assays, the same target label does not ensure that a single and a double have matched guide efficacy or control-guide burden. Preserve construct identities and flag mismatches; a residual caused by weaker delivery in a double is not evidence of a specific biological mechanism.

QC based on each cell's own count totals is allowed on test cells. Fitting an adaptive threshold using test response profiles is not. Publish per-condition retention and reasons for exclusion. Do not tune QC to favor Opal's scores or selectively remove difficult combinations.

Keep experimental batches and true biological replicates separate during profile estimation. Matched controls are required. If a batch lacks usable controls, declare its treatment rather than borrowing a reference through an undocumented rule. If treatment and batch are confounded, identify the limitation; an arbitrary correction cannot recover missing experimental information.

A pseudobulk profile in this project means the mean of library-normalized cell counts, followed by the stated transformation. It is not a bulk differential-expression replicate, and the code and report should use that definition consistently.

## 6. Normalization, gene universe, and effect scale

For cell i, let the library total t_i be computed over the source count genes before output-gene filtering. Let T_c be the median post-QC control library total in context c, frozen from allowed controls.

$$
\tilde X_{ig}=X_{ig}T_c/t_i.
$$

Applying this fixed transformation to a held-out cell uses its own counts, not a fitted test-population parameter. Save T_c and all other fitted preprocessing state. Controls and singles are allowed resources in the observed-single task.

For each condition P and matched-control stratum, estimate the mean normalized counts mu_P. When genuine replicates exist, form profiles per replicate and aggregate with prespecified equal-replicate weighting; do not let a deeply sequenced replicate dominate by accident. Use a corresponding matched-control mean mu_0.

The primary effect is

$$
\delta_{P,g}=\log_2(\mu_{P,g}+1)-\log_2(\mu_{0,g}+1).
$$

If multiple strata contribute, compute matched effects within strata before the documented aggregation. Do not take an unmatched global-control difference and label it batch-controlled.

The primary gene universe is selected from controls and observed singles only. Initial eligibility is mean normalized expression at least 0.1 in controls or an eligible single, with detection in at least 1% of cells of the qualifying condition. Freeze thresholds during the data audit. Record how much this adds relative to control-only filtering.

Retain an extended evaluation universe of source-annotated genes passing basic identifier and count validity checks. It is fixed without double-response screening. Evaluate fully combination-induced genes there as a secondary diagnostic. Distinguish that diagnostic from the primary universe and do not penalize a primary-universe model for not producing untrained output genes without stating the coverage difference.

Library normalization measures relative abundance and can remove changes in total RNA. Adding log-scale effects represents a different null from adding normalized-count effects; the pseudocount also matters for weakly expressed genes. Register a normalized-count effect sensitivity track,

$$
\delta^{count}_{P,g}=\mu_{P,g}-\mu_{0,g},
$$

and fit/evaluate its baselines and Opal consistently on that scale. An additive synthetic pair on the count scale need not have zero residual after taking logs. Unit tests must distinguish these nulls.

## 7. Eligibility and experimental coverage

Define S as the observed single conditions satisfying QC and context matching. Define C_obs as the observed double conditions. Then

$$
C_{\mathrm{eligible}}
=
\{(\{A,B\},c)\in C_{\mathrm{obs}}:
n_{AB}\geq n_{\min},\ A,B\in S_c\}.
$$

This selects observed eligible pairs; it is not the full Cartesian product of eligible singles.

Report cell and replicate counts, unique targets, eligible pairs, excluded conditions, target degree in the combination graph, and the fraction of all possible distinct-target pairs observed. Cell count is not the count of independent training combinations. Different guides, batches, or replicates of one target pair do not become new held-out pairs.

Eligibility may use condition labels and post-QC counts before split assignment. It must not use the measured residual magnitude, model error, or whether a condition looks promising.

## 8. Splits, model selection, and test access

### Primary random-pair protocol

Assign 70% of eligible doubles to development and 30% to a locked test set using a recorded seed and canonical pair IDs. Keep all eligible singles available. Group all measurements of a target pair together within a dataset so no guide or replicate variant leaks the pair across development and test.

Use fivefold cross-validation within development for architecture and hyperparameter selection. If post-QC counts cannot support five folds, register a lower fold count and its rationale before model fitting. This avoids relying on the approximately nine validation pairs the previous protocol would leave in a 131-pair screen.

Use equal condition weighting for validation. Save out-of-fold predictions, selected hyperparameters, seeds, and stopping epochs. Refit the selected configuration on development using a fixed training duration derived from development folds, then evaluate test once for the registered experiment batch.

The outer split and fold count are design choices, not evidence of sufficient statistical power. Before training, estimate precision from development condition variation and synthetic effects; report the detectable improvement. A wide interval means inconclusive evidence, not proof of equivalence.

### Robustness across splits

Register five random-pair outer seeds in advance. The primary split remains the confirmatory result. Run the fixed selection procedure on the other splits as secondary robustness experiments, with each model fitted without its own test responses. Do not select the final architecture or publication split from these test scores.

Outer splits overlap, so their scores are correlated. They are not five independent biological studies, and confidence intervals must not treat them that way.

### Combination-partner holdout

All singles remain observed, but select held-out targets whose training double partners are absent. In each of five prespecified folds, assign targets by a seeded balanced partition using pair incidence counts only. Test pairs touch that fold's held-out targets; development pairs touch neither. Every pair variant follows the same target-level grouping.

Tune within development under an analogous target-group holdout when coverage permits. Report the number of usable development/test pairs, target degree, and whether held-out pairs involve one or two interaction-unseen components. Specify a feasibility minimum before running; report infeasible folds rather than redrawing folds for favorable results.

This probes generalization from single effects when a target's combination neighborhood is unobserved. It is distinct from predicting targets whose singles were never measured.

### Leakage boundary

Training artifacts may contain controls, all eligible singles, development doubles, source metadata, and declared external annotations. They must not contain test-double expression, residuals, learned bases fitted on test doubles, or test-derived training weights.

A separate evaluator may transform test profiles with frozen preprocessing, calculate metrics, and build registered evaluation-only diagnostics. It may inspect test outcomes after predictions are finalized. File naming alone is not an access boundary; pass explicit allowed condition IDs to every fit operation and log the IDs actually consumed.

## 9. Measurement reliability and metric qualification

Estimate condition reliability with repeated seeded splits of cells, preserving real replicate or batch structure. Estimate residuals independently from both halves, including independently partitioned singles and controls. Shared component profiles induce correlated residual uncertainty across pairs.

In a shared-control stratum, epsilon_AB = Y_AB - Y_A - Y_B + Y_0. All four estimated profiles contribute measurement error. The observed-single task conditions on noisy estimates; the additive term does not make the inputs noiseless or turn a subtraction into an independently measured biological interaction. Use independent single-profile partitions and replicate sensitivity analyses to assess how much apparent residual learning depends on shared measurement error.

A technical split-half predictor is a positive measurement reference available only to the evaluator. It is not a trainable competitor, a biological replicate, or an absolute accuracy ceiling: pooling all cells can be more accurate than predicting one noisy half from the other.

Before final evaluation, qualify candidate metrics using development positive controls, additive and mean predictors, shuffled pair labels, and known synthetic interactions. A metric should detect recovery of reproducible perturbation-specific signal and resist rewards for predicting only a shared response. Retain complementary metrics rather than selecting one to make a model win.

On development residuals, compare the singular-value spectrum across independent measurement partitions and jointly resampled noise references. A rapidly decaying spectrum in one noisy residual matrix is not sufficient evidence of low-dimensional reproducible signal. Use this diagnostic to interpret, rather than replace, held-out rank selection.

For uncertainty in estimated profiles, jointly resample controls and each single condition once per bootstrap draw and reuse that resampled single wherever it appears. Resample double cells within their condition. Use replicate-level resampling where genuine replicates exist, with cell resampling nested beneath it. These estimate measurement uncertainty conditional on the experimental design.

For generalization summaries, show paired condition-level differences, seed variation, and target-group sensitivity. Conditions sharing A, B, or controls are not independent. A simple condition bootstrap is a descriptive conditional interval; also report a target-cluster or network-aware sensitivity analysis with its exact resampling design and limitations. Never count cells or repeated seed predictions as independent biological samples.

## 10. Shared representation and training inputs

Start with a common single-effect representation for controlled comparisons:

$$
e_u=E_{\mathrm{single}}(\delta_u)\in\mathbb R^m.
$$

Compare three sources:

1. PCA fitted only on eligible singles, with the saved centering/scaling and a dimension supported by the number of independent single conditions.
2. A denoising single-effect encoder with an explicit reconstruction objective.
3. Learned target-ID embeddings as a memorization and information-access control.

For the denoising encoder, use independent cell-subsampled estimates of one single condition as input and target, or genuine matched replicates where available. Minimize condition-balanced reconstruction error with regularization. Reserve measurement partitions for representation validation. Reusing one condition in multiple subsamples does not create independent single interventions.

Any feature standardizer is fitted only to allowed data. Avoid gene-wise division by nearly zero variability; floors and clipping are saved configuration values. Biological annotations or pretrained gene embeddings form a declared additional-information track. Give comparable models the same features in that track.

Freeze the PCA or single encoder for the first controlled comparison. Joint fine-tuning remains a required candidate with an auxiliary single-reconstruction loss where applicable. Representation freezing may stabilize estimation; it does not guarantee leakage freedom or prevent the residual head from memorizing training pairs.

## 11. Opal candidate architectures

Every candidate predicts a residual and adds it to the observed additive term outside the learned decoder:

$$
\hat\delta_{AB}=\delta_A+\delta_B+\hat\epsilon_{AB}.
$$

The shared output form is

$$
\hat\epsilon_{AB}=b+Bv_{AB},
\quad
B\in\mathbb R^{G\times k},
\quad
v_{AB}\in\mathbb R^k.
$$

A nonlinear interaction head is allowed inside v_AB. B is linear so its output rank has a precise meaning. The offset b is a shared residual correction, tested separately from pair-dependent prediction. Latent directions are not uniquely identifiable biological mechanisms.

### Candidate A: coordinate product

$$
q_{AB}=e_A\odot e_B,
\qquad
v_{AB}=\Lambda^\top q_{AB},
\quad
\Lambda\in\mathbb R^{m\times k}.
$$

This is the original architecture. In fixed coordinates it uses diagonal bilinear interactions. A rotation of embeddings can change what this diagonal restriction expresses, so it must not be presented as an unrestricted bilinear model.

### Candidate B: learned symmetric bilinear factors

$$
q_{AB}=(Pe_A)\odot(Pe_B),
\quad
P\in\mathbb R^{h\times m},
\qquad
v_{AB}=\Lambda^\top q_{AB},
\quad
\Lambda\in\mathbb R^{h\times k}.
$$

Each latent output is a symmetric bilinear form with

$$
W_j=P^\top\operatorname{diag}(\Lambda_{:,j})P.
$$

The factors share a learned basis. This adds cross-coordinate interactions while preserving exact symmetry. It is still a restricted family of symmetric bilinear forms.

Compare with an independently parameterized symmetric W_j when dimension and sample size permit:

$$
v_{AB,j}=e_A^\top W_j e_B,
\qquad
W_j=(H_j+H_j^\top)/2.
$$

This comparison tests whether sharing factors is useful regularization or an expressivity bottleneck.

### Candidate C: expressive symmetric residual head

Use

$$
\phi_{AB}=
[e_A+e_B,\ |e_A-e_B|,\ (Pe_A)\odot(Pe_B)],
\qquad
v_{AB}=F_\theta(\phi_{AB}).
$$

F_theta is a small regularized neural network with validation-selected width and depth. Compare against the same network without the product branch. This lets capacity grow when necessary while making the contribution of multiplication testable.

Candidate C is eligible for release as Opal v1; it is not automatically deferred because it is more expressive. The final name identifies the validated configuration, not an architecture selected before experiments.

### Output-basis comparisons

Compare a fixed SVD basis fitted on development residuals with a learned B. Fit all bases again inside each development fold. A singles-only output basis is an ablation, not the default: it may exclude directions appearing only in combinations.

For an unrestricted-output comparison, use an unfactorized learned output matrix under the same input information and declared parameter count. Output factorization ceases to restrict effective rank if k reaches the input-feature rank. Count that case honestly.

Train an extended-output sensitivity model on the fixed extended universe from section 6, using the same registered architecture-selection procedure. It provides predictions for the emergent-gene coverage diagnostic; reporting only a primary-universe model's covered genes would hide that failure mode.

## 12. Optimization and actual additive initialization

The primary training loss is equal-condition residual MSE over the primary gene universe:

$$
\mathcal L_{\mathrm{fit}}
=
\frac1{|C_{\mathrm{train}}|}
\sum_{p\in C_{\mathrm{train}}}
\frac1G\|\hat\epsilon_p-\epsilon_p\|_2^2.
$$

Regularize the interaction head and output basis, and use development folds for weight decay, stopping, and loss variants. Candidate fine-tuning includes a separately weighted single-reconstruction loss. Measurement-weighted or gene-weighted losses are secondary candidates with weights fitted from allowed development information; they cannot depend on test outcomes.

Initialize the final residual coefficient map and b to zero while keeping the upstream factors and B nonzero. This makes the initial prediction exactly additive and leaves a learnable final map. Do not initialize every multiplicative factor to zero, which can eliminate gradients. Verify both zero initial correction and a nonzero first-step gradient on the intended parameters. Compare random initialization as an optimization ablation.

For affine factorizations, regularize both factors or constrain their scale so rescaling B and its coefficients cannot evade a one-sided penalty. Record singular values and effective rank. Learned axes may rotate or rescale without changing the predictor.

Log optimizer, learning-rate schedule, batch construction, training duration, objective components, gradient failures, and numerical precision. Use deterministic CPU execution where practical and document GPU nondeterminism rather than promising identical results across hardware.

## 13. Capacity and tuning budget

Separate embedding dimension m, bilinear-feature dimension h, residual-output rank k, and nonlinear-network width/depth. Initial development search ranges are:

| Quantity | Initial candidates |
| --- | --- |
| PCA or encoder dimension m | 16, 32, 64, 128, capped by available independent single-profile rank where applicable |
| Bilinear feature dimension h | 16, 32, 64, 128 |
| Output rank k | 4, 8, 16, 32, 64, with 1 and 2 as diagnostic ablations |
| Residual network | One or two hidden layers, widths 64, 128, 256 |
| Weight decay | Log-spaced values including zero |
| Learning rate | A small registered log-spaced range appropriate to the optimizer |

These are a search space, not a full Cartesian grid. Before runs, profile the actual hardware and freeze a manifest with maximum trials, wall-clock or accelerator-hour budget, and peak-memory limit. Default to at most 24 configurations per trainable family, with one seed for screening and three for finalists. Evaluate the two strongest configurations per family on all development folds with five registered training seeds before selection, within the recorded budget. Closed-form baselines receive an appropriate regularization/rank grid rather than artificial optimizer parity.

Use the same folds, available information, and comparable search effort. Exact equality of optimizer or epoch count can unfairly handicap models that train differently. Report both controlled matched-parameter comparisons and independently well-tuned competitors.

Select by mean development error, checking fold/seed stability and complementary metrics. If smaller and larger candidates are indistinguishable within development uncertainty, prefer the less costly one. If validation improves reproducibly at the upper search boundary, expand the range before opening tests and record the amendment. Do not stop at a small size solely to justify the lab name.

Report accuracy versus trainable parameters, total checkpoint parameters, fixed bases and references, peak training memory, training time, CPU inference memory, cold-start time, and warm single/batch latency. Include external embedding storage and upstream resource access. State hardware, thread count, batch size, precision, and environment for each timing.

## 14. Baselines and information access

### Required task baselines

| Baseline | Prediction or purpose |
| --- | --- |
| Control | Zero effect; establishes the no-change reference |
| Training-mean response | Equal-condition mean of development double effects |
| Additive | Observed delta_A + delta_B |
| Mean residual | Additive plus equal-condition mean of development residuals |
| Global epistasis correction | A regularized shared linear or nonlinear function of the additive profile; tests whether a global response distortion explains apparent interactions |
| Symmetric ridge | Regularized prediction from sum, absolute difference, and product features of fixed single embeddings |
| Reduced-rank ridge | Same pair features with a fitted output basis; a strong classical approximation to the hypothesis |
| Nearest-pair residual | Transfer or average development residuals from nearest pair features using validation-selected neighbors |

The nearest-pair baseline uses only development residuals. Canonical pair labels alone are not a useful distance. Ridge and reduced-rank ridge may fit the original product candidate without iterative training; disclose when two formulations are mathematically equivalent rather than presenting duplicate methods as separate evidence.

### Required controlled neural comparisons

Compare direct-response and additive-plus-residual predictors with the same features. Compare sum-only symmetric neural prediction, product features, shared bilinear factors, independent symmetric bilinear forms, and unrestricted output. Use both frozen and fine-tuned single representations.

An ordered concatenation MLP is a secondary control. For an order-independent competitor, average its A,B and B,A outputs, or use symmetric features. Opal must not earn its main advantage solely because the comparator was given an artificial ordering problem.

### Public reference methods

Run SLIM and at least two applicable public neural methods on common pair splits. The planned neural references are GEARS and PertAdapt on genetic data, and CPA on chemical data. CPA on genetic combinations is also a useful compatibility reference. Use fixed code revisions and track native versus adapted preprocessing.

SLIM uses a bilinear predictor with a low-dimensional expression basis and gene embeddings; its combination implementation must be checked directly. It is not the same model as Opal's symmetric product of observed-single representations. [SLIM code](https://github.com/RasmussenLab/SLIM)

GEARS incorporates biological graph information, while PertAdapt adapts single-cell foundation models. Their complete input access, checkpoint resources, and supported task settings must be reported. [GEARS](https://github.com/snap-stanford/GEARS), [PertAdapt](https://github.com/BaiDing1234/PertAdapt)

NAIAD is important prior art because it combines single effects and learned intervention representations for combinatorial prediction. Inspect its supported phenotype outputs before claiming a direct transcriptome comparison. An adapted transcriptome version must be labeled an adaptation, with a matched-information experiment. [NAIAD paper](https://proceedings.mlr.press/v267/qin25g.html), [code](https://github.com/NeptuneBio/NAIAD)

Keep an input-access table for each method: observed singles, development doubles, cell-level training data, external graphs, pretrained embeddings, and foundation-model weights. Structural claims rely on common-information comparisons. Public references establish practical competitiveness even when their information budgets differ.

If a public method cannot consume the registered data contract, provide an adapter with tests or report the incompatibility and replace it with a documented applicable reference before final evaluation. Do not silently skip strong methods because a weaker baseline is easier to run.

## 15. Evaluation metrics and panels

For pair p, define a_p = delta_A + delta_B and epsilon_p = delta_AB - a_p. For any predictor,

$$
\hat\epsilon_p=\hat\delta_p-a_p.
$$

On a fixed panel, residual MSE and delta MSE are identical:

$$
\mathrm{MSE}(\hat\epsilon_p,\epsilon_p)
=
\mathrm{MSE}(\hat\delta_p,\delta_p).
$$

Keep one numerical implementation and use the residual interpretation where relevant. This is not a new independent metric.

### Primary metrics

Define E_p as gene-averaged squared effect error on the primary frozen universe. Report equal-pair mean E_p and the paired mean improvement over every required comparator. Also report additive-relative skill:

$$
\mathrm{Skill}_{add}
=
1-
\frac{\sum_p E_p^{model}}{\sum_p E_p^{add}}.
$$

One means perfect prediction, zero means no aggregate improvement over additivity, and negative values mean worse performance. Report it as undefined if the additive denominator is zero or below a numerical tolerance. This aggregate ratio emphasizes pairs with more additive error; it does not replace the equally weighted pair-error table.

The primary structural test is superiority to both additive prediction and the strongest development-selected simpler residual competitor, with paired uncertainty. The practical goal is a reproducible material improvement, not a tiny numerical win. Before test access, register a minimum relevant effect from development variability and intended use; a 5% additive-relative error reduction is the initial planning target, not a retrospectively adjustable success threshold.

### Complementary metrics

Report residual cosine alignment, residual norm/calibration error, effect Pearson correlation, MAE, and a perturbation-discrimination or retrieval diagnostic. For centered residual Pearson, explain that mean offsets are removed. For zero or near-zero residuals, report undefined alignment and coverage rather than forcing a favorable correlation value.

Report retrieval candidate sets and chance performance; near-identical conditions can make strict pair identification ambiguous. Correlation alone can reward the right direction with badly wrong magnitude, so it never substitutes for direct error.

### Component-informed panel

For single u, rank genes by

$$
\omega_{u,g}=|\delta_{u,g}|/(\sigma_{0,g}+\alpha),
$$

where sigma_0 is the standard deviation of individual normalized log control cells and alpha is a frozen numerical stabilizer. It is a ranking heuristic, not the standard error of a condition mean. Compare ranking by absolute single effect as a sensitivity analysis.

Use the union of the top 20 genes from each component, with stable gene-ID tie-breaking. Report its actual size and error. If fewer than 20 eligible genes exist, use all available genes and disclose that case.

### Emergent and benchmark panels

Use a registered evaluator-only panel of reproducible combination residual genes to test recovery of effects absent from single panels. Where cell counts permit, select this panel on one measurement partition and score on another, then swap and average. This reduces selection of the same noise later used as the target.

Test-derived panels are allowed only for this declared post-prediction diagnostic or an exact public-benchmark metric. They cannot enter model training, architecture selection, threshold tuning, or the primary panel. Distinguish benchmark top-DEG metrics from Opal's component panel; their gene sets and scales may differ.

Run the extended-output sensitivity model for genes excluded from the primary universe. Report both recovery and output coverage. A restricted panel cannot establish that emergent transcriptomic responses are predicted.

## 16. Interaction geometry and the non-additivity claim

Use RMS rather than raw vector norms for dimension-comparable summaries:

$$
N_p
=
\frac{\mathrm{RMS}(\epsilon_p)}
{\mathrm{RMS}(\delta_A)+\mathrm{RMS}(\delta_B)+\tau},
$$

$$
S_p
=
\frac{G^{-1}\langle\epsilon_p,a_p\rangle}
{\mathrm{RMS}(a_p)^2+\tau^2}.
$$

Choose tau from development scale only and save it; the initial rule is the larger of 1e-8 and 1% of the median development additive RMS. N is normalized residual magnitude. Positive S means reinforcement of the additive direction; negative S means opposition. Near-zero additive profiles require separate reporting because ratios become unstable.

The orthogonal-response diagnostic is

$$
N_p^\perp
=
\frac{\mathrm{RMS}(\delta_p-P_{\operatorname{span}(\delta_A,\delta_B)}\delta_p)}
{\mathrm{RMS}(\delta_p)+\tau}.
$$

Use a numerically stable SVD projection and a frozen singular-value tolerance. These quantities characterize response geometry on a chosen scale. They do not establish a biochemical interaction class.

Plot paired model error improvements versus N for additive and stronger comparators, with residual-reliability annotations. The additive predictor's error is determined by residual energy, so an increasing absolute advantage can arise without special success on biologically meaningful interactions. Include additive-relative skill within prespecified strata and the same diagnostics for shuffled-residual and global-correction controls.

If categories are useful, freeze magnitude thresholds from development doubles, using the 75th percentile as the initial strong-interaction threshold. Apply thresholds to test outcomes only after predictions are fixed. Report subgroup counts and uncertainty. The N-dependent trend is a secondary heterogeneity analysis; the main hypothesis cannot be rescued by a post hoc subgroup.

Use independent measurement partitions for N and error as a sensitivity analysis where feasible. The analysis still shares true biological signal; its purpose is to reduce coupling through the same measurement noise.

## 17. Required ablations and negative controls

Run the following with fixed comparison inputs and recorded parameter counts:

- Remove the explicit additive term while retaining the learned model's available single information.
- Remove multiplicative features from the symmetric network.
- Compare coordinate products with learned projections and independent symmetric bilinear forms.
- Vary output rank and compare fixed versus learned output bases and unrestricted outputs.
- Compare frozen versus jointly trained single representations.
- Remove the shared residual offset, and compare it with mean-residual and global-epistasis corrections.
- Compare PCA, single-profile encoder, target-ID features, and the declared biological-information track.
- Compare additive versus random residual-head initialization.
- Shuffle training residuals across pair labels while retaining the same data size and marginal target distribution.
- Vary observed single-profile cell counts to quantify sensitivity to noisy inputs.

The shuffled experiment is a diagnostic for meaningful pair association, not automatically an exact permutation test: pairs and residuals share targets and measurement references. Any formal permutation test needs an explicit exchangeability argument or a structure-preserving null.

A synthetic suite must include exactly additive pairs on each registered scale, known symmetric low-rank residuals, cross-coordinate interactions, high-rank interactions, shared global distortions, weak singles with emergent pair responses, and noisy shared controls. It verifies whether the implementation recovers what its hypothesis predicts and exposes situations where its bias should fail.

## 18. Sample efficiency, capacity, and stability

Use nested training-pair subsets at 10%, 25%, 50%, 75%, and 100% of the available development training pairs. Keep test pairs and registered tuning rules fixed. All singles remain available.

Use three registered subset orders so a favorable low-data subset does not define the conclusion. Report actual pair counts, target coverage, and component degrees at each fraction. Tune with folds contained in the available subset; using labels from the full development set for low-data fitting or model selection must be disclosed as a separate fixed-hyperparameter study.

Run an output-rank sweep and model-size sweep using development data, with selected configurations evaluated under the frozen protocol. Compare a smaller, selected, and larger configuration rather than only showing the one favorable rank.

Separate optimizer variability, subset variability, split variability, measurement uncertainty, and cross-dataset changes. A graph with hundreds of genes and thousands of cells can still have only tens of independent training combinations at the smallest fraction.

## 19. Reliability and uncertainty in the released model

The v1 predictor includes an ensemble of five registered independently fitted members unless development evidence shows that a cheaper alternative provides comparable accuracy and reliability. Assess the single-member and ensemble predictors separately. Apply comparable ensembling to the strongest neural comparator so averaging is not attributed to interaction structure.

Report predictive spread, single-profile measurement reliability, distance to development pair features, and each target's combination support separately. A model ensemble's disagreement is not a calibrated probability of correctness.

Evaluate whether these signals predict errors using development out-of-fold predictions, using nested selection or a held-out development partition when fitting risk thresholds, then confirm that assessment on held-out conditions. If risk thresholds are used in the interface, freeze them from development. Label low-support predictions and allow users to inspect the reason without suppressing difficult outputs from evaluation.

Prediction intervals may be added only with a documented calibration procedure, adequate calibration conditions, and measured held-out coverage and width. Do not claim gene-wise or simultaneous coverage from five ensemble members alone. Under structured shifts, state that calibration measured on random pairs may not transfer.

## 20. Statistical conclusions and decision rules

Predeclare the primary endpoint, primary outer split, selected simple comparator, uncertainty procedure, and minimum relevant effect before final tests. Report all registered comparisons and the denominator for every aggregate.

For each dataset, distinguish:

- **Supported:** the registered effect is positive with useful precision, passes the relevant baselines, and is consistent with complementary metrics and reliability checks.
- **Contradicted:** results reproducibly favor a comparator or contradict the predicted structural benefit.
- **Inconclusive:** intervals or measurement reliability do not distinguish the proposed improvement from alternatives.

Treat the independent genetic benchmark as evidence about reproducibility across genetic systems. Treat chemical retraining as evidence about method reuse across modalities. Their results can differ without invalidating every narrower finding.

Report primary comparisons with an explicit multiplicity policy, such as Holm adjustment within the small prespecified primary comparison family. Secondary ablations and heterogeneity analyses are identified as secondary, with full effect sizes and intervals. Do not replace inference with a leaderboard rank or a count of significant wins.

After test access, fixes to implementation bugs require an audit trail, a corrected rerun of affected comparators, and a disclosure of what was previously observed. A redesigned model requires a new protocol and fresh evidence; it cannot silently replace the failed confirmatory run.

## 21. Reproducible package and artifact contracts

Use a small installable Python package rather than placing all research logic in one script. The planned layout is:

    pyproject.toml
    src/opal/
        data/
        profiles/
        splits/
        representations/
        models/
        baselines/
        training/
        evaluation/
        inference/
        cli.py
    configs/
        datasets/
        experiments/
        release/
    tests/
    scripts/
    app/
    docs/
    README.md

Separate inference dependencies from dataset preparation, public-model adapters, and development tooling. Pin a reproducible research environment and compatible supported Python versions. Keep dataset caches and generated artifacts outside source control; provide retrieval commands and checksums.

Each prepared dataset writes:

    source_manifest.json
    preprocessing.json
    gene_schema.json
    condition_metadata.parquet
    profiles.npz
    profile_uncertainty.npz
    splits.json
    fit_access_log.json
    qc_report.json

Held-out double profiles are evaluator artifacts, never embedded in the training input bundle. Each run writes configuration, code revision or source hash, environment, seeds, fitted preprocessing/bases, checkpoint, selected stopping rule, predictions, per-condition metrics, timing, and a completion/failure status. Resuming a run must verify matching data and configuration fingerprints.

Planned commands are specifications for implementation, not claims that they exist today:

    opal prepare --config configs/datasets/norman.yaml
    opal audit --dataset artifacts/norman
    opal benchmark --config configs/experiments/v1.yaml --stage development
    opal benchmark --config configs/experiments/v1.yaml --stage final
    opal evaluate --run runs/selected --split test
    opal predict --checkpoint releases/opal-v1-norman --pair A B --output prediction.npz
    opal export --run runs/selected --output releases/opal-v1-norman

Final-stage execution must require a frozen selection manifest and produce a test-access record. This is an audit requirement, not a UI approval ceremony.

## 22. Verification requirements

Before model fitting, verify count-source validation, sparse/dense consistency, QC boundaries, condition parsing, guide/context grouping, matched controls, deterministic profiles, canonical pair splits, and complete single availability. Assert that no observed-response statistic determines split membership.

Add intervention tests rather than only checking printed PASS labels. With eligibility and split manifests fixed, replacing evaluator-only double expression must not change any fitted gene universe, normalizer, component panel, encoder, decoder basis, weights, or training predictions. Separately test preparation with response changes that preserve QC metadata. Changing condition counts may affect eligibility only through the documented rule; recomputing eligibility is a dataset revision, not a silent resplit.

Numerical tests must cover effect-scale nulls, residual/response MSE equality, exact pair symmetry, rank shapes, zero-head additive prediction with usable gradients, output-basis leakage, near-zero metric denominators, and correlated resampling of shared controls/singles.

End-to-end tests must exercise prepare, train, evaluate, export, load, and predict on a synthetic dataset with known interactions. Save/load must preserve predictions within a stated tolerance. Include malformed gene order, missing inputs, unsupported contexts, stale manifests, and mismatched checkpoint/profile-bank errors.

The installed package and local Space must pass smoke tests in a clean environment. Check CPU and GPU prediction agreement where both are supported, dependency compatibility, and the absence of evaluator-only responses from release references. Public-method adapters receive synthetic contract tests.

## 23. Hugging Face release and demonstration

Release a Hub model repository with safe tensor checkpoints where applicable, config, frozen preprocessing, gene schema, supported-context metadata, observed-single reference profiles or an exact retrieval recipe, example inputs/outputs, source revision, evaluation tables, licenses, and a model card.

A custom scientific model does not need a Transformer architecture to be downloadable from the Hub. Provide an explicit tested loading API and inference example. The model card identifies population-mean prediction, intervention modality, context, input access, training conditions, evaluated split, measured resources, uncertainty limitations, and known failure cases. [Hub model-card documentation](https://huggingface.co/docs/hub/model-cards)

The Gradio Space uses the same inference package and an immutable model revision. It allows users to choose a validated context and two supported interventions, inspect the additive response and learned correction, see reliability/support diagnostics, and download gene-level predictions. Also support batch pair files locally. [Gradio Spaces documentation](https://huggingface.co/docs/hub/en/spaces-sdks-gradio)

Show whether a pair was a training combination, a held-out benchmark pair, or an unmeasured combination in the source. Observed benchmark outcomes, if shown, must come from a separately identified evaluator/example asset. Do not quietly use them to produce the prediction.

The primary public research checkpoint stays trained only on its registered development combinations, so its recorded held-out results remain meaningful. A separate all-observed-data application checkpoint may be fitted after evaluation. Label it explicitly, preserve the research checkpoint, and never attach the research checkpoint's test score to the refitted weights as if that score measured them.

V1 can contain dataset-specific checkpoints for its validated contexts; sharing a family name does not imply one checkpoint generalizes across assays. New profile uploads must be validated against the input contract and identified as unvalidated when outside assessed support.

Measure the actual demo's cold start, warm latency, peak memory, and example correctness on the chosen hosting hardware. Dependency versions and model revisions are pinned. The runnable hosted Space is required for release, but any pricing or hosting plan is verified at deployment time rather than assumed here.

## 24. Milestones and completion gates

| Milestone | Work | Completion evidence |
| --- | --- | --- |
| M0: source and protocol audit | Verify all three datasets, experimental identities, licenses, counts, and matched singles; register endpoints and splits | Source manifests, feasibility report, frozen initial protocol |
| M1: benchmark implementation | Build normalization, effects, metadata, splits, reliability estimates, metrics, and synthetic fixtures | Tested deterministic artifacts and access-isolation checks |
| M2: baseline qualification | Run classical baselines, qualify metrics, integrate public references, profile hardware | Development baseline table, metric controls, tuning-budget manifest |
| M3: architecture development | Train candidates, representation variants, ranks, and capacity comparisons | Development cross-validation results and locked selection manifest |
| M4: full research evaluation | Execute primary/random/partner holdouts, independent genetic and chemical evaluation, ablations, sample-efficiency and uncertainty studies | Immutable predictions, condition-level results, intervals, all registered outcomes |
| M5: release engineering | Export research checkpoint, package/API, model card, local and hosted Space | Clean-environment tests, measured inference behavior, working Hub artifacts |
| M6: lab launch result | Write the research report, reconcile claims with evidence, document limitations and reproduction | Public report, reproducible code and artifacts, stable Opal v1 release |

The sequence is driven by dependencies, not a requirement that a particular Git commit contain no model imports. Model prototypes may be built on synthetic data while data adapters mature. Real-data model comparisons wait for the relevant benchmark gate.

Track progress in the repository by completed evidence, unresolved scientific questions, compute spent, and protocol amendments. A phase completes when its evidence exists. The launch is complete when the full research and engineering deliverables are reviewable and reproducible, even if some hypotheses are contradicted. The report must say so.

## 25. Required research outputs

Produce a report with the task and information contract, source provenance, mathematical parameterization, comparison fairness, complete results, uncertainty, and explicit conclusions for each hypothesis.

The figure set is:

1. Benchmark coverage and measurement reliability by dataset, with observed and eligible condition counts.
2. Primary baseline comparisons with paired effects and intervals, full-universe and component-panel results.
3. Residual recovery and error versus non-additivity, including stronger comparators and null controls.
4. Capacity/rank versus accuracy and total measured cost.
5. Sample efficiency with target coverage and subset variability.
6. Harder holdout and independent-dataset results.
7. Reliability calibration, representative successes, and prespecified failure analyses.

Mean-response shifts can reflect both changes within cell states and shifts in their population proportions. Use available source annotations for a prespecified mixture/state diagnostic, and avoid interpreting an average residual as proof of a cell-intrinsic mechanism. Do not automatically regress out cell cycle or differentiation signals that may be the response of interest.

Choose qualitative examples by a registered rule, such as median, best, and worst cases under a primary metric, rather than selecting only appealing plots. Biological interpretation uses recorded pathway annotations and states its multiple-testing procedure. A latent axis is not named as a biological mechanism solely because a few loaded genes look related.

Include a table of protocol deviations, unavailable comparisons, and remaining limitations. Publish negative and inconclusive outcomes alongside positive findings.

## 26. The start of the lab and the next versions

Opal v1 establishes the lab's operating standard: a concrete scientific task, enough model capacity to address it, fair comparisons, independently reproducible evidence, and an artifact people can use.

Future versions are justified by validated new capability or a material improvement in the existing task. Context transfer, unseen-component prediction, dose/time generalization, higher-order perturbations, richer distributions, and active experimental choice are possible directions. They are chosen from v1's findings rather than promised as a release schedule. A failure of low-rank residual learning can motivate another approach; it does not prevent the lab from doing useful research.

Use Opal v1 for the first model release, v1.x for compatible documented improvements, and a new major version for material changes in capability, input contract, or scientific formulation. Benchmark revisions and checkpoint revisions receive their own identifiers. No version number silently changes what a published result measured.

## 27. Primary references and what they inform

- [Norman et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/) establishes the original CRISPRa experiment and biological context.
- [GEARS study](https://www.nature.com/articles/s41587-023-01905-6) and [code](https://github.com/snap-stanford/GEARS) inform the genetic comparison and benchmark compatibility track.
- [Wessels et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36550277/) and [GSE213957 source record](https://www.ncbi.nlm.nih.gov/bioproject/883380) identify the independent RNA-targeting combination experiment.
- [CPA study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10258562/) and [combination tutorial](https://cpa-tools.readthedocs.io/en/latest/tutorials/combosciplex.html) inform chemical condition handling and compositional-model comparison.
- [Ahlmann-Eltze et al., 2025](https://www.nature.com/articles/s41592-025-02772-6) motivates serious classical baselines; it does not prove that all neural models are unnecessary.
- [Miller et al., 2026](https://www.nature.com/articles/s41587-026-03307-w) motivates positive metric controls, complementary endpoints, and evaluation beyond Norman.
- [SLIM](https://github.com/RasmussenLab/SLIM) provides a practical structured classical reference with explicit upstream embeddings.
- [PertAdapt](https://academic.oup.com/bioinformatics/article/42/Supplement_1/btag307/8726320) and [code](https://github.com/BaiDing1234/PertAdapt) provide a foundation-model adaptation reference.
- [NAIAD](https://proceedings.mlr.press/v267/qin25g.html) and [code](https://github.com/NeptuneBio/NAIAD) constrain novelty claims about using individual effects and learned representations for combination prediction.
- [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards) and [Gradio Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-gradio) define the intended public artifact format.

References were checked for this revision. Pin the exact dataset and software revisions used when implementing each experiment; a link alone is not reproducibility metadata.
