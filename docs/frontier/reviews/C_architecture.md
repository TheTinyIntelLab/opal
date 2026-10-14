# C — Independent architecture investigation

Initial investigation: 11 October 2026. Status: initial report with completed actual cross-examination and revised recommendation below. This is a research design, not an executed experiment or a claim that an architecture works. The revised recommendation supersedes the original prototype preference and provisional thresholds.

## Decision

Do not build another large residual network now. Build a small, separately identified experiment testing whether **context changes in measured singles carry transferable information about context changes in reproducible pair residuals**. Qualifying independent biological repetition is the first gate. If this information is absent or irreproducible, no transformer, graph prior, or generative decoder resolves that identifiability failure.

The eventual specialized capability is prediction of a globally unmeasured intervention pair in a biological context without measured pair labels, conditional on measured matched singles. This remains ambitious: it extends the current observed-single contract across contexts while retaining a realistic experimental input requirement. It must be described as observed-reference transfer, not zero-shot intervention discovery. Generic single-perturbation prediction and generic distribution generation now have strong competitors; simply adopting those tasks offers little differentiation.

## Evidence inspected

I read all 43 pages of the extracted current `paper/main.pdf` manuscript in `tmp/frontier_manuscript.txt`, including the variance addendum. I inspected `src/models/kernel.py`, the complete `src/series_bilinear.py`, `src/information.py`, the state-model implementation, model documentation, the information protocol/report, and `results/opal_information_v1/development/summary.json`. I did not execute new fits, inspect inaccessible independent culture measurements, or open protected historical evaluation responses. Manuscript numerical claims below are distinguished from direct saved-summary verification.

The actual kernel is centered regularized regression with an unpenalized residual mean. The symmetric product kernel averages the two component matchings; the sum kernel averages all four single similarities. Its capability is interpolation of response corrections conditional on measured singles, not encoding an intervention that has never been measured. The current bilinear series explicitly residualizes pair features against a main-effect design where available, checks unique-pair support, offers cross-fitted Base residuals, and permits a zero branch. Thus “try cross-coordinate products” or “remove the main effect first” is already implemented, not a new proposal. The context model already combines intervention/source-effect/descriptor routes, control-PCA state features, and product conditioning. A new contextual MLP alone is not a fundamentally new strategy.

The saved variance summary directly verifies primary TNF-positive partner MSE 0.0825683915 versus Base 0.0827597015, mean 0.0825437889, capacity mean 0.0825393157, and shuffle 0.0826006173. The variance head gains 0.231163% over Base but loses to both mean controls. TNF-negative partner scores coincide because support forces fallback; this is inconclusive evidence about information in that context. The summary explicitly records `information_gate_passed=false`, no confirmation, and zero new independent cultures. This rejects this projected-variance representation and finite selection menu for mean prediction; it does not reject distributions, covariances, matched biological contexts, dose response, or intervention identity information generally.

Manuscript evidence supplies the remaining motivation: independent D-SPIN procedure transfer fails; control-PCA state heads lose to source-mean/simple references; pair products fail the joint chemical pair/state gate; increased Wessels pair labels helped one common development population; residual technical-half cosine is weak; Interaction gains are below 0.1%; Guard is not promoted. None of those observations establishes that biological interaction structure is absent. Equally, none supports spending on a larger decoder with the same few pair labels.

## Four genuinely different candidates

### 1. Hierarchical statistical response model with biological replicate structure

**Hypothesis.** Apparent failures partly reflect noisy shared singles/control estimates and study calibration, while a small reproducible pair-specific signal exists. A hierarchical errors-in-variables model can separate those terms when actual independent culture measurements exist.

For gene vector measurements in context c and independently initiated culture b, write

`y_(S,c,b) = mu_(0,c,b) + sum_(u in S) m_(u,c) + q_(S,c) + eta_(S,c,b) + epsilon_(S,c,b)`.

Use actual controls/singles to estimate their means jointly, not independent pairwise residual errors. For pairs, fit `q_(AB,c)=h_c(a_(AB,c))+K_pair(AB,train) C_c`, with shrinkage on pair coefficients and a retained zero-pair branch. Technical count noise and culture deviations receive separate covariance terms. Optimize marginal Gaussian likelihood on culture pseudobulks or restricted likelihood for variance components; compare to equal-weight regularized profile kernels on the identical output scale. Do not substitute technical-cell bootstrap covariance for biological covariance.

**Inputs/data.** Controls, singles, pairs, exact dose/endpoint, randomized plate allocation, and at least three independently initiated cultures per initial context. Three cultures are a variance pilot, not a powered confirmation sample size. A qualified factorial intervention subset is more valuable here than millions of cells from one culture.

**Compute.** For hundreds of pair-context rows and 512 genes, closed-form generalized ridge/mixed effects are CPU work. Dense exact kernels scale cubically in rows; use low-rank features only if thousands of rows justify them. Count complete reference banks and covariance assets.

**Closest competition/capability.** Existing Opal covariance kernels and GPerturb are close; hierarchical uncertainty is not inherently novel. The additional capability is honest culture-level estimation and possibly uncertainty under the declared sampling regime, which current fixed-profile baselines cannot supply. It is a measurement model and strong reference rather than the main frontier contribution.

**Failure modes/minimum test.** Too few cultures make variance components unstable; batch can alias context; residuals can be scale artifacts; pooling may erase genuine context effects. First test opposite-culture prediction of pair-specific residual deviations against shared/global corrections. If repeatable pair-specific signal is below a predeclared useful margin, retain this model as the reference and stop residual-architecture development on that domain.

### 2. Transferable intervention representations from independent response screens

**Hypothesis.** A representation learned across independently measured interventions encodes relationships relevant to unseen component identities or sparse pair supervision, beyond observed-single similarity or curated class membership.

Let descriptor `d_u` contain verified gene/chemical features with explicit missingness, and let `z_u=E_phi(d_u)`. Pretrain on source single-response tasks:

`L_pre = sum_(u,c) ||D_psi(z_u,h_c)-delta_(u,c)||^2 + lambda ||phi,psi||^2`.

Freeze the encoder for the first combination test. Fit `delta_hat_(AB,c)=delta_(A,c)+delta_(B,c)+B W sym(z_A tensor z_B)` or a kernel over z. Compare to linear descriptor ridge, shuffled descriptor alignment, random features of equal dimension, observed-single kernels, and the same head without pretraining. An unmeasured intervention track replaces the additive anchors with separately evaluated estimates and never mixes its scores with measured-single prediction.

**Inputs/data.** A large independent, assay-compatible response screen, source-only descriptors and exact perturbation modality, plus a combination source with genuine identity holdout. Current Replogle CRISPRi to Norman CRISPRa transfer is a modality/scale shift, and the existing 16-dimensional pretraining study already failed against the kernel. Merely adding more updates to it is not a new hypothesis.

**Compute.** Start with frozen public vectors and a ridge probe. Small response-supervised encoders fit on an 8-GB GPU; full foundation encoders require an isolated Linux/profiling plan and storage permission. No resource justification follows from the head parameter count alone.

**Closest competition/capability.** GEARS, SLIM, PRESAGE, TxPert, PertAdapt and now PIE already occupy representations/knowledge transfer. PIE's official 5 October announcement describes population-level predictions from curated context/intervention knowledge with public code/checkpoints; that makes a generic semantic perturbation encoder especially weak as a novelty claim. A credible differentiator would be transfer to **pair-specific** response or globally unseen pairs under matched information, not another single-response benchmark.

**Failure modes/minimum test.** Pretraining contamination, mismatched perturbation efficacy, class memorization, descriptor-identity shuffles that do nothing, and failure to identify interactions from single-only supervision. The minimum test is one frozen representation versus capacity-matched controls on a new identity-held combination bank. Do not build a graph transformer before this probe passes both a materiality gate and descriptor-alignment controls.

### 3. Context-contrast composition model — preferred successor if information passes

**Hypothesis.** The *change* in singles between contexts predicts the *change* in pair residual beyond pooled pair effects and shared context calibration, and this relation transfers to globally held pair identities.

Define `r_(AB,c)=delta_(AB,c)-delta_(A,c)-delta_(B,c)`. Estimate a source-only pooled pair prediction `r_pool(AB)`, and a shared context term `g(c)` from training. The target is `t_(AB,c)=r_(AB,c)-r_pool(AB)-g(c)`. The first prototype is regularized linear prediction in a fixed small gene basis:

`t_hat_(AB,c)=B W [dA+dB, abs(dA-dB), sym(zA tensor dB+zB tensor dA)]`,

where `dU=z_(U,c)-mean_source(z_U)` and all PCA/output bases, means, scalers, W and penalties are source-only. Contexts without source means use an explicitly frozen query-reference rule. A simpler contrast-only ridge without products is mandatory. Source-only pair/context main-effect fitting and subtraction must be repeated in inner selection, and held pair outcomes cannot supply `r_pool`.

In a larger successor, replace this ridge by a low-rank context-conditioned operator `q_(AB,c)=B sum_k v_k(c)(z_A^T M_k z_B)` with symmetric M_k. This has a controlled capacity increase, not a requirement to use attention. Predict simultaneous combinations symmetrically; sequential compositions require separate temporal data and may not commute.

**Inputs/data.** Crossed independently assigned biological contexts, shared intervention identities, matched controls/singles/pairs, exact dose/time/modality and independent cultures. Minimum development qualification: enough shared pairs to create globally excluded pair folds and whole-context folds with feasible nested training; register the concrete metadata-generated folds before looking at expression. Two contexts can test one context contrast but cannot establish general held-context transfer. Source sample/lane labels cannot be silently promoted to contexts or cultures.

**Compute.** Ranks 4 and 8, three penalties, zero branch and a contrast-only reference fit on CPU. Less than 100 trainable pair-feature coefficients per output direction is the initial target, contingent on unique-pair support. Exact resource ceiling is set after metadata qualification, not assumed from an arbitrary pair count.

**Closest competition/capability.** CPA, State and Stack model composition or cross-context transfer; Opal already tested weak control-PCA state heads. Distinct evidence would be a transferred, independently repeatable pair-specific context contrast under globally held pairs. Generic context dependence or a source identity lookup is not novelty. Controls must include context-constant pair prediction, pooled profile kernel, query-single global correction, context-only correction, capacity-matched means, and context/identity shuffled alignments preserving the complete reference-sharing structure.

**Failure modes/minimum test.** Context can be confounded with assay; post-treatment cell type is not pretreatment state; context contrasts amplify noisy references; global calibration can masquerade as an interaction; source-average subtraction may leak query outcomes. First qualify cross-culture residual repeatability, then perform one contrast-transfer ridge test. If the repeatable signal exists but cannot transfer, focus next on informative local pair measurement rather than enlarging this head.

### 4. Predict response distributions or scientist-defined phenotypes

**Hypothesis.** Important combination responses change mixture proportions or heterogeneous state occupancy that mean-effect MSE cannot represent, and those changes are repeatable and transferable from measured single distributions.

The minimum distribution model is a control-trained fixed state partition, with a multinomial ridge predicting state occupancy. Let `pi_(AB,c)` be held pair proportions over K source-defined states. Fit

`pi_hat=softmax(W [features(P_A),features(P_B),context])`,

with equal-culture cross-entropy or Brier loss. Compare pooled occupancy, single-average occupancy, scalar mean/variance summaries, and identity-shuffled distribution features. K is fixed before pair access; perturbation-specific target clusters cannot define the predictor representation. Measure occupancy and expression separately. If this fails, no generative flow is justified.

If it passes and finer geometry matters, conditional flow matching predicts a population distribution: `dx/dt=v_theta(x,t,condition)` and minimizes `E||v_theta((1-t)x0+t x1,t,condition)-(x1-x0)||^2` under a declared training coupling. Frozen 32-dimensional control/source embeddings and a small MLP are enough for the first generative comparison. Cross-sectional optimal-transport coupling is predictive modeling, not identification of a cell's causal trajectory. Cell death/growth require independent abundance information; a normalized distribution cannot reconstruct lost population mass.

**Inputs/data.** Sufficient independently repeated cells per condition/culture, pretreatment context, perturbation efficiency controls and an independent biological endpoint. Matched distribution replication must precede distribution-model promotion.

**Compute.** Occupancy ridge is CPU work; 32-dimensional small flow fits on an 8-GB GPU with a strict initial GPU-hour cap. Full-transcriptome generation is deferred. Embeddings, preprocessing and the control bank count toward complete inference resources.

**Closest competition/capability.** The Theis-lab CellFlow already supplies set-condition encoders and flow matching; Arc State predicts heterogeneous populations. A second, unrelated imaging paper also uses the name CellFlow, so identify the implementation explicitly. Generating cells is not novel. The capability simple mean predictors lack is a verified shift in mixture occupancy, tails, or a predeclared functional phenotype. The scientific endpoint, rather than the generative architecture, must justify this route.

**Failure modes/minimum test.** Capturing depth, cell cycle, guide efficiency or survival selection can improve distribution metrics without useful perturbation discrimination. MMD can reward pooled generation. First show replicated pair-specific occupancy changes and a permutation-calibrated held-pair occupancy gain. Do not reuse the failed projected-variance experiment as a verdict on this task: its target was mean response and its features discarded most distribution structure.

## Central prototype contract

The preferred first research question separates measurement, information and model capacity. First estimate reproducible context contrasts across independent cultures; then evaluate whether measured single contrasts predict them. A positive result would justify a context-conditioned composition architecture. A negative repeatability result would falsify the target in that assay at its precision; a negative transfer result with repeatable labels would falsify the chosen representation, not the existence of biological context effects.

Use a newly identified run such as `frontier-context-contrast-v1`. Treat all historical outcomes as exposed development; freeze their hashes and exclude them from confirmation. New pair identities are globally held across every context/dose/culture. Biological contexts are held as whole groups. Query controls/singles are allowed only under an explicit reference contract. Inner selection repeats the same pair/context grouping and recomputes every target nuisance term. Engineering tests use synthetic known context contrasts, null context contrasts, scale distortion and shared-reference errors, with no claim of biological accuracy.

Primary loss is equal-context, equal-pair gene MSE on context-contrast residuals, alongside an absolute pair-response noninferiority criterion. The strongest comparator is a nested-selected envelope of pooled pair, context-only/global calibration and query-single reference models. Use culture-aware crossed context/target resampling and full selection/refitting where feasible; cells and overlapping folds are never counted as independent experimental units. For a very small number of contexts, report contextual effects and leave-group-out sensitivity without claiming a precise population confidence interval.

A proposed *development* promotion threshold is at least 10% reduction in context-contrast MSE versus every matched reference, a positive culture-aware uncertainty bound, disappearance of the gain after breaking context/intervention alignment, and no more than 2% deterioration in absolute pair-response MSE. These are provisional engineering/scientific triage margins to freeze before outcomes; independently justified biological usefulness and empirical power must set the later confirmatory margin. Do not tune the cutoff after seeing results. Failure ends this feature/model family for that contract; it does not trigger a wider menu, more seeds or outcome-selected panels.

## Primary sources checked

1. [Opal manuscript](../../../paper/main.pdf), particularly Methods 3.5, Tables 5–15, Sections F–I; local experimental evidence as listed above. Manuscript claims are not independent confirmation.
2. [GEARS primary article](https://doi.org/10.1038/s41587-023-01905-6).
3. [CPA primary article](https://doi.org/10.15252/msb.202211517).
4. [NAIAD primary ICML paper](https://proceedings.mlr.press/v267/qin25g.html). Its single-informed compositional and acquisition agenda constrains novelty.
5. [State official code and licenses](https://github.com/ArcInstitute/state). Transition and embedding components are separate; code and model resources have distinct noncommercial terms.
6. [State primary Cell article](https://doi.org/10.1016/j.cell.2026.07.052).
7. [Theis-lab CellFlow official implementation](https://github.com/theislab/CellFlow), [condition encoder documentation](https://cellflow.readthedocs.io/en/latest/user/genapi/cellflow.networks.ConditionEncoder.html), and [primary preprint](https://doi.org/10.1101/2025.04.11.648220). It is already a compositional distribution model.
8. [GPerturb primary article](https://www.nature.com/articles/s41467-025-61165-7). This is a relevant probabilistic reference; model uncertainty claims still require task-specific calibration.
9. [PIE official release, 5 October 2026](https://arcinstitute.org/news/pie), linked [primary preprint](https://doi.org/10.64898/2026.10.02.756297). The announcement is author evidence of release and claimed scope, not independently verified superiority.

## Preserved disagreement / open dependency

My preference places independent crossed information ahead of unseen-intervention foundation modeling or generic active acquisition. This is conditional on obtaining a qualified assay partner or sufficiently documented public crossed data. If none exists after bounded resource qualification, the honest 30-day output is a falsifiable contract and source-feasibility decision, not a pretended confirmation result. A public single-perturbation transfer probe can be an alternative development direction, but its substantial contemporary competition should lower the expectation of a distinct frontier claim.

### Initial response to the root's bridge-pair proposal

Sparse measured target-context bridge pairs are a stronger information hypothesis than another control-conditioned kernel, because they directly observe how composition changes in the target context. The old context-negative result did not test this information contract. This route deserves preference over the zero-bridge prototype if access to crossed pairs is realistic. It is also less ambitious in its information requirement: call it few-shot pair/context transfer and count every acquired anchor, its controls/singles, selection labels and replicate cultures in the experimental budget.

The dangerous alternative explanation is that four or eight target pairs estimate a shared assay correction. Therefore target shared offset, affine/global correction, component-incidence ridge and a source-free target-only kernel must receive exactly the same bridges and query singles. A source-conditioned operator earns advancement only by beating that envelope on disjoint pairs. Anchors must be fixed by source-only design; if target outcomes select them, the acquisition policy and its full budget need a separate nested evaluation. Pair-shuffled bridge alignment should retain the same target-context residual marginal and reference covariance, otherwise the shuffle can manufacture an easy negative control.

Four or eight anchors cannot identify an unrestricted low-rank matrix. A defensible minimum is a source-learned rank-two residual subspace/operator with only two target coefficients plus an intercept fitted from anchors, or a posterior update in an already fixed source kernel. Compare both to two equally free calibration coefficients. No target output PCA, rank selection or bridge-count tuning can use the sealed query pairs. If rank-two bridge adaptation beats target calibration only on pairs sharing bridge components, that is interpolation through intervention incidence; globally disjoint component or partner tests must disclose and delimit it. Repeatability qualification still precedes a frontier claim.

## Actual cross-examination and revised decision

I subsequently read specialist D's evaluation report, specialist F's systems report, and the root's concrete `docs/frontier/experiment_contract.md`. I also received explicit critiques from E and G. These reviews change my first-experiment choice: use **four measured target-context bridge pairs and a source-fixed rank-two predictor**, not my initial zero-bridge single-contrast prototype. The new information contract directly observes target composition; the available two TNF contexts cannot honestly supply nested biological-context validation for a learned zero-bridge contextual map. The public test is still exposed development and one bidirectional context contrast, not two independent biological replications.

### Review of D: evaluation

D's strongest recommendation is globally excluding query pair identities from every source context and providing every strong calibration/reference comparator the same bridge outcomes. That is essential: carrying a pair already measured elsewhere is an easier task and source-specific calibration can explain most apparent improvement. I accept response MSE as the primary endpoint and the root's frozen 5% gain in both directions, with independent-reference attribution as a separate requirement. My original 10% contrast-MSE threshold is superseded, not another gate to select after outcomes.

Two initial D recommendations need narrowing. Its multiple bridge budgets could turn this into another selection study; the root's fixed four bridges plus zero diagnostic isolates the information question. Its nested held-context selection requires more contexts than the public pilot has. The root correctly uses source-only grouped pair selection and fixes target penalty before target outcomes. Baseline selection may simulate bridge acquisition inside source pair folds, but that is source-task selection, not evidence of context-validation performance. D's independently anchored interaction diagnostic should retain separate predictor/evaluator reference estimates; if only technical partitions exist, its result is a measurement sensitivity rather than biological interaction confirmation.

D should also avoid treating a query-panel-centered error or retrieval gain as sufficient pair specificity. Target-incidence and affine controls can preserve discrimination while predicting no transferable source structure. The source-shuffle, bridge-shuffle, equal-capacity mean adapter and source-free target adapter collectively address the claim, and all are needed. A program precision-at-k or regret diagnostic can make the eventual utility connection explicit, as E requests, but only when the program is fixed before query access; there is no justification for searching pathways after scores.

### Review of F: systems

F correctly separates head values from complete source/reference storage and treats resource caps as caps rather than measured frontier requirements. Its CPU-first approach fits the numerical experiment, while the inherited approximately 75-MB artifact headroom rules out casually downloading native graphs or installing a new foundation-model environment. I accept its disagreement with a blanket block on public development: independent cultures are necessary for biological advancement, but a properly labeled public numerical falsification can proceed now.

F's revised scalar intercept proposal and its initial unrestricted two-by-two map are both superseded by the root's actual contract. The chosen model fits a **G-dimensional common bridge intercept plus two scalar pair-dependent coefficients**, so the fitted target numeric-value count is G+2, not two or three. This is still small in pair-dependent capacity, but the gene-vector intercept may absorb substantial calibration; every calibration comparator must have the same intercept freedom and all G values count in resources. The initial 8/16/32 prefixes and tuning of target regularization on bridge labels are also superseded: four bridges, fixed target penalty, source-only pair selection.

F's practical model-cost argument should not claim a larger architecture is permanently unjustified. If independent crossed labels support repeatable directions outside the source rank-two span, the prototype's failure can identify a representation bottleneck. That would justify a separately frozen higher-rank or nonlinear experiment after new information qualifies, not silent expansion within this run.

### Numerical requirements for the revised root prototype

Normalize each source-fixed mode by its source-training RMS and optimize **gene-averaged**, equally bridge-weighted squared loss plus fixed ridge penalty 1. Otherwise arbitrary SVD scaling or the number of output genes changes the meaning of regularization and the adapter/control comparison. Persist these source scales. Fit the common gene-vector intercept under the identical bridge centering rule for all arms; target genes are correlated output coordinates, not independent labels.

Check the centered two-column bridge design using relative singular-value rank tolerance 1e-8 and condition number at most 1e4; persist singular values, source RMS and the exact flattening/weighting convention. This rank check is numerical support, not a sample-size or biological power certificate. Run the four fixed leave-one-bridge fits, without selecting one, and require the root to freeze a finite stability criterion before scores—for example no query correction RMS above twice the full-fit source-normalized scale and no individual coefficient exceeding its source-fixed support limit. A stability limit is a refusal diagnostic, not another outcome-selected hyperparameter.

Report the query subset with zero components appearing in the four bridges, and the subset with one/two appearing components, with complete denominators. If no disjoint-component subset exists, the global-new-pair prototype may remain feasible, but no claim about transfer to unseen component incidence is available. A rank-supported adapter that succeeds only for bridge-sharing components is a narrower interpolation tool.

### Remaining disagreement and novelty restriction

I retain stronger emphasis than F on obtaining biological repeatability before expensive architecture development. Public bridge success is grounds to buy qualified information, not grounds to build a large model. E's warning about BATCHIE and other interaction-factor models is valid: a low-rank context/pair equation alone is not a novel contribution. The frontier claim would require sample-efficient, independently repeated transcriptomic composition transfer to globally unseen pairs and held contexts, with utility and contemporary matched comparators. If calibration alone matches the operator, abandon source-structure claims even if the calibrated predictor is useful.
