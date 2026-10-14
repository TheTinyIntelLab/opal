# Current planning scope

## Current successor status, 13 October 2026

The second-council `opal_context_contrast_v1` pipeline is implemented, with software checks only. The active plan/contract and [implementation guide](../frontier/context_contrast_implementation.md) supersede the historical bridge instructions below. SC01–SC08 now have executable qualification, preparation, contrast/reference/control fitting, sealed evaluation and readiness components. The fixed-context diagnostic uses both observed TNF contexts in training and no target bridges. Its numerical preparation, fitting and biological scoring require separate owner acceptance; independent confirmation remains unqualified. Frozen bridge configuration/results, old thresholds and failed attempts remain unchanged.

The active successor strategy is [the frontier plan](../frontier/PLAN.md), with a separately frozen bridge-information proposal. The analysis below records the earlier model-series rationale and evidence; it does not authorize another v2 grid or repeat the completed negative variance experiment.

# Historical assessment and research basis for the Opal model series

Prepared 10 October 2026. This is the historical design assessment, not a new fitted result. Its former implementation plan and handoff have been superseded by the frontier program linked above. Public model names remain Opal-Base, Opal-Interaction and Opal-Guard. References to v1/v2 describe preserved execution cycles and internal checkpoint provenance, not additional named models.

## The current series is usable, but has not established frontier performance

Opal v1 delivers real source-specific models, safe checkpoints and reproducible evaluation. It does not establish superiority over current published methods, a general biological model, or reliable improvements from its nonlinear branch. Calling it frontier would outrun the evidence. A defensible next objective is competitive prediction under a specified observed-single, small-pair-budget contract, with measured compute costs and direct comparisons to eligible published implementations.

| Frozen v1 method | Combi-seq MSE | PertSpectra TNF-positive MSE | PertSpectra TNF-negative MSE |
| --- | ---: | ---: | ---: |
| No change | 5.713889 | 0.137007 | 0.075095 |
| Additivity | 8.993460 | 0.099904 | 0.094443 |
| Base / selected reference | 3.872700 | 0.085929 | 0.082279 |
| Historical pair kernel | 4.388534 | 0.085133 | 0.082056 |
| Interaction | 3.871518 | 0.086707 | 0.083545 |
| Guard | 3.871571 | 0.086159 | 0.082651 |

Columns have different assays, genes and units and cannot be averaged. Combi-seq's nonlinear gain is about 0.03%; the two new nonlinear variants lose to Base in both PertSpectra contexts. Guard also loses to the fixed half-gate there. TNF-negative favors no change over all fitted models. In sample-label transport, Base loses to additivity and the other variants use their documented Base fallback. These results justify strengthening the reference, measuring residual signal and making learned support selection earn its place.

Evidence is in `results/opal_model_series_v1/`, including exact historical reproduction, full-budget and budget summaries, retained selection histories, independent replay, the transport diagnostic and installed-wheel records. The 1,296 budget checkpoints and 72 transport checkpoints are evaluation fits, not independent experiments or 1,368 independently trained architectures. Deduplicated experts are intentionally shared.

## What the models actually learn

Base selects a reference family from the existing bounded menu and fits its coefficients. On full-budget Combi-seq it chooses centered component-additive correction, so the effective model is measured additivity plus a shared offset and learned main effects for each intervention. It cannot express a partner-specific residual through those identity coefficients. That limited model nevertheless accounts for almost all of the new series' measured predictive performance.

Interaction adds a shared 512-to-64-to-16 tanh encoder, the elementwise product of the two encodings and a 16-to-512 projection. The branch has 42,064 trainable parameters at this endpoint. Each Combi-seq stratum has only about 249 acquired training pairs at full budget, with fewer labels at intermediate budgets. This is a parameter count and data-support observation, not proof that the network cannot generalize. Its projected correction occupies at most sixteen output directions, and its explicit product uses matching latent coordinates. A learned encoder can express rich functions, but the current experiment does not show that learning these representations from scratch helps.

The branch fits residuals from Base trained on the same branch-training pair labels. Those are legitimate development inputs, but a flexible reference can shrink the remaining in-sample signal while its out-of-fold errors remain larger. Compare in-sample residual training with an honest cross-fitted target route instead of assuming that either is superior. Cross-fitted targets introduce their own smaller-training-set distribution shift; the v2 experiment must measure that mismatch and use a nested, reproducible final-refit procedure.

Guard learns a four-feature sigmoid weight between two highly similar experts. A scalar gate cannot recover a missing output direction or learn a new residual expert. Its distance feature is often uninformative within a stratum: measured query singles already occur among training references, making their nearest-reference distances zero. Exact SampleIndex membership also forces fallback in the transport diagnostic. Source/TNF eligibility and arbitrary sample-label equality need separate contracts.

OOF expert predictions exclude their own validation pairs in v1. A further v2 audit must inspect meta-level selection: an expert predicting a gate-training row may have trained on a different row assigned to gate validation. Honest gate validation requires that *every* expert and transform used anywhere in that gate-training/validation comparison exclude the complete gate-validation population. This is a possible internal selection-dependence issue to test, not evidence that v1 outer test outcomes entered fitting. Preserve the original results and describe any verified defect precisely.

Architecture differences should produce testable capabilities, not artificially different predictions. Main effects, full cross-factor interactions and query-dependent residual retrieval provide different mechanisms. They may still collapse to the same conservative prediction when acquired labels do not support anything better. Do not add numerical noise, force nonzero gates or select seeds to make result tables look different.

## The data constrain the claims

Combi-seq contains nominal chemical pairs and three reported runs. Its aggregate-read normalization and exact-dose/vehicle/initiation limitations remain. Bad additive MSE can reflect assay arithmetic, global response attenuation, reference noise or genuinely nonadditive profiles; it does not isolate a biological interaction. A learned scalar shrinkage reference and the no-change control therefore belong inside v2 development selection.

PertSpectra supplies cell-level deposited counts, gene-label mixtures, four SampleIndex labels per TNF context and coherent stratum controls/singles. The ten-cell minimum can produce noisy aggregates. Different guide burdens, guide mixtures, coverage and reference reuse can affect residuals. These labels are not four independent cultures. Technical cell resampling can characterize sensitivity to sampling within this deposit, while shared control/single draws must be reused across every affected pair. It cannot supply independent-culture reliability or a biological noise ceiling.

Current complete-case pair populations may favor well-covered conditions. Audit coverage and keep the unchanged v1 panel as the primary comparison. Higher-cell-count and guide-compatible subsets are separately scoped sensitivities, with exclusions frozen from metadata before outcome scoring. Never change the gene panel or QC threshold because a new model lost.

Norman and Wessels already exist in this repository, along with count/log preparations, public-resource audits and architecture studies. Inspect `tools/study_architecture.py`, `src/models/advanced.py`, `results/architecture/`, `data/source_audits/architecture-20261005/` and protected expanded Wessels preparations before adding implementations. Prior attention, structured output bases, reliability weighting, linear probes and retrieval have already been explored. Their losing results count against casually proposing the same mechanisms as new discoveries. New shared-contract integrations must be distinguished from exact upstream reproductions and previously executed Opal approximations.

## Primary research and its implementation implications

The following sources were checked on 10 October 2026. Pin code revisions and assets again when executing; web summaries are not executable protocol specifications.

| Primary source | What it changes in the v2 design | Comparison boundary |
| --- | --- | --- |
| Ahlmann-Eltze, Huber and Anders, [Nature Methods 2025](https://www.nature.com/articles/s41592-025-02772-6), [author code](https://github.com/const-ae/linear_perturbation_prediction-Paper) | Retain strong linear, additive and training-mean references; test data-derived representations before assuming a pretrained encoder helps. | Its task and preprocessing are not Opal's source-specific benchmark; published scores cannot populate an Opal comparison table. |
| Miller et al., [Nature Biotechnology, 1 October 2026](https://www.nature.com/articles/s41587-026-03307-w), [author code](https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines) | Add prespecified perturbation-discrimination and metric-sensitivity diagnostics alongside MSE. Evaluate positive controls only in a separate scoring-only pathway. | Outcome-derived DEG masks and technical-duplicate controls from its protocol cannot become Opal predictor inputs or replace the historical endpoint. |
| Roohani et al., [GEARS](https://www.nature.com/articles/s41587-023-01905-6), [author code](https://github.com/snap-stanford/GEARS) | Establish a native combinatorial genetic comparator with an audited graph/coexpression data-access route. | Genetic graph inputs are an additional-information tier; GEARS does not directly supply a chemical Combi-seq comparator. |
| Lotfollahi et al., [CPA, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10258562/), [author code](https://github.com/facebookresearch/CPA) | Compare a nonlinear latent composition approach where cell-level data and matching covariates permit it. | CPA's distribution/dose task differs from aggregate change prediction. No claim of supported Opal doses follows from installing it. |
| Wenkel et al., [TxPert, Nature Biotechnology 2026](https://www.nature.com/articles/s41587-026-03113-4), [author code](https://github.com/valence-labs/TxPert) | Use matched controls, retrieval diagnostics and a bounded public-prior comparison, with data-derived and graph representations ablated. | Some strongest configurations depend on proprietary graphs. Public-only adaptations must not be called reproductions of those full configurations. |
| Adduri et al., [State, Cell 2026](https://doi.org/10.1016/j.cell.2026.07.052), [author code](https://github.com/ArcInstitute/state), [Cell-Eval](https://github.com/ArcInstitute/cell-eval) | Separate mean prediction from population-distribution and cross-context tasks; audit evaluation utilities before reuse. | Its data scale, pretraining and contexts differ sharply. Do not manufacture single-cell predictions by duplicating aggregate Opal outputs. |
| Cole et al., [2026 preprint and author code](https://github.com/genbio-ai/foundation-models-perturbation) | Treat informative external representations as a finite additional-input experiment with contamination and license audits. | Its reported benefits do not establish Opal performance; pretraining access is unequal and must be disclosed. |
| Chang et al., [PertSpectra, PSB 2026](https://psb.stanford.edu/psb-online/proceedings/psb26/chang_s.pdf), [author code](https://github.com/insitro/PertSpectra) | Attempt its native label-conditioned factorization on eligible genetic panels, with global pair exclusion during representation fitting. | Its absolute-profile correlations and filtered cell population differ from Opal's delta-MSE endpoint. |

PRESAGE's [author implementation](https://github.com/Genentech/PRESAGE) and the already cached resource audit are relevant bounded additional-input comparators. Native run status, model/data licenses and prior input provenance must remain explicit. CellFlow's [primary preprint](https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1) and [author code](https://github.com/theislab/CellFlow) inform a future distributional task, but a flow model is not a necessary repair for predicting a few hundred aggregate pair vectors.

These findings support an ambitious small-data benchmark, not a guarantee of a positive result. A field-wide frontier claim needs an adequately broad, matched comparison and reproducible superiority. A narrower efficiency claim needs actual Pareto measurements against the methods that ran under the declared contract. New seeds on exposed panels, parameter counts and architecture novelty cannot provide either by themselves.
