# Specialist E: scientific utility and discovery

Status: independent initial investigation, 11 October 2026. Primary sources and local evidence are distinguished below. This report proposes work; it does not claim that the proposed capability exists.

## Decision

Build toward **budgeted adaptation of combination-response prediction to a new biological context, then use it to prioritize interventions that move a prespecified transcriptomic program**. The dominant first experiment should test whether a small set of target-context pair measurements contains transferable, pair-specific information beyond same-budget calibration. Do not start with uncertainty-based active selection: Opal's existing uncertainty failed under shift, and existing prospective combination-screening systems already set a substantial utility standard.

The eventual scientist-facing question is: “With controls, measured singles, and a small bridge screen in this cell context, which unmeasured pairs reproducibly move this specified program while avoiding the specified toxicity or unwanted response?” This is more defensible than “discover synergy” from a log-expression residual, and more useful than lowering an all-gene error without changing which experiment a scientist runs. A transcriptomic program is an assay phenotype, not automatically a therapeutic benefit; a secondary viability or functional assay must establish the latter.

## Evidence that constrains the use case

Manuscript evidence: all 43 pages of `tmp/frontier_manuscript.txt`, extracted from the supplied `paper/main.pdf`, describe an observed-single population-mean task. Historical Wessels neural improvement over additivity loses to global correction; pair-product and state continuations mostly fail; new Interaction gains are below 0.1%; Guard remains unpromoted. Manuscript Sections C/D/E record failed D-SPIN transfer, cross-term cancellation under estimated singles, and no validated acquisition policy. Appendix I records a negative finite projected-single-variance experiment, with TNF-negative partner support insufficient. None of these establishes that independent cultures or target-context pair observations have no value.

Underlying artifacts inspected: `docs/next_cycle/opal_information_report.md`, `results/opal_information_v1/report/claims.json`, and `results/state/emeraldbay-diagnostics-v1/acquisition.json`. The information report independently confirms the failed variance triage, zero fully eligible native contemporary comparisons, and absent biological/utility confirmation. The acquisition artifact explicitly marks prospective value unestablished and acquisition not extended because neither useful state representation nor independent risk diagnostics passed. These documents are local records, not independent validation of their own scientific claims.

The manuscript's shared-distortion synthetic example and strong Base calibration gains make **calibration the leading competing explanation** for any new bridge-label improvement. An adapter that simply corrects scale or a shared offset could still be useful, but would not substantiate a new interaction-transfer capability.

## Closest prior work and novelty boundary

[NAIAD, Qin et al., ICML 2025](https://proceedings.mlr.press/v267/qin25g.html) uses measured single effects plus learned intervention embeddings, adapts capacity to available data, and recommends combinations with high predicted phenotypic effects. It evaluates large scalar-phenotype combination screens. Thus combining singles with adaptive representations, improving small-sample combination prediction, or selecting high-effect gene pairs is already covered. Its [official repository](https://github.com/NeptuneBio/NAIAD) describes scalar phenotypic modeling and experimental recommendation; a transcriptomic output comparison requires an explicitly labeled adaptation rather than a native score.

[BATCHIE, Tosh et al., Nature Communications 2025](https://www.nature.com/articles/s41467-024-55287-7) couples Bayesian combination-response modeling with batch experimental design. It reports retrospective comparisons and a prospective 206-drug, 16-cell-line screen, followed by validation of ten prioritized combinations. Its model includes drug-dose and cell-line representations and interaction terms. Consequently a low-rank multi-context response model or “active learning saves experiments” is not sufficient novelty. Opal would need to demonstrate a distinct transcriptomic capability with an explicit target-context information budget and independent replication.

[IterPert, sequential optimal experimental design of perturbation screens](https://www.biorxiv.org/content/10.1101/2023.12.12.571389v1.full) studies multimodal-prior-guided selection of perturbations and explicitly leaves combinatorial extension to future work. This is relevant competitive context, not evidence that all combinatorial transcriptomic design remains unsolved. These primary sources were checked as accessible on 11 October 2026; this report makes no exhaustive priority claim.

## Compare scientific objectives

| Objective | Why a scientist cares | Evidence needed before Opal can claim it | First-cycle decision |
|---|---|---|---|
| Difficult unseen combinations | Reduces full combinatorial enumeration | Same-information superiority with globally excluded pairs, including hard partner holdout | Necessary prediction substrate, not sufficient utility |
| Reproducible pair-specific effects | Identifies response beyond shared distortion | Independent-culture repeatability and performance after shared/main-effect calibration | Central discriminator of proposed bridge experiment |
| Response distributions | Reveals mixed or rare responding subpopulations | Distribution metrics beyond mean, independent sampling, functional value of tails | Defer; variance-feature failure does not falsify distribution modeling |
| Transfer between contexts | Reuses existing screens in a new model system | Joint pair/context exclusion with declared measured-reference and bridge allowances | Best dominant capability hypothesis |
| Rank candidate interventions | Converts predictions into actionable experiments | Prespecified biological program, hit precision and regret against equal-budget policies | Recommended eventual scientific use case |
| Select informative experiments | Learns more per assay | Equal-cost adaptive policy versus random, target-balanced and diversity controls | Conditional second stage |
| Optimize experiment design | Saves complete experimental cost | Prospective randomized campaign, audited assay/repeat/control/failed-run cost | Long-term claim only |

## Smallest decisive study

Freeze a pair panel that is absent from **all** source and bridge fitting labels, across doses and contexts. Keep target-context controls and measured singles available to every matched-information comparator. At target-context budgets of zero, one prespecified sparse bridge budget, and a larger diagnostic budget, reveal only identity-selected bridge pairs; do not add active selection in this first study. Training and inner validation must obey identical exclusions.

Compare a source-trained response basis with a small target-context coefficient adapter against: no-change; observed additivity; source-only Opal Base/kernel; target-context-only affine-additivity/shared residual; a target-context-only pair kernel fit to exactly the same bridge labels; pooled-context same-budget kernel; and a source-trained adapter with source pair-response correspondence permuted. Include a source-free adapter with the same parameter count and objective. The strongest baseline is selected within source/bridge development, not picked from outer test outcomes.

This distinguishes three mechanisms: bridge labels merely calibrate a shared response; source data supply useful interaction structure; or target-context singles already contain the predictive information. A claim about transferable interaction structure requires an advantage over both target-only fitting and shared calibration, and that advantage must disappear when source pair-specific correspondence is destroyed. Permuting only intervention names consistently is an invariant renaming control and cannot test identity information.

For the first prototype, retain population-mean loss as the primary prediction endpoint and add a preregistered program-ranking endpoint as a utility diagnostic. Define program weights from a collaborator's stated biology or source-only data, never from held pair responses. Report precision at a fixed assay budget, top-k enrichment over additive/program baselines, and simple regret using a fixed candidate pool. Resample whole independent cultures and contexts when they exist; technical sample labels and repeated folds supply only conditional development sensitivity. No unsupported fixed cell/culture count should be presented as a powered design.

**Go:** the bridge model beats the strongest same-budget reference under global pair/context exclusion; source-specific controls establish that the gain is not shared calibration; held-culture program ranking improves enough to alter a predeclared selection decision. **No-go:** if the source-free or shared-calibration comparator matches it, retain the simple reference and stop the claimed interaction-transfer architecture. If all labels are too noisy to yield stable program decisions, improve independent measurements before expanding models.

## Retrospective to prospective confirmation

1. **Retrospective development:** reproduce one sparse-budget bridge study with the complete fixed candidate population and label ledger. A curve evaluated only on measured historical candidates supports interpolation inside that candidate pool; it does not prove savings over a genome-wide screen.
2. **Blinded fixed-policy validation:** in new cultures, freeze model, program, top-k list, matched baseline lists and randomized controls before outcome access. Use identical controls, singles, guide/dose/endpoint matching and independent repeats. Confirm prespecified functional assay alongside transcriptomic program movement if the claim is biological action.
3. **Randomized adaptive campaign:** only after successful fixed-policy validation, compare an adaptive bridge/selection policy against uniform, target-balanced and strong model-based acquisition at equal total cost. Assign complete independent campaigns or contexts to policies where feasible, because testing two policies on one shared label stream is a simulation rather than randomized cost evidence.
4. **Cost/discovery claim:** count controls, singles, bridge pairs, confirmation pairs, failed cultures, sequencing, and round delays. Predeclare whether the objective is confirmed hits per currency unit, assays to a validated hit count, or ranking regret. Stop if calibrated uncertainty does not improve policy outcomes; uncertainty coverage and useful acquisition are separate gates.

The first wet-lab milestone is not “a low MSE in a new plate”; it is more confirmed program-changing pairs at a fixed budget than a strong baseline's prioritized list. A prospective hit enrichment result can establish utility even without proving causal gene interaction, provided the claim remains about reproducible outcomes.

## Red-team conclusion

Sparse bridges may teach only context-specific assay calibration; global pair holdout makes transfer much harder because the model cannot memorize a source response for the exact query pair. Existing crossed public datasets may also confound context with capture or post-treatment lineage. Even a real mean-response advantage may leave ranking unchanged. Therefore this direction is promising but unresolved, not the proven highest-yield biological strategy.

The strongest alternative is a replicated, target-context response map with economical calibrated references and a prespecified program-ranking study. If bridge information helps only a simple reference, deploy that reference for utility and drop neural novelty claims. If source structure contributes after calibration, then build a representation model and later an acquisition policy. Neither outcome justifies an indiscriminate foundation-model scale-up.

## Traceable references

- Qin, J., Wessels, H.-H., Fernandez-Granda, C., Hao, Y. (2025). Active Learning for Efficient Discovery of Optimal Combinatorial Perturbations. ICML/PMLR 267. https://proceedings.mlr.press/v267/qin25g.html
- NeptuneBio. NAIAD official code. https://github.com/NeptuneBio/NAIAD
- Tosh, C., Tec, M., White, J. B., et al. (2025). A Bayesian active learning platform for scalable combination drug screens. Nature Communications 16, 156. https://doi.org/10.1038/s41467-024-55287-7
- Sequential Optimal Experimental Design of Perturbation Screens Guided by Multi-modal Priors (2023 preprint). https://www.biorxiv.org/content/10.1101/2023.12.12.571389v1.full
- Local manuscript `paper/main.pdf`, especially Sections 4.7, C/D, E.3, F/G, I; local extraction `tmp/frontier_manuscript.txt`.
- Local evidence `docs/next_cycle/opal_information_report.md`; `results/opal_information_v1/report/claims.json`; `results/state/emeraldbay-diagnostics-v1/acquisition.json`.

## Actual cross-examination and revision

I read C's initial architecture report and D's initial evaluation report and sent each author independent criticism through the agent channel. C's rank-two source-trained subspace with two target coefficients plus an intercept is a more identifiable bridge prototype than a free target operator. I revise the minimum model accordingly: compare this source-conditioned adapter to an equally parameterized source-free calibration model, with no target output PCA. C's initial preference for zero-bridge single-contrast transfer remains a meaningful alternative, but the existing negative state/mean-feature evidence lowers its priority relative to new pair observations.

D's independently anchored evaluator residual and independent-control scoring resolve an important ambiguity: subtracting the same additive vector from prediction and truth only renames response MSE. I adopt that metric as the interaction-attribution diagnostic, while keeping response MSE primary. Its retrieval metric tests discrimination, not scientist utility; I asked D to retain the separate frozen program-ranking endpoint. Both reports need a disclosed subgroup whose held pairs share no bridge components, because gains on pairs with bridge components could be incidence calibration. Feasibility may preclude that subgroup; in that case explicitly narrow the generalization claim rather than relabeling the available split.

The 5% response-MSE and 10% contrast-MSE development cutoffs proposed in D/C are different administrative screens, not competing scientific definitions of frontier performance. The integrated contract should choose one prespecified primary development threshold and separately qualify a useful program-ranking margin with a collaborator. Do not average the cutoffs or retrofit one after outcomes.
