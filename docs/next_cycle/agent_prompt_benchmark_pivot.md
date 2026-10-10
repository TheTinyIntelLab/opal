# Agent handoff: reference-first prediction and pair-label budget benchmark

Copy the prompt below into the implementation agent that has access to this repository.

---

Work in the Opal repository and implement this pivot alongside every remaining computationally executable requirement in `PLAN.md`. The owner is changing direction: stop searching indefinitely for a dataset that can satisfy the original six-culture confirmation design. Make the active deliverable a reproducible, reference-first benchmark of observed-single pair prediction and the value of additional measured pair labels. Finish the actual runs, verification, reporting and documentation; do not stop after writing a plan or scaffold. Commit and push coherent changes to the existing remote. Ask the owner only when an indispensable fact or external authorization cannot be obtained from the repository or primary sources.

## The decision you are implementing

The new question is: **Within a declared deposited assay and source stratum, how accurately can measured singles and a limited development set of pair labels predict globally held pair profiles, and when does pair structure improve over a strong simple reference?**

This changes the estimand and deliverable. It does not pass, waive or rewrite the original prospective biological gates. The benchmark is useful if its answer is negative: it should identify whether additional pair labels improve a simple correction, whether the pair kernel adds anything, and where prediction lacks reference or graph support. Report retrospective prediction and empirical label-budget curves. Do not claim biological interaction discovery, a noise ceiling, prospective assay savings, validated uncertainty or independent culture confirmation.

The primary route must run on already available data. An external source is an extension, never a prerequisite for finishing the benchmark.

## Read and preserve the current state

Read `AGENTS.md` if present, all of `PLAN.md`, `docs/next_cycle/ceiling_execution.md`, `docs/next_cycle/execution_status.md`, `docs/next_cycle/alternative_study_designs.md`, `results/replicated_fixed_state_v1/task_status.json`, and the source-specific diagnostic documents. Inspect code and saved results before reimplementing anything.

The known starting implementation is `2798ab00cd74497d87e9919b3259e6a9040366a4`; verify current HEAD and preserve subsequent or uncommitted work. That commit has 234 passing local tests and successful four-environment Windows/Linux Python 3.11/3.14 CI. The repository is private. GitHub `v0.1.2` is already published with its exact approved nine assets from `53d6e83`; do not replace those assets, change visibility, publish to Zenodo/arXiv, register an experiment or contact anyone without separate authorization.

The original Combi-seq benchmark is complete and unfavorable to the candidate: 374 nominal pairs, three reported runs, a singles-derived 512-gene endpoint, selected reference MSE 3.872700 versus candidate 4.388534. Preserve its folds, seeds, predictions and interpretation. The donor, bulk and unmatched-array diagnostics are also complete. Do not treat new seeds, reanalysis or a new scope as fresh confirmation.

`PLAN.md` is intentionally ignored and private. Update it locally, but never force-stage it. Raw source metadata containing contacts, matrices and per-cell/per-profile prediction arrays remain in ignored local storage. Track compact aggregate results, scopes, manifests, source hashes and exclusions.

## 1. Amend the plan before new computation

Add a dated active amendment at the front of `PLAN.md` and a tracked protocol such as `docs/next_cycle/benchmark_pivot_protocol.md`. State that this owner-requested pivot supersedes the prior search-first work ordering and authorizes the finite existing-menu benchmark below. Preserve historical records and the original conditional study.

Audit every original backlog item and record one of: complete with verified evidence, an executable repair to perform now, or deferred on a named external dependency. Complete genuine remaining software, packaging, claim-provenance and reproducibility repairs. Independent cultures, empirical biological power, actual registration, collaboration and unapproved publication remain deferred; do not mark them complete through simulation or edited documentation. Keep conditional uncertainty, active-learning and architecture successors untriggered unless their distinct evidence requirements are met.

Define separate benchmark acceptance criteria: input provenance, valid reference construction, canonical pair isolation, training-only fitted transforms, complete source-stratified predictions, exact replay and an honest report. Do not reuse G1–G5 as if this deliverable satisfied them.

## 2. Build the reusable benchmark around existing implementations

Reuse `src/replicated/retrospective.py`, `src/replicated/models.py`, the preparation/access contracts, freeze/evaluator machinery and the existing sixteen-entry menu in `configs/experiments/replicated_fixed_state_v1.json`. Generalize adapters only where necessary; do not refactor frozen historical runs or overwrite their result directories.

Use Combi-seq's saved canonical bank and existing outer pair assignments for the first complete execution. Reproduce the historical full-budget result before extending it. Keep genetic and chemical identities separate. Preserve each assay's declared expression scale and reference arithmetic; never invert normalized arrays into counts or pool absolute losses across incompatible assays.

Every occurrence of a held canonical pair must be excluded from fitting, selection, response-derived transforms and representation learning, including aliases, reversed target order, guide variants and other source strata governed by the same holdout contract. Query controls and measured singles are allowed and must have explicit provenance. A fitting API must not accept query-pair outcomes. Inner selection and final refitting must consume only the currently acquired development labels.

The primary endpoint is MSE on the historical 512-gene singles-derived panel for Combi-seq, equally weighted over canonical pairs within a run, then equally over reported runs. Preserve the all-source-gene results separately where they already exist. Additional sources get their own frozen assay contract and reference-only endpoint. Report pair-level errors, per-stratum results, losses against additivity and the strongest selected reference, and the existing alignment-shuffle controls. Correlations are secondary diagnostics with explicit centering and reference reuse.

## 3. Execute a finite pair-label budget experiment

For each fixed outer split, hold its evaluation pair population unchanged. Acquire nested development subsets at 0%, 25%, 50% and 100% of the available canonical development pairs. A budget counts every unique pair label accessed for either fitting or inner validation; validation labels are not free. Include actual counts, omitted conditions and the full-development panel size in the report.

Use two label-acquisition orders: uniform random and a deterministic target-balanced order based only on intervention identities and the current acquired graph. Specify the exact target-balancing algorithm, tie breaks and three sampling seeds before scoring. Reuse nested prefixes so the budget curves measure added labels rather than unrelated populations. Acquisition must never inspect candidate unacquired outcomes, residuals or test errors. This is a retrospective comparison of fixed label-coverage rules, not a validated active-learning policy.

At zero pair labels, evaluate no-change and observed-single additivity; mark pair-supervised methods unavailable. At positive budgets, rerun the existing bounded inner selection using only acquired development pairs, then refit on that acquired set. If an inner partition lacks sufficient pairs or component support, record a prespecified unavailable result rather than leaking more labels or choosing a favorable seed. Freeze those rules before the run. Keep all declared budgets and sampling seeds, including unfavorable and unavailable cases.

Present separate curves for additivity, selected strong reference and pair candidate, plus uniform versus target-balanced acquisition. Report a descriptive marginal error change per added pair label, without translating it into wet-lab money or prospective assay savings. A method recommendation must follow the strongest-reference comparison and expose unsupported components; do not call a graph-support indicator a probability, confidence interval or calibrated risk estimate.

## 4. Attempt one bounded external extension: PertSpectra in-house

This source has already been located; do not restart broad dataset discovery:

- Author location: https://github.com/insitro/PertSpectra/tree/main/data/inhouse
- Public object: https://pert-spectra.s3.amazonaws.com/data/inhouse.h5ad
- Primary paper: https://psb.stanford.edu/psb-online/proceedings/psb26/chang_s.pdf
- Existing metadata evidence: `results/replicated_fixed_state_v1/pertspectra-obs-metadata-20261010-v3/` and `docs/next_cycle/remote_pair_metadata_20261010.md`.

The prior audit decoded selected observation metadata, not expression. Cached range blocks may contain adjacent uninterpreted expression bytes, and published result narratives have been encountered; disclose both. The deposit has 73,015 rows, eight SampleIndex labels split four per TNF context, and broad literal pair/single support. These labels are not verified independent cultures, and the paper's 65,648 filtered-cell count has not been reconciled.

First inspect author preprocessing and molecular assignments. Define control, single, pair, guide and TNF-context identities explicitly. Resolve whether a gene-level average across guide mixtures is the intended label-conditional prediction target; do not silently equate such an average with an exact matched-guide causal effect. Exclude self-pairs, ambiguous assignments and unsupported references by fixed rules, and report exclusions. Do not use expression-derived clusters, archetypes, embeddings or pretrained weights as predictor inputs.

Before decoding expression, save a source-specific access/protocol record containing the object identity, code/menu hashes, exposure declaration, allowed inputs, scale, QC rules, gene endpoint, seeds, resource limits and exclusions. Download with size/identity checks and compute a full-file SHA-256; the multipart ETag is not a file hash. Use backed/chunked processing and sparse operations rather than densifying the full cell-by-gene matrix. Check local disk and memory before transfer. Verify whether raw counts exist; otherwise retain and label the deposited transformed scale, with its upstream processing dependence.

Build aggregates separately by SampleIndex and TNF context. Preserve cell counts and guide mixtures as support information, with no claim that cells or indices are independent biological replicates. If controls or references cannot support a coherent label-conditional prediction contract, save a specific refusal and complete the existing-data benchmark anyway. Do not substitute missing measurements, invent guide equivalence or resume an indefinite source search.

If eligible, run the existing-menu within-stratum globally held-pair benchmark and budget study. An additional leave-SampleIndex-out diagnostic within each TNF context may train on the other three indices, exclude all occurrences of each test pair from training, and use the held index's controls/singles as declared query inputs. Fit transforms and select configurations inside training only. Average training strata equally rather than allowing extra cells to change their weights. Name this transport across source sample labels; it does not prove new-culture generalization or TNF-state transfer. Keep TNF contexts separate and report their own pair populations. Do not fit pooled models across unrelated historical assays.

## 5. Verify failure modes and finish the artifacts

Add meaningful tests for held-pair outcome mutation leaving fitted transforms, selection and predictions unchanged; reversed/aliased identity grouping; missing reference refusal; unacquired labels not entering budgeted fits or inner validation; unchanged outer populations across budgets; zero-budget behavior; deterministic acquisition prefixes; source-stratum weighting; invalid assay scales; and evaluator rejection of changed genes, units, inputs or prediction hashes. Verify local replay and installed-wheel execution outside the checkout, and add the new integrity tests to the existing CI matrix.

Create source-specific result directories containing the pre-access/pre-fit scope, exposure record, input and split manifests, all selection records, compact per-pair/per-stratum metrics, aggregate summaries, exclusions, runtime/resource records and verification. Keep large arrays local. Produce learning-curve and reference-versus-candidate figures from saved records, plus a concise explanation of what the results establish and what remains unknown. Never select the primary gene panel from held pair outcomes or omit losing configurations.

Update README, task status, the active plan amendment and tracked execution documents. Update the manuscript only where this new work justifies a clearly separated benchmark addition, preserving original numerical provenance and negative findings. If manuscript edits require PDF regeneration, use the existing build and visual-verification workflow. Build reviewable new release candidates if required by remaining packaging work, but do not republish or overwrite the approved v0.1.2 assets.

Run appropriate focused tests, then the full suite; perform actual source runs and installed verification. Commit and push coherent milestones, inspect CI, and fix failures. Stop only once the feasible benchmark deliverable and remaining executable repairs are complete. The final response must identify implemented artifacts, actual sources analyzed, decisive favorable or unfavorable results, tests and CI, pushed commits, and the exact externally deferred requirements. Successful completion means a usable, honest benchmark, not a forced positive predictor result.

## Primary sources and claim boundaries

The PertSpectra paper demonstrates the feasibility of an unseen-combination prediction task on this public panel; its preprocessing and reported performance must be audited rather than copied into this contract. The primary comparison at https://www.nature.com/articles/s41592-025-02772-6 supports retaining strong simple baselines and distinguishing expression prediction from interaction detection. For nested model-selection bias see https://www.jmlr.org/papers/v11/cawley10a.html; for biological pseudoreplication see https://pmc.ncbi.nlm.nih.gov/articles/PMC8479118/. These sources motivate the design; none certifies that Opal will win or converts its sample labels into independent cultures.
