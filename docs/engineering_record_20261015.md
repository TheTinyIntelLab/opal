# Closed engineering record

Snapshot of the pre-reset README. Statements of active work below are historical, not instructions. The sole current forward plan is [root plan.md](../plan.md). Original implementations, scientific results, manifests and model guides remain valid within their stated scopes.

<div align="center">
  <img src="images/logo.png" width="160" alt="Opal logo" />
  <h1>Opal</h1>
  <p>Observed-single combinatorial response prediction</p>
</div>

Combi-seq authors report three biological replicates and no randomization; initiation/reference lineage remains unverified. Primary-paper follow-up resolves experiment-level GSE95153 doses, but culture/reference lineage remains unverified and all 14 confirmation gates remain failed. Exact outreach correspondence is retained only in ignored local custody; [the sanitized status](../docs/frontier/outreach_status.md) records sends and the latest mailbox check.


<div align="center">
  <a href="https://thetinyintelligencelab.vercel.app/"><img src="images/lab_badge.svg" alt="The Tiny Intelligence Lab" /></a>
  <img src="images/status_badge.svg" alt="Private GitHub release" />
  <a href="../paper/main.pdf"><img src="images/paper_badge.svg" alt="Research preprint draft" /></a>
  <a href="../LICENSE"><img src="images/license_badge.svg" alt="Code license: MIT" /></a>
</div>

<p align="center">
  <a href="#2-model-downloads">Models</a> ·
  <a href="#3-evaluation-results">Evaluation</a> ·
  <a href="#5-quick-start">Quick start</a> ·
  <a href="#8-license">License</a> ·
  <a href="#9-citation">Citation</a>
</p>

<p align="center"><a href="../paper/main.pdf">Read the preprint draft</a></p>

Opal predicts population-mean responses to intervention pairs from matched measured singles and controls. Current code is a **0.1.3.dev0 local research snapshot**; the private 0.1.2 release is historical. No independently confirmed useful predictor or frontier claim is established.

The [closed workstream](../docs/frontier/workstream_closeout.md) records completed engineering and computational reproduction. Crossed-single nested selections, 90 final predictions and conditional statistics replay, but the registered spending filter failed. Context-contrast SC01-SC08 is implemented and fixture-tested; real fitting remains paused because no genetic pair has certified exact assignments. The original 207/256-gene refusal and all failures are preserved.

**GxE reconstruction is now complete.** The optimized CPU reader recovered aggregates from all 192,432,512 source coordinates, reproduced every one of 66 historical metric rows, and reproduced the reference-only 512-gene panel. Recovery plus panel verification used 100.3125 CPU seconds within the same 300-second envelope. The earlier failed attempt remains preserved. These are technical measurement/replay results, not independent cultures or biological confirmation.

[Reproduction instructions](../docs/frontier/reproducibility_closeout.md) distinguish fixture checks from real replay. Numerical replay requires retained local banks and manifests; raw recovery requires pinned source inputs. Raw data, numerical banks and weights are excluded from distributions/Git. Source-specific export rights remain unresolved; no public data/model release is implied. Package dependencies and optional data/research extras are in `pyproject.toml`.

[Root plan.md](../plan.md) is the repository-only authoritative forward plan; private plans/history are excluded from source distributions. The [prospective protocol](../docs/frontier/prospective_confirmation_protocol.md), [14-gate register](../docs/frontier/confirmation_readiness_20261015.md) and [external action packet](../docs/frontier/external_action_packet.md) specify the remaining real-world work. **0/14 readiness gates pass.** No model is nominated, collaboration agreed, useful margin/power selected or physical assay performed. The preceding closeout checks are in `docs/frontier/workstream_closeout_validation.json`; current external-phase checks are in `docs/frontier/external_validation_validation.json`.

The [external-validation phase](../docs/frontier/external_validation_sources.md) has now inspected three new official metadata sources, executed and independently replayed a small GSE95153 within-condition repeat-variation diagnostic, and [sent four tailored custodian inquiries](../docs/frontier/outreach_status.md). The diagnostic is descriptive, with no predictor fit or verified biological independence. No new source qualifies for context-contrast or predictive confirmation; all 14 gates remain failed. Five sends are verified, one address bounced and was rerouted, and the remaining four have no delivery proof or substantive reply/agreement.

## 1. Introduction

The [measured-single information study](../docs/next_cycle/opal_information_report.md) completed a frozen variance/mean/capacity/shuffle experiment on exposed PertSpectra development data. TNF-positive partner variance gains only 0.231% over Base and loses to matched mean controls; TNF-negative partner heads lack nested support and use Base fallback. The information gate failed, so Base remains the reference and no confirmation or downstream branch advances. New training/replay profiles are available for Opal and three disclosed comparator adaptations; canonical native competitiveness and independent biological confirmation remain unestablished.

Opal predicts the population-mean effect of an unmeasured intervention pair from measurements of each intervention on its own. The primary task requires both singles in the same biological setting and predicts their sum plus a symmetric correction:

```text
predicted pair effect = measured effect A + measured effect B + correction(A, B)
correction(A, B) = correction(B, A)
```

<img src="images/opal_overview.svg" alt="Measured singles feed an additive reference and a symmetric residual correction, producing a population-mean pair prediction." width="100%" />

The project implements neural heads, classical regression, residual kernels, and explicit reference controls. Kernels led the earlier exploratory comparisons; the later finite series selects stronger null, mean and shrinkage references on its primary panels. Historical neural gains over additivity don't establish robust pair-specific biological interaction learning, and independent context-transfer superiority remains unestablished.

This is independent research by **Arjun Vijay Prakash**, a high school student at City Montessori School, with no institutional research affiliation, funding, or competing interests. [The Tiny Intelligence Lab](https://thetinyintelligencelab.vercel.app/) is the project's independent research identity. AI assistance and author responsibility are disclosed in the manuscript.

## 2. Model downloads

The unpublished local `0.1.3.dev0` review candidate implements three trainable models. **Opal-Base** selects one of 24 main-effect/null/shrinkage references. **Opal-Interaction** adds explicit symmetric cross-coordinate bilinear features, with training-only input PCA, acquired-residual output SVD and weighted ridge. **Opal-Guard** learns a convex mixture of Base, Interaction and acquired-pair residual retrieval through fully nested honest OOF training. Unsupported experts retain their declared fallback.

| Current local evaluation | Combi-seq MSE | TNF-positive MSE | TNF-negative MSE |
| --- | ---: | ---: | ---: |
| No change | 5.713889 | 0.137007 | 0.075095 |
| Opal-Base | 2.111564 | 0.077789 | 0.071218 |
| Opal-Interaction | 2.109919 | 0.077723 | 0.071218 |
| Opal-Guard | 2.112194 | 0.078360 | 0.071218 |
| Development-selected constant mixture | 2.107765 | 0.077688 | Unavailable on four folds |
| Matched main-effect ablation | 2.105071 | 0.077789 | 0.071596 |

Each column has different genes, units and source conditions. Scores use the unchanged v1 populations, primary MSE and outer assignments, full acquisition budget, uniform order, acquisition seed 11 and the prespecified training seed. Interaction gains only 0.078% and 0.084% over Base on the first two panels; Guard loses to constant mixing. The 2% nonlinear-default promotion criterion fails. Base is the current simple fitted recommendation; no-change is available at zero labels. TNF-negative's nonlinear experts fall back to Base, so equal predictions are an honest outcome.

Pinned upstream PertSpectra, GEARS and CPA implementations completed 24 full-budget folds each under separately declared protocol adaptations. Their TNF-positive/TNF-negative MSEs are 0.182326/0.126928, 0.093978/0.079552 and 0.113226/0.071731 respectively. These use different additional inputs and optimization budgets and do not reproduce the authors' full configurations. A linear control with the same extra nineteen observed singles scores 0.082463/0.070020 and beats Base on TNF-negative. TxPert's public-only dependency closure and PRESAGE's native pair-output contract remain ineligible. This evidence establishes no frontier claim.

Actual evaluation checkpoints cover Combi-seq, both separate PertSpectra contexts, and the coherent Norman/Wessels development banks on their preserved log/count scales. Norman Guard improves retrospectively by 3.17% on log and 4.99% on count; these are two scales of one exposed source. Wessels uses Base fallback. All weights remain in ignored local storage pending source-term/publication authorization, and the published 0.1.2 assets are unchanged. Evaluation checkpoints and later deployment refits have separate roles. [The current local guide](../docs/next_cycle/opal_model_series_v2.md) describes provenance, availability and batch inference; [the execution plan](../plan.md) records work still in progress.

### Preserved first model-series evaluation

The preserved first trainable series has three source-specific variants with a shared `fit`, `predict`, `save` and `load` API, a registry, and `opal series-train`, `series-predict` and `series-evaluate` commands. Actual Combi-seq checkpoints are indexed in [the checkpoint manifest](../results/opal_model_series_v1/combi-seq/all_budget_checkpoint_manifest.json); weights remain local and aren't part of the published release.

| Trainable model | Combi-seq full-budget MSE | Recommendation |
| --- | ---: | --- |
| Opal-Base | 3.872700 | Recommended fitted reference |
| Opal-Interaction | 3.871518 | Experimental nonlinear residual |
| Opal-Guard | 3.871571 | Experimental learned OOF mixture |

The nonlinear variants reduce retrospective MSE by about 0.03% relative to Base, which is too small to warrant replacing the reference. The preserved historical pair candidate is worse at 4.388534. These are source-conditional, observed-single predictions on the unchanged 512-gene endpoint across 374 nominal pairs and three reported runs, with no independent-culture or biological interaction claim. [Model cards, actual batch examples, label-budget curves and source limits](../docs/next_cycle/opal_model_series.md) explain the contract and evidence. The series has separate [scope and result records](../results/opal_model_series_v1/combi-seq/); the original negative benchmark and approved v0.1.2 assets remain intact.

The validated GitHub release includes these two kernel references. Its downloads require repository access; the older draft is a separate preparation snapshot.

| Reference | Inputs and output | Validation scope | Availability |
| --- | --- | --- | --- |
| Norman residual kernel | Registered K562 single-effect profiles; fixed-gene population-mean pair effect | Exploratory development-selected refit, distinct from the historical primary neural result | Prepared locally |
| EmeraldBay contextual kernel | Registered exact-dose singles and controls in supported contexts | Chemical pilot; its transferred procedure failed the independent D-SPIN superiority gate | Prepared locally |

The [model guide](../docs/models.md) explains supported queries, reference access, gene order, and diagnostic metadata. Raw datasets, Wessels/D-SPIN weights, external embedding banks, and native CPA/GEARS weights aren't redistributed. Dataset-derived reference content has CC BY 4.0 terms in [the notices](../THIRD_PARTY_NOTICES.md).

There are **21 final all-development deployment checkpoints**, three per source, in the local C artifact root's `deployment-final/` directory. [The deployment manifest](../results/opal_model_series_v2/deployment-final/manifest.json) names their exact paths, source hashes, parameter/storage counts and replay results. Evaluation weights remain at their original manifest paths. These later refits have no evaluation score; deployment batch inference is supported and retrospective scoring is refused. Weights remain local under the existing source terms. The complete grid includes 858 primary and 312 retrospective conditions, 51 sensitivities and 72 upstream adaptation folds. [Installed usage and evidence](../docs/next_cycle/opal_model_series_v2.md) distinguish the review package from the published release.

## 3. Evaluation results

### Registered successor development

The new batch retains tuned linear/nonlinear shared corrections and single kernels, tests fixed context/identity/dose features and pair kernels, cross-fits technical references, and audits 16 configurations each for pure-sum shared and symmetric pair neural heads. Two seeds form the ensemble in both inner selection and outer scoring; the original schedules remain controls. All reused sources are exposed development.

| Observed-single random-pair development | Selected simple MSE | Shared neural MSE | Pair neural MSE |
| --- | ---: | ---: | ---: |
| Norman, 88 pairs | 0.006563816 | 0.007403584 | 0.007407056 |
| Wessels, 40 pairs | 0.005469709 | 0.006332205 | 0.006761126 |

EmeraldBay identity features improve on the selected shared correction by **4.06%**, below the registered 5% screen; pretreatment control features worsen it by 0.53%. Pair-product and mixed kernels do not pass on random, partner, or context folds. Separately, learned baseline-only main models are 10.40% worse than simple references on EmeraldBay and 15.30% worse on sci-Plex. No state encoder, larger pair model, calibrated risk policy, or scaling stage earned advancement.

<img src="images/stage1_results.svg" alt="Registered development gains compared with the five percent screen across genetic pair and chemical context protocols." width="100%" />

The studies consumed **631.67 aggregate study seconds**, with peak process working set **1.06 GB**, on reused CPU float64 dependencies; the available RTX 4060 was not used for fitting. Exact comparisons, paired diagnostic intervals, exclusions, component accounting, training trajectories and costs are in [comparisons.csv](../results/stage1-20261007/report-v2/comparisons.csv) and [the decision](../results/stage1-20261007/report-v2/decision.json). Technical halves are not independent cultures, dose features are not chemical descriptors, and baseline-only predictions are a separate information contract. At that study cutoff, the next justified biological work was qualifying crossed-state data with independent cultures and sealed globally held pairs, using [the screen specification](../configs/datasets/crossed-context-screen-spec.json).

### Registered dose development

The subsequent dose study globally holds out each of four doses across 24 known compounds and three sci-Plex cell lines. Source-selected smooth curves improve MSE by 8.18% at 100 nM and 45.69% at 1,000 nM, while worsening it by 35.01% at 10 nM and 12.66% at 10,000 nM. Both interpolation folds pass the development screen; both extrapolation folds fail. [Exact comparisons and diagnostic intervals](../results/dose-design-20261007/dose-v2/comparisons.csv) condition on shared controls and states. This is estimated main-effect dose transfer, separate from pair prediction and unseen chemical identity.

<img src="images/dose_results.svg" alt="Smooth dose curves pass both interpolation development gates and fail both extrapolation gates." width="100%" />

The existing sci-Plex adapter now accepts an exact dose grid. Preparation takes 60.19 CPU seconds; fitting takes 1.78 seconds. The [experimental dose model](../docs/models.md) restores both scored interpolation folds exactly and rejects unsupported states, compounds, endpoints and extrapolation. A separate [320-design precision sensitivity](../results/dose-design-20261007/precision-v1/precision-design.json) preserves state, culture, target, pair and shared-control dependence under hypothetical variance assumptions. It does not supply an empirically powered sample size. The [screen specification](../configs/datasets/crossed-context-screen-spec.json) now matches the plan's four-state, three-culture variance pilot and separate sealed confirmation requirements.

### Registered identity and provenance continuation

Publisher target/pathway annotations were qualified before a separate identity experiment. Globally held compounds gain only 2.11% over the selected simple/class-mean reference, with a diagnostic interval crossing zero; holding transitive shared-target groups worsens MSE by 3.21%. Both gates fail, so no annotation encoder or unseen-compound production route advances. Each track scores the same 288 profiles with source-only vocabularies and matched shuffled searches, using 2.12 aggregate CPU seconds. [Exact results and the decision](../results/identity-reference-20261007/report-v1/decision.json) retain both unfavorable tracks.

<img src="images/identity_results.svg" alt="Publisher annotations fail the identity and shared-target holdout development gates." width="100%" />

The primary-metadata audit matches all 799,317 cells without condition or annotation discrepancies. Every assigned 24-hour cell's culture label matches its existing plate stratum; original labels don't add independent biological units. GSE146194 metadata describes one K562 line and sequencing/platform variants, so it doesn't qualify sealed biological-state transfer. The earlier bounded inventory excluded Adamson. The newly qualified local Adamson source contains one genuine triple identity across three drug backgrounds, which remains inventory-only. [The remaining requirements](../results/identity-reference-20261007/report-v1/decision.json) still include independent culture/pair provenance, multiple sealed states, empirical covariance for power, and independently evaluated risk.

### Registered joint pair/state development

A metadata-balanced partition globally holds canonical chemical pairs across all doses and whole cell lines together. Twelve frozen outer folds query all 54 exposed EmeraldBay fitting conditions once, with inner joint holdouts and unused guard rows. Exact measured query singles and legacy query-control gene-panel adaptation are declared. The selected simple reference has MSE **0.175075515**; pair-product and mixed kernels have **0.176599048** and **0.175353284**, worsening error by 0.87% and 0.16%. Both diagnostic intervals span zero and both gates fail. This resolves the earlier sparse partition's development coverage gap, without supplying sealed confirmation.

<img src="images/joint_results.svg" alt="Pair-specific kernels fail the globally held chemical-pair and whole-state development gate." width="100%" />

The fixed search takes **1.98 elapsed seconds**, with completion RSS **276.31 MB**, not a measured peak. [Comparisons and crossed state/intervention diagnostic intervals](../results/joint-development-20261007/report-v1/comparisons.csv), per-state/pair errors, frozen splits and full selections are retained. [The decision](../results/joint-development-20261007/report-v1/decision.json) keeps shared corrections and the narrow dose route; independent cultures, multiple sealed states and globally held pair panels remain required.

### Registered scale and reporting sensitivity

Twelve additional genetic count-scale folds reuse the exact log2 development identities and source partitions. Norman mixed correction gains **9.19%** on random folds, with paired MSE-reduction interval **[-0.011700, 0.273082]**; pair-product correction gains **9.18%** on partner folds, with interval **[-0.113894, 0.496702]**. Direction changes from the log2 findings, but every count primary gate fails uncertainty. Wessels count results remain unfavorable; raw MSEs across the two scales aren't comparable.

<img src="images/panel_scale_results.svg" alt="Norman count-scale point gains fail uncertainty gates; Wessels count comparisons remain unfavorable." width="100%" />

The local Adamson UPR transfer pilot freezes Norman-selected predictions before scoring nine matched pair/background conditions over 10,423 genes. Mixed correction gains 9.37% over the source-selected correction, but its diagnostic MSE-reduction interval [-0.00015503, 0.00067825] includes zero. Additivity is stronger: MSE 0.00184324 versus 0.00292072 for the selected correction. No transfer component advances; chemical backgrounds alias capture libraries and independent cultures are unverified. [Saved comparisons and decision](../results/adamson-transfer-20261007/report-v1/decision.json) preserve all competitors.


Source-selected strong and low-single residual-enriched panels retain all comparisons, including favorable Norman count partner subset gains of 12.49% and 9.23%. These diagnostic masks don't use query labels or validate emergent biology, and can't replace the all-gene endpoint. Partner reporting now separates one/two held targets and exposes repeated-query coverage: Norman has 148 occurrences from 88 pairs; Wessels has 69 from 40. [Every comparison, interval and subset](../results/panel-scale-20261007/run-v1/comparisons.csv) and [the decision](../results/panel-scale-20261007/report-v1/decision.json) are preserved. The fitting and diagnostic runner takes **37.89 elapsed seconds**, with **243.76 MB completion RSS**, not a measured peak.

### Historical primary biological results

These are equal-pair MSEs on the original frozen held-out sets. The selected neural and simpler reference were chosen on development data; a test-best candidate isn't retrospectively promoted.

| Dataset | Held pairs | Additive | Selected neural | Selected simple reference | Neural gain over additivity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Norman | 38 | 0.008515040 | 0.007169187 | 0.007268184, reduced-rank | 15.81% |
| Wessels | 18 | 0.013936912 | 0.005423558 | 0.004214949, global correction | 61.08% |

The Norman advantage over its selected reference is inconclusive. Wessels favors a global correction, so its additive gain is compatible with shared distortion. MSEs from different datasets, scales, and output panels aren't directly comparable.

<img src="images/opal_results.svg" alt="Historical additive comparisons and the separate D-SPIN transfer test, normalized within each evaluation population." width="100%" />

### Development and independent transfer

| Evidence class | Result | Consequence |
| --- | --- | --- |
| Exploratory architecture comparisons | Kernels selected in all 40 competitive outer folds and beat the tested neural additions in aggregate | Strong development reference; larger capacity remains under-tested. |
| Wessels coverage development | Extra pair labels improve random/partner MSE by 16.05%/4.92%; extra singles give about +0.03%/-0.38% | Pair coverage was more useful in this expansion. |
| EmeraldBay observed-single pilot | Kernel 0.172217529; additive 0.210238543; adapted aggregate CPA 0.157097856 | Native CPA wins this pilot; this isn't a canonical cell-level benchmark. |
| Independent D-SPIN procedure evaluation | Frozen pooled-sum kernel 0.034024658; source-selected pooled-pair kernel 0.033107517 | Frozen procedure is 2.77% worse and fails the registered 5% superiority gate. |
| D-SPIN uncertainty diagnostic | A nominal 90% empirical source radius covers 34.3% of held conditions | No validated OOD coverage or abstention guarantee. |
| Baseline-only state development | Tested learned models are 13.91% and 22.65% worse than simple references on EmeraldBay and sci-Plex 3 | These builds don't justify learned-state adoption or scaling. |

<img src="images/opal_state.svg" alt="The tested learned state models have higher MSE than the selected simple references on both development sources." width="85%" />

D-SPIN uses a different study with whole Myeloid and NK lineage holdouts across batches. Its procedure is source-refitted and uses explicitly permitted measured held-context singles for reference normalization, so it isn't zero-shot context prediction. Native GEARS comparisons are mixed. Main-effect/residual error cancellation prevents interpreting favorable missing-single total error as reliable interaction recovery.

[The evidence guide](../docs/evidence.md), [capability ledger](../results/state/capability_ledger.json), and [manuscript](../paper/main.pdf) preserve negative results, interrupted runs, and corrections.

## 4. Training and reproduction details

The historical genetic benchmark fits residual corrections on matched population means, with source preparation, split identities, reference-only gene selection, and development-selected hyperparameters recorded separately from held-out outcomes. Later amendments use nested development comparisons; D-SPIN is the independent procedure evaluation. Cells and technical strata aren't automatically biological replicates.

Install source-data dependencies with `python -m pip install -e ".[data,dev]"` before biological preparation. Full reproduction needs separate dataset downloads, optional external resources, and track-specific CPU/GPU environments. Follow [the detailed reproduction guide](../docs/reproduction.md), the recorded dataset/experiment configs and [evidence summary](../docs/evidence.md), rather than running every study as an installation check.

Recreate README plots and manuscript PDFs from compact saved records with:

```powershell
python tools/generate_publication_figures.py
python tools/report_state.py
python tools/report_next_paper.py
python tools/build_paper.py
```

The public plot record lets the first command run without the full local prediction banks. The figure code performs no fitting and opens no held-out expression outcomes. Compilation needs an existing Tectonic installation or `--tectonic PATH`.

## 5. Quick start

### Installation

Use Python 3.11 through 3.14. From this checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\opal.exe --help
```

On Linux/macOS, use `.venv/bin/python` and `.venv/bin/opal`. The package's normal installation includes PyTorch and other declared dependencies; its download size isn't the size of the fitted kernel head.

### Synthetic execution check

```powershell
.\.venv\Scripts\opal.exe reproduce --config configs/experiments/release_quickstart.json --output results/synthetic/my-quickstart
.\.venv\Scripts\python.exe -m pytest -q
```

This runs two small synthetic scenarios and the software checks. It doesn't establish biological accuracy. Use a new output directory for each run; existing evidence isn't overwritten. `main.py` without arguments starts a much larger research workflow.

### Reference inference

Extract the reference-model ZIP from the GitHub release, then:

```python
from opal import load_model

model = load_model("models/norman-kernel")
prediction = model.predict("AHR", "KLF1")
effect, genes = prediction["effect"], prediction["genes"]
```

The output follows the fixed gene order and effect scale in the manifest. EmeraldBay uses `model.predict(a, b, context=context)` with registered contexts and both measured singles available. Check support metadata before interpreting a query.

## 6. FAQ

**Why is the package called `opal-research`, but imports use `opal`?** Distribution names and Python import names serve different purposes. The source lives directly in `src/`; setuptools maps that directory to the installed `opal` package, preserving a distinct import namespace.

**What does 0.1.2 release?** The consolidated code and comparative evidence follow the historical 0.1.1 preparation. This version does not claim a newly validated biological predictor. Its GitHub release is published in a private repository; public access, Zenodo and arXiv remain separate.

**Does the model learn biological interactions?** Insufficient evidence. A residual contains measurement error and shared distortion as well as potential pair-specific structure, and simple references explain much of the observed improvement.

**Can it predict unmeasured singles or arbitrary contexts?** The main release contract requires measured singles. Separate missing-single/state experiments are exploratory and don't establish general-purpose transfer.

## 7. Limitations

The biological pair sets are small, residual technical reproducibility is weak, and the capacity comparison isn't exhaustive. Repeated splits share data. D-SPIN has only two held lineage groups, treatment-assigned lineage labels, and no established biological replication. The tested state encoders lose to simple anchors; other representations remain untested. External priors and native adaptations don't establish general superiority.

Uncertainty outputs are diagnostic, not validated confidence percentages. Synthetic recovery, safe exports, passing tests, and CPU/GPU execution checks don't establish biological mechanisms or prospective experimental usefulness. This is a comparative research framework, with no frontier or clinical-use claim.

## 8. License

Original Opal code is [MIT](../LICENSE). Dataset-derived Norman and EmeraldBay reference content retains CC BY 4.0 terms; [third-party notices](../THIRD_PARTY_NOTICES.md) document attribution and exclusions. Template provenance and permission are recorded in the third-party notices. An Opal code license doesn't relicense external sources or pretrained resources.

## 9. Citation

The preprint hasn't been submitted to arXiv. Until an identifier exists, don't cite a fabricated arXiv number or describe it as peer reviewed. The following citation identifies the released software revision:

```bibtex
@software{prakash_opal_2026,
  author = {Prakash, Arjun Vijay},
  title = {Opal: observed-single combinatorial response prediction},
  year = {2026},
  version = {0.1.2},
  url = {https://github.com/TheTinyIntelLab/opal}
}
```

[CITATION.cff](../CITATION.cff) supplies structured software metadata. Add the actual arXiv identifier and archival DOI only after they're assigned.

## 10. Contact

Read [The Tiny Intelligence Lab](https://thetinyintelligencelab.vercel.app/) for project updates. Once the repository is public, use [GitHub issues](https://github.com/TheTinyIntelLab/opal/issues) for reproducible code or evidence problems.

## Release closeout and replicated-study workflow

The active planned successor is the [Opal model series](../docs/next_cycle/model_series_protocol.md): Base, Interaction and Guard, with actual trained checkpoints and a shared inference API/CLI required for completion. The [implementation handoff](../plan.md) covers model development and the remaining executable plan work. This scope is documented; the new models have not yet been implemented or trained.

The [closeout record](../docs/next_cycle/execution_status.md) distinguishes verified software from outstanding scientific and publication dependencies. [The fixed-state protocol](../docs/next_cycle/analysis_protocol.md) documents metadata qualification, aggregate-read preparation, bounded selection, prediction freezing, independent evaluation, and synthetic rehearsal. Original Stage 1 runs are canonical; the changed-environment recheck is a named reproducibility sensitivity. Neither expression banks nor historical fits are reopened by the reporting-only closeout.

The separately approved [retrospective Combi-seq workstream](../docs/next_cycle/retrospective_combi_20261009.md) evaluates globally held nominal pairs on already exposed counts with nested selection. Its pair-kernel candidate loses to the selected incidence reference; this is a source-conditional exploratory result and does not clear the prospective biological gates.

The [primary-donor workaround](../docs/next_cycle/donor_monocyte_diagnostic_20261010.md) analyzes one vitamin-D/LPS pair in 18 complete donor panels after two metadata-conflict exclusions fixed before expression access. Its all-source-probe, deposited-scale comparison includes same-pair transfer from other donors; it does not pass the original held-pair confirmation gate. [Continued internet search](../docs/next_cycle/public_dataset_search_20261010.md) records additional leads and qualification limits.

The [older GEO and ArrayExpress follow-up](../docs/next_cycle/public_dataset_search_older_20261010.md) adds eight sample-level GEO reviews and a six-replicate TBX2/MYCN lead. Its metadata audit separates biological source records from technical sequencing rows; neither source numbering nor multiple shRNAs establishes the original independent-culture, unseen-pair bank.

The [six-repeat LNCaP array diagnostic](../docs/next_cycle/lncap_array_unmatched_diagnostic_20261010.md) executes three pairs using averaged single/control references without assuming matched culture blocks. It reports fixed-baseline errors and exact installed-wheel replay, with no confidence interval or original confirmation claim. [Pre-2021 discovery](../docs/next_cycle/public_dataset_search_pre2021_20261010.md) records the expanded search and ten further source decisions.

The [author-hosted metadata audit](../docs/next_cycle/remote_pair_metadata_20261010.md) resolves PertSpectra's pair-label support inside its remote H5AD without decoding expression, and locates CINEMA-OT's public cytokine deposit. Metadata sample labels and donor/time files remain distinct from verified independent same-state cultures.

The [pre-2013 follow-up](../docs/next_cycle/public_dataset_search_pre2013_20261010.md) adds 172 series and eight sample-level reviews. A tested inventory merger reproduces 1,048 distinct summary identities; clone states, skipped repeat numbers and differing assay endpoints prevent those counts from becoming biological qualification claims.
