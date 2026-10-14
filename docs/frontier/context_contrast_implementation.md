# Context-contrast implementation

Updated 13 October 2026. `opal_context_contrast_v1` is implemented and software-checked, but its preparation, fitting and biological scoring have not run. The owner authorized repository implementation and bounded fixtures only. The frozen bridge remains research-retired, source-ineligible and biologically untested; its 207-gene refusal, 256-gene threshold, configuration, failed attempts and original results are unchanged.

## What is executable

`src/context_contrast/` implements the separate frozen protocol, reference-only shared panel, exact assignment/coverage graph, globally grouped pair folds, source/reference/evaluator role files, irreversible exposure records, four-coordinate PCA, eight/eighteen symmetric features, cross-fitted nuisance estimates, gene-RMS rank-two output basis and full-response nested penalty selection. The primary prediction reconstructs both observed contexts using a pooled mean residual and symmetric half-contrast. Both contexts are training information; no target bridge or held-context claim is part of this diagnostic.

The finite envelope has 55 penalty/control configurations and 21 selected arms, plus the locked selector. It includes a fold-refitted 24-configuration Opal-Base procedure, no-change/additivity, shared response/contrast offsets, scalar affine and component-incidence correction, phi8, 36-coefficient mean/context controls, pooled/per-context/joint-context symmetric kernels and response/contrast alignment controls at the three registered seeds. Every nested invocation, failed attempt, selected penalty, basis, scale, complete kernel bank and inference query is retained. Content-addressed JSON/NPZ assets forbid pickle and preserve complete dependencies; restored predictions must replay exactly.

Preparation streams the existing count cache, verifies its identity before expression access, normalizes full libraries before panel restriction, averages condition counts before log2 transformation and subtracts exactly matched controls. Only permitted single effects select the common panel; pair outcomes never select genes or representations. Controls/singles are split into physically disjoint deterministic row-parity halves. Both role swaps use the same pair-condition estimator and common complete query population. If either partition lacks ten-cell matching support, the swap remains unsupported and advancement stops. Those halves are technical measurements, not independent cultures.

Evaluation is a separate accepted command after the prediction seal. It reports equal-pair context-aggregate response MSE, MAE, full-pool retrieval with average ties and undefined records, across-perturbation correlation, contrast error, fold/stratum diagnostics, and complete fallback counts. Ninety-nine shared-block identity permutations calibrate retrieval; 2,000 intervention-product draws report conditional paired loss bounds with zero-denominator draws retained. Neither calculation estimates biological replication or selection uncertainty. Source-only exact/technical-half positives and train-mean, identity, affine and amplitude negatives qualify the metric before query ranking. Cosine retrieval's affine/amplitude invariance is disclosed; response MSE tests calibration.

Independent-anchor departure and a frozen program endpoint have evaluator APIs, but are unavailable for the current source until real provenance/weights are supplied. No utility claim is made without a defensible frozen program. Missing software/measurement qualification or unsupported controls prevent gates 2/3; a supported failure retires the finite phi18 procedure. The machine-readable decision always leaves independent confirmation/frontier gates false for this fixed-context diagnostic.

## Work-package status

| Package | Implemented deliverable | Remaining dependency |
| --- | --- | --- |
| SC01 | Cache identity verifier, reviewed rights/assignment evidence, literal graph and exact coverage/exclusion checks | Full cache hash matches the recorded identity. Rights, reagent/dose/time equivalence and source-reference support still need reviewed evidence; no source expression was decoded here. |
| SC02 | Exact logical invocation inventory, all asset classes, owner acceptance matching, live CPU/RAM/retention/reserve guards and failed-attempt receipts | A complete new numerical budget and output layout must be accepted. Historical incomplete CPU accounting is unchanged. |
| SC03 | Count streaming, one reference-only common panel, technical reference partitions and separately hashed fold roles | Authorized preparation and its actual qualification records. No new aggregate bank exists yet. |
| SC04 | Contract equations, fixed ranks/floors, cross-fitted nuisance, nested response-MSE selection and symmetric reconstruction | An authorized run with supported training designs. |
| SC05 | All same-information references, matched-capacity controls, alignment seeds and source-locked selector | Source-qualified fits and complete retained states. |
| SC06 | Small-fixture algebra, isolation, rank, serialization, null and fallback checks; practical metric qualification implementation | Real technical-half metric support and reviewed software integrity receipt before advancement. |
| SC07 | Separate sealed evaluation, both disjoint-reference swaps, full population, conditional bounds and five gates | Numerical authorization. No development result is claimed. |
| SC08 | Separate confirmation-readiness template and reviewed culture/context/assignment/reference/covariance/power/rights checks | Independent cultures, actual owner/partner arrangements, useful margin, empirical power/coverage and a separate held-context registration. |

## Commands for later authorization

Use the existing environment. These commands install nothing and download/copy nothing. Every output is new; existing outputs are refused. Replace example output roots with the exact roots in each owner acceptance.

```powershell
$python = '.venv/Scripts/python.exe'
& $python tools/opal_context_contrast.py --config configs/experiments/opal_context_contrast_v1.json protocol
& $python tools/opal_context_contrast.py source --output tmp/successor-source.json
& $python tools/opal_context_contrast.py metadata --review docs/frontier/context_contrast_source_review.template.json --output tmp/successor-metadata.json
& $python tools/opal_context_contrast.py qualify --metadata tmp/successor-metadata.json --review docs/frontier/context_contrast_source_review.template.json --output tmp/successor-qualification.json
& $python tools/opal_context_contrast.py inventory --pairs 51 --genes 512 --output tmp/successor-inventory.json
& $python tools/opal_context_contrast.py readiness-template --output tmp/successor-readiness-template.json
& $python tools/opal_context_contrast.py readiness --input tmp/successor-readiness-template.json --output tmp/successor-readiness.json
```

The supplied source-review template is intentionally incomplete and cannot qualify a source. Metadata reads access labels/counts, not expression values. The `condition_assignments` keys are compact JSON `[canonical_pair_string, context, stratum]`; each value must specify exact `reagent_ids`, `dose`, `dose_unit`, `duration`, `duration_unit`, `initiation_mode`, `assay_id` and `assignment_match=true`. Actual control/single/pair guide mixtures and original rows remain in the graph. Unknown fields are refused, never filled from a publication title.

```powershell
& $python tools/opal_context_contrast.py prepare --input reviewed-source.json --inventory tmp/successor-inventory.json --acceptance accepted-prepare.json --output E:/opal-context-contrast/preparation-001
& $python tools/opal_context_contrast.py predict --input E:/opal-context-contrast/preparation-001/prepared --inventory tmp/successor-inventory.json --acceptance accepted-predict.json --output E:/opal-context-contrast/predictions-001
& $python tools/opal_context_contrast.py replay --input E:/opal-context-contrast/predictions-001
& $python tools/opal_context_contrast.py evaluate --input E:/opal-context-contrast/predictions-001 --inventory tmp/successor-inventory.json --acceptance accepted-evaluate.json --integrity-review reviewed-software-integrity.json --output E:/opal-context-contrast/evaluation-001
```

Each acceptance names `approved=true`, actual `owner`/`approved_at`, experiment ID, operation, absolute output root, input file SHA-256, inventory file SHA-256, protocol fingerprint, `workers=threads=1`, explicit CPU seconds, RAM bytes, retained bytes, scratch bytes, failure reserve and volume reserve. It must explicitly disable installs/downloads/GPU. The input hash binds the reviewed source JSON for preparation, prepared manifest for prediction, and prediction freeze for evaluation. See `authorize` in `resources.py`; no approval file is supplied or implied by this documentation. Preparation/prediction/evaluation require separate acceptances.

The integrity review has actual reviewer/date, approved evidence file paths/hashes and `details.qualified=true` only after the listed software checks and any nominated positive controls are reviewed. Missing integrity evidence can still produce an authorized complete evaluation, but cannot pass the scientific gates. Readiness similarly validates evidence completeness; it cannot verify a reviewer's scientific judgment or authorize assays.

The installed project CLI also routes `opal context-contrast ...` to these commands. An optional source-review `program` must contain reviewed evidence, `source_only_or_biologically_prespecified=true`, literal `gene_weights` and `top_k`; preparation seals its selected-gene weights and full pool before pair expression access. There is no supplied program, and evaluator-only choice of weights is refused.

## Resource inventory and present blockers

The 51-pair/512-gene inventory schedules **53,280 logical fit invocations** when every required design is supported, including primary plus both swaps, three outer/inner folds, nested pooled nuisance selection and all Base fits. Unsupported branches retain failures and may end early; those do not create reclaimed scientific budget or a supported win. Prediction outputs contain 22 arms, giving 27,574,272 uncompressed numerical bytes per phase and 55,148,544 across prediction/evaluation copies. The deliberately conservative non-deduplicated checkpoint upper estimate is 47,800,684,800 bytes; actual content-addressed/compressed storage has not been profiled. This is an upper layout estimate, not an assertion that a 47.8 GB run is necessary or authorized.

The existing raw input is 3,142,132,144 bytes and its SHA-256 is `7b72d474be90d3e236e9b4d7dce3fdf9806e5329f84f3eaa973ec605f52ebd12`, reverified read-only during implementation. New raw/download/copy bytes are zero. The proposed scratch estimate is 111,208,000 bytes; RAM targets remain 8 GB aggregation/4 GB fitting. The 0.5–4 CPU-hour initial estimate is unprofiled; the proposed stop limit remains at most 24 core-hours, one numerical worker/thread, with at least 1,073,741,824 bytes free volume reserve. The old 25 MB pilot ceiling is unverified and smaller than logical prediction outputs alone; compression/deduplication must not be assumed to make it fit. Request an accepted complete layout before running, rather than borrowing the historical ~30.75 MB nominal headroom.

## Software validation

Fixture checks use at most 64 fitting pairs and 32 genes, except schema/isolation fixtures with 36 pairs and 256 genes that use a stub selector. Synthetic HDF5 checks use 40 cells and three genes. Tests cap saved fixture assets at 8 MB and use one BLAS thread. They check exact equations and ridge normal equations, RMS/gene-replication invariance, calibration references, full finite selection, rank refusals, full-library normalization, source-only fitting IDs, mutation isolation, fallback counts, empirical null/conditional bookkeeping, readiness refusal and tamper detection. They are software evidence, not the proposed diagnostic or biological scoring.

The current receipt is in `context_contrast_validation.json`. The historical manuscript tables and deployment/evaluation scores remain unchanged. Current successor weights, biological results, cultures, permissions, power and frontier success are not supplied by this implementation.

## Complete project build, 14 October 2026

The current package wheel/source distribution and all root/docs Markdown build locally without installing dependencies. The manuscript compiles offline with cached Tectonic resources to 44 pages; its preserved 12 October PDF is in `paper/historical/main-20261012.pdf`. Historical evidence tables and numerical bridge records are unchanged. The complete original bridge asset inventory records local banks/checkpoints/test copies without committing those payloads.

Build commands are `.venv/Scripts/python.exe -m build --no-isolation --outdir tmp/package-build-20261014` and `.venv/Scripts/python.exe tools/build_docs.py --output tmp/documentation-build-20261014/html`. Existing output directories are refused by the documentation builder. The HTML build is internal review material, not a publication. Optional documentation dependencies are declared, but none were installed here. `project_build_validation.json` binds the current build receipts.

SC01–SC08 have software implementations and executable interfaces. The complete scientific PLAN is not finished: authorized source preparation/development evaluation, independent cultures/anchors, useful margin/power, fresh-context confirmation, eligible native comparisons and scientific utility evidence remain outstanding. No software build or commit supplies those missing measurements or authorization.
