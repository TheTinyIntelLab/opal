# Opal model development

Sole authoritative forward plan, updated 20 October 2026. The completed engineering workstream and historical scientific records remain closed and preserved. Current authorization covers public-data acquisition, numerical work, tests, commits and normal push. It does not authorize purchases, external correspondence or publication.

## Audit and decision

The previous plan specified the composition objective, model, data, baselines, splits, budget and promotion rule adequately. That experiment is complete and failed. Its material forward gap was an unimplemented requirement for new pair evidence without a source, acquisition test, next model or milestones. The execution below repairs that gap. Do not repeat the completed expression fits or reopen their hyperparameter menu.

The new experiment predicts **measured double-CRISPRi growth gamma**, a functional endpoint, for held-out pairs of genes with observed single growth and training pair evidence in K562 and Jurkat. It does not predict expression or claim that fitness improvements solve transcriptomic prediction. This endpoint is justified by an acquired, much denser pair screen and directly measured functional effects; it avoids a large atlas download and an architecture sweep on the old labels.

## Acquired compatible data and terms

Official GEO GSE116198 guide-pair phenotype file: 112,096,990 compressed bytes, SHA256 ba9772f3b0e929bcf5b23bebf17d4caab86c7f2ccd50dbd62fbc47e40c2aac21. Download URL and preparation are in results/fitness-20261020/horlbeck_acquisition.json and preparation.json. Horlbeck et al., Cell 2018, DOI 10.1016/j.cell.2018.06.010, describes independent infection replicates and control-centered growth normalized by replicate doublings.

Use triple-sequencing measured gamma, averaging ordered guide pairs and two infection replicates equally. Matched single effects come only from gene/negative-control guide pairs. Published GI scores and GI correlations are excluded because their construction uses full pair outcomes. Normalization of public growth measurements remains an upstream shared technical dependency, not an independently reprocessed raw-count experiment.

Coverage-only rules require at least 20 single observations and four ordered pair observations in each context and replicate. Preparation yields 81,466 canonical pairs in both contexts, with 427 supported genes out of 467 parsed. No effect-size filter or test-driven subset selection is allowed. These are new data for Opal but one deposited screen, enriched for growth-related genes, not genome-wide generalization or independent-laboratory confirmation.

The author dataset DOI 10.17632/rdzk59n6j4.1 explicitly has CC BY 4.0, corroborated by DataCite. GEO is the same study's official public deposition; exact byte identity with the Mendeley archive cannot be checked because its file API returns 403. Retain that licensing distinction, credit the authors, and keep raw data, numeric banks and checkpoints local rather than claiming unrestricted redistribution of the mirror. The official NCI-ALMANAC compact archive was also obtained as a feasible public fallback; it is not pooled with genetic growth measurements or used for this fit.

## Chosen model and strong references

The candidate is symmetric indefinite low-rank residual completion with shared factors plus context-private factors. It predicts measured double growth after a training-fitted reference based on observed singles and gene main effects. The hypothesis is that conserved interaction structure across contexts improves missing-pair prediction while private components preserve differences.

Compare against zero, sum and minimum of measured singles; regularized scalar calibration; calibrated gene-main-effect ridge; independent-context low-rank completion; and pooled-only completion. Independent, pooled and joint modes have equal factor parameter counts, 4 * parsed_genes * rank. Joint uses common rank r plus private rank r/2 per context; independent uses rank r per context, pooled uses rank 2r. This tests sharing rather than granting the joint model a larger parameter budget. All methods access the same measured singles, permitted training pairs and validation labels.

## Locked evaluation and criteria

configs/experiments/fitness_completion_20261020.json and tools/study_fitness.py are the executable contract. Freeze source/data hashes and assignment before scoring. Seed 20261021 assigns canonical gene pairs 60/20/20 to train/validation/test, identically in both contexts. Every guide variant and orientation stays with its pair. Verify every queried gene has training-pair support. This is missing-edge prediction among supported genes, not cold-gene or unseen-cell-type prediction.

Fit scalers, calibrations, node effects and factors on training rows only. Validation chooses the initial reference per context, rank/penalty, and the best checkpoint every 25 steps. Test values are used only after persisting validation selections. Candidate menu: rank 8 or 16, penalty 0.001 or 0.01, Adam learning rate 0.03, at most 500 steps, one fixed initialization seed. The per-context strongest reference is chosen from the entire non-joint menu using validation. No rescue changes after test results.

Primary loss is mean context-normalized test MSE using training-only target variances. Report raw MSE and Pearson correlation separately for each context, all reference outcomes, fit histories, support and replicate disagreement. Promotion requires at least 2% lower MSE than the validation-selected reference in BOTH contexts and below-zero upper bounds from 2,000 conditional pair-bootstrap loss-difference draws. Those intervals condition on this pair population and shared references, not independent cultures or cold genes. Partial gains are baseline-level; a loss rejects this sharing hypothesis. No frontier claim follows from this within-source experiment.

## Resources and execution milestones

One numerical worker and one BLAS/PyTorch thread. At most one hour wall time, 2 GiB peak process RSS and 64 MiB new experiment artifacts, separate from the acquired 112 MB source. Outputs use C:/Users/USER/Desktop/opal-next-data-20261020/. C: now has several GiB free; do not delete existing scientific artifacts. Record first-fit runtime/memory before automatically continuing within caps. Save safe numeric checkpoints and exact replay receipts, with reference/schema assets required for full growth predictions.

1. Audit plan, acquire/qualify public data, stream preparation and verify source terms: complete.
2. Implement and review masked model, baselines and fixed protocol; test synthetic recovery, symmetry and serialization; freeze and commit before fitting: in progress.
3. Execute one finite registered fit/evaluation, assess sharing against strong references and measurement disagreement, record a single evidence-backed decision: pending.
4. Update the existing model-development report and README with actual results; preserve prior negative experiments; run tests, builds, data/replay checks; commit lowercase batches, push verified origin/main and check clean tree/remote/CI: pending.

## Continuation and termination

The overlap-expression candidate remains FAILED HYPOTHESIS and closed; see docs/model_development_20261020.md and results/composition-20261020/. A successful fitness result earns a separate evaluation on an independent compatible screen, not a claim about expression. A failed sharing result retains the validation-supported independent/simple reference and closes this finite menu. A new attempt requires a specific data or measured failure diagnosis, not a wider rank/seed search. Generalizing to unmeasured genes would require public gene features and a separately locked node-held evaluation; that is not silently included here.
