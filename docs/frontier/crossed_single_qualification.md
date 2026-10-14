# Distinct single-response task qualification

The PLAN specifies a crossed-single representation/ridge task when pair access fails. The current pair branch is source-ineligible, so a separate outcome-blind source qualification has now run for `tahoebio/Tahoe-100M`. This is a distinct single-response/context task; it inherits no pair-composition, held-context, biological-confirmation or frontier claim. No prediction experiment is registered until the source graph and byte-qualified subset are known.

The audit pins revision `349e2a55f5538a3e73b32a06c6bdd8e9e87011e8`, verifies declared sizes and available LFS hashes, and opens only sample and cell-line catalog fields. It fetches no expression, cell observations, differential expression, genes, weights or full atlas. Total metadata transfer was 369,683 bytes within a 4,000,000-byte ceiling and 1,000,000-byte per-file ceiling. The [pinned author card](https://huggingface.co/datasets/tahoebio/Tahoe-100M/blob/349e2a55f5538a3e73b32a06c6bdd8e9e87011e8/README.md) specifies CC0 and plate-matched DMSO_TF controls; that data permission is recorded separately from scientific eligibility.

The sample catalog contains 1,344 rows, including 1,316 literal single-treatment sample rows and 1,137 distinct drug/dose/unit assignments. Vehicle records are preserved by plate. The separate cell-line annotation catalog contains 1,000 rows and 102 unique context IDs, including repeated mutation annotations. Those IDs are catalog entries, not verified measured contexts in this experiment. The author card describes a 50-cell-line cohort, so joining catalog labels to actual observed cells is essential. Treatment-by-context edge count and independently initiated culture count remain unknown rather than inferred from a Cartesian product.

The pinned observation metadata alone is 2,293,981,573 bytes and was not fetched. No exact expression subset has been declared. Treatment duration, independent initiation, exact vehicle equivalence and condition cell support are unverified. Repeated assignments across plates are grouped and never counted as independent biological repetitions. The result is `metadata_only_unqualified`, with all exclusions and missing evidence retained. No single-response model has been fit or scored.

## Reproducible command

```powershell
.venv/Scripts/python.exe tools/qualify_crossed_single_source.py --output results/opal_context_contrast_v1/single-response-qualification-<unique-id>
```

Existing outputs are refused. Each new audit resolves and records its revision; a changed upstream revision is a new qualification, not the same frozen source. The tool uses the existing requests/pyarrow environment, installs nothing, retains partial failures, enforces metadata transfer ceilings and records process CPU/wall/current RSS and exact fetched hashes. The RSS field is a current sample, not a measured peak. It cannot pass source eligibility based on catalog data alone.

Before prediction, qualify a measured condition graph with exact treatment/vehicle/duration assignments and grouped repeated plates, finite reference support, exposure history and a complete bounded expression/aggregate export. Then register the separate identity/context estimand, outcome-isolated folds, source-only representation and descriptor ridge/kernel references with matched tuning and alignment controls. The PLAN's fixed 16-coordinate representation probe is a lead; it is not permission to invent an endpoint, context split, rank support, minimum sample size or useful margin after scoring. Independent confirmation still needs independent measurements. This audit does not rescue the failed pair claim or commission an atlas download.
