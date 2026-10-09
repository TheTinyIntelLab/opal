# Actual public Combi-seq processing diagnostic

The public GSE174695 aggregate-read matrix was downloaded and analyzed on 8 October 2026. The scope and implementation hashes were saved before expression access. This is an exploratory nominal-identity processing diagnostic, not a qualified biological measurement pilot or independent confirmation study.

The [scope record](../../results/replicated_fixed_state_v1/public-combi-processing-20261008-v1/scope_before_expression_access.json), [full report](../../results/replicated_fixed_state_v1/public-combi-processing-20261008-v1/report.json) and [new exposure ledger](../../results/replicated_fixed_state_v1/public-combi-processing-20261008-v1/exposure_ledger.json) preserve source hashes, timing, reference lineage and decisions. The matrix SHA-256 is `5524623316ae773d4d1d6ea03818934614db3a0ab3763eabc3b0e0aef5d65367`. Raw expression remains in the ignored source-snapshot directory and is excluded from releases. Original metadata-only audits remain unchanged as historical records.

## Fixed analysis and findings

The matrix contains 1,263 treatment-barcode records and 25,912 source-provided genes. All 374 distinct nonrepeat nominal pairs have positive-library pair, DMSO and identity-matched single records in each of the three reported runs. No zero-library rows or missing-reference exclusions occurred. Physical-dose units, exact vehicle compatibility and independent culture initiation remain unverified.

Each barcode library was normalized to 10,000 reads. Controls and singles were averaged before applying `log2(1 + reads)`; logged pair effects were averaged across technical variants of the same canonical pair and run. Every source-provided gene was retained, with no new response-driven gene selection. The source's upstream gene filtering does not become independent held-pair filtering by reusing this matrix.

| Reported run | Zero-effect MSE | Additive MSE | Other-run shared-offset MSE | Forbidden repeated-pair other-run MSE |
| --- | ---: | ---: | ---: | ---: |
| rep1 | 0.285204 | 0.550385 | 0.614579 | 0.816326 |
| rep2 | 0.301448 | 0.575017 | 0.625708 | 0.823186 |
| rep3 | 0.299561 | 0.569757 | 0.619437 | 0.820990 |

These are descriptive equally weighted pair/gene losses on the complete nominal panel. The repeated-pair control predicts a run's residual from the same pair measured in the other runs; it consumes pair outcomes and cannot be presented as deployable prediction. The shared offset is the mean residual across all pairs in the other runs. Neither control selects or fits a new architecture.

Mean raw-residual cosine was 0.028600, 0.026433 and 0.039769 for the three run pairs. After subtracting leave-one-run-out shared offsets, mean cosine was -0.148895, -0.145317 and -0.147208. Those vectors reuse overlapping runs in their offsets, so negative agreement can include a centering-induced dependence artifact. It is not independent evidence of biological antagonism or a calibrated absence of reproducibility.

Within this fixed all-gene diagnostic, additivity performs worse than zero effect, and the forbidden repeated-pair predictor performs worse than the shared offset. These results do not establish that biological signal is absent: low-count noise, upstream preparation and nominal reference mismatch remain possible explanations. They do not supply the positive independent-culture measurement evidence required to advance the plan. No confidence intervals, biological power claims, pair-kernel search or scientific superiority claim were produced.

## Reproduction and exposure

Run `python tools/diagnose_public_combi.py --metadata data/source_audits/replicated_fixed_state_v1/source_snapshots/combi_treatments.csv.gz --matrix data/source_audits/replicated_fixed_state_v1/source_snapshots/combi_large_scale_matrix.csv.gz --output results/replicated_fixed_state_v1/public-combi-processing-new --fetch` with a new output path. It writes the scope before reading expression, bounds the download and decompressed input, joins matrix columns by source IDs, rejects corrected/noninteger inputs and preserves exact nominal reference matches. Source-column order is never assumed to equal metadata order.

All pair outcomes in this deposit are now exposed through this diagnostic. The historical statement that no Combi-seq expression had been inspected was true before this run and is no longer the current exposure status. This source cannot now be represented as untouched confirmation. The local scope record is not external preregistration. Frozen runner and diagnostic source copies accompany the result and match the pre-access hashes.

Four new failure-oriented tests cover source joins, missing references, corrected values, technical-variant aggregation and non-advancement. The full suite passes 206 tests. Previous release archives retain their original build snapshots and validation reports; this diagnostic does not retroactively change them.

The [post-access source QA](../../results/replicated_fixed_state_v1/public-combi-processing-20261008-v1/post_access_source_qa.json) reports a median 33,434 reads and 598 detected genes per treatment barcode. It adds descriptive depth information without changing the fixed gene universe or exclusions. Sparse readouts are one reason this all-gene diagnostic cannot establish absence of biological effects.

The rebuilt review package is `releases/local-review-0.1.2-public-diagnostic/`. Installed copies on Windows Python 3.11 and 3.14 replayed the real public diagnostic with identical reference lineage and matching losses, restored all 65 historical saved queries, and passed the synthetic quickstart and archive privacy/hash checks. The [completion record](../../results/replicated_fixed_state_v1/public-combi-processing-20261008-v1/completed_verification.json) binds the wheel and exact evidence. Existing fresh dependency closures were reused; another dependency installation and local Linux execution are not claimed. The original diagnostic process did not record peak memory, and the replay environments could not measure it, so no resource-budget acceptance is claimed. These final verification records accompany the preserved build-time archive snapshot.
