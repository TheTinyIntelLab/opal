# Primary-donor single-pair diagnostic, 10 October 2026

Internet search identified [GSE78083](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78083), a primary-monocyte microarray study with matched vehicle, vitamin D, LPS and combined treatments from 20 donors. The [primary methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC4856085/) document simultaneous 24-hour treatment, 100 nM vitamin D, 10 ng/mL LPS and 1% ethanol vehicle. Three wells were pooled per donor/arm, so those wells are not independent transcriptome replicates. The donor is the biological unit.

This is a new descriptive workaround with a different assay and estimand. There is one canonical pair, so no held-pair training/validation/test graph exists. The original confirmation protocol remains blocked. Published study narratives were encountered before expression access, and this source is now expression-exposed. No fresh-confirmation or architecture-superiority claim is made.

## Metadata exclusions fixed before expression access

The first attempt rejected the 20-donor design before downloading or reading expression. `GSM2066485` and `GSM2066487` have `EL` sample titles but combined-treatment characteristics. The entire corresponding donors, `AA_185` and `EA_252`, were excluded from the revised scope without guessing labels from their outcomes. That leaves 18 donors and 72 complete profiles. The rejected attempt and subsequent scope are preserved separately.

The [metadata retrieval record](../../results/replicated_fixed_state_v1/donor-monocyte-metadata-20261010-v1/retrieval.json), [initial failure](../../results/replicated_fixed_state_v1/donor-monocyte-diagnostic-20261010-v1/metadata_validation_failure.json) and [scope before expression](../../results/replicated_fixed_state_v1/donor-monocyte-diagnostic-20261010-v2/scope_before_expression_access.json) establish this sequence. Brief GEO text requests avoid embedded expression tables during metadata review.

## Fixed diagnostic and observed results

The deposited series matrix is retained on its source LUMI variance-transformed, quantile-normalized intensity scale. There is no count reconstruction, log-CPM conversion, normalization refit, probe-to-gene remapping or outcome-based feature filter. All 47,299 deposited finite probe IDs define the endpoint, which differs from the paper's filtered gene endpoint. Source preprocessing used the full deposited study, so this is not isolated prospective preprocessing.

For each donor, the observed pair effect is combined minus vehicle. The additive prediction is vitamin D plus LPS minus twice vehicle. Residual transfer adds the mean residual from the other 17 donors to the query donor's additive prediction, excluding the complete query donor from that calculation. Other donors reveal the same pair, making this transfer ineligible under the original global held-pair protocol.

| Comparison | Equal-donor mean MSE |
| --- | ---: |
| Zero effect | 0.0993830299 |
| Matched-donor additive | 0.1039290510 |
| Same-pair residual transfer from other donors | 0.0932637703 |

The median residual cosine across all 153 donor pairs is 0.2538651178, with no undefined comparisons. These observations describe consistency and prediction error across sampled donors, combining biological differences and assay variation. They do not identify a technical noise floor or estimate independent repeated-culture reliability. There is no confidence interval, menu selection, ancestry comparison or scientific advancement decision.

The [report](../../results/replicated_fixed_state_v1/donor-monocyte-diagnostic-20261010-v2/report.json) preserves every donor loss. [Verification](../../results/replicated_fixed_state_v1/donor-monocyte-diagnostic-20261010-v2/verification.json) independently reparses the source matrix and matches every diagnostic field exactly. Raw expression and full metadata snapshots remain local and excluded from Git and release archives.

Run `python tools/diagnose_public_donors.py --metadata <samples.soft> --matrix <series_matrix.txt.gz> --output <new-directory>`. Add `--download` only when the local matrix does not exist; the runner fixes scope before downloading the named primary source and refuses overwrites. Existing local-matrix replays disclose prior access through the scope record. The committed v0.1.2 release remains frozen and unchanged.

An additional [installed-wheel replay](../../results/replicated_fixed_state_v1/donor-monocyte-diagnostic-20261010-v2/installed_replay_verification.json) matches every diagnostic field outside the source checkout. It reuses host dependencies and is not claimed as a fresh dependency installation. The development wheel was not uploaded or substituted for published release assets.
