# LNCaP six-repeat processed-array diagnostic

GSE56188 provides 42 deposited profiles, comprising a vehicle control, three singles and all three pairs, with six samples labeled as biological replicates per arm. The [GEO design](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE56188&form=html&targ=self&view=quick) specifies six hours, bicalutamide 5 uM, vorinostat 1 uM and 17-AAG 40 nM, with matched constituent doses in each combination. The [primary paper](https://link.springer.com/article/10.1186/s12885-016-2188-2) confirms the microarray bicalutamide dose differs from its earlier viability experiments. Replicate numbering does not establish matched initiation blocks, and vehicle fraction remains unknown.

Indexed GSM search results exposed small expression-table excerpts before this diagnostic was scoped. The [scope](../../results/replicated_fixed_state_v1/lncap-array-unmatched-20261010-v1/scope_before_full_matrix_access.json) records that exposure explicitly and was fixed before downloading the complete matrix. The endpoint retains all 33,297 finite unique deposited probe IDs on the source Partek RMA/GC-corrected scale. No gene filtering, count inversion, normalization refit or parameter selection occurs. The raw matrix remains local; its hash and download ledger are tracked.

Each observed joint profile is centered on the mean of all six controls, and its additive reference uses the means of all six samples for each constituent single. This avoids assuming that equal repeat numbers identify matched cultures. The same-pair diagnostic estimates a mean residual from the other five joint samples and excludes the query sample; it uses observed outcomes of that pair and is forbidden under the original held-pair prediction design. A separate fixed mean-offset baseline excludes every repeat of the query canonical pair and averages the two other pair residual means with equal pair weights. It is a descriptive baseline, with no search through model configurations.

| Comparison | Equal-pair MSE |
| --- | ---: |
| Zero effect | 0.0821773033 |
| Additive | 0.0728305991 |
| Same pair, other samples | 0.0484450244 |
| Other pairs, fixed mean offset | 0.0537437394 |

The other-pair offset improves on additivity for each of the three pairs. Within-pair residual cosine medians are 0.469036, 0.404199 and 0.470729 for 17-AAG/bicalutamide, 17-AAG/vorinostat and bicalutamide/vorinostat, respectively. These are shared-reference, source-processed profile diagnostics. Shared control/single means and study-wide source normalization create dependence, so the values do not establish independent-culture reliability, a noise ceiling or biological confidence intervals.

The [report](../../results/replicated_fixed_state_v1/lncap-array-unmatched-20261010-v1/report.json) preserves per-pair and per-sample losses, including unfavorable baselines. Independent parsing with reversed sample order reproduces every diagnostic field exactly. A development wheel installed outside the checkout also reproduces the full report and matches the pre-download implementation hash; this local wheel reused host dependencies and was not uploaded to the existing release. Tests use hand-computed errors to check both query-sample and whole-query-pair exclusion, order invariance and rejection of missing, ambiguous or nonfinite inputs.

This source is a useful six-repeat workaround, but its three pairs and unresolved initiation/vehicle metadata do not qualify the original confirmation bank. No pair-kernel candidate was selected or evaluated, no original advancement gate passed and no fresh confirmation claim follows from the baseline improvement.
