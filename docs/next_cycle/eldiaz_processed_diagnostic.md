# LNCaP processed-bulk measurement diagnostic

The previously metadata-audited El Diaz source now supplies a second real-data measurement diagnostic. The [GEO family metadata](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149428) identify tamoxifen 20 uM, mefloquine 10 uM and withaferin 5 uM, matching pair constituent doses, three repeat letters and the 0.3% DMSO time-course vehicle protocol. These are source-reported conditions, not independently verified physical measurements or a culture-initiation map.

A scope was recorded before downloading the LNCaP expression body. Published narrative results were already exposed. The matrix preserves 144 sample columns and 15,016 genes. The fixed descriptive analysis uses 126 matched controls/singles/pairs and excludes the 18 triple-treatment samples, at all six nominal times. Controls and singles match within each time and reported repeat. No model is fitted or selected, and no time point is chosen for a favorable result. The source is now explicitly expression-exposed.

The [author study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7546737/) supplies log2 CPM after TMM normalization and plate covariate adjustment. The adapter preserves that continuous scale, including negative values. It does not invert values into counts or rerun count normalization. Upstream filtering and plate correction may use the full study, so this cannot serve as confirmation-independent preprocessing. Nominal time zero is an early post-treatment collection according to the methods, rather than proof of an untreated intervention.

| Nominal hour | Zero MSE | Additive MSE | Other-repeat pair effect MSE | Other-repeat residual MSE | Median residual cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.194248 | 0.384812 | 0.268295 | 0.537814 | 0.059644 |
| 3 | 0.530772 | 0.486672 | 0.280409 | 0.554373 | 0.271831 |
| 6 | 0.969306 | 0.474049 | 0.255237 | 0.495001 | 0.227503 |
| 9 | 1.127057 | 0.528994 | 0.276137 | 0.522274 | 0.240207 |
| 12 | 1.149086 | 0.632064 | 0.365568 | 0.690196 | 0.273615 |
| 24 | 2.108794 | 1.470312 | 0.365779 | 0.640950 | 0.731437 |

Other-repeat predictors deliberately consume outcomes of the same pair in another repeat. They measure source-processed agreement and are forbidden pair-prediction controls, not deployment models. The 24-hour repeated residual control has lower MSE than a zero residual, whereas at early times it can be worse. This supports keeping measurement and prediction questions separate; it does not establish a general biological signal ceiling or invalidate the negative Combi pair-kernel result. Absolute errors are not comparable across their different assays, gene endpoints and preprocessing.

No biological intervals or power were estimated. Three repeat letters at six times do not provide 18 independently initiated cultures, and the three-pair panel is too narrow for the original broad held-pair procedure. No original gate passed. Combi-seq remains exposed, the LNCaP matrix is now exposed, and the MCF7 expression body remains unopened in this work.

Reproduce with `tools/diagnose_public_bulk.py`, supplying the local family SOFT file, local LNCaP expression file, scope and access records in `results/replicated_fixed_state_v1/eldiaz-metadata-20261009-v1`, and a new output directory. The [report](../../results/replicated_fixed_state_v1/eldiaz-processed-measurement-20261009-v1/report.json) retains every time/repeat loss, pairwise residual cosine and exact reference accession. The [scope](../../results/replicated_fixed_state_v1/eldiaz-metadata-20261009-v1/scope_before_expression_access.json) and access records bind the data hashes. Raw/derived expression remains excluded from Git. Tests reject dose, vehicle, matrix identity and nonfinite mismatches and preserve reordered source columns and separate time strata.
