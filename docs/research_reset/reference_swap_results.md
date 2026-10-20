# Frozen correction with local singles: results

Replacing the historical main-effect reference with local minimum singles does not consistently rescue the old learned correction. The correction lowers mean MSE by 0.57% and 4.66% in the two environment cohorts, with gene-resampling sensitivity intervals crossing zero, but raises it by 23.91% and 25.80% in the PARP cohorts, with intervals entirely above zero. Local additive singles plus the correction also perform worse in every primary cohort comparison.

This is exploratory successor development with extra assay-local single measurements. The historical external outcomes, earlier reset diagnostics and cross-infection development results were already known when this hypothesis was chosen. No fitting, correction-weight selection, checkpoint selection or exclusions based on these new scores occurred. These results are neither external validation of an adapted model nor independent confirmation.

## Comparison and population

The [frozen protocol](reference_swap_protocol.md) defines the correction as the unchanged historical completion prediction minus its historical main-effect prediction. Its coefficient remains exactly one. Each primary reference averages the two infection-specific minimum-single or additive-single predictions and is compared with the average pair outcome. In particular, minimum singles are computed before infection averaging. The correction is then added unchanged to the reference. No external observed pair residual enters its construction.

The exact historical absent-original-eligible-bank rows with both local singles available yield 130 pairs and 17 genes in each environment cohort, 423 pairs and 30 genes in PARP2022, and 411 pairs and 30 genes in PARP2023. All strict-novel rows have the required singles, so none were excluded for missing singles. Novelty refers to pair absence from the original eligible bank, not previously unseen genes. These small, overlapping panels do not provide four independent study replications.

## Primary scores

Positive gain means lower candidate MSE. The interval reports candidate-minus-reference squared loss under the registered 2,000-draw paired gene-resampling procedure. It is a conditional sensitivity diagnostic, with no formal population coverage guarantee or independent-culture interpretation. Tables round the machine-readable values, whose exact floats, MAE, bias, Pearson correlation, residual summaries and retained draw counts are in [results.json](../../results/research-reset-reference-swap/results.json).

| Cohort | Local reference | Reference MSE | With old correction MSE | MSE gain | Loss-difference sensitivity interval |
|---|---|---:|---:|---:|---|
| environment1 | minimum | 0.005837259 | 0.005804042 | 0.5691% | [-0.001024372, 0.001197876] |
| environment2 | minimum | 0.009145039 | 0.008719261 | 4.6558% | [-0.001807878, 0.001202613] |
| parp2022 | minimum | 0.002526220 | 0.003130288 | -23.9119% | [0.000224996, 0.001051211] |
| parp2023 | minimum | 0.002342280 | 0.002946540 | -25.7979% | [0.000264600, 0.001011470] |
| environment1 | additive | 0.018065704 | 0.018725543 | -3.6524% | [-0.001383637, 0.002313491] |
| environment2 | additive | 0.021661399 | 0.021847040 | -0.8570% | [-0.001909173, 0.001957020] |
| parp2022 | additive | 0.014202027 | 0.014864531 | -4.6649% | [-0.000061735, 0.001444025] |
| parp2023 | additive | 0.014359474 | 0.014986792 | -4.3687% | [-0.000084969, 0.001394100] |

The unchanged correction has RMS 0.0259293 in both environment panels, 0.0227334 in PARP2022 and 0.0225522 in PARP2023. Local minimum residual RMS is 0.0764020, 0.0956297, 0.0502615 and 0.0483971, respectively. Centered correction-versus-local-minimum-residual correlations are 0.175468, 0.246682, -0.071352 and -0.072280. The PARP correction is therefore substantial relative to the residual and has slightly negative centered alignment. It adds error even after replacing the historical reference with a much stronger local minimum reference.

Centered alignment with additive residuals is 0.168516, 0.262252, 0.022403 and 0.018432. Positive centered correlation alone does not guarantee lower MSE, because the uncentered residual-correction product and squared correction jointly determine the loss change. The recorded means and RMS quantities permit that calculation without fitting a rescue coefficient.

## Opposite-infection secondary scores

Direction 1 to 2 uses infection 1 singles to predict infection 2 pair outcomes; 2 to 1 reverses them. The same historical correction is used in both directions. The two directions share pairs, genes, reagents and study processing and do not count as independent replications.

| Cohort | Reference | Direction | Reference MSE | With old correction MSE | MSE gain | Loss-difference sensitivity interval |
|---|---|---|---:|---:|---:|---|
| environment1 | minimum | 1 to 2 | 0.010473242 | 0.010202719 | 2.5830% | [-0.001672437, 0.001296811] |
| environment1 | minimum | 2 to 1 | 0.003525560 | 0.003729649 | -5.7888% | [-0.000629144, 0.001211095] |
| environment2 | minimum | 1 to 2 | 0.005887530 | 0.005619028 | 4.5605% | [-0.001286005, 0.001000159] |
| environment2 | minimum | 2 to 1 | 0.014639051 | 0.014055998 | 3.9829% | [-0.002399271, 0.001493944] |
| parp2022 | minimum | 1 to 2 | 0.002490760 | 0.003197277 | -28.3655% | [0.000296132, 0.001203971] |
| parp2022 | minimum | 2 to 1 | 0.004549507 | 0.005051126 | -11.0258% | [0.000018843, 0.001033394] |
| parp2023 | minimum | 1 to 2 | 0.002385939 | 0.002993405 | -25.4602% | [0.000251083, 0.000997962] |
| parp2023 | minimum | 2 to 1 | 0.002744814 | 0.003345868 | -21.8978% | [0.000236579, 0.001037523] |
| environment1 | additive | 1 to 2 | 0.012335251 | 0.012646683 | -2.5247% | [-0.001439976, 0.001707347] |
| environment1 | additive | 2 to 1 | 0.027484164 | 0.028492411 | -3.6685% | [-0.001397098, 0.003161962] |
| environment2 | additive | 1 to 2 | 0.032271507 | 0.032681483 | -1.2704% | [-0.002106212, 0.002555469] |
| environment2 | additive | 2 to 1 | 0.014392814 | 0.014354120 | 0.2688% | [-0.001776331, 0.001482664] |
| parp2022 | additive | 1 to 2 | 0.020963445 | 0.021733434 | -3.6730% | [-0.000096854, 0.001678897] |
| parp2022 | additive | 2 to 1 | 0.010388220 | 0.010943239 | -5.3428% | [-0.000065623, 0.001243960] |
| parp2023 | additive | 1 to 2 | 0.015060715 | 0.015688502 | -4.1684% | [-0.000062835, 0.001396465] |
| parp2023 | additive | 2 to 1 | 0.014117616 | 0.014744466 | -4.4402% | [-0.000092925, 0.001365874] |

The PARP minimum-reference loss increase persists in all four secondary comparisons, and every associated conditional interval lies above zero. Environment1 changes sign across infection directions. These checks weaken an explanation based solely on averaging the two infection targets, but they do not identify whether biological context, reagent effects, coverage or protocol differences caused the mismatch.

## Ordering and reproduction

Registration completed at 2026-10-20 11:23:08.761896 UTC. Six synthetic prerequisite tests ran from 11:23:09.161097 to 11:23:11.554266 UTC and passed before the run-start receipt at 11:23:22.155975 UTC. The tests verify correction construction, infection direction, mean-target aggregation, zero-correction behavior, deterministic aggregates, invalid input rejection and failure of the prefit prerequisite. Scoring completed at 11:23:23.018924 UTC in 0.8597584 seconds with peak process RSS 102,739,968 bytes. One numerical thread was used. The 600-second, 2-GiB and 16-MiB caps are post-run checks rather than active interruption limits.

The protocol SHA-256 is `49031602d5adb3f2a98495ae526e425d714aceb81eb014cc30a782ed3b1b7aa9`. Results SHA-256 is `263f44c367aacd3af3576439150f30fd73825a8693cfbd00c658833a5c0da4b6`. Exact full scientific JSON replay succeeded initially and again during the independent review. The review also ran all four reset test modules together, with 23 tests passing. The replay updates only its completion receipt, not the protocol, scores or code.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_reset_reference_swap.py tests/test_reset_diagnostics.py tests/test_reset_support.py tests/test_reset_development.py -q
.venv/Scripts/python.exe tools/reset_reference_swap.py replay --output results/research-reset-reference-swap
```

The frozen protocol binds all four prepared-v2 banks, historical prediction arrays, original eligible bank, assignment, model assets, implementation and synthetic tests by SHA-256. The input locator is the preserved reset-diagnostics protocol, which records the local prepared and frozen prediction directories. Exact historical row identities and canonical gene alignment are verified before scoring. A different machine needs the same licensed acquisitions, preparation and expected input bytes; changing local paths does not justify substituting another bank. No source numerical measurements or newly fitted model assets are added to the repository.

## Research consequence

The combined evidence supports preserving the historical completion as a baseline while rejecting a universal unit-weight reference-replacement rescue. It does not show that pair interactions are absent. The separate [cross-infection development experiment](development_results.md) found aligned within-source gains against validation-selected node references, while a calibrated-single reference still beat one Jurkat direction. The [support audit](support_results.md) had only four original test edges whose endpoints both belonged to the external gene panel, so it cannot settle missing-clique transfer. A successor must compare against a frozen strong simple-reference policy and earn any added interaction model on an actually untouched compatible source. Further correction tuning on these exposed cohorts would remain exploratory.
