# Opal-Interaction

## Current local training cycle

Adds symmetric cross-coordinate bilinear residuals with training-only input PCA, acquired-residual output SVD and weighted ridge. The finite comparison includes in-sample and cross-fitted residual targets; unsupported ranks are unavailable.

Public names and classes are unchanged. Internal checkpoint architecture identifiers distinguish the preserved neural/sigmoid mechanism from the bilinear/retrieval mechanism. The current primary recommendation remains Base; Interaction gains about 0.08% on two exposed primary panels and Guard loses to constant mixing, so the 2% nonlinear promotion rule fails. Equal predictions on unsupported panels are declared fallback, not forced diversity.

Weights are local and source-specific for Combi-seq, separate TNF contexts and retrospective Norman/Wessels log/count banks. The original protected genetic evaluator banks remain untouched. Evaluation records are in `results/opal_model_series_v2/`; the final deployment manifest records separately fitted all-development weights under the C artifact root, with no retrospective score. See [the current guide](../next_cycle/opal_model_series_v2.md) for safe loading, shared numeric-array stores, support and expert attribution. No weight redistribution or dose/state extrapolation is authorized.

## Preserved first training cycle

Frozen Base plus a learned shared 64/16 tanh encoder, symmetric latent product and output projection. The branch uses acquired development residuals, weighted MSE, two penalties and development-only stopping.

These are actual fitted source-specific evaluation checkpoints, not universal assay weights or all-development deployment refits. Combi-seq has 374 nominal chemical pairs in three reported runs; PertSpectra has 108 TNF-positive and 52 TNF-negative gene-label-conditional pairs, each in four source sample labels. The two TNF contexts have separately fitted weights and separate reference-derived 512-gene endpoints.

| Training/evaluation source | Full-budget MSE | MSE difference from Base | Source recommendation |
| --- | ---: | ---: | --- |
| Combi-seq | 3.871518 | -0.001181 | Opal-Base |
| PertSpectra TNF-positive | 0.086707 | +0.000778 | Opal-Base |
| PertSpectra TNF-negative | 0.083545 | +0.001266 | no-change |

All rows use equally weighted canonical pairs within a stratum and then equal strata. Absolute losses cannot be compared across sources. Base is the dependable default on Combi-seq and TNF-positive; the nonlinear variants remain experimental. No-change wins on the TNF-negative endpoint, so none of the trainable series is recommended over that control there. The historical pair-kernel family narrowly improves over the selected reference on PertSpectra, while the new nonlinear variants lose; the original unfavorable Combi-seq result is unchanged.

Inputs are measured single effects in exact ordered checkpoint genes, two declared intervention identities and source stratum, with source, assay, units and reference provenance. Both sources use declared log2 library-total-10000 changes; PertSpectra normalization uses the verified deposited counts over its 13,901 retained genes. The genetic target labels average guide mixtures, which differ between single and pair labels. They are not exact matched-guide effects. Unknown dose, state and unseen-single extrapolation are not supported. Missing essential assay/reference data are refused; absent graph/context interaction support uses Base fallback.

Every fitting, stopping, selection and cross-fitting label lies inside the acquired development budget. Global canonical outer pairs are absent from every fitting stage. The complete 0/25/50/100% curves, three acquisition seeds, two identity-only orders, losing configurations, shuffle controls and unavailable partitions are retained. Architecture, optimizer, seed, penalties and finite epoch caps were fixed before training; exact settings are in `configs/experiments/opal_model_series_v1.json`.

The shared API in `opal.series` provides `fit`, `predict`, `save` and `load`; `opal.load_model` also restores this format. Installed CLI routes train, batch predict and hash-check evaluator scores outside the checkout. Prediction output includes source/split/reference provenance, support and fallback status. Checkpoints contain JSON plus safe NPZ numerical arrays with gene order, units, fitted transforms, actual coefficients/weights, architecture, source/input hashes, seed/version and selection lineage. No arbitrary executable deserialization is used.

Local paths and exact hashes are indexed in `results/opal_model_series_v1/combi-seq/all_budget_checkpoint_manifest.json` and each PertSpectra context's `checkpoint_manifest.json`. The checkpoint root is `E:/opal-model-series-20261010/`. Example bundles are under its `example/` directory and `pertspectra-banks/TNFA-positive/example/` and `TNFA-negative/example/`. See [the inference guide](../next_cycle/opal_model_series.md) for complete commands.

All saved within-stratum predictions replay exactly in the recorded environment. Combi-seq refits reproduce complete selection and epoch histories. This is software/retrospective prediction evidence, not biological interaction discovery, a noise ceiling, prospective savings, validated uncertainty or independent culture confirmation. SampleIndex labels aren't verified cultures. Weights, raw data and prediction arrays stay local pending derivative terms and publication review; the published v0.1.2 assets are unchanged.
