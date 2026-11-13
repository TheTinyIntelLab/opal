# Prediction and model availability

The GitHub source release contains inference code and evidence. Download trained models separately from [Opal Growth](https://huggingface.co/TheTinyIntelLab/opal-growth), [Opal Norman](https://huggingface.co/TheTinyIntelLab/opal-norman) and [Opal EmeraldBay](https://huggingface.co/TheTinyIntelLab/opal-emeraldbay). Preserve each repository's file layout and check its manifest and model-specific license notices.

## Growth prediction

The retained local delivery contains one original validation-selected fold-0 checkpoint for each of six assay-local tasks. The fold was fixed in advance for export; it was not chosen using test scores. The pooled five-fold benchmark is not a score of that individual checkpoint.

```python
from opal.trained_growth import TrainedGrowth

bundle = TrainedGrowth.load(
    "path/to/trained-growth",
    expected_manifest_sha256="fa8490abe83e5a5d0d0f8456639d11df16a2a6ca3ed440fb40bfd74459488e72",
)
model = bundle.model("horlbeck_K562")
genes = [g for g, ok in zip(model.gene_names(), model.support()) if ok]
results = bundle.predict("horlbeck_K562", [(genes[0], genes[1])], context="K562")
assert results[0]["status"] == "predicted"
print(results[0]["value"])
```

Use that hash only for the original October 27 bundle. A future model release must provide its own manifest receipt. Supported names and context identifiers come from the bundle, rather than from arbitrary aliases. The two Horlbeck contexts are K562 and Jurkat; the environmental and PARPi cohorts each have their own task-specific K562 context.

Each result reports status, reason, value, task, context and uncertainty. Unknown or absent genes, self-pairs, malformed names, fewer than two distinct training partners, or a different context produce an abstention with no value. Queries use the stored original measured singles. They cannot replace those singles or infer missing ones from a query. `uncertainty()` reports that calibrated per-query uncertainty is unavailable.

The original [delivery guide](model_delivery/trained_growth.md), [registration](model_delivery/registration.md) and [training-reproduction record](model_delivery/training_reproduction.md) preserve the exact source, support, fold and custody contracts. Their local-only distribution statements are historical project policy; upstream dataset rights still require separate resolution for a public full-reference bundle.

## Expression prediction

With a compatible separately obtained expression checkpoint:

```python
from opal import load_model

model = load_model("path/to/expression-checkpoint")
result = model.predict("TARGET_A", "TARGET_B")
```

Replace the placeholder targets with exact supported names. Contextual checkpoints also require the registered `context` argument. Output effects use the manifest's gene order and expression scale; an unknown target, unsupported context or mismatched supplied gene order is rejected. The [historical model cards](model_cards/opal-base.md) describe the separate source-specific model-series contracts.

Norman and EmeraldBay reference models have recorded CC BY 4.0 source provenance in [third-party notices](../THIRD_PARTY_NOTICES.md). The growth release records maintainer authorization for original Opal contributions, separate upstream reference terms and remaining source-license scope uncertainty in its [license notice](https://huggingface.co/TheTinyIntelLab/opal-growth/blob/main/LICENSE.md). Code availability, checkpoint availability and biological generalization are separate claims.
