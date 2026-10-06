# Prepared model references

The planned 0.1.1 references are residual kernels. Publication is paused; current local exports are for review. They predict population-mean effects from measured singles on the fixed assay scale and exact gene order in their manifests. They don't support arbitrary unseen interventions or arbitrary biological states.

| Directory in model archive | Scope | Data terms |
| --- | --- | --- |
| `models/norman-kernel` | Frozen exploratory development selection refitted on Norman development pairs. Its 38 query predictions were exported without scoring or fitting their held-out outcomes. It isn't the historical primary neural checkpoint. | Derived scPerturb v1.1 content, CC BY 4.0 |
| `models/emeraldbay-kernel` | Development-selected contextual chemical kernel using registered measured controls/singles and persisted fitted scales. Independent D-SPIN transfer failed against its source-selected reference. | Derived EmeraldBay content, CC BY 4.0 |

Unzip `opal-0.1.1-reference-models.zip`. With the package installed, run:

```python
from opal import load_model
model = load_model("models/norman-kernel")
prediction = model.predict("AHR", "KLF1")
effect = prediction["effect"]
```

For EmeraldBay, inspect `model.contexts` and `model.targets`, then choose a registered context with both singles available. Call `model.predict(a, b, context=context)`. A context in the reference bank doesn't mean its combinations were training labels. Read the returned support metadata.

Manifests preserve gene order, reference fingerprints, fitted state, checksums, and scope. Don't equate effects with absolute counts or silently reorder genes. Measured reference banks are part of the data-access and memory cost. Unknown interventions and unsupported contexts require compatible references and a new validation protocol.

Historical neural models remain reproducible through the original configs, but aren't the main released references. Wessels/D-SPIN, failed learned state models, and native CPA/GEARS weights aren't bundled. Historical primary and development selection records remain distinct.

Intervals and support distances are diagnostic. Source-calibrated nominal coverage failed to transfer on D-SPIN, so no calibrated OOD abstention guarantee is provided.

## Experimental routes outside the main model archive

The baseline-conditioned state route accepts a processed compatible baseline, exact gene order, assay, scale, and interventions supported by source responses. Unknown identities and doses remain unsupported. Measured-single effects are relative to the baseline, which is added once for absolute reconstruction. Two-seed NumPy exports preserve fitted preprocessing and heads without evaluator labels, but learned state models lose to simple references in the tested development comparisons.

Estimated-single inference has a separate schema and reports whether each single was measured or estimated, with its source contexts. It abstains when the exact intervention/dose has no source reference. Observed-single calibration doesn't apply to this route, and error cancellation limits interpretation of its pilot gains.

Native CPA exports require their separate recorded Python 3.10/CUDA environment and `tools/predict_native_cpa.py`; they aren't bundled in the main release. Resource costs include references, dependencies, external graphs, and pretrained resources, rather than only the fitted head. No route supplies validated OOD abstention, unseen-dose prediction, higher-order composition, single-cell distributions, or a mechanistic explanation.
