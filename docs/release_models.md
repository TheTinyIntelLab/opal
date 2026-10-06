# Released model references

The main v1 references are residual kernels. They predict population-mean effects from measured singles on the fixed assay scale and exact gene order in their manifests. They don't support arbitrary unseen interventions or arbitrary biological states.

| Directory in model archive | Scope | Data terms |
| --- | --- | --- |
| `models/norman-kernel` | Frozen exploratory development selection refitted on Norman development pairs. Its 38 query predictions were exported without scoring or fitting their held-out outcomes. It isn't the historical primary neural checkpoint. | Derived scPerturb v1.1 content, CC BY 4.0 |
| `models/emeraldbay-kernel` | Development-selected contextual chemical kernel using registered measured controls/singles and persisted fitted scales. Independent D-SPIN transfer failed against its source-selected reference. | Derived EmeraldBay content, CC BY 4.0 |

Unzip `opal-v1-reference-models.zip`. With the package installed, run:

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
