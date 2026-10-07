# Prepared model references

The 0.1.2.dev1 successor adds a pure-sum option to the existing neural correction and response-source exclusion checks across fitting, preprocessing, pretraining and calibration. It retains the state/context implementations and original schedules. None of the new pair or learned-state components passed the matched development gate, so historical reference models remain the practical references.

Local `results/stage1-20261007/exports-v2/` contains last-outer-fold inner-selected shared/pair audit ensembles for engineering verification, not newly validated models. Scored predictions replay exactly after matching float64 preprocessing; fitted-weight restoration agrees within `4.45e-16`. `exports-v1` remains as the superseded float32 replay with Norman drift around `2.44e-7`. These genetic exports and their reference banks remain local. No chemical identity transfer, independent joint context transfer, biological interaction recovery or calibrated OOD interval is established.

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

## Experimental dose interpolation

Version 0.1.2.dev2 adds a fixed log-dose reference within the existing state module. It predicts a population-mean single effect for an exact known compound and biological line at 24 hours, on the registered 512-gene sci-Plex schema. It has no learned state encoder. The two scored interpolation exports preserve source-selected ridge and kernel configurations, respectively; they are engineering replays, not newly validated release models.

```python
import opal
model = opal.load_model("results/dose-design-20261007/exports-v1/dose-100")
result = model.predict("A549", "(+)-JQ1", 100, endpoint_hours=24)
```

Dose values are explicitly nM. Unknown compounds/states, changed endpoints and extrapolation outside source support raise errors. Missing profiles cannot become zero effects. Outputs contain gene order, source doses and a development evidence label; intervals are unavailable. Source effects already include matched vehicle subtraction, so the model returns an effect and does not add a baseline. Safe exports verify schema/checksum and restore every scored interpolation prediction exactly. Each reference is about 0.85 MB; measured batches of 72 predictions take 3.80 and 5.04 milliseconds on the reused CPU environment. This route does not infer chemical identity, new state effects or combinations.

The dev3 annotation study is research tooling, not an additional inference model. Neither identity nor shared-target transfer passes the development screen; publisher annotations do not authorize unsupported compound queries through the dose model. Its support checks and unavailable intervals remain in force.

The dev4 global chemical-pair/whole-state study also adds research tooling rather than an inference route. Pair-product and mixed kernels fail against the source-selected shared/single references; no new pair head or encoder is adopted. Global pair identity includes every dose, while each permitted measured query single still requires its exact matching dose. Development query-reference adaptation is explicit, and the study supplies no calibrated intervals or independent confirmation.
