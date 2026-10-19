# Opal

Opal is The Tiny Intelligence Lab's research codebase for predicting biological perturbation responses. The current objective is a competitive specialized model for population-average double-perturbation effects given matched measured singles and examples of other pairs. Independent biological confirmation is still absent.

The first-principles reset is governed by the sole authoritative [plan.md](plan.md). Three specialist investigations inspected [official architectures and licenses](docs/research_20261020_architectures.md), [data and evaluation](docs/research_20261020_data.md), and [identifiability and measurement error](docs/research_20261020_identifiability.md). Existing neural, kernel, bilinear and transfer results informed the choice; model complexity is not a selection criterion.

## Current candidate result

**FAILED HYPOTHESIS.** Hierarchical gene-local composition learned signed overlap and cancellation terms from measured singles, with gene coefficients shrunk toward a pooled response law. Nested development evaluation did not improve over matched affine calibration or existing kernels.

| Development source | Split | Composition MSE | Inner-selected baseline MSE |
| --- | --- | ---: | ---: |
| Norman, 88 fitting pairs | Pair | 0.006213 | 0.006089 |
| Norman, 88 fitting pairs | Partner | 0.006343 | 0.006322 |
| Expanded Wessels, 110 fitting pairs | Pair | 0.008510 | 0.007115 |
| Expanded Wessels, 110 fitting pairs | Partner | 0.009281 | 0.009195 |

These are new development folds, not the historical neural comparison or independent confirmation. Each baseline is selected inside its outer training population. Historical evaluator response files were not read. Both datasets already have research exposure, and their units differ, so do not pool these scores. [The decision report](docs/model_development_20261020.md) explains uncertainty, ablations, resources and the decision to close this candidate. [Aggregate results and the frozen protocol](results/composition-20261020/) retain all comparisons and selection histories.

The experiment completed in 33.5 seconds on one CPU/BLAS worker, with about 471 MiB peak working set and 63 MiB retained artifacts. A first attempt exceeded its output cap by retaining every baseline's prediction arrays; its unchanged compact replay reproduced all fold scores exactly. Neither overlap features nor more model capacity earned promotion. The next model requires additional compatible pair-response evidence or a separately justified endpoint and dataset extraction, rather than another local architecture sweep.

## Reproduce and use the candidate

Python 3.11 through 3.14 are supported. Install the package with research dependencies in a suitable environment:

```sh
python -m pip install -e ".[data,research,dev]"
python -m pytest -q
python tools/study_composition.py freeze --output /path/to/new-attempt
python tools/study_composition.py run --output /path/to/new-attempt
```

The study needs the existing prepared norman-log2 and wessels-expanded-protected-log2 banks under data/prepared. They are not shipped. See [source acquisition/preparation history](docs/engineering_record_20261015.md), [data guide](data/README.md), [source registry](data/source_registry.json), and [third-party notices](THIRD_PARTY_NOTICES.md). The frozen runner verifies exact data/code hashes and refuses completed-output overwrite. Reproduction of the compact run requires the runner revision recorded by its protocol; future edits require a new registration. Numerical replay is conditional on platform/library arithmetic.

The new model has a small standalone API:

```python
from opal.models.composition import Composition

# singles: [training_pairs, 2, genes]; responses: [training_pairs, genes]
model = Composition(alpha=1.0).fit(singles, responses)
prediction = model.predict(query_singles)
model.save("composition.npz")
restored = Composition.load("composition.npz")
```

Inputs are matched-control effects on the same scale and in exactly the same gene order. The API checks finite values, dimensions and gene count; callers must enforce gene identities and assay compatibility with their schema manifest. It does not infer missing singles, transform doses, predict new contexts or return calibrated biological intervals. Local all-development checkpoints and separate gene-order manifests exist under the artifact root recorded in [the manifest](results/composition-20261020/artifact_manifest.json); weights and expression arrays are not redistributed.

## Preserved research

The previous engineering workstream is closed. Its complete [dated record](docs/engineering_record_20261015.md), [historical model guides](docs/models.md), [series v1](docs/next_cycle/opal_model_series.md), [series v2](docs/next_cycle/opal_model_series_v2.md), manuscript evidence and reproducibility manifests remain intact. Published v0.1.2 assets are unchanged. The [frontier evidence index](docs/frontier/README.md) points to earlier source qualifications, technical measurements and negative experiments; its dated specifications are not another active work queue.

Six obsolete planning/dispatch documents were removed after individual inspection. Scientific specifications, provenance, outreach custody and results were retained; ignored private roadmap/context snapshots were preserved outside the active tree. [The cleanup audit](results/composition-20261020/planning_audit.json) records the disposition. No private correspondence, raw data, restricted weights or credentials belong in commits.

Code is licensed under [LICENSE](LICENSE); dataset and external-resource terms remain separate.
