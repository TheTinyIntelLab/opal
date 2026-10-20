# Opal

Opal is The Tiny Intelligence Lab's research codebase for predicting biological perturbation responses. The current executed task predicts measured double-CRISPRi growth effects from matched single growth and other pairs in K562 and Jurkat. This functional endpoint follows the closed expression experiment; it does not establish improved transcriptomic prediction. The frozen predictor failed independent cross-screen superiority testing, so its supported use remains within the original assay.

The first-principles reset is governed by the sole authoritative [plan.md](plan.md). Three specialist investigations inspected [official architectures and licenses](docs/research_20261020_architectures.md), [data and evaluation](docs/research_20261020_data.md), and [identifiability and measurement error](docs/research_20261020_identifiability.md). Existing neural, kernel, bilinear and transfer results informed the choice; model complexity is not a selection criterion.

## Current measured-growth result

A new public screen supplied 81,466 canonical pairs in both contexts, with 48,879 training, 16,293 validation and 16,294 globally held test pairs. Validation-selected independent-context completion reduces test MSE by 26.44% in K562 and 14.82% in Jurkat relative to calibrated gene-main-effect references. The joint shared/private candidate fails its stronger-reference promotion criterion, so the independent models are retained. This is BASELINE-LEVEL relative to completion references, with no frontier or independent-laboratory claim.

| Growth context | Main-effect MSE | Selected completion MSE | Joint candidate MSE |
| --- | ---: | ---: | ---: |
| K562 | 0.00187049 | 0.00137593 | 0.00136626 |
| Jurkat | 0.00050049 | 0.00042630 | 0.00045576 |

The [existing decision report](docs/model_development_20261020.md#executed-follow-up-measured-growth-pair-completion) records actual acquisition, licensing qualifications, fixed splits, conditional uncertainty, all baselines and the continuation decision. [Frozen evidence](results/fitness-20261020/) records the new experiment. Source data and model assets remain local with attribution. The official author dataset is CC BY 4.0; exact equivalence to the acquired GEO mirror remains unverified, so no unrestricted raw-data redistribution is claimed.

The experiment used 18.72 seconds, one CPU worker, about 916 MiB peak working set and 40 MiB run artifacts. Named-gene inference uses the retained independent-context references:

```python
from opal.models.fitness_inference import FitnessPredictor

model = FitnessPredictor("C:/Users/USER/Desktop/opal-next-data-20261020/fitness-reference-model")
growth = model.predict([("AARS2", "AATF")], context="K562")
```

These are measured-growth predictions for supported genes in the two trained contexts. Unknown genes, new contexts and identical-gene pairs are refused. Raw expression, biological mechanisms and calibrated uncertainty are not outputs.

To reproduce a NEW registered run after acquiring the hash-matched official source:

```sh
python tools/prepare_fitness.py --source /path/to/GSE116198_sgRNA_pair_phenotypes.txt.gz --output /path/to/new-prepared
python tools/study_fitness.py freeze --data /path/to/new-prepared --output /path/to/new-run
python tools/study_fitness.py run --output /path/to/new-run
python tools/export_fitness.py --run /path/to/new-run --output /path/to/new-model
```

The source URLs and acquisition hashes are in results/fitness-20261020/horlbeck_acquisition.json. No repeat run is needed to use the existing assets. The independent screen test below is complete; no repeat fit or rank/seed search is justified.

## Independent external validation

The unchanged K562 predictor was frozen before evaluation on untreated/DMSO growth from the environmental GSE312636 and PARPi GSE284018 studies. Four cohorts each retain two independent infection replicates; they share laboratory lineage and do not establish independent-laboratory validation. Strict novel panels contain 130/130/423/411 supported-gene pairs absent from the entire original eligible bank, before any retraining or external calibration.

**FAILED HYPOTHESIS** for unchanged-model external superiority. Completion MSE is 0.0710602/0.0504016/0.1060987/0.1067118, versus frozen main effects 0.0695619/0.0493793/0.1048554/0.1055719. Completion loses by 1.08-2.15%, and original-single minimum performs better in every cohort. Conditional gene-resampling intervals include zero. Overlapping original training, validation and test pairs are reported separately, not relabeled as novel generalization.

External measured singles are much more informative than original singles, indicating a substantial source/reagent/assay shift. Their minimum baseline is an extra-information diagnostic, not an adapted Opal result. No model was refitted. Preserve the within-source predictor and close this transfer claim; any assay-adaptation hypothesis needs separate adaptation data and untouched confirmation data.

[The existing research report](docs/model_development_20261020.md#independent-external-reference-growth-validation) explains qualification, terms, replicate handling, failure diagnosis and limits. [Frozen protocol and results](results/fitness-external-20261020/) preserve the quantitative record. Raw counts, derived banks and model assets stay local. The existing README is updated in place.

## Closed expression candidate

**FAILED HYPOTHESIS.** Hierarchical gene-local composition learned signed overlap and cancellation terms from measured singles, with gene coefficients shrunk toward a pooled response law. Nested development evaluation did not improve over matched affine calibration or existing kernels.

| Development source | Split | Composition MSE | Inner-selected baseline MSE |
| --- | --- | ---: | ---: |
| Norman, 88 fitting pairs | Pair | 0.006213 | 0.006089 |
| Norman, 88 fitting pairs | Partner | 0.006343 | 0.006322 |
| Expanded Wessels, 110 fitting pairs | Pair | 0.008510 | 0.007115 |
| Expanded Wessels, 110 fitting pairs | Partner | 0.009281 | 0.009195 |

These are new development folds, not the historical neural comparison or independent confirmation. Each baseline is selected inside its outer training population. Partner folds overlap, so primary MSE scores each pair after averaging its leak-free predictions across those folds; it is not the mean single-fit fold loss. Historical evaluator response files were not read. Both datasets already have research exposure, and their units differ, so do not pool these scores. [The decision report](docs/model_development_20261020.md) explains uncertainty, ablations, resources and the decision to close this candidate. [Aggregate results and the frozen protocol](results/composition-20261020/) retain all comparisons and selection histories.

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
