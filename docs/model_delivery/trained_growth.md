# Assay-local growth checkpoint delivery

This is an assay-local research model. The all-task promotion gate failed. The bundle is not biological confirmation, not a cold-gene model, and not a cross-context model.

The exported weights are the original validation-selected fold-0 checkpoints from the executed `opal-frontier-growth-v3` run. Fold 0 is a fixed deployment choice. It is not the winner of a test-score comparison. The pooled five-fold benchmark describes that training procedure. It is not a score of this single checkpoint.

The frozen delivery registration names the original growth-v1 scientific recipe. The executed growth-v3 artifacts use the lossless `frontier_packing_v2` loader. The separate training reproduction passed for all six tasks, matching all 168 saved states and prediction vectors. Export retains the original checkpoint bytes.

## Files and local bundle

- `src/trained_growth.py` loads a bundle and answers named-gene queries.
- `tools/export_trained_growth.py` copies the original fold-0 policy checkpoint for each of the six tasks.
- `tests/test_trained_growth.py` uses source-free fixtures.
- This note.

The local bundle, which is not committed, is `tmp/model-delivery-20261027/trained-growth/`.

- `manifest.json`
- `tasks/<task>/checkpoint.npz` , original checkpoint bytes
- `tasks/<task>/schema.npz` , gene names, dropped names, stored singles, TRAIN support, TRAIN degree

## What the manifest binds

Each biological task record uses the registered assay table: task order, context, source, accession, units, assay name, bank hash, endpoint, and endpoint quality. Editing a context string makes the bundle unloadable.

The endpoint is the mean of two infection-specific gamma measurements. Its quality is a frozen processed bank. That is not an independent-infection claim. The two readings are dependent assay measurements.

The information budget is 16000 TRAIN and 4000 VALIDATION charged pair labels inside the original fold. Support is TRAIN degree of at least 2. Validation rows, test identities, and test outcomes do not grant support.

`expected_source_hashes.identity` is `original_pre_fit_snapshot`. Those hashes are the expectation file written before the original fit opened banks. They are not a statement that the finished experiment never exposed those banks.

Per-query calibrated uncertainty is unavailable. No interval wrapper is fit. Conditional gene-sensitivity intervals stay aggregate evidence for the five-fold procedure.

Item rights for weight redistribution remain unresolved. Historical growth registration allows local scientific reuse and prohibits putting these checkpoints in Git, a wheel, or an sdist.

## Prediction

```python
from opal.trained_growth import TrainedGrowth
bundle = TrainedGrowth.load(
    'tmp/model-delivery-20261027/trained-growth',
    expected_manifest_sha256='fa8490abe83e5a5d0d0f8456639d11df16a2a6ca3ed440fb40bfd74459488e72')
model = bundle.model('horlbeck_K562')
genes = [g for g, supported in zip(model.gene_names(), model.support()) if supported]
print(bundle.predict('horlbeck_K562', [(genes[0], genes[1])], context='K562'))
```

Pass the manifest sha256. Checkpoint hashes alone do not stop one self-consistent manifest from being swapped for another. The context argument, when supplied, must be the assay context (`K562` for `horlbeck_K562`), not the task id and not another task's context.

A supported pair returns the original predictor's value. An absent gene, an unknown gene, a gene without two distinct TRAIN partners, a self-pair, a malformed name, or another context returns `status='abstain'` and `value=None`. The API does not accept replacement singles and does not infer singles from the query. Nonfinite stored singles or predictions are rejected.

`uncertainty()` returns no number.

The API accepts any supported pair and does not label whether a query was a training observation. Only the original held-pair populations have the benchmark scores below. It preserves the original arithmetic; exact replay uses the original numerical stack and query batch, without claiming historical numerical invariance was solved.

## Export command

Root runs this. It reads `tmp/frontier-execution-20261026/growth-v3`, copies six original checkpoints, replays saved predictions for identity only, and stops if the output already exists. It does not fit, score, open test outcomes, or consult the training reproduction.

```text
python tools/export_trained_growth.py --source tmp/frontier-execution-20261026/growth-v3 --output tmp/model-delivery-20261027/trained-growth
```

The process is one numerical thread, with a 600-second wall cap, 2 GiB sampled memory, an 80 MiB bundle cap, and a 1 GiB free-disk reserve, under root's watchdog. The reproduction has its own 900-second and 80 MiB caps and runs separately. Keep the printed `manifest_sha256` with the bundle.

Source-free tests:

```text
pytest tests/test_trained_growth.py
```

## Installation and reproduction

Install the package from this checkout. The recorded numerical stack was Python 3.14.0, NumPy 2.4.0, SciPy 1.16.3, scikit-learn 1.8.0 and Torch 2.9.1+cpu:

```text
python -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "numpy==2.4.0" "scipy==1.16.3" "scikit-learn==1.8.0"
python -m pip install .
```

The weights are already present in this workspace at `tmp/model-delivery-20261027/trained-growth/tasks/<task>/checkpoint.npz`. The bundle occupies 4,192,322 bytes. Public weight redistribution is not authorized by this delivery. A package install supplies code; it does not supply source data, the preserved run tree or weights.

To repeat training from the preserved, hash-matched fit inputs, use a fresh directory. This runs the entire original menu for six fold-0 procedures and requires the original code/environment binding. It does not open TEST targets or compute another score:

```text
python tools/retrain_frozen_growth.py --source-run tmp/frontier-execution-20261026/growth-v3 --output tmp/model-delivery-20261027/retraining-v2
```

The completed `retraining-v1` run is preserved. A clean Git clone alone does not contain its licensed local data or historical working-file custody; the runner refuses mismatches. [Source preparation and rights](../model_development_20261020.md) and the [original registration](../frontier_execution/growth_registration.md) document those prerequisites. Training used 16,000 TRAIN and 4,000 VALIDATION pairs per task, the unchanged rank-8 signed completion recipe, original seed and all original baselines. There is no additional rank, seed or hyperparameter search.

To export another identical local bundle, use a fresh output:

```text
python tools/export_trained_growth.py --source tmp/frontier-execution-20261026/growth-v3 --output tmp/model-delivery-20261027/trained-growth-v2
```

To check the delivered bundle against the original saved predictions:

```text
python tools/verify_trained_growth_delivery.py --bundle tmp/model-delivery-20261027/trained-growth --source-run tmp/frontier-execution-20261026/growth-v3 --output tmp/model-delivery-20261027/public-api-replay.json --expected-manifest-sha256 fa8490abe83e5a5d0d0f8456639d11df16a2a6ca3ed440fb40bfd74459488e72
```

To reproduce the aggregate benchmark report without rescoring outcomes:

```text
python tools/report_trained_growth_benchmark.py --source results/frontier-execution-20261026/growth-v3/results.json --output tmp/model-delivery-20261027/benchmark-replay.json
```

## Measured results and costs

The [complete benchmark](../../results/model-delivery-20261027/benchmark.json) preserves all sixteen fixed simple references, selected simple, kernel and neural families, three shuffled policies, both infection targets, coverage, MAE, bias, correlation and conditional gene sensitivity. These are original pooled five-fold results on exposed development sources:

| Task | Selected simple MSE | Completion MSE | Relative gain |
|---|---:|---:|---:|
| K562 | 0.00190091929 | 0.00179069074 | 5.80% |
| Jurkat | 0.000507630148 | 0.000486775245 | 4.11% |
| Environmental 1 | 0.000551581712 | 0.000412898098 | 25.14% |
| Environmental 2 | 0.000727352832 | 0.000518265563 | 28.75% |
| PARPi 2022 | 0.000308097243 | 0.000254004302 | 17.56% |
| PARPi 2023 | 0.000302940513 | 0.000260679297 | 13.95% |

Completion led the finite kernel/neural menu in every task and beat every fixed simple reference and shuffled policy. Jurkat still misses the unchanged 5% all-task rule, so the model is not promoted. Coverage was 100% on these registered populations. Negative conditional gene-sensitivity intervals describe this graph, not independent-study uncertainty. Historical frozen-weight external evaluations favored simpler references, and local retraining does not reverse that transfer failure. Technical pair effects remain confounded with biological interactions.

This training reproduction took 315.106 externally measured seconds and peaked at 648,269,824 bytes process-tree memory on one CPU thread. It retained 25,811,368 bytes before its final result receipt. Export took 3.037 seconds, peaking at 284,581,888 bytes. The [training receipt](../../results/model-delivery-20261027/training-reproduction-v2.json), [state/vector result](../../results/model-delivery-20261027/training-result.json), [export receipt](../../results/model-delivery-20261027/export-v1.json) and [API replay](../../results/model-delivery-20261027/api-replay-checkout-result.json) record actual execution. API replay covered 17,176 original queries with no new score.

New biological confirmation remains blocked by source evidence. See [COLO1 and additional source routes](source_routes.md), [the corrected DOI and Parrish route](source_routes_v2.md), and [HAP1 and yeast qualification](root_alternative_sources.md). No response data from these unqualified sources entered training.
