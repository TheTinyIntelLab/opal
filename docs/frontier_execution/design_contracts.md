# Design contracts

`opal.frontier_design_contracts` is the prerequisite API `opal-frontier-design-contracts-v1`. It checks declared guide identity, canonical splits, and one caller-supplied arithmetic recipe. It does not qualify COLO1, verify an assay endpoint, prove independent measurement, recover guide nuisance, or fit a model. The historical all-task gate remains a preserved failure. This module does not rescore it, and it does not replay or refit historical checkpoints.

## Limits

| Record | Meaning |
| --- | --- |
| `colo1_qualified`, `source_endpoint_qualified`, `guide_mapping_qualified`, `independence_qualified` | False. Actual-source endpoint, guide-mapping, and independence qualification stay blocked. |
| `formula_is_verified_source_processing` | False. The log contrast is a fixture recipe, not verified pyCROQUET or day-14 processing. |
| `provenance_is_proof` | False. The four provenance flags are declarations. Accepting them does not prove the laboratory fact. |
| `library_total_excludes_held_outcomes` | False. `library_total` is used exactly as declared. A precomputed denominator can already contain held-pair outcomes; the helper cannot see how it was formed. Provenance flags do not close that gap. |
| `frozen_train_partner_support_enforced` | False. Fit-view support means a perturbation pair's endpoints have declared measured singles (`declared_measured_single_prerequisite`). The frozen tournament rule, two distinct TRAIN partners for each query gene, is a separate model check and is not applied here. |
| `ntc_pair_proven_unperturbed` | False. NTC/NTC is only a declared reference class. `unperturbed_class` is false for every class, including NTC/NTC. A nonessential target remains an intervention, not an NTC and not an unperturbed control. |
| `shared_controls_are_replicates` | False. Shared control ids are dependence keys. Component size is not a replicate count. |
| `guide_nuisance_recovery_claim`, `model_claim` | False. |
| `historical_all_task_gate` | `failed`. |

## Identity and splits

Each guide declares an id, an ACGT sequence, a target or control id, and a role: `perturbation`, `nonessential_target`, or `ntc`. Each construct declares both arms and an orientation. The orientation must match the arms; it is not inferred from a name. Guide variants and reversed orientations share one canonical key.

A public `DesignSplit` is checked against the library it is applied to. Every construct of a canonical key must carry the same partition, including TEST. Copying a same-length split from another library, or forging one variant into TRAIN and another into TEST, is refused. Held-gene ids must be declared library genes. Every construct incident to a held gene, including variants and reversed orientations, is in the `held` partition and therefore out of TRAIN, VALIDATION, and TEST. A `held` label with no incident declared gene is refused.

Degree in `train_graph` counts distinct TRAIN canonical pairs after that removal. Construct copies do not increase it. That degree is not frozen-policy support.

## Fit view

`make_fit_view` takes TRAIN and VALIDATION indices, labels, missing masks, and the measured-single support mask. TEST and held positions are returned as indices only. TEST targets are not arguments. Missing label positions are not read and are not stored as zeros. An accepted provenance tuple is sealed with `declarations_are_proof = false`.

## Fixture contrast

The only arithmetic helper is `caller-declared-raw-count-normalized-log2-median-reference-v1`:

```text
norm(count, total) = log2(pseudocount + count * scale / total)
contrast = norm(sample) - median(norm of declared TRAIN independent controls)
```

`pseudocount` and `scale` have no defaults. For an even control count, the median is the mean of the two central sorted norms. A raw count of zero is observed. A missing flag is not a zero and is not read.

With `pseudocount = 1` and `scale = 4`, TRAIN control counts 0 and 3 at library total 4 have norms 0 and 2. Their median is 1. A sample count of 0 at that total contrasts to -1, and a sample count of 3 contrasts to 1. A sample count of 1 at a declared total of 2 contrasts to `log2(3) - 1`, not to the total-4 result. The helper trusts the declared total and does not rebuild it. The total must cover the raw count; that check does not prove the total excludes held outcomes. A column tagged `held_pair` and `held` is not read. Any missing declared TRAIN control makes every contrast missing; the observed subset is not reused and a held outcome is not a substitute. A value tagged `train_reference` is admitted even if that tag is false. The API cannot detect that misdeclaration.

`fit_train_scaler` uses the same membership rule. Its `n` is the number of declared TRAIN reference rows used in the mean and root-mean-square scale. It is not a replicate count. Rows tagged `held_pair` are not read.

Selection among declared candidates breaks ties by validation score, complexity, name, then family. TEST scores are not arguments. This is not a fitted model.

Root's review added refusal of nonfinite calculated log counts and scaler state/output, including overflow from otherwise finite inputs. Read-only NumPy buffers prevent ordinary accidental assignment; they are not a security boundary against a caller deliberately replacing records or changing array flags. No declaration or digest certifies the laboratory origin of supplied numbers.
