# Independent measurement evidence package

`tools/check_frontier_readiness.py` checks whether reviewed local evidence is complete enough to draft a separate `frontier_bridge_confirmation_v1` contract. It does not score expression, validate reviewer judgments, establish biological confirmation, or authorize laboratory execution. A completed software check cannot replace the collaborator's empirical covariance/power review.

Create a blank package in a new directory:

```text
.venv/Scripts/python.exe tools/check_frontier_readiness.py --template --output results/opal_frontier_bridge_v1/measurement-template-<unique-id>
```

The blank package intentionally returns `measurement_not_ready`. Copy reviewed evidence files into that package directory, supply their SHA256 hashes, and fill each gate's reviewer, review date, explicit approval and details. Validate the completed package into another new directory:

```text
.venv/Scripts/python.exe tools/check_frontier_readiness.py --package <package-directory>/measurement_package.json --output results/opal_frontier_bridge_v1/measurement-check-<unique-id>
```

Both commands refuse overwrites. A completed negative readiness check exits successfully because it produced an honest decision. The receipt lists each failed requirement and verified evidence hash. Evidence paths must stay inside the package, so its provenance can travel with it. Neither command contacts collaborators, registers a study, buys compute, acquires cultures, or redistributes data.

The package schema is `frontier_bridge_measurement_readiness_v1`; [measurement_readiness_schema.json](measurement_readiness_schema.json) defines its structure and [measurement_readiness_template.json](measurement_readiness_template.json) contains every gate and required detail field. The structural schema accepts null template values; the CLI applies the stricter readiness rules and verifies evidence files. Evidence records use `{"path":"relative/path","sha256":"actual file hash"}`. All gates require approved reviewer evidence, and unknown values remain null until evidence exists.

| Gate | Evidence and required details |
| --- | --- |
| Independent initiation | Each culture lists `culture_id`, `context_id`, distinct `initiation_id`, `initiation_date` and `provenance_record`. Distinct SampleIndex labels with one initiation fail. |
| Pretreatment contexts | Each context lists `context_id`, `pretreatment_definition` and `definition_freeze_record`; definitions were frozen before treatment. |
| Randomization | Reviewed randomized plate/block assignments preserve actual culture and assignment provenance. |
| Literal interventions | Each assignment lists `assignment_id`, `intervention_id`, `reagent_id`, physical `dose`/`dose_unit`, `treatment_duration`/`duration_unit`, and `initiation_mode` (`simultaneous` or `sequential`). Assignment equivalence requires review. |
| Independent references | Each match lists `culture_id`, `control_match_ids`, `single_match_ids`, `reference_initiation_ids`, `endpoint_scale` and `normalization_order`. Independently measured references and shared-reference covariance must be documented. |
| Counts and cost | `counts` records observed positive integers for `contexts`, `cultures`, `pairs`, `genes`, `control_outcomes`, `single_outcomes` and `bridge_outcomes`. `campaign_costs` records `distinct_bridge_identities`, `repeated_bridge_outcomes`, `control_outcomes`, `single_outcomes`, `assay_cost` and `currency`; review must support the sampling design. No powered numerical minima are invented here. |
| Release rights | Record the actual license, permitted uses, research authorization and completed redistribution review. Completed redistribution review does not imply redistribution permission. |
| Prospective information | Freeze the full held query identities, bridge identities, global exclusion contexts, allowed query controls/singles and freeze record before query scoring. Reversed query/bridge identities cannot overlap. The reviewer must verify assignment-equivalent exclusions across all contexts/cultures and campaign-wide bridge accounting. |
| Pilot covariance | Identify the empirical pilot and covariance method; retain shared-reference covariance and its reviewed evidence. |
| Power review | A biological collaborator reviews a finite positive useful margin, metric, functional endpoint, empirical power, positive culture sample sizes by context and assay cost. The plan's illustrative workload cannot populate these fields. |
| Interval validation | Declare the estimand and claim-specific support; coverage and null type-I validation each record empirical results, reviewed acceptance criteria, replicate counts and explicit pass decisions. Fixed-panel culture inference and broader population inference require different support. |
| Owner arrangements | Identify the measurement owner and retain actual laboratory, compute and registration arrangements for review. These records do not cause the CLI to execute those arrangements. |

`ready_to_draft_confirmation_contract=true` means the package passed these local completeness checks. A separate reviewed contract still needs claim-specific design, thresholds, approved endpoints, exposure rules and execution arrangements before confirmation scoring. Independent biological confirmation and frontier claims remain false in every readiness receipt.
