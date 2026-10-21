# Workstream C evaluator audit results

Completed on 21 October 2026 from clean main `505af4d9dd6a77f73b0e8952c9f4ed8cc12725a6`. The bounded synthetic audit supports the tested arithmetic, pair/block grouping, training-only transforms, checkpoint selection and low-rank recovery. It also reproduces invalid-checkpoint acceptance and measures substantial undercoverage when conditional gene intervals are interpreted as population intervals. Neither finding changes historical scores or establishes biological replication. SPIDR and SCHEMATIC response tables and headers remained unopened; no network request or biological response file was used.

## Coverage inventory before new outcomes

Read root plan/README, block protocol/review, flexible-single protocol and relevant implementation/test files. Existing `test_block_completion.py` checks hand baselines, symmetry, roundtrip, duplicate pairs, basic held-label isolation, guard callbacks, streamed/dense ridge agreement and constant-target completion recovery. `test_scalar_screen.py` checks schema, known block hashes, support, outcome-independent strata, joint-context permutations, arithmetic/bootstrap multiplicities and hash-guarded synthetic loaders. `test_block_study.py` checks gate vetoes, file integrity, failed prerequisites and caps. Existing fitness tests already include a nonconstant signed-factor recovery case for the different two-context FitnessCompletion implementation; the new scalar case is not claimed as the project's first signed-factor test.

Existing reagent/processed-gamma tests cover construct-first minimum, exact guide/arm/transcript matching, controls, blanks and valid zero counts, sequence strata, serialization, duplicate constructs, per-replicate centering and grouped splitting. Flexible-single tests cover moments, normal equations, fixed maps, nonlinear capacity and unsupported queries. Native-resume tests have synthetic inherited-fit recovery, missing/duplicate/tampered inheritance, scored/ordering/path errors and live-PID refusal. Existing generic contract tests include pair/partner grouping; source-specific modules also contain isolation tests. This was a code inventory, not a rerun of all these suites, and assertions in these older suites are not counted as new independent evidence.

## Frozen execution and preserved failures

The protocols are [v1](wave3_evaluator_protocol_v1.md), [v2](wave3_evaluator_protocol_v2.md) and [v3](wave3_evaluator_protocol_v3.md). Each attempt binds its own code/test/protocol and generated input hashes before tests. V3 additionally hashes both earlier attempts. No frozen file was edited after its attempt. All receipts, logs, selection records and per-trial results are in [results/research-wave3-evaluator](../../results/research-wave3-evaluator/).

V1 stopped at 15 passes/one test-authoring failure: my expected construct-weighted minimum was -2.5 rather than the correct -3.25. Its launcher-only memory/CPU sampling is invalid for the scientific child; retain the 4.4824s wall receipt but do not treat 3.5MB/zero CPU as scientific usage. V2 corrected the fixture and resource accounting, then stopped at 24 passes/one test-authoring failure caused by changing the independent expected block-hash namespace during file versioning. Both attempts stopped before recovery/coverage experiments. V3 restores the frozen block namespace, retains v2 inputs and passes all 25 cases in 1.00s. Failures remain under v1/v2 and are not hidden as successful prerequisites.

Commands, each from the project root with the required Python:

```powershell
.venv/Scripts/python.exe tools/wave3_evaluator_v1.py freeze --output results/research-wave3-evaluator/v1
.venv/Scripts/python.exe tools/wave3_evaluator_v1.py run --output results/research-wave3-evaluator/v1
.venv/Scripts/python.exe tools/wave3_evaluator_v2.py freeze --output results/research-wave3-evaluator/v2
.venv/Scripts/python.exe tools/wave3_evaluator_v2.py run --output results/research-wave3-evaluator/v2
.venv/Scripts/python.exe tools/wave3_evaluator_v3.py freeze --output results/research-wave3-evaluator/v3
.venv/Scripts/python.exe tools/wave3_evaluator_v3.py run --output results/research-wave3-evaluator/v3
```

Run exit codes were 1, 1 and 0 respectively; freeze commands succeeded. These paths deliberately refuse another run/overwrite. Reproduction requires a new output directory with the unchanged frozen v3 code, and its prior-evidence v1/v2 directories present. No test cache is required. Scientific arrays/JSON are deterministic under the recorded dependency environment; timestamps, performance receipts and ZIP serialization bytes are not claimed portable.

After the final hash audit, the two obsolete failed test modules were archived byte-for-byte as `v1/test_wave3_evaluator_v1.py.frozen.txt` and `v2/test_wave3_evaluator_v2.py.frozen.txt`, then removed from live pytest discovery. [archival_v1.json](../../results/research-wave3-evaluator/archival_v1.json) binds the original paths and SHA256 values. Their frozen protocols/results were not modified. To replay either historical attempt's original signature check, temporarily restore its archived module to the recorded test path; the live v3 verifier needs no restoration. This packaging step prevents the two known test-authoring mistakes from failing the default suite, while preserving the exact failed source. The delivery inventory was created before archival and remains a valid timestamped inventory of that earlier state; the separate archival manifest records the additions/moves.

## Actual-implementation results

All fits use the same 260 training, 85 validation and 90 test canonical pairs, with measured singles identical across comparators. The twelve-reference policy is selected only on validation and written before each scenario's test metrics. In the additive null, additive singles and all aligned/shuffled completed predictions have exactly zero test MSE, so there is no improvement over the correctly specified reference.

For the nonconstant signed rank-two interaction, additive MSE is 2.037090992 and validation-selected calibration MSE is 2.038466244. Aligned completion MSE is 0.425768773, or 0.209008226 times unmodeled interaction MSE, passing the prospectively fixed ratio-below-.25 diagnostic. Control MSEs for seeds 20261023/24/25 are 1.809603231, 1.989291430 and 1.977375313. All three have 99.23% permutable training edges and moved fractions 88.85%, 90% and 85%. This tiny synthetic fit is not a historical rank-eight gate test or a promoted model.

The pair-technical-confounding case deliberately has identical observed data to the signed low-rank case with zero biological interaction. It needs no second fit: identical observations and deterministic procedures imply identical predictions. This is an observational nonidentifiability demonstration, not an empirical estimate of technical confounding in Horlbeck. The processed-gamma fixture independently exercises guide/arm nuisance, three versus one orientation weights, missing exact-control support and no zero filling; it does not train a guide-aware completion procedure.

New executable cases independently reconstruct pair and full block identities, group reciprocal/construct variants, verify all incident cold-gene outcomes are excluded by partner_splits, and show explicitly that source-keyed hashing is not source holdout. Direct dense ridge coefficients agree with the actual streamed implementation; test sentinels leave selection/predictions unchanged. The flexible map excludes unqueried-gene covariates. Completion is equivariant at .01/100 target scale, symmetric, correctly chooses its lowest recorded validation checkpoint, serializes exactly and refuses unsupported genes. Equal-pair metrics differ from construct-weighted metrics, replicate disagreement vetoes an aggregate win, and a below-5% case cannot pass. The exact-threshold case checks arithmetic near .05 without adding a tolerance or retrospectively changing historical gate behavior.

## Empirical conditional-interval behavior

Each mechanism has 128 independent synthetic trials, 512 gene draws per trial and a secondary 512-draw pair percentile interval. Mechanisms share trial seeds as common random numbers, so their estimates are correlated with each other. Candidate/baseline squared-loss differences are independently checked against generated known-truth deltas. The target estimand is the population mean difference zero across independent screens, not each screen's realized finite-panel average. No bootstrap draw was empty.

- Iid edge noise: gene coverage 128/128, 100%, Wilson 95% bounds 97.09–100%; pair coverage 125/128, 97.66%. Mean gene interval width .3150, empirical mean-estimator SD .04460.
- Shared-gene noise: gene coverage 115/128, 89.84%, bounds 83.40–93.97%; pair coverage 34/128, 26.56%. Mean gene width .9579, estimator SD .25286.
- Four shared edge-cohort effects: gene coverage 13/128, 10.16%, bounds 6.03–16.60%; pair coverage 14/128, 10.94%. Mean gene width .1787, estimator SD .48534.
- Global screen effect: gene coverage 1/128, .78%, bounds .14–4.29%; pair coverage 0/128. Mean gene width .03150, estimator SD 1.01411.

These results reject treating the implemented gene sensitivity interval as a generally calibrated 95% population interval, even for this shared-gene design. They support the existing plan's conditional-only interpretation. The four-cohort mechanism is artificial and interleaved by row index; these numerical rates cannot be transferred to real studies or used to estimate their effective replication. Full per-trial intervals, pair bounds and empty-draw counts are retained rather than summarizing only favorable mechanisms.

## Checkpoint defect and opt-in successor

The actual ScalarCompletion loader accepts a packed NaN operator, NaN scale and negative scale; these reproduce invalid or nonsensical prediction state. Existing refusals do reject unsupported format, malformed packed length, object arrays and truncated ZIP input. Historical hash-bound runner paths limit accidental artifact drift, so these probes do not show that any historical checkpoint was invalid or any historical score changed.

`strict_load_completion` in [the owned v3 tool](../../tools/wave3_evaluator_v3.py) adds finite-operator and positive-finite-scale validation, boolean support, integer metadata, selected-step range, parameter-count consistency, reconstructed finiteness and optional hash checking before returning the historical model. Seven malformed-state cases are rejected, a valid model's matrix is preserved exactly, and appended-byte drift is rejected when its expected hash is supplied. This opt-in loader is independently tested, not wired into shared src or deployed inference. Shared-source integration needs parent coordination and a new registered implementation; no shared-src edit was made. It is not a general hardened ZIP parser, and unmeasured decompression/security properties remain outside this audit.

## Resources and exact remaining gaps

V3 tests plus eight recovery/control fits and all 512 coverage trials took 7.3097s monitored wall time, 6.703125 sampled process-tree CPU seconds and 320,983,040 bytes measured peak process-tree working set, about 306.1MiB. Controller peak was 24,182,784 bytes, separately recorded. One numerical thread was enforced by environment, torch and threadpoolctl. Process-tree memory/CPU were sampled every .05s; per-process Windows high-water marks capture missed transient peaks, but summed historical peaks can conservatively exceed simultaneously live RSS. V2's failed prerequisites took 3.4553s wall, 3.0 sampled CPU seconds and 316,755,968 bytes peak. V1 scientific CPU/peak are unknown because its measurement was defective. The cap watcher terminates the scientific descendants on excess wall/memory/output; this is sampled user-space enforcement, not a kernel hard limit. The v3 receipt counted 198,231 output bytes before writing itself; the delivery inventory counted 226,522 bytes across attempts before writing itself and before archival. All measured scientific runs remained below 1GiB and 1800 seconds. V1 memory compliance cannot be established from its invalid receipt.

This is deliberately not full evaluator qualification. Exact gaps remain:

- No biological replay, new-source scoring, response/header opening, normalized-count adapter qualification, full-design streaming rehearsal or new independent source-heldout fit was performed. Canonical source hashing and cold incident exclusion were checked, but scalar completion correctly refuses cold lookup genes and no cold-gene predictive model was tested.
- No native NAIAD fit, rank-eight dense block campaign, expression evaluator campaign, adaptive label-budget selector, cross-source preprocessing pipeline or globally sealed multi-fold lifecycle was independently executed here. Existing tests were inventoried, not promoted into new evidence.
- No mechanistic separation of biological interaction from technical pair/guide effects follows. The observationally identical confounding construction proves that recovery alone cannot supply it; no explicit guide-confounded biological fit was added.
- The coverage study uses a single complete 30-gene graph, four mechanisms, 128 trials and 512 bootstrap draws. It does not establish nominal coverage on sparse/unequal-degree graphs, selection-adjusted coverage, shifted sources, heterogeneous cultures, conditional finite-panel coverage or biological prediction intervals. No replacement interval is proposed.
- Interrupted recovery is an atomic-write simulated exception plus reuse/overwrite refusal, not an OS-killed native campaign, disk-full/power-loss test or full resume qualification. The watchdog accounting has a synthetic tree probe and measured real child execution, but cap-trigger process killing was not stress-tested.
- Checkpoint validation successor is opt-in and local. Other model loaders and inference manifests have not received a comparable exhaustive malformed-state audit. The new tests do not assert an exact nominal 5% floating-point boundary or close every malformed-input path.

No commits, pushes, subdelegation, edits to plan/README or shared-src changes were made. Files outside the owned wave3 evaluator names belong to concurrent workstreams and were left untouched.
