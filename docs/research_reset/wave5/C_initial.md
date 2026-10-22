# C independent numerical proposal

Published before reading wave-five A_initial.md or A_revised.md. Starting HEAD is ab6c37cfeab75735634d0e0e364339756ee8e43e. C has read root_initial.md, root plan.md, wave-four closeout/freeze/A diagnosis/C results review, and the wave-three strong-reference protocols and closeout. No experiment, fit, outcome/header access or code change has run. This file is a planning record.

## Hypotheses

H1: v1 and v2 require separate historical reproduction. V1's recorded maximum is 1.2463305998355878e-8, while v2's is 1.245199382493567e-8. V2 state identity cannot establish v1 state identity or explain its feature-order and accumulation differences. Both remain software stops, without a scientific nuisance result.

H2: cancellation in dot products and small frozen feature scales can amplify floating-point differences past 1e-12. Better local arithmetic can improve accuracy while changing the historical result, so accuracy and compatibility require separate verdicts.

H3: fixed per-row arithmetic may give partition invariance on a specified stack, but that does not guarantee equality to the immutable legacy vector or portability across threads, libraries, platforms and dtypes. A deterministic inaccurate function can pass invariance.

H4: raw kernel roundoff, scale amplification and final cancellation must be attributed separately. A stagewise bound and signed contribution check are more informative than a final maximum alone. Normal-equation conditioning must not be conflated with v2's prediction-only discrepancy.

## Frozen proposal matrix

| ID | Explicit synthetic construction or historical review | Expected outcome and falsifier |
| --- | --- | --- |
| C1 | Binary64 dot product with terms [2^54, 1, -2^54], exact real sum 1. Compare left-associated, cancellation-first, fixed-order, compensated and high-precision diagnostic arithmetic. | Left-associated evaluation can yield 0 while cancellation-first yields 1. A fixed order is not an accuracy proof. Record actual arithmetic rather than requiring a particular BLAS result. |
| C2 | Fixed raw perturbation delta z = 2^-40, frozen scale s = 2^-20 and coefficient beta = 1. Standardized prediction perturbation is 2^-20, with no fitting. Repeat with s = 1 to isolate amplification. | Small scale alone amplifies the specified error by 2^20. This demonstrates a mechanism, not the historical scale-floor cause. |
| C3 | Four-row two-column design X = [[1,1],[1,1+e],[1,1-e],[1,1]], e = 2^-26. Examine exact/reference Gram differences and prediction sensitivity with fixed coefficients, without solving or fitting. | Forming X'X can erase small structure. Report condition diagnostics separately from prediction arithmetic; do not infer v2 coefficient instability when saved coefficients agree. |
| C4 | Repeat C1/C2 fixed rows up to 512; partitions 1, 3, 7, 64, full; reverse/permuted/restored rows, C/F arrays and noncontiguous views; binary64 primary, binary32 diagnostic only. | Fixed per-row reduction should preserve partition/order under the declared stack. Any failing candidate remains a failure; dtype changes cannot rescue binary64 legacy compatibility. |
| C5 | Two purported legacy targets for one identical query, y1=0 and y2=3e-12, with atol=1e-12 and rtol=0. Also compare one target alone to an invariant constant predictor. | The two tolerance intervals do not intersect, so no one output can satisfy both. This does not prove impossibility of matching one immutable legacy vector. |
| C6 | Review A's separate v1/v2 no-fit replay, state identity, first divergent stage, signed feature contributions and alternative arithmetic against immutable validation predictions. | Both historical maxima need separate evidence or an explicit blocked result. No test predictions, target arrays, loss or refit may enter this review. |

Alternative arithmetic is a finite diagnostic menu, never an adaptive candidate search. C1-C5 are independent synthetic counterexamples. C6 is read-only review owned by C; A owns real-state execution and tools/wave5_numerical*.py plus tests/test_wave5_numerical*.py. Proposed synthetic cases may be encoded by A only after root authorizes experimental tooling; C supplies fixed specifications and reviews them independently. No implementation decision is allowed before completed experiments and IMPLEMENTATION_AUTHORIZED.json.

## Contracts and limits

Legacy acceptance stays exactly absolute tolerance 1e-12 and relative tolerance zero against the immutable registered vector, with no changed floors, scales, state, row identity or target. A prospective invariant contract must separately name arithmetic, supported dtype, finite-input policy, batch/order/orientation/layout guarantees, stack/thread scope and tolerance. A prospective pass cannot erase a historical fail or authorize a biological successor.

C proposes one CPU thread, at most 512 rows and small fixed arrays, no optimization or fitting, 120 wall/CPU seconds per command, 600 cumulative wall/CPU seconds, 512 MiB process peak memory, 10 MiB output and 1 GiB free-disk reserve. Root's final freeze and resource slot override these proposals only prospectively. Serialize with A's numerical work. No experiments or experimental code changes before BOTH results/research-wave5/freeze.json and docs/research_reset/wave5/EXECUTION_AUTHORIZED.json exist and explicitly cover C's scope.

Stop before execution on absent authorization, hash drift, conflicting ownership, unregistered input or slot. Stop during execution on nonfinite arithmetic, resource breach, unexpected file access, any need for fitting/outcome/header access, or deviation from the frozen matrix. Preserve failure receipts; no retries, expanded menus or raised caps without a new explicit root decision that preserves the failure. Record commands, pre/post code/input hashes, dependencies/thread settings, wall/CPU/peak memory, outputs, skipped cases and limitations. Cooperative guards must not be described as interrupting watchdogs.

C stays available through A's result records and root's final review request, with waits no longer than 30 seconds. No delegation or new review round.
