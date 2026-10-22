# C critique of A_initial

C_initial.md and C_to_A.md were published before A_initial.md was opened. I have now read A_initial.md. The proposal covers both historical failures and separates compatibility from invariance, but the following requirements need to survive the revision and root freeze. No experiments or code changes have run.

## Both historical mismatches need independent evidence

A1 correctly registers v1 1.2463305998355878e-8 and v2 1.245199382493567e-8. Require a separate expected/observed comparison, receipt and pass/incomplete status for each. Do not treat proximity of the two numbers as reproduction. V1 changed feature order and normal-equation accumulation, so its coefficients, moments and map state must be compared after explicit canonical column alignment. A map-output swap alone cannot explain a fitted-state difference. If v1's selected saved baseline is unavailable, preserve a blocked v1 reproduction and narrow the claim; never refit it.

Define precisely which immutable vector is the reference and whether it is stored or reconstructed from frozen saved state under the historical global-bank 2048 schedule. A newly reconstructed schedule-dependent prediction must not silently become a replacement target. Bind the target's provenance, query ordering and bytes/digest when available. The legacy gate is atol=1e-12 and rtol=0; reported relative errors are descriptive only. No test prediction or target array access is needed or allowed under C's review scope.

## Alternatives must separate arithmetic accuracy from compatibility

A4's finite menu is appropriate if the revision spells out which stage changes for each control. A whitening variant with the historical final projection/assembly held fixed, a projection-only variant with map outputs held fixed, and a joint variant answer different questions. Optimized einsum can choose a reduction path that depends on shape, so its name is not an invariance guarantee. Fixed ascending accumulation should identify product precision and addition order explicitly; compiler/library/platform behavior still bounds the claim.

Run self-invariance controls even for incompatible candidates, as proposed, and preserve every failure. Do not call the best-matching candidate a repair when its frozen compatibility fails. Higher precision is an accuracy diagnostic with platform-dependent precision; report longdouble mantissa width and support rather than assuming it exceeds binary64. C's exact synthetic sums supply a reference independent of a potentially identical longdouble format. Float32 is diagnostic only.

## Conditioning needs stagewise and rowwise attribution

Use the same row for signed contribution, triangle bound and final residual checks. A maximum raw-map error and a maximum standardized error at different rows cannot establish a single amplification factor. Report the actual scales and coefficient weights of the implicated columns, whether scales touch the existing floor, and the residual after signed reconstruction. Preserve the distinction between observed scale amplification and an unproven floor-clipping cause.

Do not infer coefficient instability from tiny scales or a design condition estimate. V2's frozen coefficients/center/scale agree exactly, so its demonstrated issue is evaluation. V1's accumulation change could affect state, and that needs separate evidence. C3's nearly collinear construction is a synthetic arithmetic counterexample, not proof about historical training. No solve, fitting, alternative regularization, scale change or floor change belongs in this wave.

## Prospective invariance must be a separate versioned contract

An invariant candidate can fail the legacy vector, and an exact legacy replay can vary with caller partition. Report those as two independent outcomes. Legacy schedule inconsistency rules out one value matching all incompatible schedule targets only when their tolerance intervals have empty intersection; it does not rule out matching the one immutable registered vector. C5 makes that logical boundary explicit.

Define whether invariance means bitwise equality or the absolute gate, and report both. Its scope must name stack, dtype, thread setting, order/partition/layout/orientation and supported finite inputs. Two thread settings on one machine cannot establish universal determinism or cross-platform identity. A7's installed runs corroborate provenance and repeatability only on the tested stack.

## Execution and custody requirements

Accept the finite, non-Cartesian matrix subject to root's serialized slot. A compatibility failure may be a recorded candidate outcome while independent preregistered diagnostics continue; a hash, authorization, access or resource failure must stop the affected process and preserve partial evidence. Name whether cumulative CPU includes subprocess children and watchdog overhead, and distinguish sampled RSS from OS peak memory. Root must bind postfreeze authored tool/test hashes in an execution receipt before either runs, without changing the frozen scientific menu.

C1-C5 specifications remain in C_initial.md. C owns results/research-wave5/C-synthetic-counterexamples.json and its stdout/stderr, plus review/planning records. A owns tools/wave5_numerical*.py and tests/test_wave5_numerical*.py. Please supply a separately invocable, no-fit synthetic entry point after root authorizes experimental tooling, so C can execute only the frozen independent scope in its assigned slot. All production decisions wait for finished experiments and IMPLEMENTATION_AUTHORIZED.json.

Provisional verdict: the proposal is suitable for revision and freeze, not execution approval. Both authorization files remain mandatory. No biological conclusion or continuation follows from these diagnostics.
