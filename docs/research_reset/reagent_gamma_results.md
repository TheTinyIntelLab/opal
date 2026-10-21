# Exact-guide and arm matching changes the minimum baseline, not the local-supervision conclusion

Q1b completed on 21 October 2026. Applying minimum to exact-guide, arm-matched singles before averaging lowers block-test MSE by 33.3375% in K562 and 42.5378% in Jurkat versus the historical gene-average minimum. All 81,466 original pairs survive in each context, and every reconstructed infection target is exactly equal to its historical value. This identifies a consequential aggregation-order effect without changing the target population. It does not establish molecular interaction learning or explain the completion model's gain over calibrated references.

The measured evidence is exploratory reuse of GSE116198. The 16,210 original block-test pairs per context have already been exposed by Q2 and are not fresh confirmation. Shared guides, controls, orientations and infections are dependent measurements, and there are still only original-source contexts here. SPIDR response counts were not opened.

## Measured result

Losses below use the unchanged 16,210 block-test pairs, equal canonical-pair weights and the mean of the two infection targets. Both minimum predictors use measured singles without pair-label fitting. The historical predictor averages singles before minimum; the construct predictor applies minimum to the two exact-guide/arm singles and then averages predictions.

| Context | Historical minimum MSE | Construct-first minimum MSE | Relative reduction | Conditional paired loss-difference interval |
| --- | ---: | ---: | ---: | --- |
| K562 | 0.009923602593277051 | 0.006615318419545617 | 33.3375318352% | [-0.0038684957981965566, -0.0027748377755567348] |
| Jurkat | 0.0018747758713199125 | 0.0010772867967001876 | 42.5378354191% | [-0.0009803308356937093, -0.0006301810459332782] |

The intervals use 2,000 endpoint-multiplicity draws with seed 20261022 and are conditional gene-panel sensitivity bounds. They do not estimate independent-source or biological-replicate uncertainty. The full-roster minimum MSEs are 0.006733959623606163 versus 0.010080591713676049 for K562 and 0.0010677759573749295 versus 0.0018515787936931346 for Jurkat. Full-roster and block-test panels remain separate descriptions of this exposed source.

Mean-first exact-guide minimum agrees with historical gene-single minimum to floating precision on every diagnostic panel. Thus, in this source and roster, changing the ordering of the nonlinear minimum explains the observed baseline difference. The matched sum commutes with maximum absolute errors 6.106226635438361e-16 and 2.7755575615628914e-16 in K562 and Jurkat, below the preregistered 1e-12 tolerance. Sum itself remains substantially worse than minimum, with block-test MSE 0.046221620586752304 and 0.004776976763247271 respectively. These are observations under this assay endpoint, not a universal biological law.

The stronger minimum is still worse than the previous Q2 validation-selected calibrated/node reference on these same block-test pairs, approximately 0.00186564 for K562 and 0.000500916 for Jurkat, and worse than the rank-eight composite, approximately 0.00134186 and 0.000434157. Q1b therefore strengthens a simple baseline and reveals guide/aggregation structure, but does not remove the measured local completion advantage over the prior calibrated reference. A separately registered flexible singles-only comparison is the next discriminator; this no-fit analysis cannot attribute that residual advantage.

## Identity, coverage and limits

The frozen source header and literal-name join preserve K562/Jurkat, Triple Sequencing, individual infections, ordered vector arms, guide names and unique 20nt sequences. Each supported guide/arm/context/infection has 16 finite negative-control partner measurements, comfortably above the frozen minimum of two. The library contains 18 named negative-control guides. These are different counts: 18 is metadata membership, whereas 16 is finite partner support in the deposited phenotype table. The initial prepared support arrays do not retain partner identities, so that stage could not identify which two metadata controls are unavailable or establish why they are unavailable. The registered identity follow-up below resolves which controls are absent; their original experimental or processing cause remains unknown. Do not rewrite the library count as 16 or claim that all 18 were observed.

All 81,466 pairs have at least two eligible ordered constructs in each infection of both contexts, zero finite construct measurements are lost through missing matched singles, all pairs have both orientations, and the reconstructed targets match the historical infection arrays exactly. The runner streamed 922,455 source rows, including 30,240 target-control constructs, and records 160,280 distinct-target rows outside the fixed historical roster. Those rows remain outside this diagnostic, with the exclusion count disclosed rather than used to expand eligibility.

Orientation-target gap MSE on the block-test panel is 0.009102938977238682 in K562 and 0.0031839037600292107 in Jurkat. This indicates substantial orientation-associated variation in processed measurements, but it combines arm effects, guide composition and measurement noise and cannot causally isolate them. Infection-level losses, orientation means and count histograms remain available in the frozen receipts and prepared arrays. Transcript and intentional mismatch annotations remain explicitly unavailable. Deposited gamma is not raw count data, and no normalization or per-doubling calibration was reconstructed.

## Qualification, replay and the independent audit

Fifteen synthetic tests passed before scientific execution and again after freezing. Tests cover exact identities, arm/context/infection matching, control-only single estimation, double-label invariance, missingness, duplicate/truncated rejection, grouped Q2 splits, baseline aggregation formulas, full-roster exclusions, lifecycle/tamper rejection, resource guards and deterministic replay including NaNs. Registration binds the protocol, code, tests, source hashes and environment. No outcome-dependent threshold or model selection occurred.

Attempt `q1b-v1` completed its scientific run on C:, but its replay stopped at the fixed 1 GiB disk-reserve guard. That failure and its receipts remain preserved. A new attempt `q1b-v2` on E: used identical inputs, code, dependencies and recipe with a new freeze timestamp. Its run and replay passed within 54.62286919998587 aggregate wall seconds, 54.546875 CPU seconds and 146,120,704 peak bytes. Both attempts produced byte-identical scientific result JSON. The resource limits remained 1,800 aggregate seconds, 1 GiB memory, 50 MiB output and 1 GiB free-disk reserve, with one CPU thread.

The independent audit checked current frozen code hashes, original-bank and library hashes, prepared-array and result hashes, C:/E: recipe equality excluding the fresh timestamp, exact roster/target equality, every diagnostic MSE, nonlinear minimum ordering, sum commutation, unchanged test membership and exact bootstrap interval reproduction. It took approximately 0.6 seconds with one numerical thread. It did not rescan or independently rehash the large phenotype gzip; the runner's exact raw replay supplies that separate assurance. The audit receipt is [independent_review.json](../../results/research-wave2-reagent/independent_review.json), and the [attempt receipts](../../results/research-wave2-reagent/q1b-v2/results.json) preserve exact numbers.

The integrity, same-population/target and replay gates passed. This established the prerequisite for the subsequently completed Q2b on the unchanged six scalar banks and also establishes the declared prerequisite for a separately registered guide-aware historical comparison. Neither permission is evidence that a new guide-aware model has been fitted or that it will improve performance.

## Reproduce

The [frozen protocol](reagent_gamma_protocol.md) declares the complete procedure. Historical paths and hashes are in each registration. The scientific output directory is `E:/opal-wave2-20261021/q1b-v2`; inputs remain under `C:/Users/USER/Desktop/opal-next-data-20261020`. Raw source assets and numerical arrays remain outside Git with the original author-deposit versus GEO-mirror licensing caveat preserved.

For a new output directory, run these commands in order, commit the registration and passing prerequisites before `run`, and preserve every started attempt:

```powershell
.venv/Scripts/python.exe tools/study_reagent_gamma.py freeze --source-root C:/Users/USER/Desktop/opal-next-data-20261020 --output E:/opal-wave2-20261021/q1b-new
.venv/Scripts/python.exe tools/study_reagent_gamma.py verify --output E:/opal-wave2-20261021/q1b-new
.venv/Scripts/python.exe tools/study_reagent_gamma.py run --output E:/opal-wave2-20261021/q1b-new
.venv/Scripts/python.exe tools/study_reagent_gamma.py replay --output E:/opal-wave2-20261021/q1b-new
```

An independent numerical check loads `prepared.npz` and the frozen original bank, reconstructs historical singles as `bank['singles'][c][:, pairs].transpose(1, 0, 2)`, computes each prediction using the protocol formulas and verifies MSE against `results.json`. Recover the original block-test union from `block_assignments` on the unchanged genes/pairs and accession. Reproduce the interval by drawing active-gene multiplicities 2,000 times with NumPy generator seed 20261022 and weighting each paired loss difference by the product of its two endpoint multiplicities. No fitting or access to another biological source is needed.


## Registered control-identity follow-up

A separately registered identity sidecar subsequently resolved the 18-versus-16 control discrepancy. `negative_control-10016` and `negative_control-10909` occur in the guide metadata but are absent from every ordered construct identity in the phenotype source. The other 16 controls are observed and each contributes to 6,710 finite target-single cells across guide, arm, context and infection. There are no observed control identities with entirely unavailable single measurements. This establishes absence from the deposited phenotype table, without explaining the original experimental or processing cause.

The sidecar retains a uint32 control-identity bitset for every guide/arm/context/infection and exactly reconstructs every Q1b support count. The initial sidecar `q1b-partners-v1` passed its seven standalone synthetic tests, source stream and exact replay, which remain preserved. The later full-suite run exposed a test-fixture problem: synthetic lifecycle tests inherited the peak process working set of earlier ML tests, so they incorrectly exceeded the sidecar's standalone 256 MiB guard. This was a qualification-fixture failure, not a failed biological stream or a reason to relax the production guard. The successor mocks process memory only inside those synthetic lifecycle fixtures and adds explicit memory, disk and artifact guard tests. Scientific code, configuration and source inputs are unchanged. The [preserved successor record](../../results/research-wave2-reagent/q1b-partners-v1/test_qualification_successor.json) explains the transition.

Fresh registration `q1b-partners-v2` and eight passing synthetic tests preceded a new stream and exact replay. Both completed within 14.018894799999543 cumulative wall seconds, 14.0 CPU seconds and 71,667,712 peak bytes. The v1 and v2 scientific result JSON and numerical array archives are byte-identical, as recorded in the [delivery comparison](../../results/research-wave2-reagent/q1b-partners-v2/delivery_comparison.json). No new losses, models, thresholds or QC changes were introduced. Final full bitsets stay outside Git at `E:/opal-wave2-20261021/q1b-partners-v2/partners.npz`; [compact final receipts](../../results/research-wave2-reagent/q1b-partners-v2/results.json) and the [sidecar protocol](gamma_control_partners_protocol.md) retain the identities, hashes, guards and reproduction commands. This resolves the earlier audit's identity limitation while preserving its biological non-claims.
