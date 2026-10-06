# What v1 establishes

Opal is a family of symmetric output-level residual predictors with a benchmark pipeline. With both singles measured in the same biological setting, it predicts their sum plus a fitted correction. Improved total-response error doesn't by itself establish pair-specific biological interaction learning.

| Evidence class | Finding | Interpretation |
| --- | --- | --- |
| Historical primary Norman, 38 held pairs | Additive MSE 0.008515040; neural 0.007169187, 15.81% lower; selected reduced-rank 0.007268184. | Neural advantage over the selected reference is inconclusive. |
| Historical primary Wessels, 18 held pairs | Additive 0.013936912; neural 0.005423558, 61.08% lower; global correction 0.004214949. | Shared distortion can explain larger gains than the neural model. |
| Exploratory architecture development | Kernels selected in all 40 competitive outer folds and beat the tested neural additions in aggregate comparisons. | Strongest tested development family, not universal superiority. |
| Coverage development | Extra singles yield about +0.03% random and -0.38% partner improvement; extra pair labels yield 16.05% and 4.92%. | Pair coverage matters here; this isn't a general scaling law. |
| Independent D-SPIN procedure evaluation | Frozen kernel 0.034024658 versus source-selected simple reference 0.033107517. | Transfer gate failed. Procedures were source-refitted, not zero-shot frozen weights. |
| Baseline-only state development | Learned 0.073677613 versus simple 0.064679562 on EmeraldBay; 0.091868419 versus 0.074905505 on sci-Plex 3. | These encoders don't justify adoption or scaling; other representations remain untested. |
| Uncertainty diagnostics | Nominal 90% source radius covers 34.3% of held D-SPIN conditions. | No transferable coverage or abstention claim. |
| Native aggregate adaptations | CPA beats the chemical kernel in one pilot; GEARS results are mixed. | These aren't canonical cell-level leaderboards. |

Split-half residual cosines of about 0.289 on Norman and 0.084 on Wessels document technical instability. They're not biological-replicate reliability estimates and don't establish that no residual is learnable. Synthetic recovery demonstrates implementation behavior, not biological validation.

The evidence archive contains tracked compact results, protocol/selection records, reports, and manuscript evidence. The manifest records per-file hashes and the release commit. Raw data and ignored prediction arrays aren't bundled. Full reproduction requires source downloads, optional dependencies, and sometimes the recorded GPU environment. Interrupted and corrected runs are retained.

Software tests cover split isolation, exports, symmetry, and restoration. Release verification runs the installed wheel outside the checkout and replays the released references. See [the verification record](release_validation.md) for its environment, exact checks, and dependency-reuse limitation. This supports execution in the declared environment, not independent biological confirmation.
