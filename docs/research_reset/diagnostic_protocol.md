# Frozen exploratory assay diagnosis

Registered 20 October 2026 after all original and external outcomes were already exposed. This is a diagnostic reuse, with no confirmatory claim, retraining, fitted scale correction, checkpoint choice or promotion decision. The historical registrations remain unchanged.

## Questions and fixed comparisons

1. Does assay-matched single growth explain external pair growth when singles and targets come from different infections? On the historical strict-novel panel, compare the average-of-infections minimum-single prediction against the minimum from infection 1 scored on infection 2 and the reverse. Report both infections, their average loss, residual correlation across infections and raw-target correlation. Cross-infection comparisons reduce shared-infection noise but still share reagents and normalization methods.
2. Is outcome ranking a coverage or aggregation effect? Without changing historical eligibility, score fixed construct-count thresholds 2, 4 and 8 on the same strict-novel panel, using the subset with both singles measured. Report pairs, genes, MSE, MAE, bias and target quantiles for completion, original main effects, original minimum, assay-matched minimum and assay-matched additive singles. These nested subsets are sensitivity diagnostics, never a replacement primary panel. Counts represent guide constructs, not biological replicates.
3. Does a residual above assay-matched minimum singles repeat across experiments? For every unordered combination of the four cohorts, join canonical gene pairs in their historical strict-novel panels with complete singles. Score each direction without fitting: target-cohort minimum alone versus target minimum plus source-cohort observed pair-minus-minimum residual. Report paired mean-target loss differences and 2,000 gene-multiplier draws, seed 20261022, plus raw and residual Pearson correlations. A fixed weight of one is a test of unchanged residual transfer, not an estimate of optimal transfer. Include 20 shuffled-edge source-residual controls, seed 20261022. These controls destroy gene effects too, so success cannot uniquely identify pair-specific biology.
4. Are discrepancies consistent with a pure positive change of scale? Report centered correlations, means, standard deviations and quantiles. A positive scalar rescaling cannot repair a near-zero or negative centered correlation. This rules out a sole explanation, not mixed reagent, time-window, cell-state or normalization effects.

## Inputs and evaluation units

Use the exact hash-verified original protocol, model freeze, external prepared-v2 arrays, historical frozen predictions and prior scores. Keep all historical strata distinct; only the absent-original-eligible-bank stratum enters new strict-novel scores. Preserve shared-pair and shared-gene dependence. Never pool cohorts as independent biological replicates. Only two infections per cohort exist; conditional gene intervals do not estimate a population of cultures or studies.

No new gene, count or response-based filter may be chosen after viewing these results. Missing singles make a pair unavailable for a singles comparison, rather than imputing a zero. Exclude nonfinite, duplicate, noncanonical or misaligned identities with an explicit error. Do not silently drop them. Threshold subsets smaller than 50 pairs or 10 genes are marked small, not removed. Empty subsets are recorded unavailable.

## Ordering, stopping and interpretation

Executable tests in `tests/test_reset_diagnostics.py` must pass before `freeze`. Freeze hashes this protocol, implementation, tests and all numerical inputs into a new output directory. `run` refuses drift or overwriting, records runtime and peak memory, and writes only aggregate JSON results. `replay` recomputes these results and demands exact JSON equality. One CPU numerical thread, 10 minutes, 2 GiB process memory, 16 MiB output cap. Stop on integrity failures; do not repair by changing an outcome-dependent panel.

These analyses can support a successor data requirement or reject a simple explanation. They cannot resurrect the historical failed hypothesis, establish universal absence of interactions, or promote an adapted model. Confirmatory source selection must be based on provenance and compatibility before opening that source's outcomes.
