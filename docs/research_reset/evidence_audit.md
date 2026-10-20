# Evidence and evaluation audit, 20 October 2026

The frozen external-superiority hypothesis failed. This is a failure of the retained original-assay predictor under source transfer, not evidence that pair-specific biological information is absent. The primary transfer panels also concentrate on an unusual low-coverage block of the original screen, so they combine source shift with a change in graph support.

This independent audit inspected repository artifacts and local numeric banks without fitting any model. It does not replace the external registration, alter any historical outcome or constitute independent confirmation. Any use of these cohorts for successor development is exploratory.

## What was verified

The initial checkout was clean at `11b495e`, which follows `01c89b1`. Commit `b963598` contains external preparation, tests, the protocol and frozen prediction hashes; `01c89b1` subsequently records external results and descriptive diagnosis. The historical model freeze identifies `1fc0086` as its original source commit. These are different roles, not inconsistent version claims.

All 22 entries in `results/fitness-external-20261020/protocol.json` matched the current local file bytes, including original model assets, original split assignment, external preparations, frozen predictions and evaluation code. No missing or mismatched file was found. The external evaluator does not fit, select or calibrate a model on external outcomes. Its `freeze` stage generates original-input predictions, and `run` verifies the frozen hashes before calculating metrics. This supports the recorded unchanged-predictor evaluation; it cannot establish that no person previously inspected source outcomes outside the recorded workflow.

The original source's first three header rows explicitly identify K562 triple-sequencing replicate columns 13/14 and Jurkat columns 4/5, matching `tools/prepare_fitness.py`. All 922,455 parsed original construct identifiers are unique. Original preparation contains 81,466 distinct canonical pairs, finite single and replicate arrays, and eligible pair counts from 4 to 18 per context/infection. All external frozen pair names agree exactly with the corresponding prepared row identities. Canonical ordering, uniqueness and finite primary targets hold for every external panel.

The parent task owns fresh full-suite, build, replay and remote-CI verification. This audit does not reinterpret historical `validation.json` as a new test run.

## What the external result establishes

The historical artifact `results/fitness-external-20261020/results.json` gives the following unchanged-model results for the bank-novel panels:

- Environmental experiment 1 has 130 pairs and 17 genes. Completion MSE is 0.07106016738885874 against 0.06956191379413991 for frozen main effects, a 2.153841826653502% increase. The conditional gene-resampling loss-difference interval is [-0.002278234600097497, 0.00519075008154436].
- Environmental experiment 2 has 130 pairs and the same 17 genes. Completion MSE is 0.05040156439696539 against 0.04937928723562158, a 2.07025499672735% increase. Its interval is [-0.0019105159322120558, 0.00407676666395129].
- PARPi 2022 has 423 pairs and 30 genes. Completion MSE is 0.10609865972843151 against 0.10485538210148743, a 1.1857070204948972% increase. Its interval is [-0.001419125581248767, 0.003735340695665773].
- PARPi 2023 has 411 pairs and the same 30 genes. Completion MSE is 0.10671178390060439 against 0.1055719411553801, a 1.0796834203765204% increase. Its interval is [-0.001493590515174401, 0.003669637167092564].

Completion is worse than frozen main effects in both infection measurements of every cohort. Original-single minimum MSE is respectively 0.03583220305437983, 0.025507222792187808, 0.06709132510734524 and 0.0672618283874853, so the failed promotion is not merely a close comparison with one ridge reference. All four gene-resampling intervals include zero, so the artifacts support failed superiority, not statistically established general harm from completion. The cohorts share genes, pairs, laboratory lineage and original predictor inputs; their common direction does not supply four independent study-level replications.

The result does not establish unseen-gene prediction, Jurkat external transfer, independent-laboratory transfer, interaction mechanism recovery, expression improvement or an absence of reproducible interactions. The local within-source gains remain valid for their original supported-gene missing-pair task. They need stronger dependence-aware and graph-support diagnostics before supporting a broad biological claim.

## New measured graph and eligibility findings

The four primary panels contain only **509 unique gene pairs in total**. The two environmental panels are identical sets of 130. PARPi 2023's 411 are an exact subset of PARPi 2022's 423. Environmental overlap is 44 pairs with PARPi 2022 and 38 with PARPi 2023. Treating the summed 1,094 cohort-pair observations as independent would inflate precision.

Each panel is nearly a clique within a small gene set. Edge densities are 130/136 = 0.9558823529411765 for each environmental panel, 423/435 = 0.9724137931034482 for PARPi 2022 and 411/435 = 0.9448275862068966 for PARPi 2023. Minimum/maximum panel degree is 12/16, 12/16, 24/29 and 21/29 respectively.

Crucially, **every one of these 509 unique pairs occurs in the original raw Horlbeck table**. None is a previously unmeasured original gene pair. Every one fails the original requirement of at least four finite guide observations in every context/infection, so all are novel to the eligible fitting bank only. The earlier report explicitly cautioned that raw absence had not been established; this audit resolves that question using identities and finite-value availability, without computing response-magnitude summaries or fitting a model.

Among the 130 environmental pairs, 52 have at least four original observations in both K562 infections but fail only the Jurkat requirement, while 12 fail only K562 and the remainder fail both contexts. For PARPi 2022 these counts are 124 and 47; for PARPi 2023 they are 118 and 45. Therefore some exclusions from a K562 transfer panel arise entirely from original Jurkat coverage. These facts do not justify retroactively expanding training or changing the frozen primary panel.

Original active-gene eligible degrees range from 285 to 426, with median 426. The genes in every external bank-novel panel have median original degree 285, with ranges 285 to 313 for environmental panels and 285 to 312 for PARPi. Thus external bank-novel evaluation is concentrated on the least-connected original gene block. A source-generalization claim must distinguish this from the random within-source pair split, which is dominated by more densely connected genes. This observation supports a graph-support explanation as a competing hypothesis, but does not establish why the source measurements are missing.

External primary guide counts range from 2 to 8, with median 8 in every cohort. Environmental experiment 2's minimum is 3; the other minima are 2. The difference from the original minimum of 4 is prespecified and real. Equal pair weighting estimates performance on the deposited eligible pair population, rather than on an equally precise or whole-genome population.

## Preprocessing, dependence and exposure limits

`tools/prepare_external_fitness.py` applies whole-library T0 coverage filters, adds a pseudocount of ten, forms sequencing fractions, centers on double-negative controls, then divides by source-provided reference doublings. Fraction normalization cancels after control centering. A test verifies that algebra. There is no obvious sign reversal, wrong original column or pair-row alignment error in the inspected implementation. Environmental T0-only filtering deliberately differs from the published endpoint-count filter, so these are harmonized endpoints rather than exact published-score reproduction.

Environmental experiment 2 has 2,023 constructs with missing required counts. They are excluded, not imputed as zero. This prevents invented counts but does not eliminate outcome-dependent missingness if blanks encode severe dropout. Original preparation also selects pairs on availability of finite published phenotypes, and the new audit shows that the transfer panel is exactly a low-availability subset. Calling either rule independent of biological response would be stronger than the evidence supports. Document counts and selection; preserve the original analysis; use explicitly exploratory sensitivity checks rather than silently rescuing a loss.

Guide variants and orientations are subsamples, with common construct panels across infections externally. Two infections per cohort are the biological replication available here. PARPi's three sequencing runs in 2022 are merged technical measurements. Shared controls and single measurements correlate pair errors, and common normalization can correlate infections or genes beyond the edge graph. Pair resampling does not capture those dependencies. The external endpoint-multiplicity gene bootstrap is a useful conditional graph sensitivity analysis, but it is not a hierarchical biological-population interval. With only two studies and two infections per cohort, a broad study-level confidence claim is underpowered.

The original growth test, all four external cohorts, Norman, Wessels, EmeraldBay, D-SPIN and historical source diagnostics already have recorded outcome exposure. Repartitioning or using one of those external cohorts for adaptation changes its role to development or exploratory evaluation. A method can be frozen before a particular new exploratory comparison, but that does not make the source untouched. The root plan should maintain one explicit exposure ledger and keep a later, genuinely uninspected source outside selection.

## Competing explanations and discriminating work

1. **Main-effect and reagent shift is substantial.** Existing `diagnostics.json` records original/external single correlations of 0.161, 0.308, -0.030 and -0.038, with large mean changes. A positive scalar scale change alone cannot explain these low or negative correlations. Original/external pair correlations are similarly low. This favors a broad assay mismatch over an exclusively interaction-specific failure, while leaving guide efficacy, depletion floors, timing and normalization confounded. Compare same-assay singles with frozen original singles, and test opposite-infection singles to avoid shared-replicate noise. These are exploratory information-value checks, not adapted Opal validation.
2. **Graph support differs from the advertised random-pair task.** The new degree and eligibility findings show that bank-novel external pairs occupy a nearly complete hole among low-degree source genes. Report original-test error by prespecified endpoint degree and low-degree-block membership, then compare with corresponding external overlap strata. Do not select bins based on which ones make completion look good. A block-held-out development task would directly test whether completion needs within-block observations.
3. **Pair residual structure may be assay-specific or unreproducible.** Assess original completion residuals against remeasured pairs after a train-only nuisance model, and assess raw local-single residual agreement between genuinely independent infections. Include a fixed negative control that destroys pair identities while preserving gene degrees or main-effect bins. Low residual agreement would constrain how much architecture is justified, but cannot prove a biological absence when measurement reliability is poor.
4. **Simple singles may capture most transportable signal.** Same-assay minimum singles already provide much lower descriptive external MSE, but they have extra information. Compare successor models with matched local-single minimum, additive and regularized symmetric calibration in fixed development folds; compare low-rank or structured terms only after those references. Full-growth correlation alone cannot establish interaction information because main effects dominate it.
5. **Aggregation and missingness may affect rankings.** Preserve equal pair weighting as primary, report counts and guide coverage, and run a finite sensitivity set at predeclared count thresholds. Cross-infection ranking is more informative than multiplying guide rows into nominal sample size. Exact published-filter reproduction would answer a different endpoint question and should be separately labelled.

The strongest immediate objective is to establish reproducible incremental pair signal after assay-local single-effect calibration. Keep completion as a useful original-screen baseline and postpone a larger gene/pathway model until that signal survives a graph-matched development task and a later untouched source. Gene or pathway priors can help unseen identities only if evaluated on an actual unseen-gene protocol; the current lookup-factor model has no such capability.

## Reproduction of the new descriptive audit

The following PowerShell command reads only existing artifacts. It checks graph overlap, original coverage, duplicate raw identifiers and frozen hashes without fitting a predictor or calculating new outcome losses. Paths can be moved by changing the two local roots; their expected hashes remain in the registered protocol.

```powershell
@'
from pathlib import Path
import csv, gzip, hashlib, json
import numpy as np

external = Path('C:/Users/USER/Desktop/opal-external-validation-20261020')
original = Path('C:/Users/USER/Desktop/opal-next-data-20261020')
protocol = json.loads(Path('results/fitness-external-20261020/protocol.json').read_text())
mismatches = [p for p, expected in protocol['files'].items()
              if hashlib.sha256(Path(p).read_bytes()).hexdigest() != expected]
print('frozen_files', len(protocol['files']), 'mismatches', mismatches)
panels = {}
with np.load(original/'prepared/profiles.npz', allow_pickle=False) as d:
    degree = np.bincount(d['pairs'].ravel(), minlength=len(d['genes']))
    for cohort in protocol['config']['cohorts']:
        with np.load(external/'evaluation-v1'/f'{cohort}_predictions.npz', allow_pickle=False) as f:
            panel = f['pair_names'][f['strata'] == 'absent_original_eligible_bank']
            panels[cohort] = set(map(tuple, panel.tolist()))
            genes, counts = np.unique(panel, return_counts=True)
            old_degree = degree[np.searchsorted(d['genes'], genes)]
            print(cohort, 'pairs', len(panel), 'genes', len(genes),
                  'density', len(panel)/(len(genes)*(len(genes)-1)/2),
                  'degree', counts.min(), counts.max(),
                  'original_degree', old_degree.min(), np.median(old_degree), old_degree.max())
union = set.union(*panels.values())
observations = {pair: np.zeros(4, dtype=int) for pair in union}
raw_counts = {pair: 0 for pair in union}
seen, duplicates = set(), 0
with gzip.open(original/'GSE116198_sgRNA_pair_phenotypes.txt.gz', 'rt') as stream:
    reader = csv.reader(stream, delimiter='\t')
    headers = [next(reader) for _ in range(3)]
    for row in reader:
        if not row or '++' not in row[0]:
            continue
        duplicates += int(row[0] in seen)
        seen.add(row[0])
        pair = tuple(sorted(x.split('_')[0] for x in row[0].split('++')))
        if pair not in union:
            continue
        raw_counts[pair] += 1
        for i, col in enumerate([13, 14, 4, 5]):
            try:
                observations[pair][i] += int(np.isfinite(float(row[col])))
            except ValueError:
                pass
print('unique_bank_novel_pairs', len(union), 'original_raw_duplicates', duplicates)
for cohort, pairs in panels.items():
    n = np.array([raw_counts[p] for p in pairs])
    c = np.array([observations[p] for p in pairs])
    print(cohort, 'absent_raw', int((n == 0).sum()),
          'below_four_any', int((c.min(1) < 4).sum()),
          'excluded_only_Jurkat', int(((c[:, :2].min(1) >= 4) & (c[:, 2:].min(1) < 4)).sum()),
          'excluded_only_K562', int(((c[:, :2].min(1) < 4) & (c[:, 2:].min(1) >= 4)).sum()))
names = list(panels)
for i, a in enumerate(names):
    for b in names[i+1:]:
        print('panel_overlap', a, b, len(panels[a] & panels[b]))
'@ | python -
```

The source scan took about three seconds per pass on the available machine. It reads the licensed local source and exports no raw counts or response rows. The audit adds no source data, trained assets, credentials or private correspondence to Git.
