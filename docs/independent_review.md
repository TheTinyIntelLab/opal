# Independent internal manuscript review

This is an advisory assistant review of the saved implementation and evidence, not external peer certification or a declaration that the paper is ready for submission. Criteria below were written before reading `paper/main.tex`.

## Precommitted criteria

1. Every quantitative claim must agree with saved result artifacts, with its dataset, transform, split, seed, metric, comparator, and aggregation identified. Pending studies cannot support completed-result language.
2. Structural or biological interpretation must be limited to what the model and experiment establish. Predictive improvements do not establish causal biological interaction, identifiable mechanisms, or independent validation.
3. Dataset provenance must distinguish source experiments, perturbation identity and dose, control matching, normalization, replicate aggregation, and the single-to-pair prediction task. Chemical data with few held-out pairs must remain exploratory.
4. Splits must protect held-out pair outcomes. Gene selection, scale fitting, model tuning, calibration, and any adaptation must use only their declared train/validation domains. Single perturbation access must be explicit and consistent with the task.
5. Uncertainty must state its unit and dependence assumptions. Seed variation, bootstrap intervals, and conformal coverage must not be presented as broader biological replication or population guarantees without evidence.
6. The manuscript's equations, training objective, baselines, stopping/tuning protocol, and limitations must describe the implemented and actually run workflow.
7. References must support the claims attached to them. Authorship and AI declarations must identify draft status and avoid invented human approval, institutional affiliation, or validation.

## Review findings

Reviewed the manuscript, both evidence tables, primary/count/chemical result summaries, primary diagnostics, synthetic primary and secondary summaries, source audit, amendments, and fitting/preparation/evaluation implementations. No source code or manuscript was modified. Biological secondary studies are still running, so their completion is outside this review snapshot.

Severity means P1 invalidates a central result or supports an unsupported central claim, P2 needs correction before distributing this internal benchmark as an audited research artifact, and P3 is a narrower documentation or protocol clarity issue. No P1 defect was identified in the inspected completed prediction evidence.

### P2: Manuscript latency numbers do not match the current saved diagnostics

`paper/main.tex:110` reports Norman warm median/p95 of 0.625/0.715 ms and cold load of 43.1 ms, and Wessels 0.647/0.813 ms and 46.8 ms. Current `results/runs/norman-log2-primary/diagnostics.json:220`, `:224`, and `:225` instead record 47.0443 ms cold load and 0.67575/0.796535 ms warm median/p95. `results/runs/wessels-log2-primary/diagnostics.json:124`, `:128`, and `:129` record 45.7073 ms and 0.75115/1.01454 ms. Checkpoint sizes agree. Re-running `tools/analyze_results.py:145` through `:156` measures timings again and overwrites diagnostics, while the manuscript numbers are literal text, so evidence can drift even without changing predictions. Update the prose from one retained timing run, and give performance measurements their own versioned artifact or generated text include. This changes the reproducibility of the compute claim, not the predictive conclusion.

### P2: The evaluator bundle is neither frozen by checksum nor checked for pair order at final evaluation

`src/opal/training/benchmark.py:66` through `:69` fingerprint fitting files but omit `evaluator/profiles.npz`. `:194` through `:198` record selection/prediction hashes and then read evaluator `doubles` without verifying its `pair_indices` against `query_indices`. The evaluator actually contains indices, as written by `src/opal/data/synthetic.py:74` through `:75`. Replacing or reordering that file between development and evaluation could silently change all reported scores while passing the fitting-data freeze. Freeze evaluator hashes in a separate preparation manifest without exposing their expression values to fitting, verify exact index order at final evaluation, and record the evaluated file hash in `test_access.json`. Register any code change after the currently running studies rather than modifying their frozen source. This is an audit protection gap; inspection did not find evidence that existing evaluator data were altered or consumed by fitting.

### P3: Source audit still labels completed matrix checks as pending

`docs/source_audit.md:3` says expression matrix checksums and acceptance remain pending, and `:7` describes Norman count acceptance as future inspection. `paper/main.tex:46`, `:48`, and `:133` describe accepted prepared sources and recorded checksums, which is consistent with the available prepared manifests. Keep the historical discovery audit but add a dated acceptance follow-up linking the prepared manifests and QC artifacts. Without that distinction a reproducer cannot tell whether the data were accepted or whether the manuscript ran ahead of its provenance checks.

### P3: Development error is seed-average, validation-selected checkpoint error, while final error is ensemble error

`src/opal/training/benchmark.py:47` passes each CV validation response to fitting; `src/opal/training/__init__.py:59` through `:68` retain the best epoch on that same fold. `src/opal/training/benchmark.py:59` through `:61` then average member-specific CV errors over seeds and use the median best epoch for refitting. Final evaluation instead averages predictions across seeds before scoring at `src/opal/training/benchmark.py:205`. `paper/main.tex:62` mentions folds, stopping, and seeds but does not identify those aggregation and refit rules. State them explicitly and characterize development errors as model-selection scores, not unbiased validation estimates. A later ensemble-selected protocol would need a newly registered run; the current held-out evaluation remains separate and does not require retrospective reselection.

## Evidence and integrity checks that passed

- The completed primary numerical claims, count sensitivities, chemical exploratory scores, and paired uncertainty values at `paper/main.tex:88`, `:95`, `:97`, `:99`, `:101`, and `:116` agree with `results/research_summary.json` and the corresponding evidence table to their stated rounding. Synthetic MSE/skill claims and secondary skill ranges agree with their saved summaries.
- Final fitting completes predictions and writes an access marker before loading evaluator outcomes at `src/opal/training/benchmark.py:191` through `:198`. A read-only NumPy check confirmed that current evaluator `pair_indices` equal fitting `query_indices` in all five completed biological datasets. PCA, encoder inputs, and reference gene eligibility use allowed singles and controls. No inspected model-fitting path consumed final pair responses.
- The text correctly distinguishes the sum-network's absolute-difference input, scale-dependent prediction residuals, descriptive bootstrap intervals, technical split halves, and the seven-pair chemical feasibility study. It retains the negative Wessels stronger-baseline result and inconclusive Norman comparison instead of retrospectively selecting the best test candidate.
- Bibliography keys cited in the manuscript are present in `paper/references.bib`; manuscript claims about public methods are restrained to related work and unexecuted comparisons. This internal review checked consistency with the recorded source audit rather than independently re-certifying all external bibliographic metadata.
- Author, funding, conflicts, release status, and AI declarations at `paper/main.tex:132` through `:142` explicitly leave missing human information unresolved. No invented institutional approval or external review certification was found.

## Decision

The manuscript's central predictive conclusion is supported by the inspected completed artifacts. Correct the timing mismatch and close the evaluator provenance gap before calling the repository fully audited. Secondary real-data results, public-method runs, adequate chemical validation, measurement uncertainty, and responsible-author review remain completion requirements; this advisory review does not certify submission readiness.
