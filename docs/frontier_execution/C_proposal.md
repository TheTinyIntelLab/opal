# C independent framework and falsification proposal

Authored by worker C on 2026-10-26 before reading A's proposal. This is a prospective proposal, not an experiment receipt. I read `plan.md`, root's provisional interface, and implementation source only; no biological response headers, values, tables, archives or notebook outputs were opened. No numerical workloads were run.

I recommend a synthetic-only, one-initiation fitting API with physically separate scoring inputs. The eight-world comparison can test inductive bias and leakage resistance, but even perfect recovery cannot identify biology: the persistent technical world must produce exactly the same observed arrays and predictions as its biological-labeled alias.

## Reuse requires explicit boundaries

`src/models/fitness.py` supplies symmetric single features and a calibrated reference, but `FitnessCompletion.fit` explicitly requires two contexts. Do not duplicate one initiation or pass the second initiation as a second fitting context. Use the separately qualified scalar adapter and a newly named tournament contract. `src/scalar_screen.py` supplies training-only graph strata and stratified residual permutations; its historical split and eligibility rules must not silently become the tournament's rules. Its existing within-fold block construction is not a cold-gene split: a cold-gene experiment must remove every incident training and validation outcome.

`src/flexible_singles.py` fits a train-only symmetric Nystrom basis and rejects unsupported queries. It needs enough training rows and nonzero center distances, so degenerate or small-budget cases need declared refusal rather than changed basis size after failure. `src/flexible_prediction.py` defines a prospective fixed-order predictor and explicitly does not meet legacy replay. `src/evaluation_contracts.py` and `src/evaluation_precision.py` separately version finite arithmetic and underflow refusal. Preserve their historical behavior and strict expected failure; bind a new prediction precision contract before execution.

`src/replicated/data.py` separates query references from pair outcomes and checks metadata before converting counts. Preserve that ordering. Its library-total/log-after-mean endpoint is a different assay contract, so reuse its access pattern rather than its endpoint formula for COLO1. Actual constituent/control matching must remain source-specific.

The native comparator entry point `tools/study_native_blocks.py` binds pinned official code, dependencies and existing Q2 assignments, requires passing postfreeze checks, and refuses reuse of completed or partial output paths. Reuse these custody principles, but do not invoke that source-reading study entry point for the synthetic tournament. Any new composition model that does not use the pinned native implementation must carry its own name and contract.

## Inputs and selection

Use immutable copies of measured singles, permitted features, canonical pairs, and explicit availability/support masks. Fitters receive only selected TRAIN pairs and labels; selectors also receive selected VALIDATION pairs and labels. Neither object receives the world dictionary, TEST labels, second-initiation labels, full outcome summaries or a generator handle. Query objects contain identities and permitted inputs only. Every prediction has a support status or an abstention reason; no nonfinite value becomes a zero prediction.

Before applying availability, sort each partition by an independently salted stable hash of canonical pair identity and take `floor(fraction * partition_size)` rows. Missing selected observations remain charged and unavailable, with no replacement. Report nominal charged, observed usable and unsupported counts separately. Freeze the 25% convention explicitly: for the proposed partitions it charges 221 TRAIN and 88 VALIDATION pairs. A finite zero is observed; a missing observation is a separate Boolean state. All construct variants, reversed orientations and initiations inherit the pair's assignment.

Fit basis centers, feature scaling, response centering and nuisance coefficients on selected usable TRAIN only. Choose family configurations and the simple/full policy on VALIDATION only, using a frozen complexity order for exact ties. Shared main effects must be the same saved selected policy for residual families; jointly fitted variants need their own declared identity. Keep all fixed references in the score roster so selecting a weak reference cannot manufacture superiority.

## Concrete synthetic alternatives

Freeze distinct pilot/evaluation seed namespaces and separate random streams for singles, per-gene features, interactions, observation noise, masks, partitioning and optimization. Changing a mask or noise level must not redraw latent structure. Use all 1,770 canonical pairs of 60 genes and two initiations. Record the following equations or prospectively amended equations before generating evaluation outcomes.

- Additive: `mu_ij = s_i + s_j`, with singles independently measured rather than estimated from doubles. Zero-noise prediction by the fixed sum is an exact oracle.
- Signed low rank: `mu_ij = s_i+s_j + 0.5*(u_i dot u_j-v_i dot v_j)/sqrt(2)`, where each of u and v has two independent standard-normal coordinates. Preserve the same latent interaction across initiations.
- Nonlinear singles: `mu_ij = s_i+s_j + 0.5*tanh(s_i+s_j) + 0.25*(s_i-s_j)^2`. This tests flexible single calibration without asserting interaction recovery.
- Structured feature interaction: draw two-dimensional independent per-gene features z, then add `0.5*sin(z_i0+z_j0)*cos(z_i1-z_j1)`. Every entrant gets the same z, and feature-matched simple references are required.
- Sparse unpredictable edges: add `b_ij*a_ij`, with independent Bernoulli(0.05) b and standard-normal a. Persisting the edge in initiation two does not make an unobserved edge predictable.
- Guide/arm/batch nuisance: no biological pair term; add frozen arm-specific guide offsets and a batch offset to construct observations. Supply the declared nuisance covariates to every candidate and include a nuisance-adjusted reference. Hide no nuisance variable in a model-specific input.
- Persistent technical effect: byte-identical numerical arrays to the signed-low-rank world under the same seeds, with only a causal-description field changed. This is an identity control, not another independent success.
- Redrawn technical effect: share initiation-one arrays with the persistent technical world and independently redraw u,v for initiation two. Local prediction can succeed while shift prediction fails.

For each initiation/pair generate four construct errors as `noise*(sqrt(0.5)*e_pair + sqrt(0.5)*e_construct)`, making their pair-level dependence explicit. Preserve construct-level observations if claiming construct-level nuisance or aggregation tests; a two-by-pair mean alone cannot verify guide/orientation effects. Distinguish latent-mean loss from observed-outcome loss and declare the primary target before execution. Do not count 4 constructs or 2 simulated initiations as biological replication.

Use noise 0 and 0.1, three evaluation seeds and the registered 885/354/531 split. For noisy cells apply one fixed 10% independent mask and one fixed mask determined by a pre-outcome input covariate, such as the lowest input-quality decile. Freeze the exact mask unit (construct versus pair), rates and hash salts. Report all 96 cells and both budgets; report initiation-two results separately. If implementation omits construct-level nuisance or uncertainty, record the omission before freeze and narrow the deliverable rather than claiming the complete plan.

## Independent acceptance probes

1. **Access poisoning.** Replace unavailable TEST and initiation-two labels with exceptions or poison values before fitting/selection. Saved model state, validation selection and predictions must remain unchanged. Changing allowed validation labels may change selection, which is a useful positive control.
2. **Group and support checks.** Reject train/validation canonical overlap even when endpoints are reversed. Confirm no construct/orientation crosses partitions, no hidden replacement of unavailable budget rows occurs, and an isolated query gene abstains without being repaired from validation or test edges.
3. **Arithmetic oracles.** Check exact additive noiseless predictions, missing versus zero, symmetric repeated/reordered queries, train-only basis/scaler changes, and equal-pair aggregation with uneven construct counts. Source endpoints need separate hand calculations, including zero-count and pseudocount order.
4. **Causal and shift checks.** Assert identical low-rank/technical arrays, partitions, fitted checkpoints and predictions, excluding only causal labels in manifests. Assert initiation one is unchanged by redraw; score the independently redrawn second initiation as a separate task without refitting.
5. **Lifecycle refusal.** Change one prediction, checkpoint, input, code binding or roster member after sealing and require scoring/replay to refuse. Partial runs, stopped runs and existing output directories must not become valid scored runs. Installed-package replay must use the frozen saved predictions and contract, not retrain.

Root owns execution under one numerical process, one numerical thread and the 2 GiB process-tree cap. C's initial review is static. Independent tests will be authored only after implementation exists and root's interface document identifies the final callable contract; their execution must be coordinated with root's numerical slot.

## Source/control objections to resolve with A

A must establish both-arm native sequence/target/control mapping and exact endpoint lineage without opening responses. A non-essential-target partner is an intervention, not automatically a non-targeting reference. An annotation graph cannot prove measured singles or label-budget eligibility. Triplicate names cannot prove independent transductions. Unknowns must remain explicit blockers or narrow the prospective task; synthetic validators cannot pass source qualification. I have not yet read A's proposal, so these are independent criteria rather than a review of A's work.
