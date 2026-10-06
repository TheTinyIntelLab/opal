# State-response benchmark

Opal predicts condition-mean effects relative to a matched control on a declared
expression scale. Its state-conditioned prototype is experimental. The
registered development endpoint is equal-condition gene-vector MSE, with a 5%
material improvement threshold against the source-selected matched reference.
Context folds hold whole biological groups together; cells and batch suffixes
are never treated as new biological experiments.

## Information access

Every task permits exact treatment identity and pretreatment metadata, never
post-treatment expression as a baseline. Baseline controls are permitted at
inference on these reference-conditioned tasks. Source outcome access means only
labels outside the registered holdout. External resources and pretraining default
to none; a separate exposure registry must authorize them before training.

| Task | Training outcomes excluded | Query singles | Query combinations | Claim unit |
| --- | --- | --- | --- | --- |
| Unseen pair, observed singles | Held pair outcomes | Measured, explicitly allowed | Never | Unordered pair outcome |
| Held partner | All combination labels containing designated partner | Declared as measured or estimated | Never | Partner in combination training |
| Main-effect context transfer | Every query-group single outcome | Never as model input | Never | Biological state group |
| Observed-single pair context transfer | Every query-group combination outcome | Measured, explicitly allowed | Never | Biological state with measured references |
| Joint pair/context transfer | Query-group combinations and held pair globally | Explicitly declared | Never | Both state and pair identity |
| Intervention identity transfer | All forbidden response labels, including pretraining | No forbidden response reference | Never | Intervention identity, beyond familiar edges |
| Dose transfer | Registered exact doses and any forbidden related outcomes | Contract-specific | Never | Dose with units and fixed exposure, interpolation or extrapolation |
| Reliability | Disjoint risk-fit, calibration and evaluation groups | Match predictor's declared route | Never | Declared error population and target |
| Higher order | Genuine held higher-order outcomes | Measured if declared | Never | Actual simultaneous higher-order intervention |

These are benchmark contracts, not nine established capabilities. The executable
biological build evaluates main context transfer and pair/context development;
the other tasks retain existing studies or explicit unavailable status in the
capability ledger. Dose, higher-order and ordered schedules cannot be inferred
from an interface accepting a vector.

## Preprocessing and selection

The reference schema retains exact reagent, modality, dose and exposure records.
Main-response folds remove all held-group singles from every source statistic.
Control PCA, control scales, target means, response bases and hyperparameter
selection use source folds only. Biological state is encoded from matched
baseline expression, without a learned context-ID table. EmeraldBay's historical
gene panel used all declared controls; this query-control panel adaptation is
explicit and does not permit query treatment outcomes. The new sci-Plex panel
uses the same declared controls-only adaptation. Fitted PCA remains frozen.

Source inner context folds select a target mean, regularized per-target state
regression, or control kernel. Two compact learned configurations have matched
source selection. Outer comparisons use one fixed population of observed query
conditions whose exact intervention is supported in the outer source. Excluded
conditions are reported once for all models. Two neural seeds are averaged.
Response magnitude strata and per-condition failures are evaluator diagnostics,
never population-selection rules.

The main model encodes identity, numeric exact-dose descriptors with availability
masks, and source mean effects through separate paths. All encoder ablations
retain the same source-average output anchor, so "descriptor" here means a
descriptor-encoder ablation for an already source-supported exact intervention.
It does not mean prediction for an unknown intervention. Shuffled descriptors
and shuffled baseline assignments test contribution at comparable capacity.

## Combination and training comparisons

Combination experiments use only EmeraldBay's 54 source-development pair rows.
The original calibration and pair evaluator files are not opened. Each outer
fold removes a whole cell line. Source singles pretrain the main model, followed
by measured-single residual targets with the main model fixed. Sum/difference
and sum/difference/product constructions enforce pair swapping exactly. Pooled
and state-dependent versions have equal training budgets. Joint models retain
single supervision, and independent held-single errors remain visible.

Frozen-encoder joint, fine-tuned-encoder joint and no-single-pretraining joint
models use the same pair epochs and two residual seeds. A no-pretraining model
still receives the source-average anchor and fold-fitted bases; it removes neural
single-head training, not all single information. Total compute differs by the
declared pretraining stage. No external expression pretraining, mixture of
experts or learned routing is used.

The matched residual kernel is selected on inner estimated-single predictions.
Inner query singles are masked from main fitting and kernel preprocessing.
Oracle-single diagnostics reuse the trained residuals with measured query singles
and do not choose a deployment model. Every pair report includes main-effect,
residual and cross-term errors; improved total error can reflect cancellation.

## Reliability and efficiency

Risk fitting uses source-fold distance, exact-intervention support and predicted
response magnitude from out-of-fold predictions. Risk-fit, calibration and
evaluation biological groups are disjoint, but their underlying model folds
share training labels, so this is development evidence. Calibration takes one
maximum RMSE discrepancy per group. Too few groups return an unavailable radius,
without substituting per-cell sample sizes. Accept-none and accept-all policies
make unusable coverage visible. The frozen threshold is vector RMSE 0.25 with
at least 50% accepted coverage; the original D-SPIN radius remains a failed
transfer and is not re-evaluated here.

Context-efficiency curves retune simple models at each allowed source-context
budget while holding the query population fixed. Three nested orders share
labels and are dependent. A condition-assay cost unit is hypothetical, not an
observed monetary cost. More cells and independent cultures are separate budgets.
New risk-based acquisition is not justified by the failed development gates;
prior retrospective acquisition replays remain recorded, without a prospective
value claim.

## Commands

```powershell
.venv/Scripts/python.exe tools/download_state_source.py --raw data/raw/state
.venv/Scripts/python.exe tools/prepare_state_source.py --output data/prepared/sciplex-state-v1
.venv-gpu/Scripts/python.exe tools/run_state.py --dataset data/prepared/sciplex-state-v1 --output results/state/sciplex-state-v1 --device cuda
.venv-gpu/Scripts/python.exe tools/run_state.py --dataset data/prepared/emeraldbay-pilot-v1 --output results/state/new-main-and-pairs --device cuda --combinations
.venv/Scripts/python.exe tools/state_diagnostics.py --dataset data/prepared/emeraldbay-pilot-v1 --main-run results/state/emeraldbay-state-v1/main --output results/state/new-diagnostics
```

Choose new directories for another run. Optimizer, RNG and preprocessing state
are resumable only under matching data/configuration/protocol fingerprints.
Safe inference arrays exclude evaluator labels and restore without PyTorch.
Inputs must contain processed matched-baseline expression, exact gene order,
assay and scale. Effects, absolute expression, constituent contributions,
residuals, provenance and unavailable reliability status are explicit outputs.

The completed results are in [state_results.md](state_results.md). Current
two-seed exports are local under `results/state/portable-v3/*/checkpoint`; all
three pass the isolated Python/CLI checks in
`results/state/install-v5/installed_validation.json`. A baseline bundle contains
only `baseline`, `genes`, `assay` and `scale`. For example:

```powershell
opal state-predict --checkpoint results/state/portable-v3/sciplex/checkpoint --baseline my-baseline.npz --single "(+)-JQ1@1000nM:24.0" --output prediction.npz
```

Exact intervention identities come from the checkpoint manifest; compatible
controls must already be processed using the declared human-count/log2 assay
contract. Supplying a gene-shaped vector from another assay does not qualify it.
