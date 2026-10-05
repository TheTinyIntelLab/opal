# Residual kernel improvement experiments

The new model combines a shared residual offset, a reference-scaled linear correction and symmetric
single-profile kernels. Inner development folds choose between the shared correction, summed-single
kernel, pair-product kernel and mixtures. The old neural architecture is not mandatory in this search.

All fitting and selection use the original development profiles. No historical evaluator responses
were loaded by these studies. Historical test findings informed the decision to investigate shared
corrections, so these results remain exploratory and require external confirmation.

## Design and results

Three seeds use five outer random folds and five outer partner folds per dataset. Each outer fit
uses three inner folds of the same protocol. The kernel procedure searches 67 configurations.
An expanded linear comparator searches 25 configurations: a shared offset plus global, ridge and
rank-four reduced-rank models at eight penalties from 0.01 through 100,000. Its feature scaling is
fitted within each training fold. Kernel scaling uses singles only. Both searches select by inner
equal-pair all-gene MSE. These are different model spaces and budgets, not a capacity-matched test.

| Dataset | Holdout | Relative MSE reduction across three seeds | Positive studies |
| --- | --- | --- | --- |
| norman-log2 | random | 1.78% to 2.70% | 3/3 |
| norman-log2 | partner | 1.13% to 3.38% | 3/3 |
| wessels-log2 | random | 2.50% to 3.98% | 3/3 |
| wessels-log2 | partner | 0.83% to 1.00% | 3/3 |

Partner holdouts overlap. The primary nested score weights each validation occurrence equally;
the artifact also reports scores after averaging each pair's repeated predictions. On Wessels,
averaging can reverse the comparison with the reference-scaled shared global control. The reported
gain is specifically against the separately tuned expanded linear search; it is not a universal
win over every linear variant or every aggregation rule.

No native published-method scores were added. Kernels, residual learning and symmetry are established
methods; this implementation alone does not establish methodological novelty or frontier status.

## Reproduction and checkpoints

Run `python main.py improve --seed 20261005 --output results/improvements/new-study` followed by
`python tools/compare_improvement_baselines.py --study-root results/improvements/new-study`.
Repeat with seeds 20261006 and 20261007. Existing study directories are immutable and refuse reuse.
The original seed's `norman-log2/checkpoint` and `wessels-log2/checkpoint` load with the existing
`Predictor` and `main.py predict` interfaces. They contain development-only weights and observed
single profiles. Predictions on registered historical test identities are stored unscored.
They remain research candidates, and zero ensemble spread for a deterministic fit is not calibrated
uncertainty. `export_verification.json` records numerical agreement and CPU measurements.

## PertAdapt resources

The supplied checkpoint loaded with `weights_only=True` and memory mapping. The archive contains
three backbone tasks with 119,354,430 tensor elements each. The gene list has 19,264 unique entries;
the finite GO mask is 19,264 by 19,264. Its checksum manifest is in
`data/public_methods/pertadapt_resource_audit.json`. These structural checks do not establish
native training compatibility or provide a PertAdapt benchmark. The published loader returns CUDA
models and the training script uses CUDA distributed execution; this installation has CPU PyTorch.

The next decisive evidence is an information-aligned native comparison and an independent
confirmation dataset or genuinely untouched experimental conditions. Additional seeds drawn from
these same development pairs do not supply that independence.

## Frozen foundation gene-table check

The uploaded gene-task positional table was tested as 45 additional kernel candidates,
with the 67 measured-single candidates retained. Exact symbol mapping and zero external
kernel rows for unmatched targets are recorded before fitting. The same outer/inner
seed-20261005 development folds were used for the real and shuffled-gene-table searches.
The checkpoint was loaded safely; no contextual backbone forward pass was performed.

Norman selects the same measured-single models in all outer folds. Wessels' real table
changes the random-holdout score by about 0.3%, but the shuffled table does at least as
well; its partner score is slightly worse than measured singles alone. This does not
support an identity-specific benefit from the pretrained table. The measured-single
candidate remains the exported model. These findings do not test the full PertAdapt
method or rule out useful contextual foundation representations.

Full scores and source hashes are in `results/improvements/foundation_summary.json`
and the two `foundation-position-*` directories. Reproduce using
`python tools/study_foundation_positions.py --output results/improvements/new-foundation`
and add `--shuffle-seed 6143` for the negative control.
