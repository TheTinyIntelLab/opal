# Q2b fixed flexible singles results

21 October 2026. The registered 64-center flexible singles function does not
explain away the frozen completion advantage. Completion beats the
validation-selected flexible policy by 13.1324% to 34.9003% MSE in every task,
with negative upper conditional gene bounds and no infection-target loss. The
phenotype-only function fails the registered conditional noninferiority criterion
in every task. This supports retaining local pair completion for further
diagnosis; it does not identify molecular interactions or establish independent
source generalization.

## Lifecycle and unchanged comparison

[The frozen protocol](flexible_singles_protocol.md) and thirteen prerequisite
tests, passing in 10.59 seconds, preceded fitting. The actual registration uses
`E:/opal-wave2-20261021/q2b-v1`, with the passing Q1b dependency at
`E:/opal-wave2-20261021/q1b-v2`. The earlier Q1b C-volume attempt and failed disk
reserve replay remain preserved separately. Q2b waited for the new E-volume Q1b
result and exact replay rather than using that failed attempt.

The Q1b prerequisite and replay success flags, registration/log/JUnit hashes,
result and prepared-array hashes, source/code hashes and absence of failure
receipts were independently verified. This also checks the explicit success
fields that the Q2b consumer does not itself inspect. The Q2b registration SHA256
is `6faccc7dfb99526e79643e9d0ba744fd4d8dcf70058aa2527d9f24e5daa4e067`.
Scientific results SHA256 is
`0a6164d7ad2e8adc14c3d7af9725e9f27ed01997973280442b961b5dbe3ee936`.

All six original scalar banks, targets, measured singles, thirty assignments and
full test populations are unchanged. The exact original completion and
selected-simple predictions are comparators. Q1b guide matching supplies an
integrity prerequisite, not an alternative target or a reason to exclude hard
Q2b pairs. Both flexible families use the same training outcomes and shared
validation labels as the original procedures. Centers, bandwidth, coordinate and
feature moments come from training pairs only. The node family adds the same
gene incidence nuisance as the existing node baseline. No residual operator was
refitted.

## Measured results

Losses below use the mean infection target, equal canonical-pair weights and the
original fold-concatenated test order. The interval is the paired
completion-minus-selected-flexible MSE difference, conditional on the measured
gene panel. Its units are squared phenotype units. Raw losses cannot be compared
across source scales.

| Task | Test pairs | Phenotype-only MSE | Selected flexible MSE | Frozen completion MSE | Completion gain | Conditional 95% difference interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Horlbeck K562 | 16,210 | .002869775506006579 | .0018655760722959855 | .0013418626513990082 | 28.072477% | [-.0006651329415849076, -.00042383986348524394] |
| Horlbeck Jurkat | 16,210 | .0006526117235840973 | .0004997913328788439 | .0004341567404863752 | 13.132399% | [-.00008294458513421332, -.00005084283827847431] |
| Environment 1 | 9,833 | .0008367450125920616 | .0005418794384177554 | .0003655341517065346 | 32.543270% | [-.00022862865038799626, -.00013511158608405118] |
| Environment 2 | 9,833 | .0010416240484100582 | .0007139097854888607 | .00046475313111452585 | 34.900300% | [-.0003221787767041977, -.00018880283742102552] |
| PARPi 2022 | 29,016 | .00042709947452731107 | .00029571251032583975 | .00021159725552219704 | 28.444943% | [-.00010054323642418922, -.00006960459200264253] |
| PARPi 2023 | 28,651 | .0003942975299780891 | .00029171278955501107 | .00022298166618172846 | 23.561231% | [-.00008107333166084485, -.000057172050318261975] |

The selected flexible policy chooses the node family in all thirty folds. Node
penalties are 1 in sixteen folds, 10 in six and 100 in eight; phenotype-only
penalties are 1 in twenty-seven folds and 10 in three. These are validation
choices, not test-based model selections. Both infection-target point losses
favor completion over the selected flexible policy in every task. All six
completion residual gates pass. Neither the aggregate nor any individual-task
phenotype noninferiority gate passes. The outputs cover 109,753 pair-context
predictions, not that many independent interactions or biological replicates.

## Independent checks and resources

[The independent receipt](../../results/research-wave2-flexible/independent_review.json)
reconstructs all 240 validation losses from saved checkpoints with zero numerical
difference and reproduces every family/penalty choice. It verifies train-only
coordinate moments and seeded center rows, all original file hashes and the Q1b
receipt chain. Selected test predictions are bit-exact, including original test
population/order. Independent metric arithmetic reproduces primary and secondary
metrics; independent endpoint-multiplicity bootstrap and leave-one-gene-out
arithmetic reproduce all twelve primary paired intervals and scientific gate
flags within floating-point tolerance. No audit fit was performed. The audit
took 41.03 seconds on one numerical thread while the separate budget branch ran.

The final fit receipt records 52.4746824 seconds wall time and 50.984375 CPU seconds, with peak
working set 418,381,824 bytes. Fit plus scoring used 58.9834241 wall and 57.375 CPU
seconds. The complete fit, score and exact replay used 65.3896031 wall and
63.765625 CPU seconds, within the registered 1,800-second shared cap. Arrays and
scientific JSON replay exactly. Models and prediction arrays remain local;
compact lifecycle receipts and fold choices are preserved under
[q2b-v1](../../results/research-wave2-flexible/q2b-v1).

To audit the existing run without refitting:

```powershell
.venv/Scripts/python.exe tools/audit_flexible_results.py --run E:/opal-wave2-20261021/q2b-v1 --output <new-json-path>
```

To reproduce the complete registered procedure, use a fresh output path and the
same qualified inputs, environment and frozen code. Run `freeze`, `verify`,
`fit`, `score` and `replay` through `tools/study_flexible_singles.py`; `freeze`
requires `--block-run C:/Users/USER/Desktop/opal-research-next-20261020/block-v2`
and `--q1b-run E:/opal-wave2-20261021/q1b-v2`. Preserve the existing attempts.

The result narrows one competing explanation: this fixed flexible singles map,
even with additive gene nuisance, cannot reproduce the local completion ceiling.
It does not exhaust nonlinear single-effect models, eliminate technical pair or
guide effects, or prove that the completion correction transports. Sources are
already exposed and cohorts overlap; gene bootstrap sensitivity supplies no new
source or initiation replication. A residual refit against a stronger nuisance
reference would require a separate registration. Nothing here authorizes a
larger architecture or opening SPIDR responses.
