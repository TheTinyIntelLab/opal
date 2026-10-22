# Synthetic replication results, v1

The [registered 30-cell experiment](replication_protocol_v1.md) completed without changing its menu, seeds or thresholds. [All results](../../../results/research-continuation/replication-v1/results.json), [the pre-scoring prediction manifest](../../../results/research-continuation/replication-v1/all-predictions-frozen.json) and [external resource receipt](../../../results/research-continuation/replication-run-v1.json) retain inputs, every selection and every prediction. No biological file was read or scored, and no third-party simulator was executed.

The prospective synthetic recovery gate passed. Against the validation-selected reference, second-initiation observed-target MSE fell by 91.3284%, 90.3181% and 88.0715% in the three noisy persistent-biological seeds. Their mean MSE was 0.009777590556816583 for completion versus 0.0967881691976926 for the reference and 0.11983652553971043 for shuffled completion. This establishes recovery of a reusable synthetic pair surface with independently generated second-initiation measurement noise. It does not establish biological validation, independent-study confirmation or a label-efficiency claim.

| World, construct-noise SD 0.1 | Reference MSE, initiation two | Completion MSE, initiation two | Interpretation |
| --- | ---: | ---: | --- |
| Additive | 0.006466032825490419 | 0.00953945324683102 | Completion overfits residual noise. |
| Persistent biological structure | 0.0967881691976926 | 0.009777590556816583 | Prespecified structure is recoverable. |
| Persistent technical structure | 0.0967881691976926 | 0.009777590556816583 | Exactly the same predictive success cannot identify its cause. |
| Redrawn technical structure | 0.09216552908856354 | 0.18473158032657855 | Strong local prediction fails when the pair pattern changes. |
| Independent sparse edges | 0.09271449903611599 | 0.11268782058416685 | Arbitrary unseen edge effects are not inferred by this completion model. |

Each entry averages the three prespecified seeds, not biological replicates. Four constructs have correlated measurement errors and count as one pair mean per initiation. Both the noiseless and noisy grids remain in the receipt; the table does not select favorable cells. The two persistent causal worlds have byte-identical observed arrays and exactly identical predictions across every seed and noise level. An assay can therefore produce a real completion advantage that persists across initiations while leaving biology-versus-technical attribution unidentified.

All 60 completion checkpoints replayed exactly. The reference menu comprised 12 fixed candidates: zero, mean, additive, minimum and calibration/node ridge at four penalties. This is a finite comparison, not superiority to every possible nonlinear or mechanistic reference. The ideal conditional-mean oracle has access to true singles and the specified reusable structure, and is not an eligible fitted competitor. On a redrawn technical initiation it predicts zero new technical effect; on independent unseen edges it uses the realized population mean as a descriptive ideal. These oracle choices were fixed in bound code before execution.

The single process took 93.9494 measured experiment wall seconds and 92.9531 process CPU seconds, with 290,631,680-byte lifetime peak working set. The external watchdog recorded 95.5146 wall seconds, 94.3125 sampled process-tree CPU seconds and 294,862,848-byte sampled tree peak. Run artifacts occupied 2,699,151 bytes at completion. It enforced the 600-second/2-GiB caps, reported no bound-file drift and respected the unchanged disk reserve. Exact invocation:

```powershell
.venv/Scripts/python.exe tools/wave5_run_check.py --receipt results/research-continuation/replication-run-v1.json --timeout 600 --peak-bytes 2147483648 --bind tools/continuation_replication_v1.py --bind results/research-continuation/replication-freeze-v1.json -- C:/Users/USER/Desktop/code/thetinyintellab/opal/.venv/Scripts/python.exe tools/continuation_replication_v1.py
```

No production model changed. The justified next scientific decision is to qualify a source and endpoint that can test persistence with actual initiation provenance and matched controls. The negative cells rule out a general claim that completion improves arbitrary screens. They do not authorize architecture or seed sweeps to rescue the stopped biological comparisons.
