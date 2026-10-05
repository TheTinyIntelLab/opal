# Resolution of independent assistant review

The review in `independent_review.md` is an internal assistant review, not
external scientific peer review. These corrections were made after all running
biological studies completed, preserving their fitting source snapshots.

| Finding | Resolution | Verification |
| --- | --- | --- |
| Latency prose drift | Retained each primary timing observation in `inference_timing.json`; analysis reuses it; manuscript numbers now agree. Separate memory records describe a fresh process and do not overwrite latency. | Artifact/prose numeric check; paper rebuilt. |
| Evaluator identity/order missing | New preparations write `evaluation_manifest.json`; development freezes an opaque evaluator SHA; final checks it before fitting and again before reading, checks ordered pair IDs and shape, and records SHA in test access. | Altered evaluator and transport-consistent order interventions pass rejection tests. |
| Stale source discovery status | Source registry and audit now record successful count downloads, hashes, acceptance and five prepared protocols. | Preparation manifest hashes checked; raw redistribution licenses remain qualified. |
| CV aggregation unspecified | Paper states validation-selected stopping, seed-average candidate errors, median refit epoch, final prediction averaging, and overlapping partner-fold aggregation. | Compared against frozen `benchmark.py`. |

Two additional implementation corrections accompany the review: fixed-basis
rank is capped when a tiny training subset has fewer basis columns than the
requested rank, with the effective rank serialized for loading; final fitting
uses an exclusive lock and writes failure status on rejection. The rank cap does
not alter the completed large-sample runs. The preparation intervention test
changes only evaluator pair expression while preserving library size, cell IDs,
eligibility and split metadata; allowed references and preprocessing stay identical.

`tools/audit_saved_runs.py` checks all 40 completed run fingerprints, prediction
hashes, snapshot identities, evaluator index order and reported equal-pair MSE.
It records current evaluator hashes retrospectively. That is current-file
integrity evidence, not proof that historical evaluator hashes were frozen at
preparation. Original selections are retained, and unfinished historical runs
must use their saved source snapshot rather than silently crossing code versions.

Verification: 30 tests pass; the rebuilt wheel loads the Norman research
checkpoint without importing repository source, using the existing scientific
dependency installation. No GPU agreement or full clean dependency installation
is claimed.
