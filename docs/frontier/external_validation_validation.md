# External evidence phase validation

Started from clean `main` at `8a5574b`. The preceding closeout's retained artifact hashes verified, allowing recovery and crossed-single work to remain closed. Current execution records are under `results/opal_external_validation_v1/`; the machine-readable receipt is [external_validation_validation.json](external_validation_validation.json).

| Executed command or operation | Actual result |
| --- | --- |
| Bounded official GEO metadata acquisition | Three source snapshots, 161,154 compressed bytes; 15 / 1,728 / four sample records. Full source snapshots remain local. |
| Scoped public count acquisition | 367,523 compressed / 1,043,127 expanded bytes; local scope recorded before acquisition, published narrative already exposed. |
| `python -m pytest -q tests/test_external_repeat.py` | Exit 0, four passed in 0.07 seconds; real fixtures cover identities, fractional counts, normalization, missingness, zero libraries, reordering and contact stripping. |
| `python tools/review_external_sources.py --input data/source_audits/opal_external_validation_20261015 --output results/opal_external_validation_v1/metadata-review-20261015` | Exit 0; all three acquired metadata identities checked and summaries generated, without reading expression. |
| `python tools/run_external_repeat_diagnostic.py --input data/source_audits/opal_external_validation_20261015 --output results/opal_external_validation_v1/repeat-diagnostic-20261015` | Exit 0; every source row and fifteen samples retained, five within-arm variances and fifteen pairwise differences, no predictor/biological inference. |
| `python tools/replay_external_repeat_diagnostic.py --input data/source_audits/opal_external_validation_20261015 --run results/opal_external_validation_v1/repeat-diagnostic-20261015 --output results/opal_external_validation_v1/repeat-replay-20261015.json` | Exit 0; independent CSV reader/centered-sum replay, maximum discrepancy 2.7755575615628914e-17. |
| Unchanged fourteen-gate readiness checker | Exit 0 for report generation; zero gates pass. Exact requirements and resolving routes retained in `external_validation_gates.json`. |
| `python -m pytest -q` | Exit 0, 394 passed in 49.04 seconds. No scientific claim follows from fixtures. |
| `python -m build --no-isolation --outdir tmp/external-validation-package-20261015` | Exit 0, wheel and source archive built. Existing-dependency direct wheel imports and raw/private-correspondence/history exclusions pass. No clean dependency install or remote CI run. |
| Offline cached Tectonic | Exit 0, 48 pages, new evidence table visually inspected; no missing-reference/citation warnings. Existing underfull-box/fontconfig warnings remain. Exact compiled inputs/PDF are hash-bound. |

All new empirical table cells and dimensions map to retained result fields in `new_claim_map.json`; no unchanged historical scientific evaluator or costly fit was reopened. The small diagnostic used 0.03125 CPU / 0.05312 wall seconds after imports, with a 101,498,880-byte OS peak working set. GEO metadata/count payloads total 528,677 compressed bytes; two Zenodo metadata records and the publisher-checksummed supplemental PDF bring total locally acquired source payloads to 721,326 bytes. No GPU calculation, install, large archive acquisition or predictor sweep occurred.

Five messages pursued four source-specific inquiries using authorized Gmail tools, with exact recipient/SENT readback. The insitro attempt produced a verified inactive-mailbox failure; a single Brown first-author fallback was sent. The remaining four routes have no delivery proof or substantive reply. Exact messages/IDs are retained privately and excluded from distributions. No source permission, reviewer evidence, useful effect, culture or agreement was fabricated. The external dependencies remain open rather than being hidden behind software success.

Documentation/link checks, final package checks, receipt hashes and the outreach-check time are recorded in the JSON receipt after execution. Raw inputs and future private replies remain local, with compact custody receipts retained. GxE recovery code/config/measurement results and all frozen scientific configurations remain unchanged.
