# Current-state audit

Audited 11 October 2026. This is an evidence audit for a new research plan, not a new biological experiment.

## Inspection scope

The available manuscript is `paper/main.pdf`, 43 pages, SHA-256 `e8db47bfa7797d1df72028ad5af5fbbb20046a3beeaa68e168c4915b7c9faa98`. Its full extracted text was read, including the page-43 variance addendum. The originally named `main-1(1).pdf` is absent; the user's final instruction identifies this local PDF instead. Local HEAD was `d12db0e53ec841d234e6b7913b7584f248674ba3`. The local checkout was clean before planning edits. The public GitHub URL returned 404 to the web reader, so public accessibility was not established. The local checkout provides substantive source and evidence access regardless of that web failure.

The audit inspected actual implementations in `src/series_bilinear.py`, `src/information.py`, `src/contextual/model.py`, `src/contextual/benchmark.py`, `src/models/covariance_kernel.py`, and `src/replicated/models.py` and `evaluation.py`; experiment contracts; model-series and information-study summaries, claims, ledgers and status records; replication qualification and public-source diagnostics; release directories and manuscript build provenance. Specialist reports record further inspection. Saved results were read rather than generated again. Planning did not reopen protected response banks or re-score historical tests. This is not a bytewise audit of every raw matrix or every checkpoint. Existing replay/CI receipts are inspected evidence of previous verification, not tests performed by this planning cycle.

## Findings verified against saved evidence

| Claim | Evidence and audit verdict |
| --- | --- |
| Current task is primarily observed-single population-mean combination prediction | Manuscript methods and actual reference interfaces require matched singles, ordered genes and assay/scale/context support. Partner holdout keeps query singles. It does not establish unseen intervention prediction. |
| Original neural gains over additivity do not establish strong-reference superiority | Norman neural MSE 0.007169187 vs frozen reduced-rank 0.007268184 has conditional intervals spanning zero. Wessels neural 0.005423558 vs global 0.004214949 loses. Ordinary Norman ridge 0.006891604 is a descriptive test comparison and cannot replace the frozen selection. |
| Residual kernels are strong internal references | Saved nested architecture comparisons favor kernels over tested neural/GP/TabICL candidates. Search budgets differ and histories are exposed; this is not proof that all neural representations fail. |
| Interaction adds below 0.1% on responsive primary panels | `results/opal_model_series_v2/report/claims.json` records 0.0778822343% Combi-seq and 0.0838481244% TNF-positive, zero TNF-negative. Combi-seq matched main effect MSE 2.105071 also beats Interaction 2.109919. |
| Guard is not promoted | Guard loses to Base and constant mixing on the two supported primary panels. Norman retrospective gains, 3.168524% log and 4.988153% count, are views of one exposed source. TNF-negative uses fallback; incomplete constant-mix support prevents a full-population score. |
| D-SPIN frozen procedure fails superiority and shifted uncertainty | Manuscript table 12 and preserved context-transfer records give 0.03402466 vs source reference 0.03310752, about 2.77% worse. Source nominal-90% radius covers 34.3% of held conditions. Lineages/batches are not independently initiated cultures. |
| Tested additional context, identity, products and priors have no robust advancement | The manuscript preserves failures and small/scale-sensitive exceptions. Prior-shuffled controls sometimes match real-prior gains. The sci-Plex numeric-dose shuffle is a no-op at a single dose and cannot test descriptor information. None of these observations refutes all descriptors, context biology or learned representations. |
| Projected-single variance study failed | `results/opal_information_v1/development/summary.json` has `information_gate_passed=false`, 96 folds, no confirmation and zero new independent cultures. TNF-positive partner variance MSE 0.0825683915 vs Base 0.0827597015 gains 0.231163%, but mean 0.0825437889 and capacity 0.0825393157 are better. TNF-negative partner heads all use Base fallback. This rejects the tested head, while its TNF-negative information test is support-limited. |
| Additional Wessels pair coverage helped one controlled development comparison | Table 9 gives random MSE 0.005032 to 0.004224 and partner 0.009868 to 0.009382 with extra pairs, about 16.05% and 4.92%. Fixed validation, exposure history, support thresholds and weaker strict-partner sensitivity limit generalization. |
| Strong engineering does not establish frontier biology | Safe checkpoints, prediction replay, provenance, installed execution and CI exist. Native comparison eligibility, causal interaction attribution, biological replication, shifted uncertainty and prospective utility remain open. |

## Corrections to stale planning statements

The old active plan points to a 42-page PDF and an earlier hash. The current 43-page manuscript is the correct planning input. The latest variance experiment is finished and must not be assigned again. Earlier source documents describe PertSpectra as expression-unopened, while subsequent studies have exposed it; freshness must come from the latest access ledger, not an old paragraph. The original model-series router has an internal validation dependence repaired in v2, not demonstrated outer-test leakage. Exact-revision remote CI passed on `bf564476a0c079dad35b8c84a5165ba2324dea90`; that does not certify a later scientific implementation revision.

The variance precision sensitivity has a TNF-positive global-pair gain of 2.844210% over Base, yet loses to its shuffled arm and fails the full registered criterion. Reporting only this favorable value would reverse the scientific interpretation incorrectly. Similarly, a different residual target does not make fixed-additive squared loss a different statistical objective.

## Resource evidence

The saved information ledger records 2,924,808,209 retained artifact bytes under a 3,000,000,000-byte cap, leaving 75,191,791 bytes. Its 3,142,132,144-byte cached raw source is explicitly outside that historical artifact ledger. CPU accounting is an observed lower bound of 450.53125 process-tree seconds, not complete training compute; GPU seconds and historical native training RAM are unknown. Fresh planning-time volume inventory showed about 5.51 GB free on C, 0.736 GB on D and 0.636 GB on E. D/E remain read-only under the historical research allowance. A new multi-context dataset or environment needs a new storage plan; current spare volume capacity is not an implied expanded artifact allowance.

## Evidence classes

Keep historical fixed results; exposed development; newly frozen development; fresh external confirmation; independent biological replication; and engineering verification in separate records. A new split of exposed data is still exposed development. Technical sample labels, cells, genes, seeds and overlapping folds do not become biological replicates. Published method adaptations must preserve their input and protocol differences. The current cycle edits documentation only.

The machine-readable snapshot is [audit_manifest.json](audit_manifest.json). The unchanged predecessor plan is [historical/plan_before_frontier_20261011.md](historical/plan_before_frontier_20261011.md).
