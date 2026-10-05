# Manuscript evidence review

The manuscript is a substantive internal methods-and-results draft, not submission-ready. It reports the completed synthetic suite, an inconclusive Norman structural comparison, and the negative comparison on Wessels. Both genetic primary evaluations are complete. No biological state-of-the-art, clinical, zero-shot, or completed-v1 claim is justified.

## Claim and evidence matrix

| Claim | Recorded evidence | Manuscript interpretation |
| --- | --- | --- |
| The pipeline recovers compatible synthetic structure | `results/synthetic/summary.json`, low-rank skill 96.76%, cross-coordinate skill 95.63% | Engineering validation only; no biological implication. |
| Additivity is scale-dependent | Count-additive generator has 47.10% skill on log endpoint | A residual on one scale can arise without an extra interaction on another. |
| Pair models can overstate structural benefit | Global-distortion selected MSE 0.00110447 versus simpler 0.00104015 | Improvement over additivity does not establish pair specificity. |
| Low output rank is universally beneficial | Higher-rank synthetic case selects unrestricted bilinear output | Unsupported; rank must remain a selected empirical restriction. |
| Recovery survives a harder synthetic holdout | Five total low-rank random splits, including base primary; five partner folds | Synthetic generalization only; partner tests overlap. |
| Wessels selected neural correction beats additivity | MSE 0.005423558 versus 0.013936912, skill 61.08% | Supported conditional on the primary internal protocol. |
| Wessels pair-specific products improve prediction | Selected sum-network has no product block and loses to global correction, MSE 0.004214949 | Unsupported and contradicted by the selected primary comparison. |
| The network beats the stronger Wessels comparator | Improvement -0.001208610; both descriptive intervals negative | Negative result; no significance language because measurement and replication uncertainty are missing. |
| Norman selected correction beats additivity | MSE 0.007169187 versus 0.008515040, skill 15.81% | Supported conditional on the primary internal protocol. |
| Norman correction improves on strong alternatives | Development-selected reduced-rank comparator MSE 0.007268184; improvement 0.0000989974, both intervals cross zero | Inconclusive; ordinary ridge also has a better observed test score of 0.006891604, without retrospective reselection. |
| Wessels conclusion survives expression-scale sensitivity | Count selected MSE 0.19684419 versus simpler 0.16912624 | Negative comparison persists; cross-scale MSE magnitudes cannot be compared. |
| Reliability has been assessed | Split-half residual cosine 0.2891/0.0838, emergent-panel, spread/error, retrieval, examples, and CPU measurements | Diagnostics are complete but do not establish biological replication, extended-output coverage, or calibrated abstention. |
| Norman scale sensitivity establishes a structural advantage | Count sum-network MSE 1.03911249 versus selected ridge 1.25943739, both descriptive difference intervals cross zero | Inconclusive, with no multiplicative-feature selection. |
| Chemical transfer is validated | Seven-pair exploratory run uses four development and three held-out pairs; bilinear MSE 0.01222957 versus mean residual 0.00994341 | Negative exploratory comparison; too few pairs for the intended chemical validation gate. |
| Opal outperforms public methods | No aligned CPA, GEARS, SLIM, PertAdapt, or compatible NAIAD results | No ranking or state-of-the-art claim. |
| The full v1 protocol is complete | Internal CPU batch, synthetic secondary studies, primary Wessels test | Unsupported; completion gates remain explicitly listed. |

## Method checks

The prose follows the implementation's transform-after-condition-mean normalization, equal-stratum aggregation, frozen reference-only gene eligibility, exact matching of pair and single stratum support, and distinct evaluator bundle. The sum-network receives sums and absolute differences; calling it a pure sum-only predictor would misdescribe its inputs. The fine-tuning candidate updates an embedding table, not a demonstrated encoder for unseen targets. Baseline and neural selections are separate development choices, so the neural selection may lose to a simpler method at test time without replacing it retrospectively.

The bounded rank and ridge search is an implementation amendment, not the complete capacity protocol. The manuscript distinguishes fixed-hyperparameter learning curves from subset-retuned sample efficiency, optimization seeds from independent data replication, ensemble spread from calibrated uncertainty, and descriptive pair/network bootstrap intervals from population inference.

## Citation and integrity checks

All eight cited keys occur in `paper/references.bib` and were checked against `docs/source_audit.md`. Dataset, model, and evaluation claims cite the appropriate primary sources. SLIM remains explicitly a preprint. Published comparison scores are not imported into the result tables. No invented references, author identities, affiliations, funding, conflicts, ethics approvals, public archive identifiers, or chemical results are supplied.

The bibliography still requires final author-level metadata inspection before submission, particularly online-versus-issue dates and PertAdapt's issue placement. The focused implementation bibliography is not represented as a systematic review. A Chinese abstract from the general academic-paper skill is omitted because the requested manuscript is an English technical research artifact and no translated audience or venue is specified.

## Review decision and next revisions

The strongest part of the draft is the separation between synthetic recovery and biological evidence, including the inconclusive Norman and negative Wessels comparisons. Methodological scope is adequate for an internal reproducibility report. Evidence sufficiency blocks a full-v1 or comparative submission claim because independent chemical validation, aligned public-method benchmarks, and measurement/replication-aware uncertainty are missing.

Before submission, execute registered biological robustness studies, resolve the chemical source amendment, and run compatible public comparisons with disclosed external resources. Expand capacity and rank evidence before claiming efficiency or low-dimensional biology. Responsible authors must complete CRediT, affiliation, funding, conflict, archive, license, and venue-specific AI declarations. Build verification and generated tables/figures are handled by the parent implementation task; this review does not certify successful PDF compilation.
