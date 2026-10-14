# Crossed-single feasibility result, 14 October 2026

The authorized, preregistered `opal_crossed_single_v1` study actually completed preparation, nested fitting, exact saved-state replay and sealed evaluation. Its development spending filter **failed**. This is exposed single-response development across three observed target contexts, with measured source responses and target training-identity calibration. It establishes no pair composition, unseen-context, independent biological or frontier result.

| Target | Rank-16 candidate | Locked selector | Mean-profile ridge | Difference ridge | Kernel | Shuffled control | Candidate gain | Filter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A549 | 0.03816697 | 0.03918590 | 0.03966040 | 0.03825730 | 0.04687266 | 0.05311202 | 2.600% | False |
| K562 | 0.03316807 | 0.03556390 | 0.03361423 | 0.03478723 | 0.04553622 | 0.05152143 | 6.737% | False |
| MCF7 | 0.04300216 | 0.04404643 | 0.04443351 | 0.03825636 | 0.04916786 | 0.05560400 | 2.371% | False |

Each target scores all 96 declared compounds on the same 512-gene control-defined panel at 1,000 nM/24 hours, with three whole-compound outer folds and three inner folds. All 684 logical model invocations are accounted for, including 612 learned fits and zero/source-mean predictions, with every nested/final state retained. All final inference replays are exact. The primary population has 288 compound-by-target responses and 147,456 gene outputs. Each family, including fallback, has the complete population. Unsupported comparisons remain visible and cannot pass the filter. `metrics.json` preserves all reference scores, fallback counts and paired conditional bounds rather than only the candidate/selector table.

The paired bounds use 2,000 whole-compound draws coupling the targets and are conditional on exposed measurements and selected procedures. They do not measure independent culture, refitting or population-transfer uncertainty. The 5% rule is a spending filter, not a biological useful margin. Primary treatment assignments come from the pinned corrected raw release and verified primary pData; independently initiated cultures remain unverified.

Actual run resources: 520.766 CPU seconds, 530.283 wall seconds, 422,600,704 maximum RSS among retained sampling points; continuous peak is unmeasured and 91,827,750 retained bytes at the resource receipt. One worker/thread; no installs, raw downloads or GPU work. This process receipt includes raw/primary hashing, fresh streaming preparation, all nested fits, serialization and scoring. Large prepared/inference/prediction artifacts remain local and are hashed in `local_asset_inventory.json`; the complete model includes source-reference profiles, PCA bases/scales, output coefficients, kernel banks and metadata.

Registered next action: close finite rank-16 probe; no rescue sweep. An unfavorable score does not trigger extra ranks, doses, seeds or metrics. Future progress requires a new identifying hypothesis or the independent measurement/reliability panel, not another rescue sweep. The pair pipeline's source refusal and all frozen historical configurations/results are unchanged.

## Reproduction

The frozen configuration and protocol in docs/frontier/crossed_single_feasibility_protocol.md record the source hashes, endpoint, information budget, folds, model families, exact inventory and limits. Run `.venv/Scripts/python.exe tools/run_crossed_single.py --output results/opal_crossed_single_v1/<new-unique-directory>` with the existing cache/environment. Existing outputs refuse overwrites. `registration.json` seals the exact code/config hashes before new expression scoring; `prediction_seal.json` seals all target folds before evaluator access.
