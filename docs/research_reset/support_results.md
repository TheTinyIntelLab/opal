# Original-test support diagnostic, 20 October 2026

Completion retains a local-assay advantage in the broad low-degree subgroup, but the original test has only four edges joining two genes in the external bank-novel gene set. The random original split therefore cannot resolve performance inside the externally tested missing block. Reduced degree alone is insufficient to explain the whole external failure; entirely block-held transfer remains unmeasured.

This is an exploratory no-fit replay on previously exposed data, governed by [the frozen protocol](support_protocol.md). Four synthetic tests passed before registration and new subgroup-loss calculation. The initial command through system Python failed during collection because that interpreter lacks installed `opal`; the documented `.venv` interpreter passed all four tests. Freeze then hashed code, tests, protocol and all inputs in [protocol.json](../../results/research-reset-support/v1/protocol.json). Run verified those hashes before scoring. There was no model selection, retraining or tuning.

The retained predictor reproduces all 16,294 historical test baseline predictions in each context within numerical roundoff: maximum absolute difference is 5.551115123125783e-17 for K562 and 3.469446951953614e-17 for Jurkat, below the prespecified 1e-12 tolerance. This is numerical rather than bitwise identity. Complete unrounded subgroup metrics and conditional paired gene intervals are in [results.json](../../results/research-reset-support/v1/results.json).

Original TRAIN-degree tertile cutpoints are 222 and 257; ties go into the higher bin. The external bank-novel endpoint union contains 37 genes. Results below report completion and original frozen-main-effect MSE, followed by percent MSE reduction. Negative reduction means completion loses. Groups overlap and are descriptive; the original random-pair holdout is unchanged.

| Group | Pairs | Genes | K562 completion MSE | K562 main MSE | K562 reduction | Jurkat completion MSE | Jurkat main MSE | Jurkat reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 16294 | 427 | .001375929806 | .001870485727 | 26.4400% | .000426300047 | .000500492420 | 14.8239% |
| Minimum endpoint tertile 0 | 8270 | 427 | .001734367711 | .002197651098 | 21.0808% | .000567011655 | .000662392278 | 14.3994% |
| Minimum endpoint tertile 1 | 6146 | 285 | .000989083304 | .001479955317 | 33.1680% | .000273921830 | .000325515554 | 15.8498% |
| Minimum endpoint tertile 2 | 1878 | 146 | .001063510814 | .001707835179 | 37.7275% | .000305337063 | .000360181450 | 15.2269% |
| Endpoint tertiles 0,0 | 112 | 46 | .001991553637 | .002317649971 | 14.0701% | .000867698340 | .001013657541 | 14.3993% |
| Endpoint tertiles 0,1 | 4177 | 281 | .001730985356 | .002172110422 | 20.3086% | .000559949537 | .000660919006 | 15.2771% |
| Endpoint tertiles 0,2 | 3981 | 288 | .001730681017 | .002221073236 | 22.0791% | .000565962060 | .000654055716 | 13.4688% |
| Endpoint tertiles 1,1 | 2120 | 139 | .000959883383 | .001360640609 | 29.4536% | .000281737814 | .000325549550 | 13.4578% |
| Endpoint tertiles 1,2 | 4026 | 285 | .001004459318 | .001542783728 | 34.8931% | .000269806110 | .000325497653 | 17.1097% |
| Endpoint tertiles 2,2 | 1878 | 146 | .001063510814 | .001707835179 | 37.7275% | .000305337063 | .000360181450 | 15.2269% |
| Zero external-union endpoints | 14142 | 390 | .001286419153 | .001774719781 | 27.5142% | .000402675251 | .000473347951 | 14.9304% |
| One external-union endpoint | 2148 | 343 | .001952706045 | .002484007759 | 21.3889% | .000581390001 | .000679093961 | 14.3874% |
| Two external-union endpoints | 4 | 6 | .008112005338 | .010989655444 | 26.1851% | .000668460629 | .000560738348 | -19.2108% |

Every group except the four-edge two-union-endpoint group meets the prespecified minimum of 50 pairs and ten genes, so each qualifying group receives the fixed 2,000-draw paired endpoint-multiplicity bootstrap. The four-edge group is retained and flagged as insufficient, with no interval. There are no empty groups in this realized test panel; the implementation and tests preserve empty groups with null metrics if they occur.

For the minimum-degree tertile 0, conditional loss-difference intervals favor completion: K562 [-0.0005817748128202799, -0.00035312384333819865] and Jurkat [-0.00012447423088771895, -0.00007039624297070652]. One-union-endpoint intervals also favor completion in both contexts. But the narrower low/low endpoint group has K562 interval [-0.0010536582786427973, 0.0002924210974912833] and Jurkat [-0.00039195992300292696, 0.000018985231081465824], both crossing zero. Thus the broad low-degree result is mostly supported by low/high edges and does not identify a low/low interaction capability.

These intervals condition on the deposited genes and cultures. They do not account for independent culture uncertainty or give multiplicity-adjusted subgroup discoveries. The sparse four-edge group cannot adjudicate missing-clique transfer, and its opposing K562/Jurkat signs should not drive an architecture decision. A future finite block-held experiment would be informative precisely because this replay leaves that distinction unresolved; it needs a separate training registration.

The frozen run completed in 1.1962 seconds with 256,094,208 bytes peak working set, below its one-worker, 600-second and 2 GiB caps. It saved aggregate JSON only. Reproduction commands are in the protocol; the historical model assets, original source bank and original predictions remain local and unchanged.
