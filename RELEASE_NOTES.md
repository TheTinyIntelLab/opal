# Opal v1.0.0

Opal's first research source release provides expression and growth prediction code, explicit input/support contracts, the research paper, frozen protocols and compact scientific evidence. GitHub contains no trained weights or biological reference banks; model artifacts have a separate Hugging Face release.

The retained [six-task growth benchmark](results/model-delivery-20261027/benchmark.json) reports pooled five-fold procedure results, not scores of the single exported fold-0 checkpoints.

| Task | Selected simple MSE | Completion MSE | MSE reduction |
| --- | ---: | ---: | ---: |
| K562 | 0.00190091929 | 0.00179069074 | 5.80% |
| Jurkat | 0.000507630148 | 0.000486775245 | 4.11% |
| Environmental 1 | 0.000551581712 | 0.000412898098 | 25.14% |
| Environmental 2 | 0.000727352832 | 0.000518265563 | 28.75% |
| PARPi 2022 | 0.000308097243 | 0.000254004302 | 17.56% |
| PARPi 2023 | 0.000302940513 | 0.000260679297 | 13.95% |

Jurkat misses the registered 5% threshold, so the all-task promotion gate failed. Earlier frozen-weight K562 transfer also failed: completion had 1.08–2.15% higher MSE than frozen main effects, and minimum-single references performed better. Later fitting on exposed local sources does not reverse that result. Failed composition, context-sharing, low-label and numerical-compatibility experiments remain documented.

Opal supports assay-local prediction among training-supported genes. It establishes no biological discovery, causal understanding, frontier-level superiority, reliable independent-study transfer or calibrated biological prediction intervals. Technical pair effects remain confounded with biological interactions.

The six retained growth checkpoints replayed 17,176 predictions exactly during release preparation, with no new fits, scores or test-target access. [Reproducibility](docs/reproduction.md) separates source-free checks from local biological replay. The public checkout lacks assets needed to regenerate every historical biological score. [Validation](results/release-v1.0.0/validation.json) records release checks.

Original code is [MIT licensed](LICENSE); [third-party notices](THIRD_PARTY_NOTICES.md) preserve external terms. No new model-hosting URL, archive DOI or arXiv identifier is claimed.
