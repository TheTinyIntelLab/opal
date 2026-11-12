<p align="center"><img src="images/logo.png" width="140" alt="Opal logo"></p>

# Opal

Observed-single prediction of combinatorial perturbation responses. Opal v1.0.0 is the first research release from [The Tiny Intelligence Lab](https://github.com/TheTinyIntelLab).

Opal predicts population-mean expression changes or double-CRISPRi growth from measured single perturbations and a declared set of other pair outcomes. Its strongest retained results concern missing pairs within an assay, among genes already supported by training pairs. Reliable cross-study biological generalization is not established.

## How it works

Expression models add a fitted symmetric correction to measured single effects. The code includes neural, regression and residual-kernel comparisons. Growth models combine a calibrated single/main-effect reference with a symmetric learned gene-pair residual matrix. Historical "rank 8" refers to factor width: that signed symmetric operator has algebraic rank at most 16, rather than a verified biological dimension.

<img src="images/opal_overview.svg" alt="Measured singles feed an additive reference and a symmetric residual correction." width="100%">

Models require source-specific controls, endpoint definitions, gene order and information-access contracts. Growth lookup factors do not support arbitrary unseen genes, and good growth prediction does not establish good expression prediction or a causal interaction.

## Install

Use Python 3.11–3.14. From this repository:

```bash
python -m venv .venv
# Activate the environment for your shell, then:
python -m pip install ".[data,dev]"
python -c "import opal; print(opal.__version__)"
opal --help
```

The package is named `opal-research`; its import is `opal`. Install from this checkout; no v1 PyPI publication is implied. [Installation](docs/installation.md) covers CPU Torch and the recorded numerical environment.

## Use

GitHub supplies code, the paper and compact evidence. Model weights will have a separate Hugging Face release; no public v1 model URL is announced here. A package installation alone does not supply biological models or datasets.

With the retained local growth bundle:

```python
from opal.trained_growth import TrainedGrowth

bundle = TrainedGrowth.load(
    "path/to/trained-growth",
    expected_manifest_sha256="fa8490abe83e5a5d0d0f8456639d11df16a2a6ca3ed440fb40bfd74459488e72",
)
model = bundle.model("horlbeck_K562")
genes = [g for g, ok in zip(model.gene_names(), model.support()) if ok]
print(bundle.predict("horlbeck_K562", [(genes[0], genes[1])], context="K562"))
```

That hash identifies the retained October 27 bundle. Unknown or unsupported genes, self-pairs and incorrect contexts produce explicit abstentions. Calibrated per-query uncertainty is unavailable. [Prediction and model availability](docs/usage.md) describes expression inference and the full contracts.

## Results

The latest retained benchmark pools five disjoint held-out blocks per task, with 16,000 training and 4,000 validation pair labels per fitted procedure. Test genes have training support. Scores describe the procedure, not the single exported fold-0 checkpoint.

| Growth task | Selected simple MSE | Completion MSE | MSE reduction |
| --- | ---: | ---: | ---: |
| K562 | 0.00190091929 | 0.00179069074 | 5.80% |
| Jurkat | 0.000507630148 | 0.000486775245 | 4.11% |
| Environmental 1 | 0.000551581712 | 0.000412898098 | 25.14% |
| Environmental 2 | 0.000727352832 | 0.000518265563 | 28.75% |
| PARPi 2022 | 0.000308097243 | 0.000254004302 | 17.56% |
| PARPi 2023 | 0.000302940513 | 0.000260679297 | 13.95% |

**The all-task promotion criterion failed:** Jurkat did not reach the registered 5% threshold. Earlier frozen-weight K562 transfer also failed on four external cohorts, with 1.08–2.15% higher MSE than frozen main effects. Later local fitting on exposed sources does not reverse that result. [Research findings](docs/research.md) links the original ledgers and relevant negative experiments.

## Limitations

Opal does not establish biological discovery, causal understanding, frontier-level superiority, reliable independent-study transfer or calibrated biological prediction intervals. Pair supervision is substantial, technical pair effects remain confounded with biological interactions, and folds/cohorts are not independent biological replications. Restricted data and historical numerical assets limit what a public checkout can reproduce.

## Documentation and repository

- [Documentation index](docs/README.md): installation, API, findings, reproduction and archived evidence.
- [Paper](paper/main.pdf) and [LaTeX source](paper/main.tex): methods, results and limitations.
- [Reproducibility](docs/reproduction.md): public checks and licensed local replay.
- [Release notes](RELEASE_NOTES.md) and [checklist](RELEASE_CHECKLIST.md): scope and outstanding blockers.
- `src/`, `tests/`, `configs/` and `tools/`: implementation, tests, frozen configurations and utilities.
- `results/` and [data registries](data/README.md): compact evidence and provenance.

Current ledgers and runtime-required contracts stay in the active tree. Older protocols, negative findings and computational records are consolidated in [verified archives](docs/archive/README.md), with original paths and hashes available for restoration. Local builds, model banks and unfinished research are kept outside the checkout.

## License and citation

Original Opal code is [MIT licensed](LICENSE). Source data and external/model-derived assets have separate terms in [third-party notices](THIRD_PARTY_NOTICES.md). GitHub excludes raw biological data, trained checkpoints, private correspondence and local plans. [Distribution policy](docs/distribution.md) defines the separate Hugging Face model release.

Use [CITATION.cff](CITATION.cff) to cite Opal, and cite the source studies for the data you use. Arjun Vijay Prakash is responsible for the research and final text; the manuscript discloses AI assistance. No public DOI or arXiv identifier is claimed.
