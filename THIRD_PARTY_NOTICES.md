# Data and third-party notices

Opal's original code is under the [MIT license](LICENSE). That license doesn't replace source-data, external-code, or pretrained-resource terms. No raw single-cell datasets are bundled.

The Norman reference derives from the scPerturb v1.1 release, [Zenodo record 7278143](https://zenodo.org/records/7278143), specifically `NormanWeissman2019_filtered.h5ad`. The deposited collection is CC BY 4.0. Credit the scPerturb creators and Norman et al., *Exploring genetic interaction manifolds constructed from rich single-cell phenotypes*, Science, 2019. Opal aggregates retained cells into matched population means and fits a residual model. Dataset-derived profiles and fitted content are adaptations under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), separately from Opal's MIT code. Full source creators are in the linked deposit; source identifiers and checksums are in `data/source_registry.json`.

The EmeraldBay reference derives from [Tahoe Bio's EmeraldBay dataset](https://huggingface.co/datasets/tahoebio/EmeraldBay), pinned revision `f2a0be6b02f731553657f0115c345b20bb020ede`, under CC BY 4.0. Credit Tahoe Bio and the EmeraldBay creators listed in the pinned dataset card. Opal creates matched aggregate effects and fits a contextual kernel. Dataset-derived content retains CC BY 4.0 attribution requirements; exact provenance is in the checkpoint and preparation manifests.

Wessels GSE213957, D-SPIN, and other original deposits are identified in source manifests. This release includes compact computational reports and original Opal adapters, but excludes raw measurements, profile banks, prediction arrays, and their weights because redistribution terms haven't been resolved for this release.

CPA and GEARS are optional external dependencies, not vendored implementations in this release. CPA uses BSD 3-Clause and GEARS uses MIT; retain their upstream notices when downloading them. External GenePT, ESM2, PRESAGE, STRING, GO, Hallmark, PertAdapt, and TabICL resources aren't bundled. Their upstream terms and resource costs apply separately.

The manuscript and original documentation use the repository's MIT terms; this doesn't relicense cited works or source data. Predictions have no clinical-use validation.

The requested preprint style is George Kour's unmodified `arxiv.sty`, pinned at revision `920514696f6e7270cab0558fd97c44515c63b4c4` of [arxiv-style](https://github.com/kourgeorge/arxiv-style), as used by the [Overleaf template](https://www.overleaf.com/latex/templates/style-and-template-for-preprints-arxiv-bio-arxiv/fxsnsrzpnvwc). Upstream's README General Notes permits use and redistribution and disclaims responsibility; no separate LICENSE is present at that revision. The style is based on NeurIPS 2018. Opal's MIT license doesn't relabel this external style.

Source-inspection excerpts under `data/source_audits/architecture-20261005/` retain their recorded source manifest and original `LICENSE.txt`.

The GxE recovery uses locally retained GSM7056151 measurements and the published sci-Plex-GxE methods. The source paper and author code have separate terms; neither is treated as a dataset redistribution licence. Raw counts, recovered aggregate/variance banks and derived weights are not included in source distributions. The corrected sci-Plex single-response source is scPerturb v1.4, Zenodo 13350497; source-specific attribution/rights apply separately from MIT code. Numerical reproduction does not establish legal or institutional approval for public data/model export.
