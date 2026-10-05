# Source and citation audit

Audit date: 4 October 2026. Norman and original Wessels counts have been downloaded, checked and prepared on log2 and normalized-count effect scales. Original ComboSciPlex counts have been prepared for a restricted exploratory protocol. These are source and preparation acceptance decisions, not completion of biological Opal v1 validation. Checksums, source manifests, gene schemas, QC reports and split artifacts are recorded locally and indexed in `data/source_registry.json`.

**Norman.** The original study is a K562 CRISPR activation experiment. GEO accession GSE133344 and the authors' analysis repository provide the original route. Original matrix filenames were verified through the GEO text endpoint after the HTML listing presented a browser challenge; local experiments use the pinned secondary count release. [Original study](https://pubmed.ncbi.nlm.nih.gov/31395745/), [author code](https://github.com/thomasmaxwellnorman/Perturbseq_GI), [BioProject](https://www.ncbi.nlm.nih.gov/bioproject/551220).

A practical secondary source is pinned scPerturb Zenodo record 7278143, version 1.1. The downloaded 698,680,199-byte H5AD matches publisher MD5 `c870e6967d91c017d9da827bab183cd6`; local SHA-256 is `efde6f5301fe256725dce1d980f37bd96a13481a9a16135515897368e631affc`. Original transformation and local preparation establish the count source. It remains a harmonized filtered release, distinct from GEARS and the original unfiltered GEO matrix. The prepared protocol has 105 singles and 126 eligible doubles, rather than imposing the roadmap's 131-double processed-track reference. [Pinned record](https://zenodo.org/records/7278143).

**Wessels.** THP1 metadata and the shared THP1/HEK293 matrix archive from GSE213957 have been downloaded and extracted; the registry records both file SHA-256 values. The local THP1 matrix keeps GEX features and metadata-matched barcodes, with SHA-256 `8dd92310769c79c352700b6519ae66da28bec717b17006fa6fe3a1197b43a66a`. It supplies 16 eligible singles and 58 eligible doubles after QC and availability checks. The separate small pooled-screen counts CSV is not used as a single-cell expression matrix. [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213957).

The hosted scPertEval Wessels release is unsuitable for Opal's observed-single input: it excludes all single-target cells and carries log-normalized expression rather than raw counts. Its documentation recommends preparing the raw source to retain singles. This rules out recovering additive inputs from hosted double targets. [Preparation documentation](https://scperteval.readthedocs.io/en/latest/user-guide/datasets.html).

**ComboSciPlex.** The tutorial-associated Drive URL returned a 2,453-byte HTML interstitial, not a valid H5AD, so that download is rejected. Original GSE206741 counts and metadata are used instead. Primary methods verify 1 micromolar per component, A549 and 24-hour exposure, but only seven doubles have both corresponding singles in this source; acceptance is restricted to the exploratory seven-pair protocol. [Official tutorial](https://cpa-tools.readthedocs.io/en/latest/tutorials/combosciplex.html).

**Reuse and immutable provenance.** The downloaded Zenodo API record explicitly assigns CC BY 4.0 to record 7278143. Explicit redistribution licenses for original GEO matrices and CPA Drive data remain unresolved; scientific attribution is not itself a redistribution grant. Article and software licenses do not automatically license associated data. Download records preserve byte counts and SHA-256, and preparation manifests bind count sources, identifiers, guide mappings and normalization history. Transport records use `downloaded_not_accepted` before preparation; the registry records subsequent protocol-scoped acceptance without rewriting those earlier records.

**Literature strategy.** Searches used the roadmap's exact article titles, accession numbers and model names, then resolved publisher, PubMed, PMLR and author-maintained documentation. Sources were included for dataset provenance, comparator methods or evaluation calibration; secondary aggregators were used only for discovery. DOI/title verification is recorded through the links below; this is a focused implementation bibliography, not a completed systematic review.

| Citation key | Verified primary reference | Relevance and limitation |
| --- | --- | --- |
| norman2019 | [Norman et al., Science, 2019](https://doi.org/10.1126/science.aax4438) | Original genetic experiment; does not establish Opal performance. |
| wessels2023 | [Wessels et al., Nature Methods, 2023](https://doi.org/10.1038/s41592-022-01705-x) | Independent Cas13 benchmark; online publication occurred in 2022. |
| cpa2023 | [Lotfollahi et al., Molecular Systems Biology, 2023](https://doi.org/10.15252/msb.202211517) | Compositional comparator and chemical source. |
| gears2024 | [Roohani et al., Nature Biotechnology, 2024](https://doi.org/10.1038/s41587-023-01905-6) | Knowledge-graph comparator; online publication occurred in 2023. |
| miller2026 | [Miller et al., Nature Biotechnology, 2026](https://doi.org/10.1038/s41587-026-03307-w) | Published 1 October 2026; supports metric calibration, not a universal model ranking. |
| naiad2025 | [Qin et al., ICML/PMLR, 2025](https://proceedings.mlr.press/v267/qin25g.html) | Single-effect-informed combination learning; check scalar phenotype versus transcriptomic task compatibility before a direct comparison. |
| slim2026 | [Hu et al., bioRxiv, 2026; author repository](https://github.com/RasmussenLab/SLIM/blob/main/README.md) | DOI and citation verified from authors; preprint server access was blocked. External STRING embeddings must be disclosed. |
| pertadapt2026 | [Bai et al., Bioinformatics, 2026](https://pubmed.ncbi.nlm.nih.gov/42412811/) | Foundation-model adaptation; upstream resources and combination support require protocol alignment. |

Two open evidence questions follow directly from this audit: whether gains persist with observed singles and strong classical comparators under identical data access, and whether any residual signal remains reproducible across raw-source independent contexts. Neither can be settled from the cited literature alone.

Author identity, contributions, conflicts and funding for Opal have not been supplied. No declaration about them is inferred from another paper.

## Follow-up: actionable source mappings

**Norman provenance.** The GEO text endpoint succeeded after the HTML challenge. It lists `GSE133344_filtered_matrix.mtx.gz`, filtered barcodes, filtered genes and filtered cell identities; corresponding unfiltered files also exist. The author's README identifies these as Cell Ranger outputs. The scPerturb processing notebook reads the count matrix with `mmread`, joins cell identities through the barcode list, and writes the transposed matrix without an expression normalization in that block. Its gene table provides gene symbols and Ensembl identifiers. These sources support the locally accepted count-derived preprocessing. [Author README](https://github.com/thomasmaxwellnorman/Perturbseq_GI), [scPerturb transformation](https://github.com/sanderlab/scPerturb/blob/master/dataset_processing/notebooks/data_processing_JS.ipynb).

The scPerturb transformation parses targets before `__`, strips `NegCtrl` components, and maps `no_reads_found` to control. Preserve guide IDs and explicitly report the no-read control decision; don't treat an unassigned guide as a verified matched control solely because the harmonized label says control. Its source mapping notebook identifies the original gene and barcode filenames. [Source mappings](https://github.com/sanderlab/scPerturb/blob/master/dataset_processing/notebooks/data_access_and_preprocessing_JS.ipynb).

**Wessels actual metadata.** The retrieved THP1 TSV has 30,707 unique cell barcodes: 28,095 `Dual`, 2,188 `Single` and 424 `NT` in `Guide.Class`, spanning 187 `GenePair` labels before matrix join or QC. The header omits the first row-name column; load it with `index_col=0` or prepend a barcode field explicitly. Use `GenePair` for targets, `CRISPR.Array` and `ID.g1`/`ID.g2` for guide provenance, and `TenX.Lane` plus the barcode for joins. Controls are exactly `NT_NT`; remove only exact `NT` components from singles, so `INTS1` is never misclassified as a control. The archive has four 10x lane prefixes `THP1-CaRPool-seq_and_HEK293FTstabRNA.GEXGDO{lane}.`; join to metadata using `L{lane}_{10x_barcode}` and retain only metadata-matched THP1 cells. [Original metadata](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213957/suppl/GSE213957_THP1-CaRPool-seq.metadata.tsv.gz), [independent loader example](https://github.com/larslorch/lcd-clipr/blob/main/lcd_clipr/realworld_data/wessels.py).

**Chemical limitation uncovered.** Original ComboSciPlex is GSE206741, with a 343.4 MB compressed count matrix and separate cell/gene tables. Its original cell metadata has 63,430 unique cells, 32 conditions and 96 wells; `Drug1` and `Drug2` provide component labels, `Cell` identifies matrix rows and `Well` preserves experimental layout. The primary methods state A549, 24-hour exposure and 1 micromolar for each compound. This supports component-dose equality for this experiment. [Original GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206741), [primary methods](https://doi.org/10.15252/msb.202211517).

Only six drugs have single profiles within that original experiment: Alvespimycin, Dacinostat, Givinostat, SRT2104, dasatinib and panobinostat. Of 25 double conditions, only seven have both singles: Alvespimycin+SRT2104, Alvespimycin+panobinostat, Dacinostat+dasatinib, Givinostat+SRT2104, Givinostat+dasatinib, SRT2104+panobinostat and dasatinib+panobinostat. This is too few for the intended chemical model selection and independent test protocol. A larger suite requires documented matching to external singles, with batch/context differences treated explicitly, or a prespecified replacement. It cannot be made adequate by inventing missing singles. The cell table contains trailing nonbreaking whitespace in a Sorafenib label and both PCI-34051/PCI-34501 spellings; retain original identifiers until a source-backed mapping resolves these. [Original cell table](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206741/suppl/GSE206741_cell_metadata.tsv.gz).

The scPerturb chemical preparation retains original counts but drops some original metadata columns, so use the original tables for dose, component and well auditing. The tutorial's prepared gene subset remains a separate compatibility source. [Chemical preparation](https://github.com/sanderlab/scPerturb/blob/master/dataset_processing/notebooks/LotfollahiTheis2023.ipynb).

## Current prepared artifacts

The following values are taken from local preparation, QC, gene-schema and split manifests. They describe the actual retained task, not published dataset-wide counts. All tracks use outer seed 1729; gene selection uses allowed controls and singles, and effect scales remain separate.

| Prepared track | Source cells | Eligible singles | Eligible doubles | Output genes | Development/test pairs | Control library normalizer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `norman-log2` | 111,445 | 105 | 126 | 10,623 | 88 / 38 | 14,048 |
| `norman-count` | 111,445 | 105 | 126 | 10,623 | 88 / 38 | 14,048 |
| `wessels-log2` | 30,707 | 16 | 58 | 10,590 | 40 / 18 | 10,167 |
| `wessels-count` | 30,707 | 16 | 58 | 10,590 | 40 / 18 | 10,167 |
| `chemical-exploratory` | 63,430 | 6 | 7 | 5,972 | 4 / 3 | 1,957 |

Norman filters 20,277 cells through metadata eligibility rules before numerical preparation; its per-condition QC report records subsequent retention and five excluded doubles. Wessels retains all 30,707 metadata-matched source cells at the initial metadata stage, then applies numerical QC and requires eligible matched singles. Sixteen retained singles explain why only 58 doubles enter the local task despite much broader original pair coverage. No missing single is reconstructed from a double response.

The chemical original count file has SHA-256 `95d5dc4b28eb407449b8976f79e89000ec89eee620e5f3b819c8d68aa454dc6e`. Its cell and gene metadata hashes are `67593f3785759bf9f2089cd369ab6f3d5aca76555fa1a31fd7ba781a79b20765` and `5beab60fb97830d1fcc11276c3f200e96a93bbb96bccd65186784f928fa81bdd`. Preparation retains 62,891 cells, uses equal well weighting within conditions and an equal control-well plate reference, and stores a threefold exploratory development amendment. This amendment enables pipeline testing on seven matched pairs; it does not satisfy the planned chemical confirmatory benchmark.

For exact exclusions, source versions, condition maps and normalization settings, use each track's `source_manifest.json`, `qc_report.json`, `preprocessing.json`, `gene_schema.json` and `splits.json` under `data/prepared/`. The registry stores their preparation manifest SHA-256 alongside source count hashes. The audit accepts source construction for these local protocols; unresolved source redistribution terms, guide efficacy differences, limited chemical coverage and the remaining release validation requirements still apply.

## New chemical sources, 6 October 2026

[EmeraldBay](https://huggingface.co/datasets/tahoebio/EmeraldBay) is pinned at
revision f2a0be6b02f731553657f0115c345b20bb020ede under CC BY 4.0. The pilot reads
six verified shards (2,585,018,634 bytes, 94,746 cells), selects eight eligible
cell lines by condition coverage, and retains 17 exact-dose pair conditions
and 99 context/pair rows. Its 1,024-gene panel uses vehicle variation only.
Controls and each query's measured singles are allowed; this exposed subset
supports exploratory evaluation rather than new independent confirmation.

The independent [D-SPIN release](https://data.caltech.edu/records/2cjss-wgh69)
provides 173,412 cells and a 772,564,529-byte count matrix. Publisher MD5 is
3aedeecc18b8acb54c7a97ffc3943ef7; recorded SHA256 is
9cdff701a44a23e0032f6ab3fe75177b2e23d1808698f5cbcd6f4788dbce53c8.
The dosage-combination task uses stimulated controls, 55 exact-dose pairs,
five lineage labels across two batches and 1,011 genes intersecting the frozen
pilot vocabulary. Whole-lineage holdouts cover 108 of 273 pair/context rows.
Published lineage assignments are post-treatment; batch copies and dose-grid
rows are not independent biological replication. One shared endpoint supports
no time-transfer claim. The independent procedure test was negative.
The [primary Cell paper](https://doi.org/10.1016/j.cell.2026.04.028) describes
the source experiments; configs/datasets/dspin-confirmation-v1.json records
matching and raw storage, and tools/download_dspin.py verifies both checksums.
