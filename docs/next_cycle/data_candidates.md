# Source qualification, 8 October 2026

Current exposure update: the [actual public-count processing diagnostic](public_combi_diagnostic.md) now consumes all large-screen Combi-seq pair outcomes. It is a nominal-reference diagnostic and cannot become untouched confirmation. Statements below about unopened expression describe the earlier metadata-only audit, whose artifacts remain preserved.

Combi-seq is a metadata-qualified lead, not an accepted confirmation source. The primary [GEO large-screen subseries GSE174695](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174695) belongs to GSE174696. The retrieved SOFT description and treatment metadata identify K562, 12-hour exposure, and aggregate treatment-barcode RNA reads. This needs an aggregate-read assay contract; the historical 50-cell threshold does not apply.

Each of rep1, rep2, and rep3 has 421 metadata rows: 377 nonrepeat pair treatment rows, 39 monotherapy rows, three repeated-target rows, and two DMSO controls. Canonicalizing drug identity gives 374 distinct nonrepeat pairs among 36 components, with identity-matched monotherapies for every pair in each run. Partner degree is 19 minimum, 20 median, and 35 maximum; 374 of 630 possible distinct pairs are represented. These are potential identity matches. Exact component doses, reagent support, ordered variants, and control compatibility have not yet been verified, so the exact-dose eligible pair count remains unknown.

The primary reporting summary explicitly calls the three runs biological replicates. The selected drug workbook maps all 36 components to GR35 annotations, but its dose units and exact combination-to-single assignments remain unverified. Run labels and an author-level replicate claim do not supply the per-culture initiation map required here. Even three accepted independent runs cannot provide both three development and three disjoint confirmation cultures for the current procedure. See the [primary-source audit](../../results/replicated_fixed_state_v1/qualification-primary-final/primary_methods.json). The main study remains blocked on primary initiation provenance, exact-dose assignments, derivative redistribution terms, and empirical precision. Expression matrices were not downloaded or opened. The [machine-readable card](../../results/replicated_fixed_state_v1/qualification/qualification.json) retains each potential control/single match and the unresolved requirements; the [source manifest](../../data/source_audits/replicated_fixed_state_v1/source_manifest.json) records URLs, file sizes and hashes. Original metadata snapshots remain local because SOFT includes contact/account details.

The owner states that this codebase is all available prior work. Searches of the pre-implementation tracked source, tools, configs, docs, results, registries and manuscript, plus all-branch Git content history, found no Combi-seq names or accessions. Exposure is recorded as no prior use found in the supplied codebase, without claiming knowledge of unavailable pretraining data.

## Backup decisions

| Source | Role | Current decision |
| --- | --- | --- |
| [El Diaz bulk RNA, GSE149428](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149428) | Fixed-time bulk measurement diagnostic | SOFT metadata retrieved. Small pair panel is unsuitable for the main superiority study; initiation, doses and terms still need qualification. |
| [CoMPoSE](https://escholarship.org/uc/item/8dj4g8fb) | Deposit/access lead | Current usable deposit and replicate provenance unresolved. No expression download or fitting. |
| [Arc context challenge](https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge-2026) | Later context lead | A matched pair-response panel is not established. Defer this source for the fixed-state task. |
| [OP3, GSE279945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279945) | Later donor/context lead | Donor replication does not establish a compatible combination panel. Defer. |
| Norman, Wessels, EmeraldBay, sci-Plex, D-SPIN, Adamson | Exposed rehearsal only | Prior analyses prevent an untouched confirmation claim. |

## Retrieval and rerun

The [expanded public search](public_dataset_search.md) adds primary-source audits of GSE291757, the three GSE119135 rat substudies, GSE174693, the 2026 ProPer-seq successor and statin–ezetimibe transcriptomics. Public files can be retrieved without owner-supplied uploads. The rat source provides documented animal replication and exact treatment doses, but needs a separate tissue/microarray/sequential-intervention contract. These new leads do not establish the cultured-cell confirmation design automatically.

`python tools/qualify_replicated_sources.py --fetch --snapshots data/source_audits/replicated_fixed_state_v1/source_snapshots --output results/replicated_fixed_state_v1/qualification-new` retrieves only bounded metadata resources. Keep previous cards; use a new output directory. Source eligibility cannot be selected from favorable expression responses.
