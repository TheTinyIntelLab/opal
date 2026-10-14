# Official GxE measurement acquisition and technical pilot

`opal_gxe_measurement_v1` obtains and checks real public chemical-combination measurements rather than generating missing provenance. Its numeric registration is `configs/experiments/opal_gxe_measurement_v1.json`. This is a technical measurement-reliability pilot, with no predictor training, candidate nomination, independent confirmation or frontier claim. The earlier rank-16 probe remains closed. The original TNF pair protocol and its minima remain unchanged.

## Source and exact information

The author [repository](https://github.com/cole-trapnell-lab/sci-Plex-GxE/tree/63eed120697611d8baf4a207a797a147308a6674/combinatorial_chemical_screen), pinned at commit `63eed120697611d8baf4a207a797a147308a6674`, identifies [GSM7056151](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7056151) as the combinatorial chemical screen. Official GEO files provide full sparse counts, cell/gene annotations, hash counts and the assignment sheet. An earlier local “combo sci-Plex” cache belongs to GSE206741 and cannot substitute for these records. It includes a failed HTML download, which remains preserved.

The [rendered primary methods](https://cole-trapnell-lab.github.io/pdfs/papers/mcfaline-sci-plex-gxe.pdf), page 24, specify 72-hour exposure, matched compound/trametinib concentrations at 0.01/0.1/1/10 micromolar, and 0.2% DMSO across single and combination exposures. The actual glyph is micromolar; PDF text extraction renders it incorrectly as “mM.” The author cleanup script assigns A172/T98G/U87MG by RT barcode allocation, with wells 1–96 retained. Context assignment here uses that physical allocation, not post-treatment UMAP labels. Other RT origins are explicitly excluded.

Literal sample labels specify plate, trametinib dose, companion drug, companion dose and nominal replicate. These are condition assignments, not verified culture initiation IDs. Multiple hash oligos share some condition names, so the assay grouping is at deposited assignment-label level, not claimed unique physical well lineage. Plate/replicate labels, barcodes and cell partitions are not invented culture records. Independent initiation, dates, campaign costs, reviewed useful margin and power remain absent.

Repository code is MIT, while the article has its own CC-BY-NC-ND terms. No dataset licence is inferred from either. The publicly deposited raw inputs are acquired for local research with attribution; raw measurements and fitted assets are not redistributed. This custody/technical readout does not pass the frozen experiment's rights, native-comparison or confirmation requirements.

## Acquisition and custody

The official count file is exactly 590,579,159 compressed bytes. All five acquired inputs total 611,638,699 bytes. HTTPS source, advertised size, gzip magic and acquired SHA256 are retained; no publisher cryptographic checksum is claimed where none is supplied. Gzip CRC is checked as each file is decoded. The byte ceiling is 700 MB, with at least 1 GiB writable reserve and immutable output paths. Counts remain compressed and are never expanded into a full on-disk matrix or copied into Git.

```powershell
.venv/Scripts/python.exe tools/acquire_gxe_measurement.py --counts --output data/raw/gxe-measurement-<unique-id>
.venv/Scripts/python.exe tools/run_gxe_measurement.py --input data/raw/gxe-measurement-<unique-id> --output results/opal_gxe_measurement_v1/technical-<unique-id>
```

The deposited triplets have no MatrixMarket header. The author `makeCDS.R` reads one-based gene/cell/count triples with annotation-supplied dimensions. The reader retains the first entry, checks integer nonnegative counts and every coordinate bound, and streams bounded blocks. The parser interprets the headerless hash-table fields as library, cell, assignment, family and count: the fourth field matches the sample-sheet family code, while the fifth supplies the nonconstant count. Its complete field schema remains a requested source record, so this is a declared technical interpretation, not an independent provenance certification. Hash quality requires at least five counts and top/second ratio at least 2.5. Earlier unsupported RT/schema attempts are retained, with no measurement results fabricated.

## Technical measurement readout

The finite pilot uses exactly 1 micromolar and 72 hours. All 22 literal companion/trametinib pair identities are retained in population accounting across three cell origins. Twenty-two canonical pairs are not 88 independent pairs because there are four doses. The source therefore cannot meet the unchanged 36-canonical-pair diagnostic minimum, and it is a chemical star rather than the frozen genetic TNF task.

Split cells by original count-column parity into two disjoint technical partitions. Require ten cells per partition for the pair, its companion single, trametinib single and exact plate/replicate vehicle. Unsupported strata stay explicit; a pair/context is not scored as fully supported if any declared stratum lacks the required matching. No difficult row silently disappears from the declared population.

First obtain full-library totals. Select up to 512 literal assay genes, preserving the 256-gene minimum, from equal-context within-context measured-single variance only, after equal-stratum aggregation by literal single identity; pair outcomes never choose the panel. Normalize cell counts to 10,000 using full-library totals, average condition counts before log2(1+mean), subtract matched controls, and average supported strata equally. The only outputs are technical-half response/residual MSE, zero-effect comparisons and complete support/exclusion accounting. These quantify sampling/assignment/reference agreement, not independent culture covariance or predictive biological superiority. There is no architecture fitting or parameter/endpoint rescue sweep.

The first attempt stopped at unallocated RT origins; the second refused the initially assumed matrix encoding. A third attempt was stopped to repair single-identity panel weighting before any pair metrics were written; its source-only partial panel is preserved. The fourth attempt captures the corrected code/config identities. The controlled stop has incomplete final CPU/peak accounting, which remains explicitly unknown rather than reclaimed as budget. These are preparation repairs, not endpoint/architecture searches after a favorable result.

One process/thread has a two-CPU-hour, 8 GB RAM and 300 MB output stop, with a writable reserve. Acquisition and pilot resources are receipted separately. Software fixtures contain at most three coordinate entries and do not access source expression. The pilot cannot supply independent confirmation, a powered biological useful margin or a collaborator agreement; those remain real measurement/review dependencies.
