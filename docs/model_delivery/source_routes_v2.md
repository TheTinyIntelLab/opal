# Source-route audit, v2, 27 October 2026

Parrish and colleagues, DOI [10.1016/j.celrep.2021.109597](https://doi.org/10.1016/j.celrep.2021.109597), is a distinct two-gene growth screen. It stays unqualified. The count file has no item license in the GEO series record, the guide map was not inventoried, and the twelve sample accessions have no characteristics. No count file, score file, response header, archive member, or outcome filter was opened. COLO1 file 45433417, the HAP1 qGI route, CellMap, Dryad, Horlbeck `GSE116198`, and the exposed `GSE312636` and `GSE284018` banks were not requested.

The retained receipt is [source_routes_v2.json](../../results/model-delivery-20261027/source_routes_v2.json): 6 requests and 370,929 bytes, under the 6-request and 2 MiB cap. Each request was written into that receipt before the socket opened and again after the response. Nothing was retried.

## Index DOI

Crossref for the prior index DOI [10.1093/nar/gkag161](https://doi.org/10.1093/nar/gkag161) returns a different article: Aguilar-González and colleagues, "A novel Dual-guide CRISPR–Cas13 strategy improves specificity for single-nucleotide variant detection," *Nucleic Acids Research*, issued 2026-02-24, CC BY-NC 4.0. The abstract describes two RNA guides, dcrRNA and dtracrRNA, recognizing one RNA target, with analytical assays for SARS-CoV-2, KRAS G12D/G12C, and Leishmania sequences. That design supplies no two-gene growth endpoint, matched single-gene growth measurements, or pooled-screen count object. No paper body or data file was requested. This DOI is blocked before any file inventory.

The PMCID previously attached to that DOI, PMC12464814, is the Ajmal article below. Its XML does not contain the string `gkag161`.

## Ajmal article and code

The primary article is Ajmal, Nandi, Kebabci, and Ryan, "Benchmarking genetic interaction scoring methods for identifying synthetic lethality from combinatorial CRISPR screens," *NAR Genomics and Bioinformatics*, DOI [10.1093/nargab/lqaf129](https://doi.org/10.1093/nargab/lqaf129), PMID 41018953, PMC12464814. The article states CC BY 4.0. Data availability names the code repository [cancergenetics/Benchmarking-GI-Scores](https://github.com/cancergenetics/Benchmarking-GI-Scores) and two Figshare share links. The share link described as scoring-method results was not requested. The code share link was not requested either; the repository README was.

The article's dataset table names five pooled combinatorial knockout studies. Each is described as having pair constructs, single-gene constructs with a non-targeting control, and read counts at an early and a later time. These are article sentences, not file inventories. Published library sizes were exposed and are kept in the receipt.

| Study | Citation | Lines named in the article table | Held in this pass |
| --- | --- | --- | --- |
| Dede | [10.1186/s13059-020-02173-2](https://doi.org/10.1186/s13059-020-02173-2) | A549, HT29, OVCAR8 | README names `counts.txt` only |
| CHyMErA | [10.1038/s41587-020-0437-z](https://doi.org/10.1038/s41587-020-0437-z) | HAP1, RPE1 | HAP1 route left closed |
| Ito | [10.1038/s41588-021-00967-z](https://doi.org/10.1038/s41588-021-00967-z) | eleven lines, including A549 and MEWO | README names `Count_data_ParalogV1.xlsx` only |
| Parrish | [10.1016/j.celrep.2021.109597](https://doi.org/10.1016/j.celrep.2021.109597) | PC9, HeLa | Followed below |
| Thompson | [10.1038/s41467-021-21478-9](https://doi.org/10.1038/s41467-021-21478-9) | MEWO, A375, RPE | Closed Figshare 25954027 was not requested and was not equated with this citation |

The README blob `48e5d4a526057018e265df4af5ea5a2f62e80843` says the MIT license applies to the repository except subfolders that carry their own license. The LICENSE file was not fetched. The README lists external inputs and does not itself contain counts. It names Parrish raw counts as `InputData/Parrish/GSE178179_pgPEN_counts_PC9.txt`. It also names a Thompson guide-sequence workbook and raw-count input. Those files were not fetched. DepMap TPM, SLKB scores, and the De Kegel and Köferle label files were not requested.

Figshare article 27868350, DOI [10.6084/m9.figshare.27868350.v2](https://doi.org/10.6084/m9.figshare.27868350.v2), is public, version 2, CC BY 4.0, with 55 files totaling 33,756,509 bytes. The description says the files are genetic-interaction scores, with and without the methods' default filters. The README's raw-count filenames are absent. No file body was downloaded. This deposit is a scored-output catalog. It is not a raw-count endpoint, and the CC BY 4.0 grant covers that score deposit.

## Parrish pgPEN

GEO series [GSE178179](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178179) is public as of 31 August 2021. The title and PMID 34469736 match the Parrish paper. The series summary says pgPEN assays paralogs in single-knockout and double-knockout configurations in two cell lineages. That summary also contains a published hit-rate sentence; the number is not transcribed. Overall design is one sentence and does not name times, replicates, or initiations.

The series record lists one supplementary file, `GSE178179_pgPEN_counts.txt.gz`, plus BioProject PRJNA737515 and SRA SRP324035. The gzip, the SRA runs, and platforms GPL16791 and GPL19604 were not requested. The README filename `GSE178179_pgPEN_counts_PC9.txt` is a different name from the series supplement. The series record has no license field. Sample accessions are GSM5381351 through GSM5381362, with no titles or characteristics.

The article license line is CC BY-NC-ND 4.0. The key-resources table assigns "Raw and analyzed CRISPR screen data" to GSE178179 and assigns PC9 and HeLa RNA-seq to GSE120703. GSE120703 was not opened. The same table names the plasmid library as Addgene 171172 and the counting code as Zenodo [10.5281/zenodo.5081113](https://doi.org/10.5281/zenodo.5081113). Neither record was requested.

Methods describe the library as Brunello guides in 33,170 paired constructs: 1,030 paralog pairs, sixteen 4-by-4 double-knockout constructs per pair, sixteen single-knockout constructs per family formed by pairing one targeting guide with a non-targeting guide, and 500 double non-targeting constructs. A given gene's guide is described as occupying the same site in every construct. The article names this library as Table S1. Table S1 was not requested.

PC9-Cas9 cells were transduced at a multiplicity of about 0.3, selected with puromycin, and split into three biological replicates after infection and before selection. Early DNA was taken after selection, and the endpoint followed about twelve population doublings. HeLa/iCas9 cells stayed in one pool through selection, producing one early sample, then were induced with doxycycline and split into three replicates for the same doubling window. MAGeCK is described as comparing plasmid-pool abundance with the early and late samples. Those sentences do not bind any GSM accession, and the replicate splits occur after a single transduction.

The methods also describe score scaling from double non-targeting and pan-essential medians, a further shift from unexpressed-gene medians, and residual interaction scores. That processing uses the screen-wide response distribution. The resulting scores were not opened. Results prose in the same XML contained published correlations and test statistics; those numbers are not transcribed.

## Missing companion

The next documentary objects, in order, are a names-and-license catalog of Table S1, a names-and-license catalog of Zenodo 5081113, and GEO characteristics for GSM5381351–GSM5381362. The count gzip stays closed until those records show a separated construct map, a count-file rights statement, and a sample key for line, time, and replicate. Ajmal score files stay closed.

Root subsequently made one separately capped catalog request to [Zenodo 5081113](https://zenodo.org/api/records/5081113). The [receipt](../../results/model-delivery-20261027/parrish-catalog.json) records 3,831 bytes, MIT, and `bergerbio/pgRNA_sequencing_analysis-v1.0.0.zip`, 415,579 bytes. The description identifies counting scripts and associated files. No archive member was opened, and this grant does not establish the rights or identity of the GEO count object. The guide map, sample characteristics and exact count-file scope remain unresolved.
