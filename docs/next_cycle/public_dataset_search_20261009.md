# Continued public-source qualification, 9 October 2026

Public data access is available for additional combination-expression sources. This search established more access paths and checked their design metadata, but did not establish a source meeting the current independent-culture development/confirmation contract. The [seven candidate decisions](../../results/replicated_fixed_state_v1/public-search-20261009-v1/qualification_decisions.json) distinguish access, scientific fit and prior exposure. This is a bounded search, not proof that no suitable public source exists.

## Matched-dose glioblastoma combinations

[GSE225775, sci-Plex-GxE](https://pmc.ncbi.nlm.nih.gov/articles/PMC10879025/) includes a chemical validation experiment deposited as [GSM7056151, sciPlex_4](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7056151). The methods describe three glioblastoma lines, four equal constituent-dose settings, 72-hour exposure and equal DMSO across single and combination treatments. The public hash sample sheet contains 384 barcode rows covering 22 partner compounds, trametinib and vehicle controls. These are barcode-design counts, not evidence of measured expression coverage or biological independence.

The [reproducible support audit](../../results/replicated_fixed_state_v1/public-search-20261009-v1/sciplex4_support_audit.json) finds only `rep1` and `rep2`. Under the authors' processing-code semantics, their encoded panels contain 87 and 88 pair/dose conditions respectively; each has encoded same-label singles and vehicle support. Two labels cannot supply the plan's minimum six disjoint independent culture blocks. Pooling three cell lines would change the state rather than provide more cultures of the same state. Every combination shares trametinib, which also limits the pair graph for component-generalization experiments.

GEO describes the second label field as compound dose and the fourth as trametinib dose, but the authors' [processing notebook](https://github.com/cole-trapnell-lab/sci-Plex-GxE/blob/main/combinatorial_chemical_screen/Notebook_1_cds_cleanup) assigns the reverse. The notebook ordering yields the expected single-agent and paired conditions; the audit records this conflict rather than treating either description as silently corrected. Plate10 also contains 95 `rep1` rows and one `rep2` row. The source labels remain intact, and independent initiation remains unestablished. Expression matrices were not opened.

Reproduce the design audit from the ignored source snapshot with:

```powershell
.venv/Scripts/python.exe results/replicated_fixed_state_v1/public-search-20261009-v1/reproduce_sciplex4_design.py
```

The script checks the retrieved sheet's SHA-256, validates label structure and oligonucleotide uniqueness, and records its own hash. The [retrieval record](../../results/replicated_fixed_state_v1/public-search-20261009-v1/sciplex4_design.json) retains the source URL and checksum. Author code and sample metadata have separate retrieval records.

## Public PertSpectra source located

The [2026 PertSpectra paper](https://psb.stanford.edu/psb-online/proceedings/psb26/chang_s.pdf) describes an A549 NF-kB-pathway combinatorial dataset with 65,648 cells and 182 conditions. Conditions are not a verified count of pairs. Its [repository](https://github.com/insitro/PertSpectra/tree/main/data/inhouse) supplies `s3://pert-spectra/data/inhouse.h5ad`; an HTTPS HEAD request succeeded and reports a file of 3,142,132,144 bytes. The [Zenodo record](https://zenodo.org/records/14740509) also lists a data archive. Public access is established without asking the repository owner to supply files.

The inspected methods and repository metadata do not establish an independent culture/block map, matched reagent/time/control support or six disjoint cultures. The response body, checkpoint archives and precomputed metrics were not downloaded. The paper's all-sample variable-gene selection cannot be adopted as a confirmation-independent gene universe. This is a concrete public processing lead, with qualification still required before opening a response bank.

## Other leads and alias corrections

| Source | Primary evidence and decision |
| --- | --- |
| [GSE164996, breast tumor suppressor CROP-seq](https://pmc.ncbi.nlm.nih.gov/articles/PMC9762330/) | The transcriptome library contains 72 guide-pair combinations, with four populations differing in time or medium. The large fitness screen is a separate experiment. Four populations and 20 GEO libraries do not establish four same-state independent culture blocks. Retain as a context-specific exploratory lead. |
| [GENEVA / GSE283335](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035475/) | The article describes ARS-1620 combinations with antimycin A1, galunisertib or INK128 in mosaic mouse tumors. All five GEO subseries were checked: in-vitro expression designs describe single-agent treatments, while the guide screen measures abundance. The xenograft sample lists multiple sequencing libraries but broadly labels ARS-1620 exposure. Its exact combination/animal map remains unresolved; mouse tumor profiling requires a separate population and unit contract. |
| [ComboSciPlex / GSE206741](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206741) | GEO reports triplicate treatment. The repository already records a seven-matched-pair exploratory study in `configs/experiments/chemical-exploratory.json`; this is an exposed source, not fresh confirmation. |
| [CaRPool-seq / GSE213957](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213957) | Accession matching identifies the already exposed Wessels dataset. The paper name was a new search hit, but the source is historical. No response bank was reopened. |
| [Dual Perturb-seq / GSE229505](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE229505) | “Dual” denotes host-and-parasite profiling rather than two simultaneous matched perturbations. It does not supply the required pair-response intervention design. |

## Verification and exposure boundary

Retrieval artifacts record successful source bytes and hashes, plus failed requests. The Nature HTML request returned a short page without useful deposit information; the article was instead obtained from Europe PMC. CaRPool Europe PMC requests failed, while accession matching in the repository and primary GEO records resolved the alias. Successful retrieval must not be inferred from a failed request or search snippet.

No new expression matrices, checkpoints or fitted models were consumed in this search. Published narrative results were encountered, so these sources are not described as wholly outcome-unseen. The earlier Combi-seq diagnostic has already exposed its measured pair outcomes; its separate exposure ledger remains authoritative. Public availability and private diagnostic access are distinct from permission to redistribute derivative data.

The source hashes and metadata-only audit were verified locally. Runtime implementation is unchanged; the previously completed 206-test suite and installed-package checks belong to the earlier software snapshot. This metadata update does not claim a new release build, biological gate, power estimate, preregistration or confirmation experiment. No commit was made.
