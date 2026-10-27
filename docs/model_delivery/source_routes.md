# Source-route audit, 27 October 2026

COLO1 remains unqualified for response access. The alternative primary record inspected here also fails the pair-growth prerequisite. No response table, response header, checkpoint, archive member, or outcome-derived filter was opened. The closed Figshare file route and the root HAP1, CellMap, and Dryad catalogs were not requested.

The retained receipt is [source_routes.json](../../results/model-delivery-20261027/source_routes.json): 20 requests and 100,632 bytes. An earlier process followed the same metadata sequence, received a Crossref HTTP 400, and exited during local projection before writing a receipt. Its request and byte counts were not preserved, so this pass cannot claim compliance with the original total 20-request cap. The retained counts are a lower bound on total access. Those unsaved responses are not used as scientific evidence. This accounting defect is disclosed rather than resetting the budget; the next bounded source round writes its request ledger before access.

## Alternative blocker

The first retained pair-growth index choice with a DOI, after dropping COLO1 and the root-owned Costanzo/Billmann/Boone/CellMap/Dryad routes, was Purman and colleagues, DOI [10.1177/25731599261448141](https://doi.org/10.1177/25731599261448141), PMID 42136104, titled "Optimization of Genome-Wide CRISPR Screens Using Dual-Guide RNA Infection with Cas9 Electroporation (DICE)."

Europe PMC core metadata for that DOI gives this design, with aggregate performance sentences omitted:

- SLICE is a single-guide lentiviral infection followed by Cas9 protein electroporation.
- DICE expresses two guides from the same lentiviral construct that target the same gene.
- The named genome-wide readouts are essential genes and modulators of PD-L1 expression in interferon-gamma-activated THP1 cells.

The data-link record for PMID 42136104 returns that PMID and no GEO, SRA, ENA, or other screen accession. The core record has `isOpenAccess` N, an empty license, and `hasSuppl` N. No supplement or article body was requested.

This blocks qualification before any file inventory. A same-gene dual-guide essentiality and PD-L1 assay does not supply two perturbation arms, constituent single measurements, a growth-pair endpoint, sample or initiation keys, or a rights-bearing data object.

An earlier title match, Li and colleagues, DOI [10.1002/advs.77130](https://doi.org/10.1002/advs.77130), PMC13460280, was opened at abstract and data-link level and set aside. Its abstract describes concordant single-gene effects across two Cas9 insect cell lines under cold stress. Its data-link record names PMC13460280 and no screen accession. It is not a human genetic-pair growth source.

The same index contains an unopened combinatorial-screen record: Ajmal, Nandi, Kebabci, and Ryan, "Benchmarking genetic interaction scoring methods for identifying synthetic lethality from combinatorial CRISPR screens," DOI [10.1093/nar/gkag161](https://doi.org/10.1093/nar/gkag161), PMC12464814, open-access flag Y, supplements flag Y. Only those index fields were seen. Its abstract, license text, supplements, and any underlying deposits were not requested, so it is not qualified and has no measured blocker of its own yet.

Root follow-up: that index DOI/PMCID association was wrong. The [second primary-source audit](source_routes_v2.md) resolves the Ajmal manuscript to DOI `10.1093/nargab/lqaf129` and identifies the unrelated Cas13 article at `10.1093/nar/gkag161`. The index-only statement above is preserved as the initial lead, not a verified citation.

## COLO1 catalogs accessed

| Route | What the primary record contains | Class |
| --- | --- | --- |
| [Figshare article 25533091](https://api.figshare.com/v2/articles/25533091) | Version 1, dataset, CC BY 4.0. Title `MAPPING.zip`. Description: "Mapping files resulting from sequencing to measure fitness after dual CRISPR KO." One file, id 45433417, 1,153,775,580 bytes, catalog MD5 `45c84b5ec11dc17bf5967968c7097ea0`. | Response-free catalog. Container license only. |
| [Version list](https://api.figshare.com/v2/articles/25533091/versions) | One version. | Response-free catalog. |
| [DataCite 10.6084/m9.figshare.25533091.v1](https://api.datacite.org/dois/10.6084/m9.figshare.25533091.v1) | Same title, size, and CC BY 4.0 legal-code URI. The only related identifier is `IsIdenticalTo` 10.6084/m9.figshare.25533091. | Rights metadata. |
| [Commit path for `IDs_colo1.rds`](https://github.com/ibarrioh/DualGuide_COLO1/commit/4a14cf0252e32dff69e0c234539beb5e8cff22db) | One commit at the pinned history: `4a14cf0252e32dff69e0c234539beb5e8cff22db`, 2024-04-02, message "updates in folder s and readme." | Repository metadata. |
| [Repository](https://github.com/ibarrioh/DualGuide_COLO1) and [main branch](https://github.com/ibarrioh/DualGuide_COLO1/tree/cfb5eb9a39ff1ec43642fba9485270ae200e81ae) | License field empty. Default branch `main` SHA `cfb5eb9a39ff1ec43642fba9485270ae200e81ae`, the existing pin. | Repository metadata. |
| [ENA PRJEB102587 XML](https://www.ebi.ac.uk/ena/browser/api/xml/PRJEB102587?includeLinks=false) | Project title `ENCORE_Pilot__New_Targets_for_Effective_Combination_Therapies_in_Tumors_with_Unmet_Medical_Need`. No study attributes. | Study metadata. |
| [BioStudies search ERP183979](https://www.ebi.ac.uk/biostudies/api/v1/search?query=ERP183979&pageSize=5) | `hitCount` 0. | Negative search. |
| [Zenodo version index 17191951](https://zenodo.org/api/records/17191951/versions?size=10) | Record 17191951, CC BY 4.0, one unopened zip of 89,422,914 bytes. Record [10912780](https://zenodo.org/records/10912780), CC BY 4.0, one unopened zip. | Response-free catalog. |
| [Zenodo 10912780](https://zenodo.org/api/records/10912780) | DOI 10.5281/zenodo.10912780, published 2024-04-03, description "No description provided." File `ibarrioh/DualGuide_COLO1-Releas1.1.zip`, 62,144,079 bytes, catalog MD5 `05885d1549bcbae37fad466e5c4dc592`. Related URL `https://github.com/ibarrioh/DualGuide_COLO1/tree/Releas1.1`, page not opened. | Response-free catalog. |
| Figshare search `DualGuide COLO1` | HTTP 200 body `[]`. The direct article id still resolves the known deposit, whose title is `MAPPING.zip`. | Uninformative search. |
| [crispy `notebooks/dualguide` listing](https://github.com/EmanuelGoncalves/crispy/tree/master/notebooks/dualguide) | Seven Python files. Opened only `Library2Composition.py`, 13,937 bytes, content SHA256 `ef4f5eec055ee85808658c19de226e093c7b1ba0bd34c55b4a564376e56afffd`, git blob `dc9a601dae72e4a8c67bf4760f7e323344dbe060`. | Listing, plus one source file. |

The Figshare download URL still matches closed file 45433417. It was not requested. `MAPPING.zip` was not opened, so the description's phrase "mapping files resulting from sequencing to measure fitness" is a catalog sentence, not a member inventory or a demonstration that any member is response-free.

`Library2Composition.py` parses as Python. The article code-availability statement already assigns this path to pilot library design. Function names present include `generate_guides_gi`, `generate_anchors`, `get_nontargeting_sgrnas`, `get_nonessential_sgrnas`, and `generate_negative_controls`. The text also contains the symbols ADAD1, CYLC2, KLK12, HT-29, anchor, and nontargeting. The source was not reproduced or executed, and literal tables inside it were not extracted. Nothing in this pass joins the script to the acquired 78,920-row annotation. The crispy repository license was not requested, so this file's rights stay unknown. The article's CC BY 4.0 sentence applies to the Zenodo code deposit.

These crispy files were listed and left unopened: `GILibsRepresentation.py` (5,465 bytes), `LibRepresentation.py` (4,434), `LibraryGICoreadComposition.py` (10,464), `ParseFASTQ.py` (5,234), `ParseFASTQPlot.py` (9,413), and `PilotComparisons.py` (21,573). A filename does not classify their contents.

## Missing items

- Native guide identifier, sequence, and target or control assignment for both COLO1 arms.
- A join from those identifiers onto the acquired annotation, including orientation.
- Generating inputs, invocation, row selection, and QC for `IDs_colo1.rds`. The only path commit message is "updates in folder s and readme."
- Culture, transduction, and independent-initiation keys. The project XML title and a zero-hit BioStudies search do not supply them.
- Member names and member-level rights for `MAPPING.zip` and for `DualGuide_COLO1-Releas1.1.zip`. Record-level CC BY 4.0 does not identify members.
- A separated response-free library file. The Figshare version list and DataCite relations do not name one.
- DICE pair identity, constituent singles, growth endpoint, sample keys, data accession, and article or data license.
- Any qualification of Ajmal and colleagues, which remains index-only.

## Next action

Register a names-and-sizes catalog of Zenodo 10912780, then a code-only read of `LibraryGICoreadComposition.py` and `GILibsRepresentation.py` if their roles are still limited to library construction. Stop before any count, log-fold-change, score, or fitness member. Leave file 45433417 closed.

For a pair-growth source, start at the data-availability and license statements of DOI 10.1093/nar/gkag161. Stop if that paper's primary data are the closed paralog deposit, the already exposed Horlbeck growth matrices, or the root HAP1, CellMap, or Dryad routes. DICE needs no further retrieval for this task.
