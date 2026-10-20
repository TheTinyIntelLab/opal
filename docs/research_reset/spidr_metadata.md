# SPIDR metadata-only qualification

On 20 October 2026, SPIDR passed a narrow metadata-overlap check. It remains an unscored source candidate whose LFC endpoint, observed coverage and replication do not yet qualify it as external confirmation of Opal's gamma prediction task.

The acquired public asset is Fielden et al., *Comprehensive interrogation of synthetic lethality in the DNA damage response*, Nature 2025, DOI [10.1038/s41586-025-08815-4](https://doi.org/10.1038/s41586-025-08815-4), Supplementary Table 2. The [publisher ZIP](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-08815-4/MediaObjects/41586_2025_8815_MOESM4_ESM.zip) is 19,932,183 bytes with SHA256 `c5bb66b0f3bbd1d1677ce862b5188a6a9201c47cf68e27af56e4a9a62d52a9a6`. Its only member is `Supplementary Table 2/Supplementary Table 2.csv`, 190,604,826 uncompressed bytes. The library archive stays outside the repository.

This library includes historical `predicted relative activity (mean)` and `mean relative gamma` annotations. The source-specific projector scans CSV structure and retains only `sgRNA name`, `gene`, `perfect match sgRNA`, `sequence`, `transcript` and `mismatched` for each arm. It discards the other field values without converting them to numeric arrays, logging them, exporting them or using them to choose guides. Reading the archive necessarily processes its raw bytes. No source-response/count table was downloaded or opened, no SPIDR model predictions were scored, and no target-pair outcome was used for eligibility or selection. Do not describe this step as acquiring an entirely outcome-free file.

Before qualification, all three projector tests passed. They check selected-field equivalence to standard CSV parsing, quoted commas and escaped quotes in discarded fields, and rejection of changed widths or multiline records. The qualifier verifies the registered archive hash, exact member and column contract, consistent repeated guide identities, ZIP CRC and original supported-gene count. `results/research-reset-source/qualification.json` records the full receipt.

| Metadata quantity | Observed count and implication |
| --- | --- |
| Library constructs | 697,233 rows, 1,165 unique guide identities and 549 gene labels including controls; these are design counts, not observed T0/T14 coverage. |
| Unordered label pairs | 150,975 including controls and same-gene constructs; these are not independent biological observations. |
| Original supported genes | 427 genes with at least 20 original single observations in every context and replicate represented in the original bank. |
| Shared supported genes | 41, giving 820 distinct unordered supported gene pairs in the library. This uses literal source gene labels, without retrospective alias rescue. |
| Pair overlap | 684 of those 820 pairs are present in the original eligible bank and 136 are absent. Source transfer on previously seen pair identities and source transfer on new pairs must be reported separately. |
| Exact guide-sequence overlap | 21 guide identities spanning 21 supported genes have a sequence matching an original guide assigned to the same literal gene label. |
| Gene pairs with sequence overlap | 210 pairs have both genes represented by at least one exact-matched guide. This does not certify that every construct or observed double for those pairs uses matched guides. |

The original bank hash is `7a0a6a4c7f79e79983071deb59b18d52493f4333133dfa89c100d6720b12cb94`; original guide metadata hash is `84a785108a5bf830ce2ee569e14757fe5c31738ba6d88c95206f38a0b1bbc150`. The receipt includes all 41 shared genes and the 21 sequence-matched genes. It contains no source-response values.

Reproduce from the repository root, with the existing original bank and library archive at the paths below. The qualifier refuses to overwrite an existing receipt, so preserve the recorded output and use a new output filename for an independent replay.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_spidr_metadata.py -q
.venv/Scripts/python.exe tools/qualify_spidr_metadata.py --archive C:/Users/USER/Desktop/opal-research-reset-20261020/sources/41586_2025_8815_MOESM4_ESM.zip --original-root C:/Users/USER/Desktop/opal-next-data-20261020 --output results/research-reset-source/qualification.json
```

The actual run returned `3 passed in 0.02s`, then the counts above. The registered receipt was created once; there was no previously completed qualification output or running Python qualification process when this resumed run started.

The [primary article XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12018271/fullTextXML) records CC BY 4.0 unless an item states an exception. Public publisher access required no credentials here. Raw sequencing is listed under [PRJNA988447](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA988447); its separate reuse terms were not verified and no SRA data were acquired. Attribute the source and check any item-specific exception before redistributing supplement-derived material. The repository receives only compact counts, gene identities, hashes and acquisition references, not the library CSV.

SPIDR uses hTERT RPE-1 TP53-knockout cells and includes deliberate guide mismatches. Guide sequence equality does not match knockdown level, oxygen, culture duration or cell state. Its T14/T0 LFC is not the original per-doubling gamma scale. The two cultures were split after transduction during selection, so they do not establish independent infection initiations. These distinctions rule out calling this an interchangeable gamma benchmark or treating guide constructs as independent biological replicates.

Before opening Supplementary Table 8, register a separate LFC task, freeze source-independent predictions or the explicitly permitted target-single adaptation procedure, define control/T0 coverage eligibility and guide strata, and state how repeated constructs and split cultures enter uncertainty. Preserve perfect and mismatched guides as separate strata, with a sequence-only inclusion rule. The paper's depletion-based removal of strong perfect guides would make eligibility outcome-dependent if copied. T0 coverage can establish measurement availability, but T14 response must not choose the confirmatory panel. Any protocol adaptation after source-pair outcomes are inspected makes the affected source evidence exploratory.
