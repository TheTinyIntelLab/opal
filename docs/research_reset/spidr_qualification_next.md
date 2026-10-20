# Reserved-source qualification after Q1

Metadata/code review on 20 October 2026 identifies two concrete reasons to keep
SPIDR responses closed. Its published count supplement is already normalized,
which the registered raw-count adapter correctly rejects, and its frozen primary
construct rule restricts the designed panel to pairs incident on nine genes.
Neither source has received a new biological fit, score or response-table read.
This record does not amend the registered endpoint, stratum or promotion gates.

## SPIDR design support is narrower than the gene count suggests

The existing hash-verified library archive was reprojected with
`tools.qualify_spidr_metadata.project_line`, retaining the existing allowlisted
identity fields only. Historical guide activity fields were discarded without
numeric conversion, selection or use. The source hash remains
`c5bb66b0f3bbd1d1677ce862b5188a6a9201c47cf68e27af56e4a9a62d52a9a6`.

| Metadata quantity | Exact count |
| --- | --- |
| Targeted guides with explicit mismatch status and transcript | 1,150, comprising 258 mismatched and 892 perfect guides |
| Negative-control guides with blank mismatch and transcript fields | 15, all labeled `negative_control` |
| Genes with at least one mismatched guide | 249 |
| Genes with two mismatched guides | 9 |
| Designed distinct both-mismatched gene pairs | 30,876 |
| Designed both-mismatched pairs with at least two ordered constructs | 2,196 |
| Genes represented by those 2,196 pairs | 249 |
| Degree distribution in that restricted primary design | 240 genes have degree 9; nine genes have degree 248 |
| Designed mixed-stratum pair identities | 105,327 |
| Designed perfect/perfect-stratum pair identities | 149,878 |

The nine genes with two mismatched guides are BOD1L1, DDB1, ERCC4, KIF22,
POLR2D, POLR2E, RPS3, SMARCB1 and UBA52. Every one of the 2,196 primary-design
pairs touches at least one of them. The arithmetic is
`9*240 + 9*8/2 = 2,196`; the other 28,680 both-mismatched pairs have only one
ordered construct and fail the frozen minimum before count QC.

These are design counts, not observed coverage or test eligibility. They do not
include T0 filtering, exact-single availability, observed count linkage, train
support or the final pair-hash split. A panel can exceed ten genes while depending
on very few hubs. Such a result would establish performance on this restricted
design, and endpoint-multiplicity uncertainty would be sensitive to those nine
genes. It would not establish broad dense-block prediction among 249 genes.
The source's 548 targeted genes therefore cannot be used as the effective primary
gene-pair population under the frozen both-mismatched/two-construct rule.

Reproduce the design calculation without opening response data:

```powershell
@'
import io, json, zipfile
from collections import Counter
from tools.qualify_spidr_metadata import project_line, SAFE_COLUMNS, MEMBER, sha256, ARCHIVE_HASH
path = 'C:/Users/USER/Desktop/opal-research-reset-20261020/sources/41586_2025_8815_MOESM4_ESM.zip'
assert sha256(path) == ARCHIVE_HASH
pairs, guides = Counter(), {}
with zipfile.ZipFile(path) as archive:
    with io.TextIOWrapper(archive.open(MEMBER), encoding='utf-8-sig', newline='') as stream:
        stream.readline()
        stream.readline()
        for line in stream:
            row = project_line(line, SAFE_COLUMNS, 25)
            for start in (1, 13):
                guides[row[start]] = tuple(row[start+j] for j in range(6))
            a, b = row[2], row[14]
            if a == b or 'negative_control' in (a, b):
                continue
            flags = row[6], row[18]
            assert all(f in ('TRUE', 'FALSE') for f in flags)
            stratum = ('perfect_perfect', 'mixed', 'both_mismatched')[sum(f == 'TRUE' for f in flags)]
            pairs[(tuple(sorted((a, b))), stratum)] += 1
mm = Counter(v[1] for v in guides.values() if v[5] == 'TRUE')
primary = {p for (p, s), n in pairs.items() if s == 'both_mismatched' and n >= 2}
degree = Counter(g for p in primary for g in p)
print(json.dumps({'mismatch_genes': len(mm),
                  'multi_mismatch_genes': {g: n for g, n in sorted(mm.items()) if n > 1},
                  'primary_design_pairs': sum(s == 'both_mismatched' for p, s in pairs),
                  'primary_two_construct_pairs': len(primary),
                  'degree_histogram': dict(Counter(degree.values()))}, sort_keys=True))
'@ | .venv/Scripts/python.exe -
```

## What the official methods and code resolve

The [primary methods](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12018271/fullTextXML)
specify 15 non-targeting partners per guide in both orientations and 225
non-targeting/non-targeting constructs. Arm one uses mU6/CR3 and arm two hU6/CR1.
This supports the roles of the 15 library guides labeled `negative_control`;
mapping observed count-row identifiers to those constructs remains unverified.

The methods explicitly identify Supplementary Table 8 as median-ratio-normalized
counts, with size factors estimated using non-targeting guides. Their T0 rule is
at least 50, whereas Opal froze strictly above 50 raw counts. Cultures split
during selection after a common transduction. T0 is four days post-transduction;
T14 follows fourteen more days, approximately ten doublings. This is not a
measured per-doubling gamma denominator. These facts are methodological, not new
performance measurements. [Primary methods](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12018271/fullTextXML).

The official
[Supplementary File 1](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-08815-4/MediaObjects/41586_2025_8815_MOESM1_ESM.docx)
was fetched into memory and only its methods function text was inspected. Its
3,500,612 bytes have SHA256
`27b145b688fef7a805d2b78178eb81d51ebce3b1895abfa80b88fb16d161904e`.
It also contains flow-cytometry gating and uncropped-blot images; those images
were not inspected, exported or used. The modified function defaults
`normalize=TRUE`. With `normalize=FALSE`, it uses the input matrix directly,
adds the pseudocount before log transformation and subtracts early-time log
counts. Its early/late sample grouping depends on `replicate.map`; multiple
columns can be averaged within sample. The supplement supplies the function,
not the actual invocation or sample map. The intended avoidance of double
normalization does not independently establish the flag used by an omitted call.
No complete raw-count-generation or size-factor-export implementation was
located in the inspected code-availability record.

The [ENA metadata-only run report](https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA988447&result=read_run&fields=run_accession,sample_accession,experiment_accession,experiment_title&format=tsv)
contains four runs whose submitted experiment titles map to:

| Culture | T0 | T14 |
| --- | --- | --- |
| 1 | SRR25050635 / SAMN36019608 | SRR25050633 / SAMN36019610 |
| 2 | SRR25050634 / SAMN36019609 | SRR25050632 / SAMN36019611 |

Those mappings qualify archive metadata, not the unopened table's column names.
No FASTQ was acquired. Public methods/narrative and supplementary code are
exposed; the response-count matrix remains unopened. This is not whole-source
blindness.

## Remaining SPIDR blockers and smallest resolutions

| Blocker | Smallest resolution before any count acquisition |
| --- | --- |
| Table 8 is normalized, contrary to the tested raw-count input contract | Obtain a documented raw construct-count matrix and its sample map, or register/test a distinct normalized-count endpoint with known size-factor/pseudocount semantics. Never round normalized values into raw counts. |
| Exact response header and construct linkage remain unknown | Obtain a schema-only companion or official processing/sample-map code. A future header-only access needs an explicit exposure record and must not silently include the first data row. No such access occurred here. |
| Published invocation and size factors are missing | Establish the actual normalization flag, non-targeting reference set, size-factor rule and early/late culture mapping. The function definition alone is insufficient. |
| Nine-hub primary design | Decide whether that narrow capability is scientifically worth reserving a source for, before opening it. A changed stratum/construct threshold requires a new registration and claim. |
| No independently initiated cultures | Keep split cultures as conditional measurements and obtain another independently initiated study or prospective assay for a broader replication claim. |
| Raw-data redistribution scope is not source-specifically cleared | Preserve the archive policy and any source-specific terms; do not apply the article's license automatically to raw FASTQ. |

The [article license](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12018271/fullTextXML)
is CC BY 4.0 with material-specific exceptions. [NCBI's molecular-data
policy](https://www.ncbi.nlm.nih.gov/home/about/policies/#data) adds no repository
reuse restrictions but says submitter intellectual-property claims can remain.
That supports public archive access without constituting a source-specific
blanket redistribution license. No raw sequencing asset or private material was
added to the repository.

## SCHEMATIC is a separate fallback, not a qualified replacement

Fong et al., *A multilineage screen identifies actionable synthetic lethal
interactions in human cancers*, DOI
[10.1038/s41588-024-01971-9](https://doi.org/10.1038/s41588-024-01971-9), supplies
an independent laboratory and a different CRISPR-knockout growth task.
The [NCBI article XML](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=12266356)
was accessible; EuropePMC XML returned HTTP 500. The design uses 67 by 176
gene-axis positions, three guides per gene and nine ordered guide combinations
per axis-pair, with non-targeting and AAVS1 safe-editing controls. Seven cell lines
were screened in duplicate at four times across 24–28 days. Fitness is
sequencing-abundance LFC versus the plasmid pool. The publication does not
establish separate infection initiations. This is not CRISPRi gamma or SPIDR T14/T0
LFC, and its axis overlap still needs a metadata-only guide manifest.

The same primary methods define published single fitness using medians over all
double partners. Those values would leak held-pair outcomes if used as measured
single inputs. They cannot be imported. The designed target/non-targeting
constructs could instead support separately declared exact-guide singles, but
their actual count coverage remains unobserved. The paper's outcome-based
timepoint filtering also cannot select a future confirmatory panel.
[Primary methods](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=12266356).

The official [SCHEMATIC repository](https://github.com/samsonfong/SCHEMATIC/tree/218cf1bb8c83204da846256951ba5f6b0092890b)
was inspected at `218cf1bb8c83204da846256951ba5f6b0092890b` using its file tree,
README, dependency metadata and Python source, without cloning datasets or
notebooks. It expects a five-field count index, a `plasmid` reference column and
sample names carrying cell/time/culture information. Its
[fitness pipeline](https://github.com/samsonfong/SCHEMATIC/blob/218cf1bb8c83204da846256951ba5f6b0092890b/src/decipher/scores/fitness.py)
uses pseudocount one, whole-pair variance scaling, partner-median singles and
outcome-dependent sample filtering. Its control-only aggregation helper exists,
but the main pipeline does not use it for singles. A new evaluator must therefore
define its own train-safe transformations and exact measured-single inputs,
rather than calling the published pipeline unchanged.

Correct the earlier literature record when selecting a source: **Supplementary
Data 1 is the read-count asset; Supplementary Data 2 contains interaction
scores**, as labeled in the primary article's supplement metadata. Data 2 must
not be treated as an independent raw-count source. The supplementary tables
workbook includes guide metadata alongside other tables; it was not acquired.

The [ENA metadata-only project report](https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA701256&result=read_run&fields=run_accession,sample_accession,experiment_accession,experiment_title&format=tsv)
lists 56 paired-end amplicon runs. For example, BioSample metadata identify
[SAMN17859973](https://www.ebi.ac.uk/ena/browser/api/xml/SAMN17859973) as
`A549_t0_1` and
[SAMN17859979](https://www.ebi.ac.uk/ena/browser/api/xml/SAMN17859979) as
`A549_t3_1`. These are symbolic time labels, not verified elapsed-day or infection
records. They do not establish that culture 1 and culture 2 originated in
separate transductions.

The inspected manuscript license permits text mining and use consistent with
fair use; it is not an article-wide CC BY grant. The pinned repository tree has
no license file and GitHub reports no recognized license. Do not vendor its code
or redistribute its assets on an assumed permissive license. Raw molecular-data
access follows the repository policy above; supplement-specific rights still
need verification. No SCHEMATIC response asset was acquired or opened.

SCHEMATIC's nine-construct design is a better metadata lead for broad pair
coverage than SPIDR's frozen nine-hub primary panel. Its strengths do not resolve
the missing control/count schema, exact-axis overlap, elapsed-time map,
initiation lineage or reuse terms. Qualify those metadata first, then register a
distinct KO-growth endpoint with target/NT singles, a fixed final-time rule and
canonical-pair grouping. Neither source currently qualifies independent
confirmation of an Opal predictor.
