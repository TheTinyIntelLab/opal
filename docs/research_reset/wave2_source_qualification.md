# Wave two independent-source metadata qualification

21 October 2026. SPIDR remains the preferred independent-source candidate, but
it is not ready for response access. This wave corrects one unnecessary blocker:
raw-SRA redistribution rights do not block a supplement-only route that never
uses FASTQ. The remaining endpoint and evaluator blockers are substantive.
Neither SPIDR nor SCHEMATIC responses or response headers were acquired or opened.
There were no fits, predictions, biological metrics or new source selection from
outcomes. Public article narratives were already exposed and were revisited here;
this is response-table reservation, not whole-source blindness.

## Executed checks and limits

[The registration](../../results/research-wave2-source/registration.json) binds
the code and eleven passing synthetic tests before the bounded qualification
run. This is a metadata audit registration, not a new prediction protocol. The
[receipt](../../results/research-wave2-source/readiness.json) records seven
successful requests, URL-specific byte counts, SHA256 hashes and structural
facts. The allowlist contains article XML, archive run metadata and pinned code
only. Requests and redirects outside it fail before opening a resource, including
response and header-only URLs. No response link from the article was followed.

The run used 8.0721935 seconds wall time, 0.21875 seconds CPU and sampled peak RSS
33,251,328 bytes. It transferred 453,343 metadata/code bytes, within the declared
30-minute wall/CPU, 1 GiB RSS, 25 MiB transfer and 50 MiB output caps. These are
cooperative checks after each bounded request, not operating-system limits;
requests have a 45-second timeout and 8 MiB body cap. Earlier manual metadata
discovery is not included in this run's transfer/runtime totals. No source code,
article body, count file or sequencing asset was added to Git.

Tests cover response/header URL refusal before network calls, unsafe redirects,
bounded reads, complete evidence schemas, deterministic checklist replay,
structural projection without narrative outcome retention, malformed archive
schemas and truncated repository trees. A manifest declaring every item supported
still returns `review_required` and never authorizes count access. Human evidence
review, a tested endpoint and the development gates remain necessary.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_source_readiness.py -q
.venv/Scripts/python.exe tools/check_source_readiness.py --output <new-directory>
```

The tests passed 11/11 in 0.03 seconds. Repeating remote requests can change hashes
because archive row order and article metadata can change. The receipt preserves
the observed bytes' hashes; it does not claim immutable archive serialization.
The deterministic replay test covers the local reviewed checklist and guard,
not byte-identical future HTTP responses.

## SPIDR qualification

The [publisher methods and supplement index](https://www.nature.com/articles/s41586-025-08815-4)
again identify Supplementary Table 8 as median-ratio-normalized sgRNA counts and
the published LFC calculation as using pseudocount ten. The existing raw-integer
adapter therefore remains incompatible. The article provides the modified
function, but its definition does not supply the actual invocation, complete
sample-column map or count-key schema. This wave deliberately did not inspect
Table 8, including its header.

The article's CC BY 4.0 notice applies subject to material-specific exceptions.
The publisher index has no Table 8 exception notice visible in that article
record. This supports a supplement-only access route with attribution and a final
item-notice check, without importing unrelated raw-FASTQ redistribution questions.
It does not establish that all assets in the sequencing archive share that
license. The [article XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12018271/fullTextXML)
was 228,206 bytes, SHA256
`4bed7fdb1bcecc9d6737322b51188036c8b636a4bccf9927181491f519681e99`.

The [ENA run metadata](https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA988447&result=read_run&fields=run_accession,sample_accession,experiment_accession,experiment_title&format=tsv)
still contain four runs for T0/T14 in cultures one and two. They do not identify
the unopened table's actual sample columns. Split cultures after common
transduction do not establish independent experimental initiations. The existing
library design audit supports 121,198 perfect/perfect pairs with at least two
ordered constructs across 548 genes, before observed counts, control matching,
single availability, training support or QC. That count is not the eligible test
population.

The smallest remaining path is a response-free schema/sample-map companion or
official invocation that identifies normalized columns, ordered construct keys,
NT reference roles and early/late mapping. Then register a normalized-count
endpoint, its pseudocount and control centering, its missingness/QC rule and
equal-construct/equal-culture aggregation. Test exact-guide/arm joins and a
full-design resource rehearsal. Completion, native and simple references must
receive identical allowed information and test pairs. Freeze the learning
procedure, budget and evaluator only after Q1b/Q2b/Q3b gates. Opening responses
to infer units or choose these rules would destroy the reserved evaluation.

## SCHEMATIC fallback

The [primary manuscript XML](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=12266356)
still has a text-mining/fair-use notice rather than article-wide CC BY. The pinned
[official repository](https://github.com/samsonfong/SCHEMATIC/tree/218cf1bb8c83204da846256951ba5f6b0092890b)
contains 40 tree entries and no license path. Its README describes installation
but supplies no actual response-column/sample map. No code was vendored.

The inspected [fitness implementation](https://github.com/samsonfong/SCHEMATIC/blob/218cf1bb8c83204da846256951ba5f6b0092890b/src/decipher/scores/fitness.py)
expects a five-field count index and a plasmid reference, and contains the
partner-median singles calculation. Such singles use held-double outcomes and
cannot be imported into Opal's measured-single comparison. A new KO-growth task
would need target/NT measurements, a fixed elapsed-time rule, train-safe
transformations and verified ordered-guide axes. Its 20,206 code bytes hash to
`51103dbf99a3e6a7a4f7255ddb1eaa2700d3e6d465de1284ff02c2d4dda2d755`.

The [ENA project report](https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA701256&result=read_run&fields=run_accession,sample_accession,experiment_accession,experiment_title&format=tsv)
still lists 56 runs. Their titles do not resolve actual count columns, elapsed
days or independently initiated infections. Supplement-specific reuse terms and
the control/count schema remain unresolved. This is a useful metadata lead for a
separate CRISPR-knockout endpoint, not a qualified replacement confirmation set.

The research decision is unchanged: retain SPIDR reservation, close the irrelevant
FASTQ-rights dependency for the supplement route, and finish source semantics and
development gates before committing to a confirmatory access protocol.
