# Licensed COLO1 annotation audit results, v1

One response-free annotation object is now acquired, byte-verified and structurally inspected. This resolves the earlier member-association/acquisition gap, but it does not qualify source responses or establish a biological result. The actual object is a 78,920-by-7 R character matrix with fields ID, Note, MyNote, sgRNA1_ID, sgRNA2_ID, Gene1 and Gene2. Six fields differ from the predeclared documentation-based allowlist. The separately registered matrix adapter refuses field-value conversion and aggregation. Guide, single/control and sample joins remain unestablished.

## Prospective sequence and negative results

1. The [validator diagnostic](colo1_validator_diagnostic_v2.md) froze two HTTP requests, one documentation query and at most two primary documentation pages before execution. The technical file catalog returned HTTP 200 and 980 bytes. Its entries are a list, contrary to the tool's mapping expectation; the original receipt retains that mismatch. The separate final-22-byte range request returned HTTP 206 with the exact Content-Range and Content-Length, but no ETag. Zero archive body bytes were read. This new observation does not reconstruct or retroactively pass the earlier rejected ETag.
2. The [container catalog registration](colo1_container_catalog_v1.md) used the actual primary file identity and its documented container link. It returned a 59,765-byte JSON inventory with 123 entries, total 123 and truncated false. Entries include directories, so 123 is not a count of regular files. The specific row-annotation member has declared size 277,295 and CRC32 508845907. No archive or member body was opened during inventory. Other catalog filenames were not treated as response-free authorization.
3. [Opaque acquisition v1](colo1_annotation_acquisition_v1.md) received HTTP 200 for the exact metadata member, but Content-Length was absent. Its frozen envelope gate closed before reading any body. Preserve this failure. [Acquisition v2](colo1_annotation_acquisition_v2.md) registered a different complete-object identity contract: bounded read plus pre-existing size/Git-blob/CRC identities, refusing contradictory headers, status, MIME, redirects, truncation, corruption and oversize. It passed with exactly 277,295 bytes. It does not retroactively pass v1, relax a scientific tolerance or authorize any other member.
4. The [first schema audit](colo1_annotation_schema_v1.md) pinned two pure-Python wheels before download and installed them into ignored temporary storage with no dependency resolution or byte compilation. Its synthetic gzip fixture failed because GzipFile received BytesIO as filename instead of fileobj. Fifteen other cases passed. A root command-sequencing error nevertheless launched the audit, which returned TypeError at that same decompression call before third-party deserialization. A successful watchdog exit merely records a normally reported refusal; it is not a successful audit.
5. [Schema v2](colo1_annotation_schema_v2.md) corrects only the demonstrated gzip call and adds refusal-stage reporting. Its fixtures passed before a separate actual invocation. Decompression yielded 10,519,604 bytes, but the data-frame-only adapter refused at structural_schema. V1 code/tests/binding/failure bytes remain unchanged. One exact legacy gzip case is now a strict expected TypeError through the collection hook in tests/conftest.py. An unexpected pass fails, and all successor gzip fixtures must pass normally.
6. The [structural v3 diagnostic](colo1_annotation_structure_v3.md) retained partial schema without value conversion. It established root STR with dim/dimnames attributes, no explicit class and no unsupported object kinds. Thus the earlier data-frame hypothesis is false for this object; the problem is representation, not an identified executable payload or missing byte integrity.
7. The [matrix v4 registration](colo1_annotation_matrix_v4.md) froze a two-dimensional R character-matrix adapter, tested column-major ordering and dimension/field refusals, then executed exactly once. Dimensions are 78,920 by 7, storage length 552,440, with the field names listed above. Only ID belongs to the unchanged allowlist; the other six cause unknown_field. No field values were converted, no mapping counts were calculated, and the finite local annotation menu stops here. A plausible name is not evidence for guide identity, arm semantics or a gene mapping.

## Identity, rights and transport

The [primary Zenodo record](https://doi.org/10.5281/zenodo.17191951) declares CC BY 4.0 and associates the Release2 archive with the [source-owner release](https://github.com/ibarrioh/DualGuide_COLO1/tree/Release2), pinned commit cfb5eb9a39ff1ec43642fba9485270ae200e81ae. Its primary input-role documentation classifies IDs_colo1.rds as row annotation and separately classifies count, LFC, scaling, score and QC objects. Those other objects remain unopened. The actual member inside the licensed deposit matches the pre-existing GitHub catalog identity:

| Identity | Verified value |
| --- | --- |
| Member | ibarrioh-DualGuide_COLO1-cfb5eb9/Analysis/ENCORE/input/IDs_colo1.rds |
| Bytes | 277295 |
| Git blob SHA1 | b88d727fdc3f34109d07a94cfae44aad99405990 |
| CRC32 | 508845907 |
| SHA256 | 13746b0357683c690b3066da194d58451ed2bb500cd5ea66f5c44efaa2ed0870 |

The archive's catalog MD5 was not verified by downloading its mixed 89,422,914-byte body. Complete-member identity is established separately; it does not imply whole-archive verification. Raw catalogs, the annotation, wheels and installed dependencies remain local under ignored tmp, outside Git and distributions.

Zenodo's [file-management documentation](https://help.zenodo.org/docs/deposit/manage-files/) permits file correction and exceptional support-mediated changes, while its [version documentation](https://help.zenodo.org/docs/deposit/manage-versions/) describes distinct records/files/PIDs for versions. A DOI is not a byte hash. The compact [documentation observation](../../../results/research-continuation/colo1-documentation-observations-v1.json) records the finite query/pages; the web service did not supply retained raw response hashes or local resource measurements. Do not invent them or treat this as a legal conclusion beyond the catalog's declared grant.

## Actual coverage and remaining qualification gaps

The 78,920 annotation rows are neither the 100,136 designed vectors nor a verified eligible response population. Their selection/QC/derivation lineage remains unknown. No ID uniqueness, ordered-guide-to-sequence mapping, target consistency, canonical pair count or exact matched-single/control eligibility was computed because the field gate refused. The names Gene1/Gene2 and sgRNA1_ID/sgRNA2_ID do not by themselves establish the documented guide-arm convention or distinguish WGE identifiers from internal labels. Note/MyNote cannot be silently interpreted as design or selection roles.

The next source step needs primary derivation/field-role evidence for these exact columns and annotation subset, followed by a separately frozen semantic adapter if justified. It also needs actual sample/time/culture/initiation keys, exact control roles, normalization invocation and train-safe preprocessing. COLO1 singleton controls target nonessential genes; they cannot silently become unperturbed controls. No new fit or response ingestion is justified by this audit. SPIDR, Shen 2017 and SCHEMATIC response tables, headers and values remain unopened. Paralog METADATA access and historical GitHub searches remain as previously reported; this is no new mirror or qualification claim for those sources.

## Resources, tests and custody

Five source HTTP requests were executed across the frozen transport/catalog/acquisition phases, with one complete metadata member body acquired. Two PyPI catalog requests and two hash-pinned wheel downloads were separately authorized, plus the finite documentation query/pages. There were no new source requests in the parser audits. The gzip-defective v1 invocation did not deserialize; the corrected v2, structural v3 and matrix v4 each parsed only the same classified metadata object. No R interpreter, study script, R function or biological analysis ran.

Parser pins are rdata 1.1.0 and xarray 2026.9.0. The isolated install manifest hashes 835 files; active host dependencies reused numpy 2.4.0, pandas 2.3.3, typing_extensions 4.15.0 and packaging 25.0. Host versions and isolated file bytes are checked; this is not a claim that every host dependency file was cryptographically pinned. No .venv modification occurred. The 64-MiB expansion bound, 150,000-row bound and 1-GiB reserve remained unchanged. Setup limits were 120s/512MiB, audits 180s/1GiB, with sampled external process-tree monitoring. CPU/peak samples can miss exit tails and are not service-side measurements.

Exact argv, working directory, code/freezes, start time and log hashes are retained in each command receipt and its .started.json companion. The table links each authoritative command receipt; these are measured resource values, not estimates.

| Command receipt | External wall seconds | Sampled CPU seconds | Sampled peak bytes | Watchdog result |
| --- | ---: | ---: | ---: | --- |
| [colo1-validator-tests-v2](../../../results/research-continuation/colo1-validator-tests-v2.json) | 0.9742845 | 0.750000 | 83279872 | passed |
| [colo1-validator-command-v2](../../../results/research-continuation/colo1-validator-command-v2.json) | 0.7819169 | 0.093750 | 32505856 | passed |
| [colo1-container-tests-v1](../../../results/research-continuation/colo1-container-tests-v1.json) | 0.9869142 | 0.765625 | 82259968 | passed |
| [colo1-container-command-v1](../../../results/research-continuation/colo1-container-command-v1.json) | 1.3436501 | 0.125000 | 32186368 | passed |
| [colo1-annotation-acquisition-tests-v1](../../../results/research-continuation/colo1-annotation-acquisition-tests-v1.json) | 0.9817992 | 0.687500 | 83275776 | passed |
| [colo1-annotation-acquisition-command-v1](../../../results/research-continuation/colo1-annotation-acquisition-command-v1.json) | 1.2398733 | 0.125000 | 31780864 | passed |
| [colo1-annotation-acquisition-tests-v2](../../../results/research-continuation/colo1-annotation-acquisition-tests-v2.json) | 0.9735481 | 0.765625 | 83423232 | passed |
| [colo1-annotation-acquisition-command-v2](../../../results/research-continuation/colo1-annotation-acquisition-command-v2.json) | 1.6651920 | 0.125000 | 32440320 | passed |
| [colo1-parser-catalog-command-v1](../../../results/research-continuation/colo1-parser-catalog-command-v1.json) | 0.9235050 | 0.125000 | 37068800 | passed |
| [colo1-parser-install-command-v1](../../../results/research-continuation/colo1-parser-install-command-v1.json) | 4.6416435 | 1.203125 | 103723008 | passed |
| [colo1-schema-tests-v1](../../../results/research-continuation/colo1-schema-tests-v1.json) | 1.0274485 | 0.875000 | 97460224 | failed |
| [colo1-schema-command-v1](../../../results/research-continuation/colo1-schema-command-v1.json) | 1.0295171 | 0.812500 | 118013952 | passed |
| [colo1-schema-tests-v2](../../../results/research-continuation/colo1-schema-tests-v2.json) | 1.0376310 | 0.812500 | 97378304 | passed |
| [colo1-schema-command-v2](../../../results/research-continuation/colo1-schema-command-v2.json) | 3.4102338 | 3.296875 | 299782144 | passed |
| [colo1-structure-tests-v3](../../../results/research-continuation/colo1-structure-tests-v3.json) | 0.9164512 | 0.687500 | 83623936 | passed |
| [colo1-structure-command-v3](../../../results/research-continuation/colo1-structure-command-v3.json) | 3.7151879 | 3.453125 | 317296640 | passed |
| [colo1-matrix-tests-v4](../../../results/research-continuation/colo1-matrix-tests-v4.json) | 0.9176637 | 0.687500 | 83648512 | passed |
| [colo1-matrix-command-v4](../../../results/research-continuation/colo1-matrix-command-v4.json) | 3.7174658 | 3.562500 | 316194816 | passed |
| [colo1-annotation-custody-command-v1](../../../results/research-continuation/colo1-annotation-custody-command-v1.json) | 0.3462948 | 0.171875 | 26312704 | passed |

The initial validator/container/acquisition fixture suites passed 9, 13, 9 and 7 cases respectively. Schema v1 had 15 passes and one failure; v1 plus v2 then had 31 passes and one strict expected failure. Structural and matrix fixtures passed 5 and 14 cases. These fixtures test software safeguards, not source eligibility or biological validity. The [local custody verifier](../../../tools/verify_continuation_colo1_annotation_v1.py) makes no requests and deserializes no data; its [receipt](../../../results/research-continuation/colo1-annotation-custody-v1.json) binds the frozen sequence, logs, exact object identities, dependencies and final refusal. It does not independently rederive the study's annotation.

All work in this follow-up is root-authored and executed. The prescribed four-worker proposal/critique/revision history is preserved in the earlier Wave Five records. The later four-worker completion round failed at provider limits before execution, so no independent executed final review is claimed here. No replacement workers were spawned. The broader research goal remains active and unachieved.

## Scientific decision

Rank the actual licensed-object identity and actual-schema observation above the transport details: they remove acquisition ambiguity and falsify the assumed data-frame schema. The field mismatch is useful negative evidence because it prevents invented guide/control joins. All mapping statistics, response ingestion, biological fits and scores were skipped under explicit failed or missing prerequisites. No production numerical implementation changed; the legacy 1e-12 absolute, zero-relative tolerance and all prior numerical failures remain unchanged. No model is promoted, no biological interaction is established, and synthetic recovery remains synthetic evidence.

## Delivery validation

The full local suite passes 1,074 tests with one narrowly scoped strict expected legacy gzip failure and five unchanged warnings, four from the optional local NAIAD checkout and one deliberate duplicate-name ZIP fixture. External wall is 91.9928778s, sampled CPU 67.28125s and peak 728,723,456 bytes. The read-only custody verifier checks 885 distinct bound files, including 835 isolated parser files, and retains the expected failed fixture command. [Delivery validation](../../../results/research-continuation/colo1-annotation-delivery-validation-v1.json) records the separately observed build, documentation and artifact-inspection scope. Hosted clean-install status must be verified for the actual delivered commit; earlier green CI is not evidence for new files.
