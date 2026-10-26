# Grok COLO1 source prerequisite v1

Cursor Grok 4.7 continuation on 2026-10-26 after worker A reached capacity. This is not an A completion. No A file, root model, README, or plan was modified. No commit or push. Source qualification stays blocked. The synthetic architecture tournament can run while it is blocked.

## Exact directory blocker

One registered range GET ran after the local audit. URL `https://ndownloader.figshare.com/files/45433417`, `Range: bytes=1153775558-1153775579`, no retry. The response was HTTP 403, `text/html`, `Content-Length: 146`, `Content-Range` absent, redirects 0. Body bytes read: 0. The 146-byte HTML body was not read. Central-directory names: 0. Member bodies: 0. Whole-archive MD5 `45c84b5ec11dc17bf5967968c7097ea0` remains unverified. Receipt: `results/frontier-execution-20261026/Grok_source_requests_v1.json`. Registration SHA256 `fa112026f702adebaafec818da04df44e35492a540dc7590948efad0ab4c074c`. Code SHA256 `2c8caaf1181a5c4c337613c218078a7c144c7de6f373a5cad2f58c4354464c9c`. Wall time 0.617 s. This URL is closed; it was not repeated.

## Cached identity

`results/frontier-execution-20261026/Grok_source_audit_v1.json` was written before the request. It still says `archive_directory_request: not_started`. All 14 hash checks passed, including `tmp/colo1_frontier_v3/methods_projection.json` and `tmp/colo1_frontier_v6/item_1.raw`. The audit did not parse `article.xml`.

| Item | Identity | Class |
| --- | --- | --- |
| Figshare 10.6084/m9.figshare.25533091.v1 | File 45433417, `MAPPING.zip`, 1,153,775,580 bytes, catalog license CC BY 4.0 | Catalog known. Body absent after HTTP 403. |
| ENA ERP183979 / PRJEB102587 | 36 run rows, 36 experiments, 33 samples, source SHA256 `3ebfa8eef43f85c56a435fd6bc41d82c6824201bf6401307cb2d7f38747947fa` | Allowlisted metadata known. |
| Four sample and two experiment XML records | V8 hashes match cached files | Allowlisted projection known. |
| PMC12808759 methods projection | SHA256 `b71b56d4f3421f883dce98498f1568197bf13f5ae9b2ac5e847afa40e5afbae5` | Article narrative already exposed. |
| Pinned documentation at `cfb5eb9a39ff1ec43642fba9485270ae200e81ae` | Four blobs re-hashed to A's v2 receipt | Documentary. |
| Historical annotation object | 78,920 unique construct IDs, SHA256 `13746b0357683c690b3066da194d58451ed2bb500cd5ea66f5c44efaa2ed0870` | Opaque labels known. Sequences unknown. |

## Documentary claims

The pinned COLO1 README states 100,136 vectors, 40 anchor genes, and 404 library genes. Its three stated categories multiply to 960 anchor singletons, 4,848 library singletons, and 64,640 library combinations, summing to 70,448. The README total exceeds that sum by 29,688 vectors. The methods projection states 100,136 vectors and 444 library genes. Both statements are present in the cached texts. The 404-versus-444 difference is unresolved.

The README places anchor singletons with the anchor guide in position 2 and ADAD1, CYLC2, or KLK12 in position 1; library singletons reverse those positions; library combinations use a library guide in position 1 and two of the four anchor guides in position 2. Those are design sentences. They are not a native join from guide ID to sequence to target.

The methods narrative names pyCROQUET v1.6.0, exact guide-pair matches, removal below 10 counts, sum normalisation, scaling by 1,000,000, and a day-14 comparison with day 3 or plasmid, then a log transform. It also says the screens were biological triplicates in HT-29 at 100X GI coverage. That is article prose. Verified invocation on the deposited objects is unknown. A's v4 projection counted 98 code lines and 6,127 bytes from the cached R script and did not retain the lines. This pass did not reopen the script. The script's historical reading imports `IDs_colo1.rds` already prepared.

Article text and earlier aggregate metrics are already exposed. This pass did not open a new results table.

## ENA naming, without an initiation claim

All 36 experiment aliases match eight name patterns: 12 `d14-{1,2,3}-{100x,500x}`, 6 `d14-R{1,2,3}`, 4 `d3-{100x,500x}`, 2 plain `d3`, 2 `lib-GI`, 2 `lib-PILOT`, 2 plain `lib`, and 6 `SC_EXP_35467_{1,2}`. Each pattern counted here includes both `_R1_` and `_R2_` alias suffixes. Thirty rows are SINGLE / NovaSeq 6000 with a blank library name. Six rows are PAIRED / HiSeq 2500. Library names `NT1647659R`, `NT1647660K`, and `NT1647661L` each occur twice.

Thirty samples have one experiment. Three samples have two: SAMEA120519615 (`NT1647659R`, title CPID1020), SAMEA120519616 (`NT1647660K`, title CPID1023), and SAMEA120519617 (`NT1647661L`, title CPID1026), on multiplex labels `35467_1` and `35467_2`. SAMEA120519624 is title `6268STDY16379586` and is the sample descriptor on `EXP_d14-R1`. The four sample projections contain zero values for the allowlisted cell-line, time, replicate, culture, transduction, and passage tags. Alias words and shared sample accessions are the observed keys. Independent initiation, shared transduction, and culture history remain unknown.

## Rights and missing companions

Container license evidence is the Figshare version field CC BY 4.0 and the article CC BY 4.0 projection. The historical Zenodo annotation chain stays historical. No member inside `MAPPING.zip` was named or classified, because the directory was not returned. Per-item rights for an unretrieved member are unknown. Guide sequences, arm orientation on the acquired annotation, annotation QC lineage, processing invocation, reference identity, zero handling, and missingness remain unknown.

## Exposure and tests

New requests: 1. New body bytes: 0. Response tables, count headers, count values, notebooks, and archive members opened: 0. Invented-fixture tests: `python -B -m pytest tests/test_colo1_frontier_grok_zipdir_v1.py tests/test_colo1_frontier_grok_audit_v1.py -q` passed 23 tests. One warning came from building a duplicate-name fixture. Free space at the audit was 6,398,779,392 bytes, with the 1 GiB reserve unused by this request.

## Next primary action

Obtain a 206 response with `Content-Range: bytes 1153775558-1153775579/1153775580` for file 45433417, or a separately documented metadata route to the same version, before any member name is interpreted. Until that exists, do not open a companion body and do not treat COLO1 as qualified. Root can continue the registered synthetic tournament.
