# Grok Figshare central-directory registration v1

Attributed continuation by Cursor Grok 4.7 on 2026-10-26, after worker A reached capacity. This file is not an A completion and does not modify A's registrations, tools, or receipts. Frozen before the request below. A local audit of cached projections precedes any network call.

The exact Figshare version already cataloged by A is DOI 10.6084/m9.figshare.25533091.v1, article id 25533091, file id 45433417, title and member name MAPPING.zip, catalog size 1153775580 bytes, catalog MD5 45c84b5ec11dc17bf5967968c7097ea0, download URL https://ndownloader.figshare.com/files/45433417. A's receipt records that version's license as CC BY 4.0. This partial read cannot verify the MD5. The Zenodo COLO1 range attempt, the paralog Figshare file 46763992, and the publisher article route stay closed.

Authorize at most two ranged GETs of that one URL. Send `Range`, `Accept-Encoding: identity`, and no retries. On 301, 302, 303, 307, or 308, read zero body bytes, require one https Location without userinfo, and repeat the same Range once. A second redirect is a blocker. Do not store a redirect query string.

Read a body only when the status is 206, `Content-Range` is exactly `bytes {start}-{end}/1153775580`, `Content-Length` equals the requested count, `Content-Encoding` is absent or identity, and the MIME type is application/zip, application/octet-stream, application/x-zip-compressed, or binary/octet-stream. Any other status, including 200, is a blocker; do not read or retain that body. When both ranges carry an ETag, the values must be identical or the second body is not read. A missing ETag is not itself a blocker. Timeouts are 30 seconds per request and 90 seconds overall. Memory target is 256 MiB. Keep at least 1 GiB free plus this stage's raw bytes. Accepted raw ranges stay in ignored tmp/colo1_frontier_grok_v1 only.

First range: bytes 1153775558-1153775579, exactly 22 bytes. Accept only a single-disk, comment-free, non-ZIP64 end-of-central-directory record whose entry count is from 1 through 4096, whose directory length is from 1 through 262144, and whose directory is contiguous with this record (`directory_start + directory_length == 1153775558`). Otherwise stop. Do not widen the tail and do not scan backward. A rejected 22-byte buffer is hashed, then discarded, and is not written to disk.

Second range, only after that record parses: the central-directory span and nothing beside it. Project each record's name, regular-or-directory kind, compression method, compressed size, uncompressed size, CRC32, and local-header offset. Do not request a local header or any member byte. Maximum filename length is 240 bytes. Refuse encryption, ZIP64 sentinels, methods other than store or deflate, unsafe or duplicate paths, truncated records, and any member whose local offset plus 30 plus compressed size extends past the directory start. Do not project extra-field or comment text. ZIP64 extra fields are not interpreted.

Before any body access, classify each exact basename:

- colo1_plm.rds, colo1_plm_scale.rds, colo1_c91.rds, and colo1_plm_GEN.rds: documented log-fold-change objects, sealed.
- IDs_colo1.rds: documented row-annotation object, sealed.
- COLO1_RUNMERGED_EXACT_ANNOTATED.txt: documented raw counts, sealed.
- lib-COLO-1.stats and All.stats: documented stats objects, sealed.
- paralog_scores_ryan.csv and SCORE_median_lfc.rds: documented external score or fold-change objects, sealed.
- Every other name: unclassified filename, sealed.

A filename does not establish a schema, a guide sequence, a target join, or response-free contents. The container CC BY 4.0 statement does not authorize a member extraction. No count-matrix header or value, notebook output, mixed member, or full archive is authorized. No biological fit is authorized.
