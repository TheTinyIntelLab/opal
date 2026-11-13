# Research archive

The active checkout contains the v1 guides, current result ledgers and contracts required by working tools. Older research is preserved here as verified archives rather than thousands of loose files.

## Full scientific history

[research-records.zip](research-records.zip) contains 3,992 historical documents and result files. The [manifest](research-records.json) records every original path, byte count and SHA-256 hash. It preserves protocols, negative findings, provenance, code snapshots and computational evidence without rewriting experimental history.

The numerical arrays in this archive are the original source-free synthetic replication fixtures under `results/research-continuation/replication-v1/`. They are explicitly identified and hash-pinned in the manifest. No biological dataset or trained biological checkpoint is included.

Restore this collection into a separate directory before replaying a historical workflow:

```bash
python tools/restore_archived_docs.py --research --output ../opal-history
```

Overlay the restored files onto a separate copy of the appropriate source revision. The tool checks archive membership and every file hash, and refuses overwrites. Keep the active v1 checkout lean.

## Retired planning records

[development-records.zip](development-records.zip) and [its manifest](retired-documents.json) preserve 71 earlier planning, coordination and status records needed for custody audits. Eleven private planning/contact records remain outside the repository. Restore the public records with:

```bash
python tools/restore_archived_docs.py --output ../opal-planning-history
```

These are closed historical records, not a current roadmap or experiment queue.

## Earlier documentation and manuscript

[pre-v1-documentation.zip](pre-v1-documentation.zip) and [its manifest](pre-v1-documentation.json) preserve the old README and model/evidence/reproduction guides under their original names.

The [pre-v1 paper](../../paper/historical/v0.1.3-pre-v1/main.pdf), [source](../../paper/historical/v0.1.3-pre-v1/main.tex) and original build manifests remain under `paper/historical/`. Rebuild that source in a separate copy with the retained bibliography, style and evidence inputs.

Use [current research findings](../research.md) for the release interpretation. Raw data, trained models, local builds and unfinished next-phase work are held outside this checkout; original contents have been preserved.
