# Historical scientific archive

Scientific protocols, results and files used directly by reproduction tools remain at their original paths. Superseded planning, coordination and status documents have been removed from the public folders. Failed hypotheses and unfavorable scores remain part of the release evidence.

## Retired planning and coordination

[development-records.zip](development-records.zip) preserves 71 historical documents under their original repository-relative names and bytes. This replaces loose copies of superseded plans, proposals, agent handoffs and status snapshots. The [retirement manifest](retired-documents.json) records each disposition and SHA-256 checksum. Eleven private planning/contact documents are backed up outside the repository and are absent from the public archive.

For a historical custody audit, restore the public records into a new directory:

```bash
python tools/restore_archived_docs.py --output ../opal-historical-docs
```

The [restoration tool](../../tools/restore_archived_docs.py) verifies archive membership and every file hash before writing, and refuses to overwrite existing files. To replay an old frozen experiment, overlay the restored files onto a separate copy of its source revision. Do not reintroduce the obsolete plans into the active release checkout. Runtime-required proposals and source-attempt records remain in place so current tools continue to work.

| Record | Contents and role |
| --- | --- |
| [October 20 development](../model_development_20261020.md) | Expression composition, original growth, external transfer and exploratory diagnostics. |
| [Research reset](../research_reset/summary.md) | Protocols, block studies, guide/label-budget controls, numerical failures and source qualification. |
| [Frontier execution](../frontier_execution/growth_registration.md) | Frozen modeling contracts and six-task lineage. "Frontier" is a historical directory name, not a performance claim. |
| [Earlier evidence audit](../frontier/evidence_audit_20261015.md) | Measurement, source eligibility, missing biological replication and evidence limits. |
| [Model delivery](../model_delivery/registration.md) | Original fold-0 export, exact replay and historical reproduction. |
| [Earlier model series](../next_cycle/opal_model_series.md) | Registered model comparisons, negative gates and source-specific contracts. |
| [Pre-v1 paper](../../paper/historical/v0.1.3-pre-v1/main.pdf) | Full manuscript before the release revision. |

[pre-v1-documentation.zip](pre-v1-documentation.zip) preserves the previous README and model/evidence/reproduction guides under their original repository-relative names. [Its manifest](pre-v1-documentation.json) records SHA-256 values. Extract it into a separate copy of the matching repository to resolve historical relative links; it is not active documentation.

The archived manuscript source is [main.tex](../../paper/historical/v0.1.3-pre-v1/main.tex). To rebuild it, copy it as `paper/main.tex` in a separate repository copy with the retained `paper/evidence/`, bibliography and style inputs. Archived build manifests identify original inputs. Do not overwrite the v1 manuscript to inspect history.

Private correspondence, local forward plans, raw data and model banks are excluded from the public release. Unavailable historical local paths are reproducibility limitations, not implied downloadable assets.
