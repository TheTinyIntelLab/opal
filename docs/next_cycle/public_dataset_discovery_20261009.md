# Bounded public discovery follow-up, 9 October 2026

Two fixed GEO searches returned 296 unique series metadata records. Nine leads received sample-level SOFT review; none qualifies for the original fixed-state held-pair confirmation design. This is a bounded search result, not proof that no suitable public study exists. No new expression matrices or model weights were downloaded. Public study descriptions were encountered, so metadata-only access does not establish outcome-naive exposure.

The [inventory](../../results/replicated_fixed_state_v1/public-discovery-20261009-v1/inventory.json) preserves translated queries, counts, retrieval URLs, byte budgets and SHA256 hashes. Its text-based ranking prioritizes inspection and never certifies cultures. The [qualification decisions](../../results/replicated_fixed_state_v1/public-discovery-followup-20261009-v1/qualification_decisions.json) distinguish the nine reviewed sources from the other 287 summary-only records.

| Source | Design evidence and consequence |
| --- | --- |
| GSE264728 | 144 samples mix cell lines, organoids, genetic interventions, estrogen conditions and times. The principal chemical comparison is PF-9363 plus SNDX-5613; repeated labels do not establish a complete independent-culture bank or a held-pair graph. |
| GSE272885 | 48 samples cover four genotypes and four co-culture conditions with three repeat labels. NR4A1/SOCS3 supplies one double knockout, so there is no held-pair training/validation/test graph. |
| GSE315341 | All 60 deposited expression samples report either DMSO or SNDX-50469 alone. A combination-focused study title does not supply joint expression observations. |
| GSE305378 | 108 samples cover two cell lines, multiple doses and three collection days. The series design resolves D1/D2/D3 as days; they cannot be counted as culture replication. Only one chemical pair is present. |
| GSE298247 | 36 macrophage samples cover vehicle and four treatments at separate time strata. Four reported repeats and one DHA/dexamethasone pair do not supply six qualified cultures or a held-pair graph. |
| GSE288795 | 111 mouse samples mix three tissues, sex and treatment groups. Rapamycin/trametinib supplies one pair; tissues and animals cannot be pooled into the existing culture estimand. |
| GSE281604 | 74 samples measure tumor barcode abundance and fitness. This is not the matched gene-expression endpoint required by the plan. |
| GSE282705 | 90 samples mix xenograft models and BLU-222 doses with one palbociclib pair. Series-level fulvestrant language is not resolved by sample treatment labels, and animal/model replication requires a different contract. |
| GSE334500 | 96 wells cover cardiac and retinal organoids, five drugs and five neurotransmitters. Pool-1 is cardiac and Pool-2 retinal; a sequencing depth rerun is technical. The 25 nominal combinations are a possible descriptive successor, but do not establish six independent same-state cultures. |

Each accession links to its primary GEO record through the decision ledger. The local metadata snapshots remain excluded from release archives because full SOFT records may contain contact details. The committed sample summary retains characteristic counts and hashes, without contact fields or expression arrays.

Reproduce discovery with `python tools/discover_public_pair_sources.py --output <new-directory> --snapshots <new-local-directory>`. Recompute sample counts from downloaded SOFT metadata with `python tools/audit_geo_family_design.py --snapshots <local-directory> --output <new-json-file>`. New output paths are required to preserve earlier records. Scientific decisions remain explicit review statements rather than automatic deductions from sample counts.

The existing Combi-seq retrospective benchmark and LNCaP processed measurement diagnostic remain the completed empirical workarounds. Fresh prospective confirmation still needs a source with a sufficiently broad pair panel, exact matched references and documented independent culture initiation. Registration and secure external custody also remain unmet. This search does not change the frozen v0.1.2 release or authorize additional publication destinations.
