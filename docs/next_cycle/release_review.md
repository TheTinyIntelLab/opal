# Opal 0.1.2 release review

The package consolidates the comparative study. Original frozen Stage 1 runs are canonical; the changed-environment rerun is a reproducibility sensitivity. Publication remains paused for owner review. Dirty-workspace packages are local review candidates, not immutable public releases.

## Asset policy

| Asset | Decision and support |
| --- | --- |
| Code, wheel, sdist, original docs | MIT code; Python 3.11 through 3.14 declared. Clean Windows 3.11/3.14 checks are recorded separately. |
| Compact evidence | Original/rerun roles, amendments and unfavorable outcomes retained. Selected comparison CSV names are allowlisted; expression/prediction arrays are excluded. |
| Manuscript and standalone TeX | Comparative research preprint; extracted bundle compiles locally. arXiv server build and submission are not claimed. |
| Norman kernel export | Existing scoped observed-single log2 reference; scPerturb v1.1, Zenodo 7278143, CC BY 4.0. Source attribution and manifest hashes retained. |
| EmeraldBay kernel export | Existing observed-reference context export; pinned Tahoe Bio revision, CC BY 4.0. No arbitrary context or missing-single prediction claim. |
| Wessels/D-SPIN banks and weights | Excluded because derivative redistribution terms are unresolved. |
| Combi-seq/bulk expression banks | GSE174695 was downloaded for the scoped public processing diagnostic; raw/derived arrays remain excluded. Derivative terms and culture/dose provenance remain unresolved. |
| External models, priors, embeddings, raw data | Excluded; upstream terms remain separate. |
| Private execution/outreach files | Case-insensitive exclusion of plan, outreach, context and roadmap names. Metadata snapshots with contact details remain local. |

See [third-party notices](../../THIRD_PARTY_NOTICES.md) and each export manifest for exact source versions, terms and support. Archive allowlisting is separate from the Git-tree review; the pre-change tracked tree had no private-name matches. Every archive path is portable, rejects traversal/drive/backslash paths, and excludes symlink members.

## Reproduction routes

Saved-evidence reporting regenerates selected numerical claims and publication figures without fitting or historical evaluator access. Installed inference restores the two permitted reference exports against saved predictions and checks symmetry, gene order, scale, missing singles and unsupported contexts. Full research reproduction remains a separate route with its source access, licensing and fitting dependencies.

Build with `python -m build --outdir dist/closeout-0.1.2`. Run `tools/closeout_reporting.py` with a new output directory and `--render`, then its `--check` route. Build the standalone source with `tools/build_arxiv_submission.py --config configs/releases/v0.1.2.json --output releases/arxiv/opal-0.1.2-arxiv-source.zip`. Package with `tools/package_research_release.py --config configs/releases/v0.1.2.json --dist dist/closeout-0.1.2 --output releases/local-review-0.1.2 --allow-dirty`; validate using `tools/validate_research_release.py --assets releases/local-review-0.1.2`.

`tools/check_clean_install.py` creates a fresh dependency environment outside the checkout, verifies installed import provenance, runs the actual synthetic quickstart and checks dependency closure. The Windows/Linux Python 3.11/3.14 CI matrix is configured, but Linux checks are not claimed until CI actually runs.

## Owner-operated publication

Review the exact manifest, assets, manuscript, model cards and logs, then resolve the existing publication pause. On 9 October the owner authorized committing and pushing the implementation, superseding the earlier no-commit instruction. Git publication of this work is now authorized; external release uploads, Zenodo archival, actual DOI insertion, arXiv submission and identifier updates remain separately pending. Earlier dirty-input package records retain their historical scope. No invented DOI, preprint identifier, registration or collaborator commitment is used.

Suggested wording: Opal evaluates population-mean pair-response prediction when matched single effects are observed. Its controlled comparisons show that improvement over additivity can occur without demonstrated incremental pair-specific learning. The release includes scoped simple references, reproducible comparative records, and unresolved transfer and measurement findings.
