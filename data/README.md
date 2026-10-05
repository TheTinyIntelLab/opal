# Data and reference resources

`source_registry.json` names the accepted expression releases and source terms.
Raw files go in `data/raw/`; preparation writes development profiles and separate
historical evaluator responses in `data/prepared/`. Neither directory is committed.
Download commands and observed-single acceptance rules are in the root README.

The architecture build also needs the supplied annotation files in
`OneDrive_1_10-5-2026/`: `go_mask_19264.npz` and `scfoundation_genes.txt`.
Their exact checksums and the generated-resource lineage are retained in
`results/architecture/resource_manifest.json`. The foundation checkpoint and
`go.csv` are audited resources, but the current architecture studies use the
sparse mask directly and do not execute the native contextual backbone.

`python main.py frontier` downloads the publisher-verified PRESAGE archive and
independent Replogle essential-gene expression matrix into
`data/architecture_resources/`. It extracts only declared public priors and
independent-screen resources. DEG rankings are not substituted for expression.
External semantic derivatives retain CC BY-NC 4.0 terms; the code license does
not relicense them. Large resources and trained banks stay local and ignored.
