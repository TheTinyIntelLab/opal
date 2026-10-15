# Reproducing the closed workstream

Use an existing environment with the package's declared scientific dependencies plus `data`, `research`, `dev` and `docs` extras as appropriate. Current local validation used Windows Python 3.14; no dependencies were installed in this closeout. CI configuration covers Linux/Windows and Python 3.11/3.14, but this revision has not been run remotely. Paths below are relative to a repository checkout. Every output must be new.

## Without biological assets

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe tools/plan_confirmation_power.py --output tmp/new-power-sensitivity.json
.venv/Scripts/python.exe tools/build_docs.py --output tmp/new-docs-build
.venv/Scripts/python.exe -m build --no-isolation --outdir tmp/new-package-build
```

Tests use bounded fixtures, including source/query isolation, reference-only panel selection, unsupported ranks, complete fallback and decision gates. The analytic power table is a sensitivity calculation, not empirical culture variance or a recommended sample size. An installed wheel alone does not contain repository tools/tests, raw data, numerical banks or private plans.

## With retained prediction banks

```powershell
.venv/Scripts/python.exe tools/audit_research_evidence.py --output results/opal_evidence_audit_v1/new-audit
```

This verifies source/code/state identities, all 594 inner crossed-single losses, 90 final exact replays, complete populations, selectors, fallback counts, primary metrics and conditional bounds. It also verifies retained GxE aggregates and panel custody when the version-2 bundle is present. It never refits models, downloads inputs or creates reviewer/owner approvals. The original historical biological evaluators are not reopened.

## With recovered measurement assets

```powershell
.venv/Scripts/python.exe tools/replay_gxe_aggregates.py --input results/opal_gxe_reconstruction_v2/recovery-20261015 --output tmp/new-gxe-replay.json
```

Required local assets are the roughly 5.56 MB uncompressed aggregate NPZ and its hash-bound axes, metrics, registration, decision and manifest. All 66 rows, including 19 unsupported rows, must match the original metric file. The variance NPZ and panel decision under `panel-verification-20261015` retain the independently reprocessed full-gene reference rule's ranking evidence. Recovered arrays are not raw measurements or new biological replicates; they remain local under unresolved dataset redistribution terms.

## Reconstructing from the original source

The completed operation used the immutable `opal_gxe_reconstruction_v2.json` and `tools/reconstruct_gxe_measurements_v2.py`. It requires all five hash-pinned files under `data/raw/gxe-measurement-20261014`, the original a4 registration/panel/qualification, exact historical transformation code and the unchanged original metric report. It needs no new download or count-file copy. CPU and wall flags may lower limits but cannot exceed the original 300-second/1-GB/10-MB envelope. Progress is diagnostic, not a resumable checkpoint. A failure retains its receipts and never reports complete replay.

No routine count reprocessing is needed after this successful recovery. The panel verifier charges the parent recovery CPU and was run once, with a combined 100.3125-second result; repeated launches must not be used to evade a campaign budget. A deliberately different resource boundary or scientific estimator requires a separately identified operation and genuine authorization. Scientific context-contrast fitting still requires source qualification and its exact accepted inventory; these measurement commands do not grant that permission.

## Evidence and source limitations

`results/opal_gxe_reconstruction_v2/local_asset_inventory.json` binds local bank/variance bytes; compact metadata are committed, numerical banks excluded. Historical inventories preserve failures. Current validation records bind the PDF, source inputs, package and command results. A fresh checkout without retained arrays can run fixtures and inspect compact evidence, but cannot truthfully claim numerical reproduction. Source-specific research/export rights, culture independence and complete hash-table semantics remain distinct from checksum and replay integrity.
