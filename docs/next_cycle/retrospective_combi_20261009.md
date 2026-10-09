# Retrospective Combi-seq study, 9 October 2026

The approved next workstream has executed on already exposed GSE174695 aggregate reads. This is a nominal-identity, within-source prediction comparison. It does not qualify exact physical doses, independent culture initiation, biological interactions or fresh confirmation. The original prospective protocol remains unchanged and conditional on qualified data.

## Design and executable evidence

The run used 1,263 source rows, three reported runs and all 374 pairs with complete nominal controls and monotherapies. Technical variants were collapsed before modeling. The fixed endpoint is 512 genes ranked by variance of unique observed-single profiles, with lexical ties, out of 25,912 source genes. All query single references are allowed under the observed-single task; pair outcomes do not select genes. This endpoint differs from the earlier all-gene processing diagnostic, so its absolute MSE cannot be compared directly with that report.

Three seeded outer folds hold each canonical pair out across every run. Models fit separately within run, select the existing 16-entry menu on a rotating inner holdout, and refit on both outer development folds. Kernel scales and distances use only the relevant training references. This is nested holdout selection inside outer cross-validation, not a three-fold inner cross-validation. No outcome-based reruns or architecture additions were made. Three fixed whole-profile shuffles each repeat selection. All nine aligned outer fits selected the component-additive reference and the global-plus-pair candidate with penalty 0.1; each shuffled fit also selected the component-additive reference.

Run `tools/run_retrospective_combi.py --metadata data/source_audits/replicated_fixed_state_v1/source_snapshots/combi_treatments.csv.gz --matrix data/source_audits/replicated_fixed_state_v1/source_snapshots/combi_large_scale_matrix.csv.gz --output <new-directory>` with the repository Python environment. The tool writes an already-exposed scope before fitting and refuses to reuse an output directory. Local expression predictions remain ignored by Git. Verify local prediction losses with `tools/verify_retrospective_combi.py <directory>`. Sources are the same downloaded files used by the prior diagnostic; no new expression source was opened.

## Descriptive results

| Run | Zero | Additive | Selected reference | Pair-kernel candidate |
| --- | ---: | ---: | ---: | ---: |
| rep1 | 4.992125 | 7.995942 | 3.675410 | 4.142605 |
| rep2 | 6.271724 | 9.772758 | 4.033414 | 4.588619 |
| rep3 | 5.877819 | 9.211679 | 3.909276 | 4.434377 |
| Equal-run mean | 5.713889 | 8.993460 | 3.872700 | 4.388534 |

The candidate has 13.32% higher error than the selected reference and is worse in all three reported runs. That is evidence against advancing this candidate on this endpoint. Component identity remains known across train and query pairs, so incidence correction can generalize to unseen pairs of known components; this benchmark does not test unseen-component transfer.

The shuffled reference means are 4.022376, 4.039852 and 4.024636; shuffled candidate means are 4.561668, 4.581613 and 4.574068. Alignment gives a modest descriptive reduction relative to these shuffles, but the pair candidate still loses to the aligned reference. These three controls are sensitivity comparisons, not a null distribution or significance test. Neither the poor additive baseline nor the model comparison proves absence of biological interaction.

## Acceptance and limits

The [scope](../../results/replicated_fixed_state_v1/retrospective-combi-20261009-v1/scope_before_fit.json) binds source, code and menu hashes before fitting. The [report](../../results/replicated_fixed_state_v1/retrospective-combi-20261009-v1/report.json) retains every inner configuration comparison, every outer fold, the gene panel, pair assignment and source lineage. The [verification](../../results/replicated_fixed_state_v1/retrospective-combi-20261009-v1/verification.json) checks 96 saved fold/run prediction losses, menu selections and complete outer coverage. The inherited selection payload describes refitting as "train plus validation, all development cultures"; here those arrays contain only the one named processing run, as the runner and fold record specify.

Tests verify held-pair response isolation, global folds across runs, technical-collapse enforcement, reference-only gene selection invariance to duplicate profiles, shuffle reselection, and rejection of tampered saved losses or scientific gate flags. The earlier diagnostic tests continue to pass after extracting shared nominal preparation. Original published-study exposure cannot be undone by nested evaluation; selection bias is why model selection stays inside the outer evaluation partition ([Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)).

No biological intervals or superiority claim were computed. Processing runs, folds, genes, cells and barcodes do not establish new independently initiated cultures. The original scientific and measurement gates remain false. The next source search should resolve concrete metadata gaps in already identified public leads before opening additional expression arrays; a future external-development/new-study test requires its own qualified transfer contract. More same-source architecture tuning is not supported by this result.

All 210 local tests pass. The original processing report was replayed after the shared preparation extraction and every original field matched exactly. Five byte-preserved code snapshots match the pre-fit scope hashes. The current runner additionally copies these snapshots automatically; the exact executed version remains in the evidence directory.

Implementation commit `f289616` passes [all four Windows/Linux Python 3.11/3.14 CI jobs](../../results/replicated_fixed_state_v1/ci-f289616/verification.json), each with a fresh CPU dependency installation, 103 integrity tests and the standalone synthetic assay rehearsal. The committed retrospective runner reproduces all saved predictions and model selections exactly. These checks do not change the scientific claim boundary.
