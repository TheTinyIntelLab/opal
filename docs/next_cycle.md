# The next Opal cycle

The next cycle is executable, and its first real chemical transfer pilot has completed. The master-plan document remains local and excluded from Git.

## What was built

`main.py next` archives historical metadata and source snapshots, verifies the saved uncertainty studies, audits source compatibility, runs development-only kernel attribution and retrospective acquisition, and exercises the contextual pipeline on a labeled synthetic fixture. Historical evaluator responses are not opened. Completed output directories are immutable.

The contextual package in `src/opal/contextual/` supports measured-single prediction across declared contexts. It compares shared correction, global regression, pooled and context-conditioned sum kernels, and pooled and context-conditioned pair kernels. Singles and controls are explicit references; unavailable singles have a separate boolean mask. The schema rejects missing singles, mixed modalities, mixed assays, ambiguous intervention conditions, repeated targets and gene-order changes.

Preparation physically separates fitting, calibration and held-context responses. Model selection holds out fitting contexts. Calibration reserves entire unordered intervention-pair groups across source contexts. Predictions and selection are saved before evaluation access. Exports contain observed references and fitted numeric state in safe NPZ files with checksummed manifests; they contain no raw combination-response bank. Inference reports support and its exploratory reliability status.

## The downloaded chemical pilot

[Tahoe Bio EmeraldBay](https://huggingface.co/datasets/tahoebio/EmeraldBay) supplies raw single-cell counts for compound treatments and combinations across cancer cell lines. Its dataset card declares a five-day endpoint and CC-BY-4.0. The pinned revision is `f2a0be6b02f731553657f0115c345b20bb020ede`. Attribute Tahoe Bio when redistributing derived material; the Opal software license does not replace the source-data terms.

The registered pilot uses the first six expression shards, totaling **2,585,018,634 bytes** and **94,746 cells**. Every shard passed the publisher's LFS SHA-256 and size checks. The adapter downloaded the gene vocabulary, cell-line metadata, dataset card and license too. Raw downloads stay in ignored `data/raw/emeraldbay/`.

The pilot selects up to eight cell lines using condition identities and cell counts, requiring at least 20 cells for the matched vehicle, each exact-dose single, and each pair. Compound identity includes dose, unit and endpoint; unmatched doses and higher-order mixtures are excluded. It yields 17 matched pair conditions and 99 pair/context observations across eight cell lines. The final two sorted eligible cell-line identifiers, NCI-H1573 and SW1417, are reserved for evaluation: 54 fitting observations, 18 calibration observations, and 27 evaluation observations.

The RTX 4060 performed the preprocessing in an isolated CUDA 12.6 PyTorch environment. Each cell is library-normalized to 10,000, transformed with `log2(1+x)`, and averaged within its condition; effects subtract that cell line's matched vehicle mean. The 1,024-gene feature set is selected from vehicle controls only. Context features are the same control profiles. Target-context controls and measured singles are explicitly allowed. Cells are pooled across the bounded shards; this is not an independent biological-replicate design.

| Fitting-selected family | Held-cell-line MSE |
| --- | ---: |
| Additive prediction | 0.210239 |
| Shared correction | 0.203791 |
| Global regression | 0.174979 |
| Pooled pair kernel | 0.173233 |
| Context-conditioned pair kernel | 0.174601 |
| Pooled sum kernel, selected on fitting contexts | **0.172218** |
| Context-conditioned sum kernel | 0.182929 |

The selected pooled sum kernel reduces additive MSE by 18.1%, but provides zero gain over the strongest fitting-selected simple reference because it is that reference. This pilot does not justify adding the context kernel. Scores describe these two exposed evaluation cell lines and the selected control-expression genes; they are not evidence of genetic combination transfer, all-gene performance or broad biological generalization.

Four calibration pair groups cannot supply a finite 90% error radius under the registered rank rule. The export reports `finite_radius_unavailable` and `exploratory_no_validated_abstention`; it supplies no confidence guarantee. Independent source/replicate confirmation and native information-aligned comparators remain unfinished.

The immutable biological results are in `results/next/emeraldbay-pilot-v1/`: `results.json`, selection, calibration, access record, source audit, dataset protocol, source snapshot and checkpoint. Subsequent `main.py next` runs reuse the recorded result after verifying the dataset protocol hash, without reopening its evaluator. Reusing evidence does not turn it into a new confirmation.

## What the earlier data explained

The attribution study compares independently tuned components, fixed mixed-model branch removal, shuffled single-feature identities, and matched near/random pair removal on development data. Wessels random-fold MSE is 0.005031 for the single-sum kernel versus 0.005192 for pair-product geometry; partner-fold MSE is much worse at 0.010053 and 0.010149 respectively. The gain largely follows measured-single geometry under these tests. Pair multiplication does not earn its complexity here, and weak partner transfer remains visible.

Retrospective acquisition starts every policy from the same five labeled pairs and freezes model selection using those labels. It compares uniform acquisition, target balance, diversity and conditional kernel variance on three dependent seeds and a common fixed validation set per seed. On Norman at 40 labels, mean MSE is 0.008218 for uniform and 0.007556 for diversity. On Wessels at 20 labels, uniform is 0.006238, diversity is worse at 0.007049, and target balance is 0.005631. This supports dataset-specific acquisition evaluation; it does not establish a universal or prospective policy. Infeasible budgets are reported instead of silently capped.

All 12 stored ceiling studies passed the metadata, split, full-size-control and export checks. Their older evidence and source files remain separate from the new pilot. `cycle-v4` is the integrated next-cycle run; the earlier runs are preserved with their original source snapshots and status.

## Run it locally

Use the existing CPU environment for the audit and registered studies:

```powershell
.venv/Scripts/python.exe main.py next --output results/next/my-new-cycle
```

The prepared pilot and historical references are local prerequisites. Choose a new directory for each run. To recreate the chemical preparation with CUDA, use the pinned environment specification in `configs/environments/rtx4060-cu126.txt`; install CUDA PyTorch from its official index, then the remaining packages:

```powershell
C:/Python314/python.exe -m venv .venv-gpu
.venv-gpu/Scripts/python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.venv-gpu/Scripts/python.exe -m pip install numpy==2.5.3 scipy==1.18.1 pandas==3.0.6 pyarrow==25.0.1 threadpoolctl==3.7.0 psutil==7.2.2
.venv-gpu/Scripts/python.exe tools/prepare_emeraldbay.py --output data/prepared/emeraldbay-new-copy
```

This command verifies cached files and downloads missing ones. The pilot config, source revision, minimum counts, subset and normalization are explicit; preparation fails if matched reference coverage is inadequate. The existing CPU environment and graphics driver are unchanged.

For a newly prepared dataset, benchmark through the CLI:

```powershell
.venv/Scripts/python.exe main.py context-benchmark --dataset data/prepared/emeraldbay-new-copy --output results/next/emeraldbay-new-run
```

This reproduces the already-exposed pilot and must remain labeled retrospective. It does not create new held-out evidence.

Use the completed checkpoint directly:

```powershell
.venv/Scripts/python.exe main.py context-predict --checkpoint results/next/portable-checkpoints-v1/emeraldbay --context CVCL_1478 --pair "Gemcitabine@10uM:5d" "Paclitaxel@0.5uM:5d" --output tmp/chemical-prediction.npz
```

`--single` and `--control` provide the explicit reference routes. The command writes arrays and a JSON support sidecar and refuses overwrite. Python callers can use `from opal.contextual import ContextPredictor` and the same checkpoint.

## Validation and remaining gates

The full suite passes **104 tests**. Tests cover kernel symmetry/PSD, safe exports, missing-versus-zero singles, condition metadata, gene/context integrity, fitting/calibration/evaluation isolation, finite-rank calibration, selective-risk ties, attribution isolation, acquisition isolation, exact-dose source matching, lineage-wide fold isolation and estimated-single provenance. Release wheels are installed into the isolated environment and exercised from outside the source checkout; the contextual inference route does not import PyTorch.

The completed pilot checkpoint is about 1.7 MB, including references and state. A fresh process measured about 65 MB peak working set after removing an unused import that pulled in PyTorch, versus about 556 MB before that correction. Fifty warm single-pair predictions have median latency about 6 ms with one numerical thread on this PC. These figures apply to this 1,024-gene pilot and include support metadata; they are not large-corpus estimates.

The original CPU environment's import probes still lack native dependencies. The separate Python 3.10 environment now runs CPA and GEARS on the RTX 4060. CPA's aggregate-input pilot MSE is **0.157098**, compared with the kernel's **0.172218**. A no-label restore of its exported weights reproduces all predictions exactly. GEARS uses the native architecture and loss with aggregate observations and graphs derived only from allowed single profiles. Its random/partner development MSE is **0.011199/0.011707** on Norman and **0.007094/0.008419** on Wessels. Neither adaptation is a canonical cell-level benchmark, and the chemical pilot does not make genetic-method comparisons applicable to chemicals.

The frozen kernel procedure was evaluated on independent D-SPIN data after holding out whole Myeloid and NK lineages across both batches. Its MSE is **0.034025**, versus **0.033108** for the source-selected simple reference, so it fails the registered material-skill gate. The nominal 90% empirical source radius covers only **34.3%** of held conditions. Positive transfer confirmation and calibrated abstention remain unestablished. More complex representations are conditional on improving the simple reference; these results do not justify them. Higher-order mixtures and sequential treatment remain unsupported.

## Independent source and native runs

The D-SPIN source has 173,412 cells and a publisher-verified 772 MB count matrix, stored on D: to avoid crowding the system disk. The recipe verifies MD5 and SHA256; change `raw_path` in a new config when using another machine. The processed task has 10 lineage/batch contexts, 273 pair/context observations and 1,011 genes in the intersection with the frozen EmeraldBay vocabulary. Published lineage assignments are post-treatment, and neither batches nor repeated doses establish independent biological replication. The source study reports 24-hour drug exposure; the prepared manifest conservatively records one shared source endpoint and does not make a time-transfer claim. [Primary methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC13387596/), [data release](https://data.caltech.edu/records/2cjss-wgh69).

```powershell
.venv/Scripts/python.exe tools/download_dspin.py
.venv-cpa/Scripts/python.exe tools/prepare_dspin.py --output data/prepared/dspin-new-copy
.venv/Scripts/python.exe tools/evaluate_confirmation.py --output results/next/dspin-new-run
```

The confirmation tool reads the registered `dspin-confirmation-v1` dataset and freezes the EmeraldBay-selected procedure. For another prepared directory, register a new config/tool invocation rather than silently replacing that evidence. Repeating this exposed source is retrospective.

Native CPA requires Python 3.10 and CUDA PyTorch 2.0.0+cu118. Its upstream package pins PyTorch at most 2.0.1, while torchaudio 2.0.1 requires 2.0.0. The installed core versions are frozen in `configs/environments/native-common-win310.txt`; install those exact wheels with the CUDA 11.8 PyTorch index, then install the pinned `tools/external/CPA` with `--no-deps`. The optional Ray[data] resolver constraints conflict with the Arrow runtime used here; this environment uses Ray 2.9.3 without those extras and Arrow 14.0.2. No algorithm source was changed. The original Opal wheel targets Python 3.11+, so these native research tools run from source in their separate environment.

```powershell
.venv-cpa/Scripts/python.exe tools/study_native_cpa.py --output results/next/cpa-new-run
.venv-cpa/Scripts/python.exe tools/study_native_gears.py --dataset norman-log2 --output results/next/gears-new-norman
.venv-cpa/Scripts/python.exe tools/study_native_gears.py --dataset wessels-log2 --output results/next/gears-new-wessels
.venv-cpa/Scripts/python.exe tools/predict_native_cpa.py --checkpoint results/next/native-cpa-release-v1/checkpoint --context CVCL_1478 --pair "Gemcitabine@10uM:5d" "Paclitaxel@0.5uM:5d" --output tmp/native-cpa-prediction.npz
```

The full 19,264-gene PertAdapt adapter failed the recorded float32 forward/backward feasibility probe with CUDA out of memory. Its distributed launcher also requires NCCL, unavailable on this Windows build. No PertAdapt biological score is reported. A full gene-aligned backbone run requires a compatible environment and memory budget; dropping genes or substituting random backbone features would change the method.

## Estimated singles and research artifacts

Missing-single inference is implemented under a separate experimental schema. The imputer uses measured instances of the exact intervention/dose in other contexts, identifies their source, and abstains if no such reference exists. It never changes measured-availability flags. Source-only single validation chooses between averaging and control-feature kernels. The best control kernel improves source validation only **0.5%**, leaving the representation complexity gate closed.

The 27-query pilot's total MSE is **0.092829**, with main-effect MSE **0.165359**, residual MSE **0.183338** and cross term **-0.255868**. Large error cancellation means this result does not establish reliable main-effect transfer. The estimated route has no calibrated interval. Its export round trip agrees within `7e-15`.

```powershell
.venv/Scripts/python.exe tools/study_main_effects.py --output results/next/missing-single-new-run
.venv/Scripts/python.exe main.py context-predict --estimated-singles --checkpoint results/next/missing-single-pilot-v3/checkpoint --context CVCL_1478 --pair "Gemcitabine@10uM:5d" "Paclitaxel@0.5uM:5d" --output tmp/estimated-prediction.npz
```

The model card is `docs/model_card_next.md`; the recoverable evidence ledger and
report are in `results/next/research-ledger-v2`. `cycle-v7` integrates the
registered extensions by reading saved records without reopening evaluation
outcomes. Checkpoint state version 3 persists fitted gene scaling and kernel
distance normalization, preventing predictions from depending on BLAS thread
roundoff when restored. The portable observed checkpoints and masked-single
export are validated outside the source checkout.

The manuscript includes a context-transfer amendment and a table generated only
from saved result JSON. Regenerate and compile with the existing Tectonic runtime:

```powershell
.venv/Scripts/python.exe tools/report_next_paper.py
.venv/Scripts/python.exe tools/build_paper.py
```

This changes the report, not the historical experiments. Raw data, arrays,
checkpoints and archive bundles remain local; compact score records and the
paper are versioned. The source configs contain recorded machine paths; use new
configs and output directories on another machine rather than overwriting an
exposed run.
