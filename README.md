# Opal research

Opal predicts population-mean effects of an unmeasured unordered intervention
pair from measured singles. It adds a learned symmetric correction to the
observed additive prediction. This repository implements the research workflow
in [roadmap.md](roadmap.md); independent biological validation remains unresolved.

The new [state-response benchmark](docs/state_benchmark.md) tests baseline-only
main-effect transfer before expanding the model. It includes a compact state
encoder, separate intervention input paths, symmetric residuals, staged and joint
training, and NumPy inference for compatible new baselines. On EmeraldBay and a
publisher-checksum-verified corrected sci-Plex 3 source, selected simple state
models improve on target averaging by only 0.12% and 0.01%; learned models are
14% and 23% worse than those references. No representation scaling is justified.
The new pair development comparison also favors the matched residual kernel.
These are development results, with no new independent confirmation or reliable
intervals. See the [capability ledger](results/state/capability_ledger.json),
[completed results](docs/state_results.md), [architecture](docs/state_architecture.md)
and [confirmation design](docs/state_confirmation_design.md).

The next-cycle implementation, commands and evidence are in [the next-cycle guide](docs/next_cycle.md).
It covers the historical closeout, crossed-context data requirements, contextual models,
evaluation gates and release work without changing the historical protocol.

The latest architecture batch adds joint reference uncertainty, GO/Hallmark/
GenePT/ESM2 priors, structured decoders, signal-weighted objectives, independent
single-screen transfer, and kernel/GP/retrieval/TabICLv2 comparisons. Within that historical architecture batch, the residual kernel gives the best
aggregate scores; extra training coverage and measurement weighting give the
useful gains. These are exploratory development results, and
frontier quality hasn't been established. Exact scores and exposure limits are in
[the architecture study](docs/architecture_experiments.md).

Run `python main.py frontier` to reproduce the architecture batch. It downloads
verified public resources, trains the independent encoder, repeats nested
ablations and controls, checks exports and tests, and regenerates the paper.
Install the optional frozen tabular comparator with `python -m pip install -r
requirements-tabular-lock.txt`, or pass `--skip-tabular`. `--skip-transfer` omits
expression pretraining but retains semantic-prior comparisons; skipped stages
produce an explicitly partial report. The GO graph and gene dictionary currently
come from the supplied `OneDrive_1_10-5-2026/` resources. Keep those files locally;
the public archive doesn't contain a replacement expression matrix or this exact
GO input. Large data, model weights, caches and generated prediction banks are
ignored by Git. PRESAGE-derived semantic weights retain CC BY-NC 4.0 terms.

The working layout is `data/`, `paper/`, `results/`, `src/`, `tests/`, and `tools/`,
with registered dataset and experiment contracts under `configs/`.

## Current context-transfer results

The contextual pipeline is implemented with exact intervention/dose matching,
separate fitting, grouped calibration and evaluation artifacts, and safe inference
exports. The EmeraldBay chemical pilot's selected kernel MSE is 0.172218 versus
0.210239 for additivity; native CPA on aggregate inputs improves it to 0.157098.
This pilot is exploratory. On independent D-SPIN data, the frozen kernel
procedure has MSE 0.034025 versus 0.033108 for the source-selected simple
reference, failing the registered 5% improvement gate. Its nominal 90% empirical
source radius covers only 34.3% of held conditions, so transfer validation and
calibrated abstention remain unresolved.

Native GEARS development runs use aggregate observations and reference-derived
graphs. GEARS beats the kernel on Wessels partner folds and loses on its random
folds; the kernel wins both Norman protocols. These adapted comparisons do not
establish canonical public-method rankings. Estimated-single inference has a
separate experimental contract: its favorable pilot error includes substantial
main-effect/residual error cancellation and does not establish reliable transfer.

Use `python main.py next --output results/next/new-cycle` for the integrated
registered extension. Inference can run without PyTorch:

```powershell
.venv/Scripts/python.exe main.py context-predict --checkpoint results/next/portable-checkpoints-v1/emeraldbay --context CVCL_1478 --pair "Gemcitabine@10uM:5d" "Paclitaxel@0.5uM:5d" --output tmp/chemical-prediction.npz
```

Large checkpoints and datasets remain local; a fresh checkout must reproduce or
restore the documented artifacts before running this example. The [next-cycle
guide](docs/next_cycle.md) records download, GPU environment, training and
estimated-single commands. The [model card](docs/model_card_next.md) defines
supported inputs and limitations; `results/next/research-ledger-v2` and
`results/next/cycle-v7` retain the compact evidence. The suite currently passes
the historical 104-test suite, and installed-wheel checks reproduce observed and estimated
predictions within `7e-15`. The [manuscript](paper/main.tex) and its
[compiled PDF](paper/main.pdf) include the new findings as a separate amendment.

## Install and run

Use Python 3.11–3.14 with a CPU-compatible PyTorch installation. The recorded
experiments use Python 3.14.0 and the versions in `requirements.txt`.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e ".[data,dev]"
.venv/Scripts/python.exe -m pytest -q --basetemp=.pytest-tmp
.venv/Scripts/python.exe main.py
.venv/Scripts/python.exe main.py reproduce --config configs/experiments/synthetic.json
```

`requirements-lock.txt` records the full scientific/data dependency closure from
this Windows CPU environment; local `+cpu` PyTorch builds can use the PyTorch CPU
wheel index. Other Python/platform combinations require compatible wheels and
have not been exhaustively tested.

`main.py` with no arguments runs the registered synthetic and genetic experiments,
count sensitivities, chemical feasibility run, biological split studies,
technical bootstraps, adapted SLIM references, diagnostics, the architecture amendment and paper build. Use `--skip-architecture` to run only the historical batch. `main.py research
--development-only` stops before opening tests; `--skip-synthetic` uses the real
pipeline directly. `--skip-secondary` runs the primary batch, and `--skip-paper`
preserves the source without requiring a compiler. The full CPU batch can take substantial time. Completed
stages are preserved and fingerprinted, and interrupted evaluations require an
explicit corrected run rather than silently reopening tests.

The synthetic suite exercises eight known response generators, five registered
outer random-pair seeds on the low-rank generator, five partner holdouts, neural
and classical controls, and nested fixed-hyperparameter learning curves.
Generated outputs are explicitly synthetic. Existing frozen run directories are
preserved; choose a new `--output` directory for another reproduction.

## Real data

Source identities, original URLs, licenses and acceptance limitations are in
`data/source_registry.json` and `docs/source_audit.md`. Raw caches are ignored by
Git. Downloads stream to disk and record checksums; data acceptance additionally
requires count provenance, matching singles, controls and context metadata.

```powershell
.venv/Scripts/python.exe tools/download.py norman
.venv/Scripts/python.exe tools/download.py wessels-metadata
.venv/Scripts/python.exe tools/download.py wessels-counts
.venv/Scripts/python.exe tools/prepare_sources.py norman
.venv/Scripts/python.exe tools/prepare_sources.py wessels
.venv/Scripts/python.exe main.py audit --dataset data/prepared/norman-log2
.venv/Scripts/python.exe main.py benchmark --config configs/experiments/norman-log2.json --stage development
.venv/Scripts/python.exe main.py benchmark --config configs/experiments/norman-log2.json --stage final
```

Original expression data for all three sources downloaded successfully. The Wessels prepared download used by some benchmarks lacks singles and is
rejected. The original ComboSciPlex source supports only seven matched pairs,
so chemical release validation requires an explicit amendment or a qualified
replacement. It is not silently trained with missing singles.

Preparation preserves sparse counts, freezes normalization and gene selection
from controls/singles, matches control strata, and separates test responses in an
`evaluator/` directory. Development writes fivefold out-of-fold predictions and
a frozen selection manifest. Final execution checks source and fitting-data
fingerprints, saves predictions before accessing test responses, and refuses a
second test access. Each run saves a source snapshot and environment record.

## Predictions and manuscript

```powershell
.venv/Scripts/python.exe main.py predict --checkpoint results/runs/norman-log2-primary/checkpoint --pair CBL CNN1 --output results/prediction.npz
.venv/Scripts/python.exe main.py evaluate --run results/runs/norman-log2-primary
```

Use `import opal; model = opal.load_model(path)` and
`model.predict(target_a, target_b, context=...)` for local inference. Outputs
include effects, additive terms, residuals and uncalibrated ensemble spread.
Checkpoints contain only observed-single references and development-fitted
weights, stored as safe NumPy arrays with JSON manifests. Controls and singles
have explicit routes; unsupported contexts, repeated targets and missing
profiles are rejected. Checkpoint gene order is immutable.
New exports report whether a query is a development combination, registered test
combination, or another pair in the supported context, using identities only.
Historical exports that lack test identities report the broader third category;
they do not infer source coverage or calibrated reliability.

`paper/main.tex`, `paper/references.bib` and generated `paper/main.pdf` form the
research report. Result tables and figures are generated from saved artifacts;
source-level feasibility, negative findings and incomplete comparisons remain
visible. The draft is not submission-ready while author declarations and
required biological evidence remain unresolved.

For the baseline-conditioned route, `opal.state.StatePredictor` accepts a
processed matched baseline, exact gene order, assay and scale. `opal state-predict`
reads those fields from an NPZ, with an exact single or pair identity and optional
explicit measured-single effects. The two-seed exports under
`results/state/portable-v3/{emeraldbay,sciplex}/checkpoint` and
`results/state/portable-v3/pair/checkpoint` stay local. The pair export restores
the scored last development fold, rather than a model chosen on its outer score.
Unknown interventions, unsupported doses, incompatible assays and repeated
targets are rejected. Clean NumPy-only installations reproduce the exported
predictions exactly outside this checkout. Model variation is uncalibrated,
and inference reliability remains explicitly unavailable.

The manuscript uses the app's built-in source editor. A multi-file PDF build can
also use an existing Tectonic executable:

```powershell
.venv/Scripts/python.exe tools/build_paper.py --tectonic PATH/TO/tectonic.exe
.venv/Scripts/python.exe main.py research --tectonic PATH/TO/tectonic.exe
```

The first build retrieves TeX resources into a workspace cache; later builds reuse
them. No system TeX installation is performed. Compiler input and PDF hashes are
in `paper/build_manifest.json`. Rendered pages were inspected for layout.

See `docs/protocol_amendments.md` for changes from the initial specification and
`docs/progress.md` for evidence-based completion status. The MIT software license
does not relicense source datasets or external methods.

## Joint uncertainty and retuned learning curves

`.venv/Scripts/python.exe main.py ceiling` reproduces the shared-reference covariance extension, three nested split seeds, technical refitting sensitivity, subset-retuned learning curves, isolated error-radius diagnostics, tests, evidence figures and the paper. `main.py frontier` also includes these stages. Completed studies validate their input hashes and seed before reuse; choose a new output folder for changed experimental protocols.

The covariance extension does not consistently beat diagonal weighting, and partner transfer remains weak. Its error-radius diagnostics are empirical and do not establish calibrated abstention. See [the measured extension](docs/ceiling_experiments.md) for results and limitations. Large joint bootstrap arrays and checkpoints stay local; their hashes, source snapshots and compact numerical records are versioned.
