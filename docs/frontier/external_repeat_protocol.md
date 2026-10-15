# Scope of the executed public-repeat diagnostic

`configs/experiments/opal_external_repeat_diagnostic_v1.json` is the frozen local scope, recorded before acquiring GSE95153 counts. It changes no historical experiment or successor contract. This is an exposed-narrative descriptive measurement study, not a preregistered predictive confirmation. The source table, acquisition scope, metadata and code are hash-bound in the execution registration. No candidate is nominated.

The exact endpoint, retained population, transformation, missingness/refusal rules and limits are in that configuration. The unit is one deposited sample. Culture independence, matched cross-arm blocks, exact dose/vehicle lineage and repeat-specific initiation remain unverified. All five arms and all source genes are reported. Untreated is not substituted for vehicle. There is no train/validation/test split because no procedure is fitted or predicted; introducing one would imply a different study. There is no predictive comparator, capacity control, candidate tuning budget or success margin to optimize. Whole-column normalization is deterministic, without cross-sample fitted parameters. The fifteen within-arm pairwise differences are measurement diagnostics, not independent samples or negative-control significance tests.

Scientific acceptance means only that source/table identity, explicit fifteen-column mapping, numerical validity, whole-population accounting and independent summary replay pass. Numerical mismatch, missing/ambiguous identities, invalid counts or budget violation fails the operation. Successful execution never passes a biological gate. New laboratory measurements, false source labels, or unsupported precision/power are not supplied by this calculation.

Reproduce with existing scientific/data dependencies, pinned local inputs and new output paths:

```powershell
.venv/Scripts/python.exe tools/review_external_sources.py --input data/source_audits/opal_external_validation_20261015 --output tmp/new-external-metadata
.venv/Scripts/python.exe tools/run_external_repeat_diagnostic.py --input data/source_audits/opal_external_validation_20261015 --output tmp/new-repeat-diagnostic
.venv/Scripts/python.exe tools/replay_external_repeat_diagnostic.py --input data/source_audits/opal_external_validation_20261015 --run results/opal_external_validation_v1/repeat-diagnostic-20261015 --output tmp/new-repeat-replay.json
```

The tools do not download data or send messages. The source-specific acquisition receipts identify official URLs, compressed/expanded sizes and SHA256; source arrays and full contact-bearing metadata stay out of Git and distributions. No numerical rerun is needed after successful replay. Software fixtures work without biological inputs; a fresh source checkout cannot reproduce real statistics without the pinned count/metadata files.

The next predictive protocol remains [the prospective independent-culture design](prospective_confirmation_protocol.md). Its separate pilot, nomination, power-design and fresh confirmation observations, useful margin, comparator information, blinding, complete inference freeze, culture-level endpoint and multiplicity rules are unchanged. This diagnostic cannot fill those evidence fields. A custodian response must establish which biological units and references exist before a more specific existing-data registration is defensible. If no such records exist, obtain a reviewed physical precision design and quote rather than interpreting these five variances as power for an unspecified Opal predictor.
