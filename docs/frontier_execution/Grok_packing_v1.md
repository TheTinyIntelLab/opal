# Packing correction for the failed v1 growth profile

This is a storage correction for the already measured `opal-frontier-growth-v1` profile. The model contract remains `opal-frontier-synthetic-v1`. The v1 growth tool, tournament module, and that profile's bytes are not modified. No biological bank is read and the synthetic profile is not refit.

Completion checkpoints store only a bitwise-symmetric native float64 upper triangle, including the diagonal, under `opal-frontier-packing-v1`. Scalar bits are copied. The matrix is not quantized, averaged, or rewritten to force symmetry. Every other array, including a Fortran base kernel basis, keeps its C or Fortran layout. A non-completion checkpoint stays on the original predictor contract. Loading a packed file rebuilds that original `.npz` in a temporary directory of at most 32MiB outside the historical tree, then calls `Predictor.load`. The temporary directory is removed by its context manager.

Each NPY member is checked from its header before the described array is allocated. Object, structured, complex, and other unsupported dtypes are rejected. The declared shape's `product * itemsize` must equal the remaining payload and fit the uncompressed cap. ZIP member sizes are capped as well. Packed completion checkpoints are limited to 600 genes. File identity is the lowercase sha256, and an existing path is not overwritten.

`tools/frontier_growth_v2.py` is the v1 adapter with a new protocol id, `save_checkpoint` / mixed load, packing fields, and binding entries. The menu, labels, gate, 512MiB program cap, and 64MiB profile cap are unchanged. A fit now also stops when eligible genes exceed 600 or the test exceeds 10000, before `fit_models`. `profile` does not refit. `qualify-packed` freezes a separate output directory first, checks the v1 code binding and the stored qualification, and refuses any biological stage marker.

The qualification reuses the v1 measurements. It does not open fit arguments or sealed targets except to hash them. It packs the four completion checkpoints in each of the four profile procedures, leaves the other checkpoints and inputs in the v1 tree, and replays all 112 models with `numpy.array_equal`. Per-procedure storage becomes the measured v1 artifact size minus the full completion files plus the packed files. The forecast is `30 * max(adjusted procedure bytes) * 1.25 + retained v1 profile bytes + new packed profile bytes`. Packing and replay time is added onto each measured procedure time. The recorded v1 peak and the failed v1 qualification stay in the result as prior evidence. `prior_profile_qualified` is false. A storage miss is written as `qualified: false`; it does not raise the caps.

Root can run this after installing a wheel that contains `frontier_packing.py`, from outside the checkout:

```
python tools/frontier_growth_v2.py qualify-packed --output <new-v2-directory> --prior-profile tmp/frontier-execution-20261026/growth-v1
```

`fit` still requires a reviewed `profile-qualification.json` in that v2 directory. This stage writes `packed-qualification.json` and does not start a biological fit. The retained profile field is whatever `profile_bytes` the v1 qualification records, rechecked against the profile and profile-sealed directories. The known v1 value is 49336536. Whether the packed profile still fits in the remaining 64MiB, and whether the 512MiB forecast passes, is determined by those measured file sizes. This document does not invent that result.

Focused packing and v2 fixture tests cover bit-exact completion roundtrip, signed-zero asymmetry, Fortran basis layout, a huge-shape tiny NPY payload refused before allocation, the gene and test guards, the forecast formula, and biological-marker refusal. They do not train a model or read the 49MB profile.
