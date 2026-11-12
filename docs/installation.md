# Installation

Use Python 3.11–3.14 and install from the repository root. `opal-research` is the package name; `opal` is the Python import and CLI. No v1 PyPI publication is implied.

```bash
python -m venv .venv
```

Activate the environment using `.venv/Scripts/Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install ".[data,dev]"
python -m pip check
python -c "import opal; assert opal.__version__ == '1.0.0'"
opal --help
```

For CPU-only Torch, install it first from the official CPU index:

```bash
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[data,dev]"
```

The base package supplies prediction and numerical dependencies. `data` adds AnnData, tabular/HDF5 adapters and thread controls; `dev` adds pytest and resource measurement. `docs` adds Markdown rendering. Optional native comparators require their separately obtained pinned upstream code and licenses; they are not necessary for ordinary prediction.

The recorded growth-delivery stack was Python 3.14.0, NumPy 2.4.0, SciPy 1.16.3, scikit-learn 1.8.0 and Torch 2.9.1+cpu. `requirements.txt` and `requirements-lock.txt` preserve historical versions rather than promising cross-platform lockfile portability. SciPy support is below 1.18 because AnnData 0.12.4 backed sparse access fails with the observed SciPy 1.18.1 environment. The data extra also requires pandas below 3 because AnnData 0.12.4 cannot serialize the observed pandas 3.1.1 default string index. Newer combinations must pass preparation and inference checks before those bounds are relaxed.

Installing code does not download datasets or models. See [model availability](usage.md) and [reproduction prerequisites](reproduction.md).
