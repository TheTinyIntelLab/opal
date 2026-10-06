# Publishing Opal v1

The release is experimental research, tagged `v0.1.0` with package version `0.1.0`. The commit and tag are pushed and a GitHub draft contains the assets. Publishing the draft and making the private repository public are your remaining owner actions. Don't create another tag or rebuild the files.

## Publish the prepared release

1. Open https://github.com/TheTinyIntelLab/opal while signed into GitHub. Click **Settings**, then **General**. Scroll to **Danger Zone**, find **Change repository visibility**, click **Change visibility**, and choose **Make public**. Follow the confirmation prompts. This exposes the repository and its history, so do it when you're ready for the public research release. If the control is unavailable, an organization owner must change it. [GitHub's visibility instructions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility).

2. Open https://github.com/TheTinyIntelLab/opal/releases. Find **Opal v1 research release**, marked **Draft**, and click its edit pencil. Check that the existing tag is `v0.1.0` and the title is **Opal v1 research release**. Keep **This is a pre-release** checked. The package version is intentional; v1 names the first research cycle, not production maturity.

3. Check for eight uploaded files: the wheel, source distribution, source ZIP, evidence ZIP, model ZIP, manuscript PDF, manifest JSON, and `SHA256SUMS.txt`. GitHub also adds its own source ZIP and tarball links automatically; those don't contain the separate evidence and model assets.

4. Read the description, then click **Publish release**. Don't click **Generate release notes**, because the prepared notes distinguish historical, development, and independent negative results. [GitHub's release instructions](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).

5. Open https://github.com/TheTinyIntelLab/opal/releases/tag/v0.1.0 in a signed-out or private browser window. Check that the page loads and its assets download. Publishing a release alone doesn't make a private repository public.

## What each download is for

| File | Use |
| --- | --- |
| `opal_research-0.1.0-py3-none-any.whl` | Install the Python package. |
| `opal_research-0.1.0.tar.gz` | Build/install the Python source distribution. |
| `opal-v1-research-source.zip` | Source, tools, configs, tests, and manuscript files. |
| `opal-v1-evidence.zip` | Compact saved protocols, selections, results, and reports, including failed gates. |
| `opal-v1-reference-models.zip` | Norman and EmeraldBay kernel references and data notices. |
| `opal-v1-manuscript.pdf` | Research manuscript with author declarations and limitations. |
| `opal-v1-manifest.json` | Exact commit, file inventory, sizes, and checksums. |
| `SHA256SUMS.txt` | SHA-256 for each asset. |

Local copies are in `releases/v0.1.0/`. Large assets go to GitHub Releases rather than Git. The next-cycle plan stays local and is excluded from Git and archives.

## Try the released package

Use Python 3.11 through 3.14. In a new directory, download the wheel and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install .\opal_research-0.1.0-py3-none-any.whl
.\.venv\Scripts\opal.exe --help
```

Normal installation resolves PyTorch and the declared dependencies, so the initial download can be substantial. Don't use `--no-deps` unless you already maintain a compatible environment. Extract the source ZIP, enter its directory, and run the small execution check using the full path to your environment's `opal.exe`:

```powershell
opal reproduce --config configs/experiments/release_quickstart.json --output results/synthetic/my-quickstart
```

This checks two synthetic scenarios, not the biological study. Running `main.py` without arguments starts a much larger workflow, so don't use it as an installation check. Follow [the model guide](docs/release_models.md) for inference and [the evidence guide](docs/release_evidence.md) for scientific scope.

## Add an archival DOI later

A DOI is useful for citation but isn't required for this GitHub research release. Sign into [Zenodo](https://zenodo.org), use **New upload**, and upload all eight assets. A GitHub URL alone doesn't archive the model/evidence files. Enter title **Opal v1: experimental research release**, version **0.1.0**, creator **Arjun Vijay Prakash**, and no institutional affiliation. Mention that you're a high school student at City Montessori School in the description, with no funding or competing interests. Use the release notes as the description and add the GitHub tag URL as a related identifier.

Preserve mixed licensing explicitly: original code is MIT, and data-derived model/reference content has CC BY 4.0 terms in `THIRD_PARTY_NOTICES.md`. Don't relicense the entire deposit under one code license. Review and publish the deposit, then add its DOI to `CITATION.cff` and README in a follow-up metadata commit. Keep the v0.1.0 tag and asset checksums unchanged. Follow [Zenodo's upload guide](https://help.zenodo.org/docs/deposit/create-new-upload/) if form labels differ.

## How to describe it

Use: "Opal is an experimental observed-single combinatorial response benchmark and residual prediction framework. Kernels are its strongest tested development family; independent transferable superiority and reliable OOD uncertainty remain unestablished."

Don't claim robust biological interaction learning, canonical CPA/GEARS superiority, calibrated abstention, frontier status, or arbitrary new-intervention/context support. Preserve negative results. Read the AI disclosure and retain responsibility for the manuscript's scientific claims.

## Maintainer recipe

New scientific work gets a new protocol and version. Run tests, build distributions, commit release inputs, package from that clean commit, validate installed inference, push the commit and a new tag, then draft/publish the next release. Don't silently replace results or move this tag.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m build --no-isolation
.venv/Scripts/python.exe tools/package_research_release.py --output releases/new-version
```

For a later version, update the explicit version recipe and distribution names first. The packager rejects dirty inputs and existing output directories, verifies model checksums, and uses tracked allowlists. It doesn't refit models or open evaluator outcomes. Model building also requires the separately licensed local exports; a source-only checkout doesn't contain those arrays.
