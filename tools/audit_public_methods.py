"""Pin upstream source identities and inspect required resources without guessing."""
from pathlib import Path
import sys
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import write_json

REPOS = {"SLIM": "RasmussenLab/SLIM", "GEARS": "snap-stanford/GEARS",
         "PertAdapt": "BaiDing1234/PertAdapt", "CPA": "theislab/cpa", "NAIAD": "NeptuneBio/NAIAD"}


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "OpalResearch-source-audit", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    output = {}
    for method, repo in REPOS.items():
        print(f"Auditing {method}", flush=True)
        try:
            info = get_json(f"https://api.github.com/repos/{repo}")
            branch = info["default_branch"]
            commit = get_json(f"https://api.github.com/repos/{repo}/commits/{branch}")
            revision = commit["sha"]
            tree = get_json(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1")
            files = tree["tree"]
            output[method] = {"repository": repo, "revision": revision, "license": info.get("license"),
                              "source_url": f"https://github.com/{repo}/tree/{revision}", "status": "source_pinned_not_run",
                              "resource_files": [{"path": f["path"], "size": f.get("size"), "git_sha": f["sha"]} for f in files if any(s in f["path"].lower() for s in ["embedding", "checkpoint", "requirement", "setup.py", "pyproject", "model_combo", "reproduce"])]}
            write_json(ROOT / f"data/public_methods/{method.lower()}_tree.json", tree)
        except Exception as e:
            output[method] = {"repository": repo, "status": "audit_failed", "reason": f"{type(e).__name__}: {e}"}
    write_json(ROOT / "data/public_method_registry.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
