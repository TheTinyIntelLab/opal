"""Record observed native dependency/resource failures without inventing scores."""
from pathlib import Path
import argparse
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import read_json, write_json, environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", action="store_true")
    args = parser.parse_args()
    records = {}
    for name, module in [("GEARS", "gears"), ("CPA", "cpa")]:
        script = f"import sys; sys.path.insert(0, {str(ROOT / 'tools/external' / name)!r}); import {module}"
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        records[name] = {"native_import_exit_code": result.returncode, "diagnostic": result.stderr,
                         "scores": None, "status": "native_not_run"}
    records["CPA"]["declared_python"] = ">=3.9,<3.11 (pinned upstream pyproject)"
    records["GEARS"]["additional_work"] = "isolated legacy environment and fold-safe raw-cell/graph adapter; dependencies alone do not establish protocol compatibility"
    records["PertAdapt"] = {"status": "native_not_run", "scores": None,
        "required_files": ["model.ckpt", "go.csv", "go_mask_19264.npz", "scfoundation_genes.txt"],
        "cuda_available": __import__("torch").cuda.is_available(),
        "source": "pinned README and scFoundation adapter scripts"}
    if args.network:
        url = "https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/IgDPdzo_YIYxQrXsvlJj5XGJAY3JZS3lyGlx_sQguzA-c2g?e=esfrXG"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "OpalResearch-resource-audit"})
            with urllib.request.urlopen(request, timeout=30) as response:
                prefix = response.read(4096)
                records["PertAdapt"]["resource_probe"] = {"status": response.status, "content_type": response.headers.get("Content-Type"),
                    "direct_model_download": False, "html": b"<html" in prefix.lower() or b"<!doctype html" in prefix.lower()}
        except Exception as error:
            records["PertAdapt"]["resource_probe"] = {"error": f"{type(error).__name__}: {error}", "direct_model_download": False}
    registry = read_json(ROOT / "data/public_method_registry.json")
    for name in records:
        records[name]["revision"] = registry[name]["revision"]
    write_json(ROOT / "data/public_methods/native_feasibility.json", {"environment": environment(), "methods": records})
    print({name: record["status"] for name, record in records.items()})


if __name__ == "__main__":
    main()
