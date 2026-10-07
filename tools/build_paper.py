"""Compile the multi-file manuscript and record all evidence inputs.

Use an existing Tectonic executable or --tectonic PATH. No TeX installation is
performed. Dependencies are cached inside this research workspace.
"""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import write_json, sha256


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tectonic")
    p.add_argument("--cache-directory", type=Path)
    args = p.parse_args(argv)
    binary = args.tectonic or shutil.which("tectonic")
    if binary is None and (ROOT / "tools/vendor/tectonic.exe").exists():
        binary = str(ROOT / "tools/vendor/tectonic.exe")
    if binary is None:
        raise FileNotFoundError("Supply an existing Tectonic binary with --tectonic PATH")
    env = dict(os.environ)
    env["TECTONIC_CACHE_DIR"] = str(args.cache_directory or ROOT / "tools/vendor/tectonic-cache")
    result = subprocess.run([binary, "--untrusted", "--keep-logs", "--keep-intermediates", "main.tex"], cwd=ROOT / "paper", env=env,
                            capture_output=True, text=True)
    log = result.stdout + result.stderr
    (ROOT / "paper/build.log").write_text(log, encoding="utf-8")
    manifest = {"compiler": binary, "cache_directory": env['TECTONIC_CACHE_DIR'], "return_code": result.returncode,
                "inputs": {str(f.relative_to(ROOT / "paper")): sha256(f) for pattern in ["*.tex", "*.bib", "*.sty", "evidence/*.tex", "evidence/*.pdf", "evidence/next_cycle_manifest.json"] for f in (ROOT / "paper").glob(pattern)},
                "pdf_sha256": sha256(ROOT / "paper/main.pdf") if result.returncode == 0 and (ROOT / "paper/main.pdf").exists() else None}
    write_json(ROOT / "paper/build_manifest.json", manifest)
    print(log)
    if result.returncode:
        raise SystemExit(result.returncode)
    print(ROOT / "paper/main.pdf")


if __name__ == "__main__":
    main()
