"""Download registered public sources with streaming checksums and an audit record."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import read_json, write_json, sha256


def download(url, path, md5=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit = {"url": url, "destination": str(path), "status": "running"}
    started = time.time()
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "OpalResearch/0.1 source-audit"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as f:
            audit["final_url"] = response.url
            audit["headers"] = {k: response.headers[k] for k in ["Content-Type", "Content-Length", "Last-Modified", "ETag"] if k in response.headers}
            total = 0
            last = time.time()
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                f.write(chunk)
                total += len(chunk)
                if time.time() - last >= 20:
                    print(f"Downloaded {total / 1024 ** 2:.1f} MiB", flush=True)
                    last = time.time()
        if path.suffix == ".h5ad":
            with temporary.open("rb") as f:
                if f.read(8) != b"\x89HDF\r\n\x1a\n":
                    raise ValueError("Download is not HDF5; provider returned a warning/login page")
        if md5:
            h = hashlib.md5()
            with temporary.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != md5:
                raise ValueError("Publisher MD5 mismatch")
            audit["md5"] = h.hexdigest()
        temporary.replace(path)
        audit.update({"status": "downloaded_not_accepted", "sha256": sha256(path), "bytes": path.stat().st_size})
    except Exception as error:
        audit.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    audit["seconds"] = time.time() - started
    write_json(path.with_suffix(path.suffix + ".download.json"), audit)
    print(json.dumps(audit, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["norman", "wessels-metadata", "wessels-counts", "chemical", "chemical-original", "chemical-cells", "chemical-genes"])
    args = p.parse_args()
    registry = read_json(ROOT / "data/source_registry.json")["datasets"]
    if args.dataset == "norman":
        entry = registry["norman_scperturb_v1_1"]
        result = download(entry["download_url"], ROOT / "data/raw/NormanWeissman2019_filtered.h5ad", entry["publisher_checksum_md5"])
    elif args.dataset.startswith("wessels"):
        entry = registry["wessels_original"]["files"][args.dataset == "wessels-counts"]
        result = download(entry["download_url"], ROOT / "data/raw" / entry["filename"])
    elif args.dataset in ["chemical-cells", "chemical-genes"]:
        kind = "cell" if args.dataset == "chemical-cells" else "gene"
        filename = f"GSE206741_{kind}_metadata.tsv.gz"
        result = download(f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206741/suppl/{filename}",
                          ROOT / f"data/raw/source_metadata/combosciplex_{kind}_metadata.tsv.gz")
    elif args.dataset == "chemical-original":
        entry = registry["combosciplex_original"]
        result = download(entry["download_url"], ROOT / "data/raw/GSE206741_count_matrix.mtx.gz")
    else:
        entry = registry["combosciplex_cpa_prepared"]
        result = download(entry["download_url"], ROOT / "data/raw" / entry["filename"])
    sys.exit(0 if result["status"] == "downloaded_not_accepted" else 1)
