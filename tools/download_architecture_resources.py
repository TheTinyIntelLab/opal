"""Fetch pinned public priors and independent single-screen resources."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / "data/architecture_resources"


def fetch(url, destination, size=None, md5=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists():
        if size is not None and destination.stat().st_size != size:
            raise ValueError("Existing resource has unexpected size")
        if md5 is not None:
            digest = hashlib.md5()
            with destination.open("rb") as f:
                for chunk in iter(lambda: f.read(1024*1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != md5:
                raise ValueError("Existing resource checksum mismatch")
        return destination
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "OpalResearch/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request_url = url + ("&" if "?" in url else "?") + f"opal_range={start}" if start else url
    request = urllib.request.Request(request_url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        resumed = start and response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
            raise ValueError("Unexpected server range response")
        if not resumed:
            start = 0
        total, last = start, start
        with partial.open("ab" if resumed else "wb") as f:
            for chunk in iter(lambda: response.read(1024*1024), b""):
                f.write(chunk)
                total += len(chunk)
                if total-last >= 128*1024*1024:
                    print(destination.name, round(total/1e6), "MB", flush=True)
                    last = total
    if size is not None and partial.stat().st_size != size:
        raise ValueError("Publisher size mismatch")
    if md5 is not None:
        digest = hashlib.md5()
        with partial.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != md5:
            raise ValueError("Publisher checksum mismatch")
    partial.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--presage", action="store_true")
    parser.add_argument("--replogle", action="store_true")
    args = parser.parse_args()
    fetch("https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020", RESOURCE / "hallmark.txt")
    if args.presage:
        manifest = fetch("https://zenodo.org/api/records/15587986", RESOURCE / "presage.json")
        data = json.loads(manifest.read_text())
        f = next(f for f in data["files"] if f["key"] == "cache.tar.gz")
        fetch(f["links"]["self"], RESOURCE / "presage_cache.tar.gz", f["size"], f["checksum"].split(":")[-1])
        print("Verified PRESAGE archive", flush=True)
    if args.replogle:
        manifest=fetch("https://zenodo.org/api/records/7278143",RESOURCE/"replogle_7278143.json")
        data=json.loads(manifest.read_text())
        f=next(f for f in data["files"] if f["key"]=="ReplogleWeissman2022_K562_essential.h5ad")
        fetch(f["links"]["self"],RESOURCE/f["key"],f["size"],f["checksum"].split(":")[-1])
        print("Verified independent Replogle expression screen",flush=True)


if __name__ == "__main__":
    main()
