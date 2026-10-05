"""Retrieve exact pinned comparator source blobs into an ignored local cache."""
import argparse
from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import read_json, write_json, sha256
from download import download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=["SLIM", "GEARS", "CPA", "PertAdapt", "NAIAD"])
    args = p.parse_args()
    registry = read_json(ROOT / "data/public_method_registry.json")
    entry = registry[args.method]
    tree = read_json(ROOT / f"data/public_methods/{args.method.lower()}_tree.json")["tree"]
    base = ROOT / "tools/external" / args.method
    files = []
    for blob in tree:
        name = blob["path"]
        include = (name.endswith(".py") or name in ["LICENSE", "LICENSE.md", "README.md", "pyproject.toml", "requirements.txt"])
        if args.method == "SLIM" and name == "data/gene_string_embeddings.v0.3.h5":
            include = True
        if not include or blob["type"] != "blob" or blob.get("size", 0) > 30_000_000:
            continue
        path = (base / name).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError("Unsafe upstream path")
        if not path.exists():
            result = download(f"https://raw.githubusercontent.com/{entry['repository']}/{entry['revision']}/{name}", path)
            if result["status"] != "downloaded_not_accepted":
                raise RuntimeError(f"Could not retrieve pinned source: {name}")
        data = path.read_bytes()
        git_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if git_sha != blob["sha"]:
            raise ValueError("Upstream Git blob checksum mismatch")
        files.append({"path": name, "sha256": sha256(path), "git_sha": git_sha})
    write_json(ROOT / f"data/public_methods/{args.method.lower()}_local_manifest.json", {"revision": entry["revision"], "files": files})


if __name__ == "__main__":
    main()
