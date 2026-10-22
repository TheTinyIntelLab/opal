"""Bounded documentary acquisition only; never fetch response assets."""
import hashlib
import json
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent.parent / "opal-continuation-documentary-20261023"
REQUESTS = {
    "colo1-tree": "https://api.github.com/repos/ibarrioh/DualGuide_COLO1/git/trees/main?recursive=1",
    "harle-tree": "https://api.github.com/repos/team113sanger/harle_dgCRISPR_paralogs/git/trees/develop?recursive=1",
    "synulator-tree": "https://api.github.com/repos/hart-lab/SYNULATOR/git/trees/main?recursive=1",
    "colo1-zenodo-record": "https://zenodo.org/api/records/17191951",
}


def main():
    DEST.mkdir(exist_ok=True)
    receipt = ROOT / "results/research-continuation/documentary-fetch-v1.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    started = time.monotonic()
    records = []
    for name, url in REQUESTS.items():
        record = {"name": name, "url": url, "scope": "repository tree or deposit metadata only"}
        try:
            if shutil.disk_usage(DEST).free < 2**30 + 2**21:
                raise RuntimeError("unchanged one-GiB reserve plus request cap unavailable")
            request = urllib.request.Request(url, headers={"User-Agent": "Opal-metadata-qualification/1"})
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = response.read(2**21 + 1)
                if len(payload) > 2**21:
                    raise ValueError("metadata request exceeds two MiB")
                record["final_url"] = response.url
                record["status"] = response.status
            data = json.loads(payload)
            target = DEST / (name + ".json")
            with target.open("xb") as stream:
                stream.write(payload)
            record.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), path=str(target))
            if "tree" in data:
                record["commit"] = data.get("sha")
                record["truncated"] = data.get("truncated")
                record["entries"] = [{k: item[k] for k in ("path", "type", "sha", "size") if k in item} for item in data["tree"]]
            else:
                record["rights"] = data.get("metadata", {}).get("license")
                record["files"] = [{k: item.get(k) for k in ("key", "size", "checksum")} for item in data.get("files", [])]
        except Exception as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)
        records.append(record)
    result = {"schema": "opal-documentary-fetch-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "elapsed_seconds": time.monotonic() - started, "response_assets_opened": False, "requests": records}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
