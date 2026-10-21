"""Bounded metadata-only source audit. This tool cannot open response assets.

Supported metadata is evidence for a later protocol, never permission to acquire
counts. Every network request and redirect must match the fixed metadata list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from urllib.request import HTTPRedirectHandler, Request, build_opener

SPIDR = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12018271/fullTextXML"
SCHEMATIC = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=12266356"
PIN = "218cf1bb8c83204da846256951ba5f6b0092890b"
REPO = f"https://api.github.com/repos/samsonfong/SCHEMATIC/git/trees/{PIN}?recursive=1"
README = f"https://raw.githubusercontent.com/samsonfong/SCHEMATIC/{PIN}/README.md"
FITNESS = f"https://raw.githubusercontent.com/samsonfong/SCHEMATIC/{PIN}/src/decipher/scores/fitness.py"
ENA = "https://www.ebi.ac.uk/ena/portal/api/filereport?accession={}&result=read_run&fields=run_accession,sample_accession,experiment_accession,experiment_title&format=tsv"
URLS = (SPIDR, SCHEMATIC, REPO, README, FITNESS,
        ENA.format("PRJNA988447"), ENA.format("PRJNA701256"))
MAX_BYTES = 8 * 1024 * 1024
MAX_TOTAL = 25 * 1024 * 1024
MAX_SECONDS = 1800
CHECKS = {"units", "rights", "controls", "sample_mapping", "ordered_linkage",
          "construct_aggregation", "adapter", "evaluator", "development_gates"}


def require_metadata_url(url):
    if url not in URLS:
        raise ValueError("Unregistered metadata URL; response and header access forbidden")
    return url


class MetadataRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_metadata_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_metadata(url, opener=None):
    require_metadata_url(url)  # Must precede any opener or request.
    opener = opener or build_opener(MetadataRedirects())
    with opener.open(Request(url, headers={"User-Agent": "Opal-source-metadata-audit"}),
                     timeout=45) as response:
        require_metadata_url(response.geturl())
        content = response.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("Metadata byte budget exceeded")
    return content


def assess(checks):
    """Validate a reviewed checklist, without certifying its scientific truth."""
    if set(checks) != CHECKS:
        raise ValueError("Incomplete or unexpected source checks")
    for item in checks.values():
        if set(item) != {"status", "evidence", "resolution"}:
            raise ValueError("Invalid evidence schema")
        if item["status"] not in {"supported", "blocked", "not_run"}:
            raise ValueError("Invalid status")
        if not all(isinstance(item[k], str) and item[k].strip()
                   for k in ("evidence", "resolution")):
            raise ValueError("Every check requires evidence and resolution")
    blockers = sorted(k for k, v in checks.items() if v["status"] != "supported")
    return {"checklist_complete": True, "blockers": blockers,
            "response_access_authorized": False,
            "qualification": "blocked" if blockers else "review_required",
            "limitation": "Checklist validation does not certify source or authorize access"}


def item(status, evidence, resolution):
    return dict(status=status, evidence=evidence, resolution=resolution)


def reviewed_checks():
    """Dated scientific review, not inferred automatically from fetched text."""
    spidr = {
        "units": item("blocked", "SPIDR methods identify median-ratio normalized Table 8; raw adapter rejects it", "Register a normalized endpoint and actual invocation/pseudocount semantics"),
        "rights": item("supported", "Article/supplement CC BY 4.0 absent item exception; no exception visible in article Table 8 credit", "Record attribution and check any item notice before supplement-only ingestion; no FASTQ route"),
        "controls": item("blocked", "Library has 15 NT guides and 225 NT/NT designs; observed coverage unknown", "Verify role and exact-guide/arm matching with schema, then preregister QC"),
        "sample_mapping": item("blocked", "ENA four runs map two cultures and T0/T14; unopened count columns not mapped", "Obtain a schema-only sample-column map and modified-function invocation"),
        "ordered_linkage": item("blocked", "Library has ordered sgRNAs; actual count key linkage unopened", "Obtain response-free key/schema companion and test exact library joins"),
        "construct_aggregation": item("blocked", "Perfect/perfect primary has 121198 designed two-construct pairs, not observed eligibility", "Register equal construct/culture aggregation, missingness and pair grouping"),
        "adapter": item("not_run", "Existing tested adapter accepts raw integer counts only", "Implement distinct normalized input contract and synthetic full-design resource rehearsal"),
        "evaluator": item("not_run", "No qualified real-source equal-information evaluator", "Test identical populations, guide information, train-only transforms, leakage, serialization and replay"),
        "development_gates": item("not_run", "Q1b/Q2b/Q3b precede Q6 in plan.md", "Complete gates and freeze full learning procedure, budget, splits and uncertainty before access"),
    }
    schematic = {k: item("blocked", "SCHEMATIC source review; no response asset opened", "Qualify a distinct KO-growth task") for k in CHECKS}
    schematic.update({
        "rights": item("blocked", "Pinned source tree has no LICENSE; manuscript is not article-wide CC BY", "Establish supplement-specific rights; do not vendor unlicensed pipeline"),
        "units": item("blocked", "Fitness pipeline uses plasmid-relative LFC and global pair transformations", "Freeze train-safe endpoint independent of globally processed GI scores"),
        "controls": item("blocked", "Published singles use all-partner medians and would leak pair outcomes", "Use explicit target/NT constructs with exact ordered guide singles"),
        "sample_mapping": item("blocked", "56 ENA runs have symbolic time/culture names, not elapsed-day/initiation proof", "Obtain actual columns, elapsed times and initiation lineage"),
        "ordered_linkage": item("blocked", "Pipeline expects a five-field pair index; actual response schema not inspected", "Obtain metadata-only guide axes/key schema, overlap and sample map"),
    })
    return {"spidr": spidr, "schematic": schematic}


def metadata_facts(url, content):
    """Extract structural metadata only, never article biological effect sizes."""
    text = content.decode("utf-8")
    if url in (SPIDR, SCHEMATIC):
        root = ET.fromstring(content)
        licenses = [" ".join("".join(node.itertext()).split()) for node in root.iter("license")]
        paragraphs = ["".join(node.itertext()) for node in root.iter("p")]
        return dict(license_mentions_ccby4=any("Attribution 4.0" in s for s in licenses),
                    license_mentions_text_mining=any("text mining" in s for s in licenses),
                    normalized_table8_method=any("median ratio" in s and "Normalized" in s for s in paragraphs))
    if url == REPO:
        obj = json.loads(text)
        if obj.get("truncated"):
            raise ValueError("Repository tree truncated")
        paths = [v["path"] for v in obj["tree"]]
        return dict(repository_pin=PIN, license_paths=[p for p in paths if "license" in p.lower()],
                    paths=len(paths))
    if url in (ENA.format("PRJNA988447"), ENA.format("PRJNA701256")):
        import csv
        rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
        if not rows or set(rows[0]) != {"run_accession", "sample_accession", "experiment_accession", "experiment_title"}:
            raise ValueError("Archive metadata schema changed")
        return dict(runs=len(rows), run_accessions=sorted(r["run_accession"] for r in rows))
    if url == FITNESS:
        return dict(five_field_count_index="index_col=[0, 1, 2, 3, 4]" in text,
                    plasmid_reference='counts["plasmid"]' in text,
                    partner_median_code="smf.append(f.median(axis=axis))" in text)
    return dict(readme_only=True)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import psutil
    process = psutil.Process()
    peak = 0
    fetched = []
    total = 0
    for url in URLS:
        if time.monotonic() - started > MAX_SECONDS:
            raise RuntimeError("Metadata wall budget exceeded")
        try:
            content = fetch_metadata(url)
            total += len(content)
            if total > MAX_TOTAL:
                raise RuntimeError("Total metadata byte budget exceeded")
            fetched.append(dict(url=url, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), status="fetched",
                                metadata_facts=metadata_facts(url, content)))
        except (OSError, ValueError) as exc:
            fetched.append(dict(url=url, status="unavailable", error=str(exc)))
        peak = max(peak, process.memory_info().rss)
        if time.monotonic() - started > MAX_SECONDS or sum(process.cpu_times()[:2]) > MAX_SECONDS:
            raise RuntimeError("Metadata time budget exceeded")
        if peak > 1024 ** 3:
            raise RuntimeError("Metadata memory budget exceeded")
    checks = reviewed_checks()
    receipt = dict(protocol="metadata-only-source-v1", response_access=False,
                   response_header_access=False, fits=0, sources={k: dict(checks=v, assessment=assess(v)) for k, v in checks.items()},
                   requests=fetched, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   resources=dict(wall_seconds=time.monotonic()-started, cpu_seconds=process.cpu_times().user+process.cpu_times().system,
                                  sampled_peak_rss_bytes=peak, fetched_bytes=total, max_wall_seconds=MAX_SECONDS,
                                  max_rss_bytes=1024**3, max_transfer_bytes=MAX_TOTAL, max_outputs_bytes=50*1024**2))
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if len(encoded.encode()) > 50*1024**2:
        raise RuntimeError("Output budget exceeded")
    (output / "readiness.json").write_text(encoded, encoding="utf-8")
    print(json.dumps({k: v["assessment"] for k, v in receipt["sources"].items()}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
