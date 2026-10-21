"""Wave-three metadata qualification only. No response input or endpoint exists.

Reads the hash-bound Table 2 library using the historical allowlist projector.
Historical activity fields are scanned as bytes, discarded, and never converted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile
from urllib.request import HTTPRedirectHandler, Request, build_opener
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.qualify_spidr_metadata import ARCHIVE_HASH, MEMBER, SAFE_COLUMNS, SAFE_NAMES, project_line, sha256

PROTOCOL = "wave3-spidr-metadata-v1"
LIBRARY = Path("C:/Users/USER/Desktop/opal-research-reset-20261020/sources/41586_2025_8815_MOESM4_ESM.zip")
ARTICLE = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12018271/fullTextXML"
SUPPLEMENT = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-08815-4/MediaObjects/41586_2025_8815_MOESM1_ESM.docx"
ENA = "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA988447&result=read_run&fields=run_accession,sample_accession,experiment_accession,experiment_title&format=tsv"
URLS = (ARTICLE, SUPPLEMENT, ENA)
STRATA = ("perfect_perfect", "mixed", "both_mismatched")
BOUND = ("tools/wave3_spidr_qualification.py", "tests/test_wave3_spidr_qualification.py",
         "tools/qualify_spidr_metadata.py", "tools/check_source_readiness.py", "src/reagent_screen.py",
         "docs/research_reset/wave3_spidr_protocol.md", "plan.md", "README.md",
         "results/research-reset-source/qualification.json", "results/research-reset-reagent/design_coverage.json",
         "results/research-wave2-source/registration.json", "results/research-wave2-source/readiness.json")


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(obj, sort_keys=True, indent=2) + "\n")


def require_url(url):
    if url not in URLS:
        raise ValueError("Not allowlisted metadata; response/header access forbidden")


class Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, opener=None):
    require_url(url)
    opener = opener or build_opener(Redirects())
    with opener.open(Request(url, headers={"User-Agent": PROTOCOL}), timeout=30) as response:
        require_url(response.geturl())
        data = response.read(8 * 1024**2 + 1)
    if len(data) > 8 * 1024**2:
        raise ValueError("Metadata body exceeds cap")
    return data


def split(pair):
    a, b = sorted(pair)
    value = int(hashlib.sha256(f"opal-reset-v2|PRJNA988447|{a}|{b}".encode()).hexdigest()[:8], 16) % 10
    return "train" if value < 6 else "validation" if value < 8 else "test"


def exact_key(identity):
    name, gene, perfect, sequence, transcript, mismatch = identity
    if not name or not gene or not sequence or any(c not in "ACGT" for c in sequence):
        raise ValueError("Invalid guide identity/sequence")
    control = gene == "negative_control"  # Explicit reviewed Table 2 + methods role.
    if control:
        if transcript or mismatch:
            raise ValueError("Unexpected control annotations")
    elif not transcript or mismatch not in ("TRUE", "FALSE"):
        raise ValueError("Unknown targeted transcript/mismatch")
    return gene, sequence, transcript, mismatch, control


def projected_rows(path):
    # A caller cannot substitute an unopened response file, even with a new name.
    if sha256(path) != ARCHIVE_HASH:
        raise ValueError("Unregistered library bytes")
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != [MEMBER] or archive.getinfo(MEMBER).file_size != 190604826:
            raise ValueError("Unexpected library member")
        with io.TextIOWrapper(archive.open(MEMBER), encoding="utf-8-sig", newline="") as stream:
            group = next(csv.reader([stream.readline()]))
            header = next(csv.reader([stream.readline()]))
            if (len(group) != 25 or group[1:13] != ["gene 1"] * 12 or group[13:] != ["gene 2"] * 12
                    or len(header) != 25 or any(tuple(header[i:i+6]) != SAFE_NAMES for i in (1, 13))):
                raise ValueError("Library arm/header schema drift")
            for line in stream:
                row = project_line(line, SAFE_COLUMNS, 25)
                yield tuple(tuple(row[start+j] for j in range(6)) for start in (1, 13))
        # Reading to EOF validates the member CRC, not just the ZIP directory.


def analyze(rows, check=lambda: None):
    identities, keys, key_ids = {}, [], {}
    constructs = set()
    for number, arms in enumerate(rows, 1):
        ids = []
        for identity in arms:
            name = identity[0]
            if name in identities and identities[name] != identity:
                raise ValueError("Inconsistent guide identity")
            identities[name] = identity
            key = exact_key(identity)
            if key not in key_ids:
                key_ids[key] = len(keys)
                keys.append(key)
            ids.append(key_ids[key])
        encoded = ids[0] * 65536 + ids[1]
        if max(ids) >= 65536 or encoded in constructs:
            raise ValueError("Duplicate ordered exact reagents or oversized guide roster")
        constructs.add(encoded)
        if number % 4096 == 0:
            check()
    partners = defaultdict(set)
    nt_pairs, same_gene = 0, 0
    for value in constructs:
        left, right = divmod(value, 65536)
        a, b = keys[left], keys[right]
        if a[4] and b[4]:
            nt_pairs += 1
        elif a[4]:
            partners[(right, 1)].add(left)
        elif b[4]:
            partners[(left, 0)].add(right)
    all_pairs, matched_pairs, exclusions = Counter(), Counter(), Counter()
    for number, value in enumerate(constructs, 1):
        left, right = divmod(value, 65536)
        a, b = keys[left], keys[right]
        if a[4] or b[4]:
            continue
        if a[0] == b[0]:
            same_gene += 1
            continue
        pair = tuple(sorted((a[0], b[0])))
        stratum = STRATA[(a[3] == "TRUE") + (b[3] == "TRUE")]
        all_pairs[(pair, stratum)] += 1
        missing = (not partners[(left, 0)], not partners[(right, 1)])
        if any(missing):
            exclusions[(stratum, "both" if all(missing) else "arm1" if missing[0] else "arm2")] += 1
        else:
            matched_pairs[(pair, stratum)] += 1
        if number % 4096 == 0:
            check()
    strata, assignments = {}, []
    for stratum in STRATA:
        designed = {p: n for (p, s), n in all_pairs.items() if s == stratum}
        two = {p: n for p, n in designed.items() if n >= 2}
        matched = {p: matched_pairs[p, stratum] for p in two if matched_pairs[p, stratum] >= 2}
        degree = Counter(g for pair in matched for g in pair)
        support = {g for pair in matched if split(pair) == "train" for g in pair}
        strata[stratum] = dict(role="primary" if stratum == "perfect_perfect" else "secondary",
            designed_distinct_pairs=len(designed), designed_constructs=sum(designed.values()),
            excluded_fewer_than_two_designed_construct_pairs=len(designed)-len(two),
            two_designed_construct_pairs=len(two),
            exact_arm_nt_matched_two_construct_pairs=len(matched),
            excluded_after_exact_arm_nt_design_matching_pairs=len(two)-len(matched),
            missing_exact_arm_nt_constructs={reason: exclusions[stratum, reason] for reason in ("arm1", "arm2", "both")},
            matched_design_genes=len(degree), degree_histogram=dict(sorted(Counter(degree.values()).items())),
            construct_histogram=dict(sorted(Counter(designed.values()).items())),
            design_split_counts=dict(Counter(split(p) for p in matched)),
            design_lookup_unsupported_pairs=sum(not set(p) <= support for p in matched if split(p) != "train"),
            observed_t0_coverage=None, observed_exact_singles=None, observed_test_eligibility=None)
        if stratum == "perfect_perfect":
            assignments = [(a, b, split((a, b)), matched[a, b]) for a, b in sorted(matched)]
    targeted = [i for i, key in enumerate(keys) if not key[4]]
    summary = dict(rows=len(constructs), named_guides=len(identities), exact_reagents=len(keys),
        targeted_guides=len(targeted), nt_guides=sum(k[4] for k in keys), nt_nt_constructs=nt_pairs,
        same_gene_target_constructs=same_gene,
        target_nt_constructs=sum(len(v) for v in partners.values()),
        target_arm_nt_partner_histogram={f"arm{arm+1}": dict(sorted(Counter(len(partners[i, arm]) for i in targeted).items())) for arm in (0, 1)},
        strata=strata, observed_coverage="unknown; design membership supplies no counts, times or cultures")
    check()
    return summary, assignments


def metadata_facts(url, data):
    if url == SUPPLEMENT:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > 8 * 1024**2:
                raise ValueError("Supplement XML exceeds cap")
            root = ET.fromstring(archive.read(info))
        paragraphs = ["".join(p.itertext()) for p in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")]
        # Restrict review to function/code text, never images or narrative effect values.
        code = "\n".join(p for p in paragraphs if any(s in p for s in ("function", "normalize", "replicate.map", "pseudocount", "size.factor")))
        return dict(function_excerpt_sha256=hashlib.sha256(code.encode()).hexdigest(),
                    contains_normalize_true="normalize = TRUE" in code or "normalize=TRUE" in code,
                    contains_replicate_map="replicate.map" in code,
                    actual_invocation_verified=False, complete_sample_map_verified=False,
                    limitation="Definition/signature is not evidence of actual invocation; no endpoint inferred")
    if url == ENA:
        rows = list(csv.DictReader(data.decode().splitlines(), delimiter="\t"))
        expected = {"run_accession", "sample_accession", "experiment_accession", "experiment_title"}
        if not rows or set(rows[0]) != expected:
            raise ValueError("ENA schema drift")
        return dict(runs=rows, response_columns_mapped=False)
    root = ET.fromstring(data)
    ns = "{http://www.w3.org/1999/xlink}href"
    sections = [s for s in root.iter("sec") if s.find("title") is not None and "code availability" in "".join(s.find("title").itertext()).lower()]
    return dict(code_availability_links=sorted({n.attrib[ns] for s in sections for n in s.iter() if ns in n.attrib}),
        has_median_ratio_method=any("median ratio" in "".join(p.itertext()) for p in root.iter("p")),
        actual_invocation_verified=False, response_schema_verified=False)


def register(output):
    output.mkdir(parents=True, exist_ok=False)
    if sha256(LIBRARY) != ARCHIVE_HASH:
        raise ValueError("Library integrity failure")
    write_new(output / "registration.json", dict(protocol=PROTOCOL, registered_utc=datetime.now(timezone.utc).isoformat(),
        base_commit="505af4d9dd6a77f73b0e8952c9f4ed8cc12725a6", bound={p: sha256(ROOT/p) for p in BOUND},
        library=dict(path=str(LIBRARY), bytes=LIBRARY.stat().st_size, sha256=ARCHIVE_HASH),
        urls=URLS, response_access=False, response_header_access=False,
        primary="perfect_perfect metadata role only; no normalized endpoint registration",
        splits="opal-reset-v2|PRJNA988447|geneA|geneB SHA256 first 8 hex modulo 10; 0-5/6-7/8-9",
        eligibility="Design-only >=2 unique ordered exact constructs, each arm with >=1 exact target/NT partner; observed eligibility unknown",
        metrics="Exact metadata coverage/exclusions/histograms; no biological metrics or uncertainty",
        uncertainty="Future paired identical-information MSE; 2000 conditional endpoint-gene draws seed 20261022, not biological replication",
        selection="No outcome-dependent selection, no fitting; no secondary stratum rescues primary",
        gates="Historical 5% paired-MSE/upper conditional bound/replicate/shuffle/all-reference gates unchanged; no source access authorized",
        limits=dict(wall_seconds=1800, cpu_threads=1, rss_bytes=1024**3, outputs_bytes=100*1024**2, network_bytes=24*1024**2)))


def run(output):
    import psutil
    registration = json.loads((output / "registration.json").read_text())
    if registration["protocol"] != PROTOCOL:
        raise ValueError("Protocol drift")
    for path, digest in registration["bound"].items():
        if sha256(ROOT/path) != digest:
            raise ValueError(f"Registered input/code drift: {path}")
    started, cpu = time.monotonic(), time.process_time()
    process = psutil.Process()
    peak = 0
    def check():
        nonlocal peak
        memory = process.memory_info()
        peak = max(peak, memory.rss, getattr(memory, "peak_wset", 0))
        if time.monotonic()-started > 1800 or time.process_time()-cpu > 1800 or peak > 1024**3:
            raise RuntimeError("Resource cap exceeded")
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 100*1024**2:
            raise RuntimeError("Output cap exceeded")
    try:
        summary, assignments = analyze(projected_rows(LIBRARY), check)
        with (output / "primary_design_splits.tsv").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("geneA\tgeneB\tdesign_split\texact_nt_matched_designed_constructs\n")
            for a, b, group, n in assignments:
                stream.write(f"{a}\t{b}\t{group}\t{n}\n")
        requests = []
        for url in URLS:
            check()
            try:
                data = fetch(url)
                requests.append(dict(url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                                     facts=metadata_facts(url, data), status="fetched"))
            except (OSError, ValueError, ET.ParseError, zipfile.BadZipFile) as exc:
                requests.append(dict(url=url, status="failed", error=str(exc)))
        check()
        write_new(output / "qualification.json", dict(protocol=PROTOCOL, status="source_blocked", summary=summary,
            requests=requests, library_sha256=ARCHIVE_HASH, library_crc_validated=True,
            registration_sha256=sha256(output/"registration.json"), splits_sha256=sha256(output/"primary_design_splits.tsv"),
            response_access=False, response_header_access=False, normalized_endpoint_implemented=False,
            resources=dict(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                peak_rss_bytes=peak, peak_method="Windows lifetime peak_wset plus sampled RSS; cooperative 4096-row/stage checks",
                network_bytes=sum(r.get("bytes", 0) for r in requests), threads_for_numerical_libraries=1)))
        check()
    except Exception as exc:
        write_new(output/"failure.json", dict(error=repr(exc), wall_seconds=time.monotonic()-started,
                  cpu_seconds=time.process_time()-cpu, peak_rss_bytes=peak))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (register if args.action == "register" else run)(args.output)
