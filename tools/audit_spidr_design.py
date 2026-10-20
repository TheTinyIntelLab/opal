"""Count designed guide-stratum coverage without reading response counts."""
import argparse
from collections import Counter
import io
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.qualify_spidr_metadata import project_line, SAFE_COLUMNS, MEMBER, ARCHIVE_HASH, sha256
from opal.artifacts import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--library', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError('Preserve metadata evidence')
if sha256(args.library) != ARCHIVE_HASH:
    raise ValueError('Library hash differs')
counts, guides = Counter(), {}
with zipfile.ZipFile(args.library) as archive:
    with io.TextIOWrapper(archive.open(MEMBER), encoding='utf-8-sig', newline='') as stream:
        stream.readline()
        stream.readline()
        for line in stream:
            row = project_line(line, SAFE_COLUMNS, 25)
            for start in (1, 13):
                guides[row[start]] = tuple(row[start+j] for j in range(6))
            a, b = row[2], row[14]
            if a == b or 'negative_control' in (a, b):
                continue
            flags = row[6], row[18]
            if any(flag not in ('TRUE', 'FALSE') for flag in flags):
                raise ValueError('Unknown targeting mismatch metadata')
            stratum = ('perfect_perfect', 'mixed', 'both_mismatched')[sum(flag == 'TRUE' for flag in flags)]
            counts[(tuple(sorted((a, b))), stratum)] += 1
result = {}
for stratum in ('perfect_perfect', 'mixed', 'both_mismatched'):
    pairs = {pair: n for (pair, group), n in counts.items() if group == stratum}
    eligible = {pair: n for pair, n in pairs.items() if n >= 2}
    degrees = Counter(gene for pair in eligible for gene in pair)
    result[stratum] = dict(designed_pairs=len(pairs), two_construct_pairs=len(eligible),
        genes=len(degrees), degree_histogram=dict(sorted(Counter(degrees.values()).items())),
        construct_count_histogram=dict(sorted(Counter(pairs.values()).items())))
write_json(args.output, dict(library_sha256=ARCHIVE_HASH, code_sha256=sha256(__file__), strata=result,
    evidence='design metadata only; no response/count access or numerical historical activity use',
    eligibility='at least two ordered constructs by design, before observed QC/single matching/splits',
    primary_changed=False, confirms_predictive_performance=False))
print(result)
