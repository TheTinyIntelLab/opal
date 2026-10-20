# Exact-guide scalar-screen adapter

Historical contract note, 21 October 2026: this raw-count adapter retains the original both-mismatched primary. The later metadata-only [design decision](spidr_qualification_next.md) and authoritative root plan supersede that future primary with perfect/perfect. Actual normalized SPIDR ingestion remains blocked pending a separately registered endpoint/role contract and equal-information tests.

Q1 now has an executable source-neutral contract and synthetic aggregation QA in
[reagent_screen.py](../../src/reagent_screen.py). It does **not** have a qualified
parser for SPIDR's real response table. No response/count table was acquired,
opened, normalized, fitted or scored in this work. The numerical examples below
are hand-constructed software fixtures, not biological results.

## Contract and averaging

`ScreenContract`, `Guide`, `Construct`, `CountObservation` and `ReagentScreen`
declare source/context, culture IDs and initiation lineage, source-schema review,
raw-count semantics, T0/T14 times, ordered vector arms, exact sequence,
transcript, mismatch flag and explicit control annotation. No gene-label heuristic
assigns controls. Unknown targeting-guide mismatch status fails rather than
becoming perfect-match. An explicitly identified non-targeting control can use
`mismatched=None`, meaning not applicable. A missing control transcript needs an
explicit reviewed `not_applicable` mapping; an empty target transcript fails.
Unknown/additional serialized identity fields fail, including activity columns.

Only nonnegative integer counts or explicit `None` are accepted. Empty strings,
fractional or normalized counts and incompatible units/normalization stop.
`None` excludes a construct, while an observed endpoint count of zero remains a
count. Each construct needs both declared cultures, nonmissing counts and T0
strictly above 50 in both cultures. The source adapter must establish that the
supplied integers really are raw counts; integer-valued normalized counts cannot
be identified from their values alone.

For each surviving construct and culture, `aggregate` computes
`log2((T14+10)/(T0+10))` and subtracts the median of surviving explicitly annotated
non-targeting/non-targeting constructs in that culture. Missing controls stop the
entire endpoint. Single estimates average surviving guide/control-partner
construct LFCs equally, matching the target guide by gene, exact sequence,
transcript, mismatch annotation, arm, declared time window and culture.
Different-arm, different-transcript or different-sequence measurements cannot
substitute for a missing single. Missing exact singles produce construct-level
exclusion records.

The reference API offers zero, sum, minimum or an already-frozen scalar callback.
It predicts each ordered double from its own two measured singles first. Then it
gives surviving constructs equal weight within canonical gene pair, metadata-only
mismatch stratum and culture, requires at least two such constructs in each
culture, and gives cultures equal weight. It never takes the minimum of averaged
guides. Reciprocal orientations remain distinct ordered constructs. Duplicate
constructs, including renamed identical ordered reagents, fail.

Primary results use both-targeting-guides-mismatched constructs. Perfect/perfect
and mixed strata are separate secondary outputs and cannot rescue the primary
population. Canonical pair hashes use the exact `opal-reset-v2` contract in
[plan.md](../../plan.md), so guide identity, orientation, stratum and culture
cannot move a pair into another split. The adapter reports individual construct
predictions, per-culture summaries, pair summaries, input guide-arm identities,
construct/single coverage and exclusion reasons. It does not calculate an
independent-biological-replicate confidence interval.

## Executed QA

From the repository root:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_reagent_screen.py tests/test_spidr_metadata.py -q
```

This run passed **35 tests in 0.07 seconds**, including 32 adapter tests and the
three existing metadata-projector tests. Tests preceded any source-response
access, and there were no fits. A fixture with two guide combinations has
guide-specific singles `(-2, 0)` and `(0, -2)`: averaging construct minima gives
`-2`, while taking the minimum after averaging guides would incorrectly give
`-1`. Another fixture shifts every culture-two raw LFC by one and confirms its
own control median removes that technical offset before equal-culture averaging.
Other tests cover count/mismatch failures, exact-arm/transcript/sequence
matching, two-T0 coverage, missing singles/controls, nonfinite predictions,
duplicate identities, split grouping and deterministic JSON/aggregation replay.

`ReagentScreen.to_json()` and `.from_json()` round-trip exactly without pickle or
implicit flag coercion. The class is deliberately independent of a CSV/Excel
parser. Synthetic fixtures do not prove that a real source satisfies the schema,
provides usable controls, or contains enough eligible primary pairs.

## Real-source readiness and stop conditions

The hash-verified local SPIDR library was scanned with the existing allowlist
projector from [qualify_spidr_metadata.py](../../tools/qualify_spidr_metadata.py).
Only identity, sequence, transcript and mismatch metadata were retained while
scanning. Historical guide activity values were discarded without numeric
conversion or use. The compact
[metadata receipt](../../results/research-reset-reagent/metadata_readiness.json)
records the code/library hashes and metadata groups. All 1,165 guide identities
have explicit uppercase `TRUE`/`FALSE` mismatch literals except 15 blank flags;
892 are `FALSE`, 258 are `TRUE`, and 15 transcripts are blank. These are library
design metadata counts, not observed counts or endpoint eligibility. The 15
blank flags and transcripts all belong to the literal `negative_control` label;
none belongs to the 1,150 targeted guides.

Reproduce the receipt's metadata groups without opening a response table:

```powershell
@'
import io, json, zipfile
from collections import Counter
from tools.qualify_spidr_metadata import project_line, SAFE_COLUMNS, MEMBER, sha256, ARCHIVE_HASH
path = 'C:/Users/USER/Desktop/opal-research-reset-20261020/sources/41586_2025_8815_MOESM4_ESM.zip'
assert sha256(path) == ARCHIVE_HASH
guides = {}
with zipfile.ZipFile(path) as archive:
    with io.TextIOWrapper(archive.open(MEMBER), encoding='utf-8-sig', newline='') as stream:
        stream.readline()
        stream.readline()
        for line in stream:
            row = project_line(line, SAFE_COLUMNS, 25)
            for start in (1, 13):
                guides[row[start]] = tuple(row[start+j] for j in range(6))
groups = Counter(('negative_control' if v[1] == 'negative_control' else 'targeted',
                  v[5], bool(v[4])) for v in guides.values())
print(json.dumps([dict(label_class=k[0], mismatch_literal=k[1], has_transcript=k[2], guides=v)
                  for k, v in sorted(groups.items())], sort_keys=True))
'@ | .venv/Scripts/python.exe -
```

The [existing metadata record](spidr_metadata.md) and [primary paper](https://doi.org/10.1038/s41586-025-08815-4)
establish RPE-1 TP53-knockout, deliberate mismatches, T14/T0 LFC, and culture
splitting after transduction. This work adds no new biological replication or
raw-data license determination. It preserves the distinction between CC BY 4.0
article/supplement terms, any item-specific exception and unverified raw-SRA
terms.

Real SPIDR ingestion remains blocked until metadata review supplies the actual
response/count column schema, construct-to-library linkage, raw versus normalized
count semantics, usable denominators, exact T0/T14 culture identities, and a
methods-backed mapping of control and single-partner constructs. The library's
`negative_control` label alone does not prove each required control role. Blank
control mismatch/transcript annotations require explicit not-applicable handling,
never conversion into a targeted perfect guide. Do not open response values to
guess these semantics. Seek a header-only schema or official processing code;
if adequate raw counts/control semantics are unavailable, stop or register a
different endpoint/source.

After metadata qualification, a real `ScreenContract` requires a reviewed schema
SHA256 and explicit control-evidence reference instead of synthetic defaults.
That field is a provenance declaration, not an automated certification of its
truth. A source-specific parser, frozen endpoint/split/evaluator registration,
rights check and Q2/Q3 procedure gates must still precede Q6 response access.
This synthetic QA does not satisfy those gates or establish real primary-panel
size, native predictor superiority, calibration or independent confirmation.
Full-source memory, wall time and output size are also unprofiled; this run makes
no claim that the in-memory adapter meets the later Q6 resource cap.
