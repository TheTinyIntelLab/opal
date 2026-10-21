# Wave three SPIDR metadata qualification v2

This prospectively supersedes v1 for the metadata run only. All scope, splits,
coverage stages, gates, resource caps and response/header prohibitions in
[v1](wave3_spidr_protocol.md) remain unchanged. No normalized endpoint exists.

V1 stopped before processing its first complete construct because the library
represents some DNA bases in lowercase. Its registration, executable, tests and
failure remain frozen. A metadata-only first-failing-row inspection found a
20-base sequence containing lowercase `g`; only allowlisted identity fields were
inspected. This diagnostic made no requests and accessed no response values.

V2 uppercases DNA for exact sequence identity, while preserving gene, transcript,
mismatch annotation, arm and control role in joins. Case is annotation, not a
distinct nucleotide. Repeated names must still retain identical source metadata;
renamed identical ordered exact reagents still fail as duplicates. New synthetic
tests cover DNA case, retained mismatch distinctions, bounded fetches and final
URL rejection before reading. Initial memory is sampled before streaming so an
early failure records an RSS measurement. No other eligibility rule changes.

Register v2 code/tests plus frozen v1 evidence before running. Use a fresh
`results/research-wave3-spidr/metadata-v2` directory. Preserve any further failure
and stop if the branch cannot be justified within the original 30-minute cap.
