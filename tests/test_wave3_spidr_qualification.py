"""Synthetic metadata only; fixtures contain no source response values."""
import csv
import io
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools import wave3_spidr_qualification as q


def guide(name, gene, seq, mismatch="FALSE", transcript="tx"):
    return name, gene, seq, seq, transcript, mismatch


A = guide("a", "A", "AAAA")
B = guide("b", "B", "CCCC")
N = guide("nt", "negative_control", "GGGG", "", "")


def test_construct_counts_exact_arm_matching_and_split_grouping():
    summary, pairs = q.analyze([(A, B), (B, A), (A, N), (B, N), (N, A), (N, B), (N, N)])
    primary = summary["strata"]["perfect_perfect"]
    assert primary["designed_distinct_pairs"] == 1
    assert primary["two_designed_construct_pairs"] == 1
    assert primary["exact_arm_nt_matched_two_construct_pairs"] == 1
    assert primary["observed_exact_singles"] is None
    assert primary["observed_test_eligibility"] is None
    assert pairs == [("A", "B", q.split(("A", "B")), 2)]
    assert q.split(("B", "A")) == q.split(("A", "B"))
    assert summary["nt_nt_constructs"] == 1


def test_missing_arm_is_not_rescued_by_other_arm_or_other_sequence():
    other = guide("other", "A", "TTTT")
    summary, pairs = q.analyze([(A, B), (B, A), (N, A), (B, N), (N, B), (other, N)])
    primary = summary["strata"]["perfect_perfect"]
    assert primary["missing_exact_arm_nt_constructs"]["arm1"] == 1
    assert primary["exact_arm_nt_matched_two_construct_pairs"] == 0
    assert primary["excluded_after_exact_arm_nt_design_matching_pairs"] == 1
    assert not pairs


def test_transcript_is_part_of_exact_single_key():
    other = guide("other", "A", "AAAA", transcript="other_tx")
    summary, _ = q.analyze([(A, B), (B, A), (N, A), (B, N), (N, B), (other, N)])
    assert summary["strata"]["perfect_perfect"]["exact_arm_nt_matched_two_construct_pairs"] == 0


def test_strata_are_secondary_without_rescuing_primary():
    am = guide("am", "A", "TTTT", "TRUE")
    bm = guide("bm", "B", "ATAT", "TRUE")
    rows = [(A, B), (am, bm), (bm, am), (A, bm), (bm, A)]
    rows += [(g, N) for g in (A, B, am, bm)] + [(N, g) for g in (A, B, am, bm)]
    summary, pairs = q.analyze(rows)
    assert not pairs
    assert summary["strata"]["perfect_perfect"]["excluded_fewer_than_two_designed_construct_pairs"] == 1
    assert summary["strata"]["both_mismatched"]["role"] == "secondary"
    assert summary["strata"]["mixed"]["exact_arm_nt_matched_two_construct_pairs"] == 1


@pytest.mark.parametrize("rows", [[(A, B), (A, B)], [(A, B), (guide("renamed", "A", "AAAA"), B)]])
def test_duplicate_exact_constructs_fail_even_if_renamed(rows):
    with pytest.raises(ValueError, match="Duplicate"):
        q.analyze(rows)


def test_inconsistent_identity_unknown_mismatch_and_control_annotations_fail():
    with pytest.raises(ValueError, match="Inconsistent"):
        q.analyze([(A, B), (guide("a", "A", "TTTT"), B)])
    with pytest.raises(ValueError, match="Unknown"):
        q.analyze([(guide("a", "A", "AAAA", ""), B)])
    with pytest.raises(ValueError, match="control"):
        q.analyze([(A, guide("n", "negative_control", "GGGG"))])


def test_activity_sentinels_discarded_without_float_conversion():
    row = ["POISON_ACTIVITY_NOT_A_NUMBER"] * 25
    for start, g in ((1, A), (13, B)):
        row[start:start+6] = g
    stream = io.StringIO()
    csv.writer(stream).writerow(row)
    projected = q.project_line(stream.getvalue(), q.SAFE_COLUMNS, 25)
    assert "POISON" not in repr(projected)
    assert tuple(projected[i] for i in range(1, 7)) == A


@pytest.mark.parametrize("url", ["https://example.com/counts", q.SUPPLEMENT.replace("MOESM1", "MOESM9"), q.ARTICLE + "?header=1"])
def test_response_and_header_urls_rejected_before_network(url):
    opener = Mock()
    with pytest.raises(ValueError, match="forbidden"):
        q.fetch(url, opener)
    opener.open.assert_not_called()


def test_redirect_refused_before_following():
    with pytest.raises(ValueError, match="forbidden"):
        q.Redirects().redirect_request(None, None, 302, "", {}, "https://example.com/counts")


def test_write_refuses_overwrite_and_unregistered_library(tmp_path):
    path = tmp_path/"result.json"
    q.write_new(path, {"x": 1})
    with pytest.raises(FileExistsError):
        q.write_new(path, {"x": 2})
    with pytest.raises(ValueError, match="Unregistered"):
        list(q.projected_rows(path))


def test_summary_deterministic_when_row_order_changes():
    rows = [(A, B), (B, A), (A, N), (B, N), (N, A), (N, B), (N, N)]
    assert q.analyze(rows) == q.analyze(reversed(rows))


def test_resource_failure_propagates():
    def stop():
        raise RuntimeError("cap")
    with pytest.raises(RuntimeError, match="cap"):
        q.analyze([(A, B)], stop)
