"""Metadata qualification never authorizes response access or trusts declarations."""
import copy
import io

import pytest

from tools import check_source_readiness as source


@pytest.mark.parametrize("url", ["https://media.springernature.com/counts.txt", source.SPIDR + "?header=1", "file:///tmp/counts", "https://www.ncbi.nlm.nih.gov/sra/SRR25050635"])
def test_response_and_header_urls_fail_before_request(url):
    class NeverOpen:
        def open(self, *args, **kwargs):
            pytest.fail("Network was attempted")
    with pytest.raises(ValueError, match="forbidden"):
        source.fetch_metadata(url, NeverOpen())


def test_redirect_to_unregistered_asset_fails():
    with pytest.raises(ValueError, match="forbidden"):
        source.MetadataRedirects().redirect_request(None, None, 302, "redirect", {}, "https://media.springernature.com/counts.txt")


def test_bounded_metadata_fetch_checks_final_url_and_size(monkeypatch):
    class Response(io.BytesIO):
        def geturl(self):
            return source.SPIDR
    class Opener:
        def open(self, *args, **kwargs):
            return Response(b"12345")
    monkeypatch.setattr(source, "MAX_BYTES", 4)
    with pytest.raises(ValueError, match="byte budget"):
        source.fetch_metadata(source.SPIDR, Opener())


def test_checklist_requires_all_checks_and_evidence():
    checks = source.reviewed_checks()["spidr"]
    with pytest.raises(ValueError, match="Incomplete"):
        source.assess({})
    checks["units"]["evidence"] = ""
    with pytest.raises(ValueError, match="evidence"):
        source.assess(checks)


def test_complete_declarations_do_not_authorize_counts():
    checks = copy.deepcopy(source.reviewed_checks()["spidr"])
    for value in checks.values():
        value["status"] = "supported"
    result = source.assess(checks)
    assert result["qualification"] == "review_required"
    assert result["response_access_authorized"] is False


def test_review_is_deterministic_and_both_sources_remain_blocked():
    first = {k: source.assess(v) for k, v in source.reviewed_checks().items()}
    assert first == {k: source.assess(v) for k, v in source.reviewed_checks().items()}
    assert all(v["qualification"] == "blocked" for v in first.values())
    assert "rights" not in first["spidr"]["blockers"]
    assert "rights" in first["schematic"]["blockers"]


def test_structural_extraction_discards_narrative_outcomes():
    content = b'<article><license>Attribution 4.0</license><p>OUTCOME_123</p></article>'
    result = source.metadata_facts(source.SPIDR, content)
    assert result["license_mentions_ccby4"] is True
    assert "OUTCOME" not in repr(result)


def test_truncated_tree_or_wrong_archive_schema_fails():
    with pytest.raises(ValueError, match="truncated"):
        source.metadata_facts(source.REPO, b'{"truncated":true}')
    with pytest.raises(ValueError, match="schema"):
        source.metadata_facts(source.ENA.format("PRJNA988447"), b'counts\n123\n')
