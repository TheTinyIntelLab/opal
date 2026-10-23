"""Retain the frozen, demonstrated v1 adapter failure without editing its bytes."""
import pytest


def pytest_collection_modifyitems(items):
    historical = ('tests/test_continuation_colo1_schema_v1.py::'
                  'test_bounded_decompression_and_nested_refusal[compress0]')
    for item in items:
        if item.nodeid == historical:
            item.add_marker(pytest.mark.xfail(
                strict=True, raises=TypeError,
                reason='Frozen v1 gzip filename/fileobj defect; v2 has passing regression coverage',
            ))
