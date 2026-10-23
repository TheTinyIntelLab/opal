# Archive safeguard fixture amendment, v2

The registered source request ended with HTTP403 before any redirect or archive acquisition. All three pure helper identities matched. No retry, alternate download route or member/table opening is authorized by this amendment.

The first ten-case synthetic archive suite passed nine tests and failed while constructing the oversized-header fixture: Python3.14 TarFile.addfile refuses a nonzero-size regular member without fileobj. Inventory did not run on that fixture. Preserve the failed test bytes and receipt. This is a fixture defect, not evidence that the size guard accepted an oversized member.

Replace only that fixture's construction with a gzip stream containing a manually serialized USTAR header declaring the same16MiB+1size and end markers, with no member body. The intentionally incomplete synthetic object must be refused from its header before body consumption. Keep all original path/link/decompression guards and thresholds, and the assertion expecting the size-cap refusal. Bind corrected test/code hashes before the v2 test command. Do not rerun network acquisition or edit the frozen v1 registration, bindings or receipts. The archived failed bytes supply v1 custody; v2 is the current maintained regression suite.
