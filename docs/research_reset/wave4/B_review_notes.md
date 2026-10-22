# B inspection findings before freeze

The frozen `tools/check_source_readiness.py` accepts nonblank evidence prose and supported status without claim-specific scope, although it correctly never authorizes responses. Its SPIDR rights row is supported while its own resolution still requires the item-notice check. The new contract must preserve that unresolved prerequisite explicitly. Do not alter the frozen checklist to make the historical result look stricter than it was.

`src/reagent_screen.py` enforces raw integer counts, raw log2-plus-ten NT-pair centering and the old hardcoded both-mismatched primary. This is appropriate for its frozen contract but incompatible with SPIDR's normalized perfect/perfect prospective task. No adapter modification is proposed in this phase. Its exact reagent key correctly includes sequence, transcript, arm, mismatch and control flag, which the metadata-only synthetic fixtures should preserve.

The existing v2 SPIDR tests already reject wrong-arm/sequence/transcript joins, duplicate exact constructs and unregistered network targets. They do not establish actual response-column mapping or real-source endpoint compatibility. Reusing them in a focused regression protects the boundary while new tests address sample/culture/initiation and evidence-scope contradictions.

Inspection command `Get-Content src/source_readiness.py` failed because that file does not exist; the actual tool is `tools/check_source_readiness.py`, which was then read. This was a read-only path error, not a diagnostic or fit. Several combined reads returned exit 1 because pending shared review/freeze files were absent; absence is not authorization.
