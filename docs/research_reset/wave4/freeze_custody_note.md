# Freeze custody correction

The initial freeze commit 1ccba8d normalized some text line endings in Git.
Local bound bytes did not change. Commit a16c35c added exact-byte attributes
and staged the original local bytes. A subsequent hash check found D_to_B.md
had received D's final coordination update after the freeze. The complete
updated message is preserved as D_to_B_final.md, and D_to_B.md was restored
to the exact pre-update bytes, verified against its already-frozen SHA256.
No proposal or experimental criterion changed. D_final_review.md separately
records the final endorsement of B's revised blocked decision.

The first Git-blob verification command failed because Windows path separators
were passed to git show; the corrected command used POSIX separators. The next
check exposed the coordination-file append. Attempts to reconstruct uniform
newlines did not match the bound hash; recovery of the original mixed newline
bytes did match. These were custody checks, not numerical experiment attempts.
All local freeze inputs now match. Final Git-blob verification follows staging
the restored coordination file. New diagnostic implementations still bind
their own exact code hashes before execution.
