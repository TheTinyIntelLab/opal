# v0.1.0 release verification

Verification was performed on Windows with Python 3.14.0 on 2026-10-06. The software suite passed all 117 tests. These are engineering checks, not independent scientific replication.

The wheel installed into a fresh target directory outside the checkout. Package import was asserted to resolve to that installation, and the existing host dependencies were reused. This doesn't establish a fully clean installation of the complete dependency graph or compatibility on every supported Python version and operating system.

The installed Norman kernel reproduced all 38 previously saved unscored query predictions with maximum absolute error about 2.03e-15. The EmeraldBay kernel reproduced all 27 saved selected predictions with error about 1.33e-15. Pair reversal was checked for both. These comparisons restore frozen models; they neither refit nor select models using evaluator outcomes.

The installed CLI completed the two-scenario synthetic quickstart. The built source distribution and wheel were inspected for private-plan, website, external-code, and raw-data exclusions. ZIP integrity and SHA-256 checks passed. The release archives use explicit tracked-file allowlists and separately approved model exports. The 77 source snapshots already tracked in the repository were also checked for private-plan and website members; none were found.

The 32-page manuscript compiled successfully. Its author page and declaration page were rendered and visually checked. Recorded scientific outcomes weren't changed. The final asset manifest and checksum list identify exact downloadable files and the committed release revision. `tools/validate_research_release.py` writes its detailed local report beside the assets; it isn't an additional scientific experiment.

To replay these packaging checks, obtain the source and the eight release assets, restore the local saved prediction/query records, and run the validator. Those ignored prediction records aren't bundled under unresolved redistribution terms, so the public release alone doesn't reproduce every local numerical-restoration check. Source acquisition and full experiment reconstruction are documented in the track-specific tools and configs.
