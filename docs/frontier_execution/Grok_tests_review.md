# Tournament tests and delivery runner

Cursor Grok 4.7, 26 October 2026. This is the authorized implementation round for the independent tournament fixtures and `tools/frontier_delivery.py`. It is not worker A, not C, and not the COLO1 source-parser stream. A's capacity failure remains recorded. No commit or push. No biological response was opened. The numerical lease stays with root for pilot v2.

## Attribution

This worker authored `tests/test_frontier_tournament_independent.py` and `tools/frontier_delivery.py`. Root then edited the test and `tools/frontier_tournament.py`, and root executed the fixtures. This worker did not edit those root fixes and did not rerun pytest.

The authored tests cover generator alias and nested budgets, feature and split refusal, shuffle exclusion from the deployed policy, support fallback, checkpoint schema, inaccessible-label invariance, a validation-label selection control, a 50-step completion checkpoint control, and lifecycle refusal for an altered seal, a partial roster, skipped score and replay, duplicate replay markers, code-binding drift, a missing forecast, and an occupied model directory. Fixture seeds in that file are 77111, 88021, and 88022. Menu fits use `controls=False` with 5 completion steps and 5 neural steps. The registered recipe remains 500 and 300 with controls on.

Root replaced the authored source-text scan for evaluation and pilot seed literals. That scan also contained the literals, so it could not measure isolation. The executed test compares the fixture values with `SEEDS`. The test file hash is the same on both root receipts below, `a61bda2c254186ce491bb5c14d9872ce4c7646ffd837b16ef67368eeecdf5db5`.

Root's first execution failed in `test_lifecycle_seal_roster_skips_and_binding`. `verify_seal` compared Windows `str(path.relative_to(root))` backslashes with the manifest key `models/marker.txt` and raised `Unsealed roster artifact`. Nineteen tests had passed. Root preserved that receipt and changed manifest, binding, and procedure paths to `as_posix`. `src/frontier_tournament.py` is unchanged between the two receipts (`a1c54b4668de6fc64de358cfaac5048c282ab8b88576c7ee1f59db6330686b98`). `tools/frontier_tournament.py` changed from `166285f84d4e497675515c0124b3eebe40c414dfc105e66c53ec2dbabce9b4ab` to `15f527107d006d39b9bcfe03c389093aebfc34bc080b10e513762340c0637318`.

## Root execution

| Receipt | Result |
| --- | --- |
| `results/frontier-execution-20261026/independent-fixtures-root-v1.json` | Preserved failure. Pytest log: 1 failed, 19 passed in 2.15s. Return code 1. |
| `results/frontier-execution-20261026/independent-fixtures-root-v2.json` | Pass. Pytest log: 20 passed in 2.07s. Measured receipt wall 3.560s. Return code 0. |

Both commands were root's `.venv` pytest of `tests/test_frontier_tournament.py` and `tests/test_frontier_tournament_independent.py`.

Pilot v1 fit remains unscored and unqualified. Root's ordering note says the fit started before this review, the recipe is unchanged, and v1 outputs stay preserved. Pilot v2 is root's current numerical job. This worker did not score, forecast, or refit either pilot.

`replay` still writes `replay-start.json` before it checks `--installed-root`. A wrong install path leaves the marker and the exclusive create blocks the real replay. That is a replay-stage fix for root. It does not block the pilot fit. This worker left the function unchanged.

## Delivery runner

`tools/frontier_delivery.py` is ready and has not been run. Invoke it only after pilot v2 releases the numerical lease, with the venv interpreter, one process, and a new output directory:

`.venv\Scripts\python.exe tools/frontier_delivery.py --output <new-directory> [--wheel <existing.whl>] [--installed-root <directory-outside-checkout>] [--tournament-run <sealed-run>]`

The tool builds a wheel and sdist with `python -m build --no-isolation` unless `--wheel` is set, then installs with `pip --no-deps --target` into an empty directory outside the checkout. An existing outside tree that already contains `opal/__init__.py` is reused and pip is skipped. A non-empty unrelated directory is refused. No virtualenv is created, nothing is deleted, and existing outputs are not overwritten. Each command stores argv, cwd, log, sha256, wall time, and return code. Missing historical custody is a ledger blocker. The historical paths are the ones recorded in `tools/validate_wave4_delivery.py`. `--tournament-run` is required for installed tournament replay; replay markers `replay-start.json` and `replay.json` block a second launch. The count-assay script still inserts the checkout on `sys.path`, so that step is recorded as a synthetic CLI check with the installed tree present. Hosted CI remains the clean-install path.
