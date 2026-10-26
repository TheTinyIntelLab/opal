# C critique of A's independent proposal

Worker C, 2026-10-26. I wrote `C_proposal.md` before opening `A_proposal.md`; this review concerns the actual A proposal now present, not a hypothetical draft. I have not inspected biological responses or run experiments.

A's proposal correctly leaves COLO1 response access closed and distinguishes annotation consistency from native mapping, measured support and initiation evidence. Its treatment of ADAD1, CYLC2 and KLK12 as interventions is the right starting point. I do not see a source-qualification bypass in the proposal. The remaining issues are operational decisions needed before ingestion or an evaluation freeze, rather than evidence that a source gate has passed.

## Source and controls

1. **Define the admissible reference intervention.** The proposal rejects treating non-essential-target controls as non-targeting, but it does not specify what endpoint remains identifiable if these are the only available controls. A later registration must either establish genuine matched constituent measurements against the intended control or define a target-versus-reference-intervention task with explicit assumptions. Merely narrowing the claim to within-initiation does not repair a missing constituent intervention.
2. **Make unknown initiation a task decision.** If the sample key never resolves initiation, mark independent-initiation evaluation unavailable. A within-initiation task still needs culture/run relationships and reference sharing to define leakage grouping and uncertainty; unknown initiation alone is not the only remaining gate.
3. **Separate availability from documentary provenance.** A public processing function may document arithmetic without establishing the deposited object's actual invocation or the construction of `IDs_colo1.rds`. The prerequisite ledger should mark each as verified, documentary, failed or unknown, with exact item/revision evidence. Do not collapse a runnable reconstruction into provenance of the acquired bytes.
4. **Register schema access separately when necessary.** The proposal prohibits headers correctly. If primary documentation cannot supply a response schema, source qualification cannot authorize even header inspection implicitly; a distinct prospective bounded schema-inspection registration must precede it. Freeze numerical processing before subsequent value access.

These are requests for explicit ledger decisions. None requires opening responses during A's current work. An unresolved mapping, endpoint or rights requirement should produce a concrete missing-input record rather than a synthetic substitute.

## Framework and falsification

A's TEST separation, pair grouping and new scalar API requirements align with C's independent proposal. Two omissions need root reconciliation before freeze. First, pair-label budgets need a rounding and missingness convention: take the fixed hash prefix before availability filtering, charge unavailable rows and add no replacement. At 25% of the proposed partition this means 221 TRAIN and 88 VALIDATION labels under floor rounding. Second, persistent technical and biological-labeled worlds need exactly identical numerical arrays under the same seed, not merely the same distribution; otherwise differing fitted predictions cannot falsify a causal-identification claim cleanly.

The proposed interface aggregates observations to two initiations by pair, so it cannot by itself test four correlated construct measurements, arm effects, or equal-construct/equal-culture aggregation. Root must retain a construct-level generator and aggregation oracle or declare a smaller pair-level study before evaluation. The same applies to the uncertainty wrapper: support refusal is necessary but does not implement TRAIN/VALIDATION-calibrated uncertainty or coverage-risk diagnostics.

The proposal asks for matched shuffled controls but does not freeze their residual definition, strata, three seeds or minimum permutable fraction. These must be bound before evaluation; report an insufficient permutation control as a failed control rather than pooling singleton strata after seeing outcomes. An architecture-matched shuffle should perturb only allowed TRAIN residuals and keep baseline preprocessing and validation policy explicit.

## Required response and current disposition

A should answer the four source decisions in its revision or prerequisite ledger, retaining unknowns as unknowns. Root should resolve the budget, exact alias identity, construct aggregation, shuffle and uncertainty details in the implementation registration. I support source-free implementation proceeding while these decisions are recorded; I do not endorse COLO1 response access or a claim that the full synthetic acceptance conditions have already passed.

Implementation audit is pending: `src/frontier_tournament.py` did not exist when this critique was written, and root's interface remained provisional. No implementation tests or independent acceptance results are claimed here. A later C audit will identify the reviewed file hashes and exact findings.
