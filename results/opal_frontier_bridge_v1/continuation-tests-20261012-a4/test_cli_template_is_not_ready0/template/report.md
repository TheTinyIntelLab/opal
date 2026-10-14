# Independent measurement readiness

measurement_not_ready.

This checks reviewed local evidence; it does not authorize laboratory execution or establish biological confirmation.

- independent_initiation: review approval missing
- independent_initiation: reviewer missing
- independent_initiation: reviewed_at missing
- independent_initiation: cultures missing
- independent_initiation: initiation_provenance_reviewed must be explicitly true
- independent_initiation: reviewed evidence files missing
- independent_initiation: cultures must be a nonempty list
- pretreatment_contexts: review approval missing
- pretreatment_contexts: reviewer missing
- pretreatment_contexts: reviewed_at missing
- pretreatment_contexts: contexts missing
- pretreatment_contexts: definitions_frozen_before_treatment must be explicitly true
- pretreatment_contexts: reviewed evidence files missing
- pretreatment_contexts: contexts must be a nonempty list
- randomization: review approval missing
- randomization: reviewer missing
- randomization: reviewed_at missing
- randomization: plate_block_assignments missing
- randomization: randomized_assignment_reviewed must be explicitly true
- randomization: reviewed evidence files missing
- literal_interventions: review approval missing
- literal_interventions: reviewer missing
- literal_interventions: reviewed_at missing
- literal_interventions: assignments missing
- literal_interventions: assignment_equivalence_reviewed must be explicitly true
- literal_interventions: reviewed evidence files missing
- literal_interventions: assignments must be a nonempty list
- independent_references: review approval missing
- independent_references: reviewer missing
- independent_references: reviewed_at missing
- independent_references: reference_matches missing
- independent_references: independently_measured must be explicitly true
- independent_references: shared_reference_covariance_retained must be explicitly true
- independent_references: reviewed evidence files missing
- independent_references: reference_matches must be a nonempty list
- measurement_counts: review approval missing
- measurement_counts: reviewer missing
- measurement_counts: reviewed_at missing
- measurement_counts: counts missing
- measurement_counts: counts_supported_by_review must be explicitly true
- measurement_counts: campaign_costs missing
- measurement_counts: reviewed evidence files missing
- measurement_counts: counts.contexts must be an observed positive integer
- measurement_counts: counts.cultures must be an observed positive integer
- measurement_counts: counts.pairs must be an observed positive integer
- measurement_counts: counts.genes must be an observed positive integer
- measurement_counts: counts.control_outcomes must be an observed positive integer
- measurement_counts: counts.single_outcomes must be an observed positive integer
- measurement_counts: counts.bridge_outcomes must be an observed positive integer
- measurement_counts: campaign_costs.distinct_bridge_identities missing
- measurement_counts: campaign_costs.repeated_bridge_outcomes missing
- measurement_counts: campaign_costs.control_outcomes missing
- measurement_counts: campaign_costs.single_outcomes missing
- measurement_counts: campaign_costs.assay_cost missing
- measurement_counts: campaign_costs.currency missing
- release_rights: review approval missing
- release_rights: reviewer missing
- release_rights: reviewed_at missing
- release_rights: license_id missing
- release_rights: permitted_uses missing
- release_rights: research_use_authorized must be explicitly true
- release_rights: redistribution_review_completed must be explicitly true
- release_rights: reviewed evidence files missing
- release_rights: permitted_uses must be a nonempty list
- prospective_information: review approval missing
- prospective_information: reviewer missing
- prospective_information: reviewed_at missing
- prospective_information: held_query_pair_ids missing
- prospective_information: bridge_pair_ids missing
- prospective_information: global_exclusion_context_ids missing
- prospective_information: allowed_query_references missing
- prospective_information: freeze_record missing
- prospective_information: query_outcomes_unscored must be explicitly true
- prospective_information: reviewed evidence files missing
- prospective_information: held_query_pair_ids must be a nonempty list
- prospective_information: bridge_pair_ids must be a nonempty list
- prospective_information: global_exclusion_context_ids must be a nonempty list
- prospective_information: nonempty held query and bridge identities required
- pilot_covariance: review approval missing
- pilot_covariance: reviewer missing
- pilot_covariance: reviewed_at missing
- pilot_covariance: pilot_id missing
- pilot_covariance: covariance_method missing
- pilot_covariance: shared_reference_covariance_retained must be explicitly true
- pilot_covariance: empirical_covariance_reviewed must be explicitly true
- pilot_covariance: reviewed evidence files missing
- power_review: review approval missing
- power_review: reviewer missing
- power_review: reviewed_at missing
- power_review: useful_margin missing
- power_review: margin_metric missing
- power_review: functional_endpoint missing
- power_review: sample_size_by_context missing
- power_review: empirical_power_reviewed must be explicitly true
- power_review: assay_cost_reviewed must be explicitly true
- power_review: biological_collaborator missing
- power_review: reviewed evidence files missing
- power_review: reviewed useful margin must be finite and positive
- power_review: reviewed sample size requires positive culture counts per context
- interval_validation: review approval missing
- interval_validation: reviewer missing
- interval_validation: reviewed_at missing
- interval_validation: estimand missing
- interval_validation: coverage_validation missing
- interval_validation: null_type_i_validation missing
- interval_validation: shared_reference_covariance_retained must be explicitly true
- interval_validation: claim_specific_support_reviewed must be explicitly true
- interval_validation: reviewed evidence files missing
- interval_validation: coverage_validation needs empirical result, reviewed criterion, replicates and passed=true
- interval_validation: null_type_i_validation needs empirical result, reviewed criterion, replicates and passed=true
- owner_arrangements: review approval missing
- owner_arrangements: reviewer missing
- owner_arrangements: reviewed_at missing
- owner_arrangements: measurement_owner missing
- owner_arrangements: laboratory_arrangements_reviewed must be explicitly true
- owner_arrangements: compute_arrangements_reviewed must be explicitly true
- owner_arrangements: registration_arrangements_reviewed must be explicitly true
- owner_arrangements: reviewed evidence files missing

## Second council annotation, 12 October 2026

This file is the retained output of an intentionally incomplete readiness-template test. Its measurement_not_ready result is not a supplied experimental bank, biological rejection or prospective study. Original validator output remains unchanged above.

The current council separately inspected the real engineering closeout and readiness evidence. Independently initiated cultures, pretreatment context definitions, randomized literal assignments, independent reference matching/covariance, rights/custody, a useful biological margin and reviewed power/interval validation remain unavailable. Passing repository tests cannot create them.

The [active plan](C:/Users/USER/Desktop/code/thetinyintellab/opal/docs/frontier/PLAN.md) requires those empirical records before sealed confirmation. Metadata fields must reference actual reviewed evidence; a coding agent must not populate this template with guessed sample counts, reviewers or approvals. A broader mixed-context claim cannot reuse the fixed-state evaluator without a separate validated contract. This annotation changes no readiness decision and authorizes no assays or numerical work.
