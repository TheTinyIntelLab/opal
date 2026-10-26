from tools.colo1_frontier_grok_audit_v1 import group_aliases, sample_experiment_counts


def test_invented_alias_groups_leave_unknown_names_unmatched():
    aliases = [
        'EXP_d14-1-100x_S2_L001_R1_001',
        'EXP_d14-R3_S4_L001_R2_001',
        'EXP_d3-500x_S5_L001_R1_001',
        'EXP_d3_S1_L001_R2_001',
        'EXP_lib-GI_S1_L001_R1_001',
        'EXP_lib-PILOT_S2_L001_R2_001',
        'EXP_lib_S5_L001_R1_001',
        'SC_EXP_35467_2#3',
        'EXP_unlisted_culture_001',
    ]
    counts, unmatched = group_aliases(aliases)
    assert unmatched == ['EXP_unlisted_culture_001']
    assert sum(counts.values()) == 8
    assert counts['d14_index_coverage'] == 1
    assert counts['sanger_sc'] == 1


def test_invented_rows_do_not_treat_repeated_names_as_a_sample_join():
    rows = [
        {'sample_accession': 'S1', 'experiment_accession': 'E1'},
        {'sample_accession': 'S1', 'experiment_accession': 'E2'},
        {'sample_accession': 'S2', 'experiment_accession': 'E3'},
    ]
    summary = sample_experiment_counts(rows)
    assert summary == {'samples': 2, 'experiments_per_sample': {1: 1, 2: 1}}
