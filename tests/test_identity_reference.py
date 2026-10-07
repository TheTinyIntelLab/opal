import numpy as np
from tools.run_identity_reference import annotation_tokens, target_components, source_features, predict


def test_multitarget_holdout_uses_transitive_components():
    rows = [dict(target=t, pathway_level_1='shared', pathway_level_2='shared') for t in ['A,B', 'B,C', 'C', 'D']]
    assert target_components(rows) == [{0, 1, 2}, {3}]
    # Pathway overlap is allowed information, but does not merge target components.
    assert 'pathway_level_1:shared' in annotation_tokens(rows[3])


def test_unknown_annotations_are_explicit_and_query_effects_isolated():
    rows = [dict(target=t, pathway_level_1='path', pathway_level_2='path') for t in ['A', 'A', 'B']]
    x, q, vocabulary = source_features(rows, [0, 1], [2])
    assert 'target:B' not in vocabulary
    assert q[0, -1] == 1 and np.all(x[:, -1] == 0)
    y = np.ones((3, 12, 2)); y[2] = 100000
    config = dict(kind='ridge', alpha=.1)
    a = predict(rows, [0, 1], [2], y, config)
    y[2] = -100000
    np.testing.assert_array_equal(a, predict(rows, [0, 1], [2], y, config))
    np.testing.assert_array_equal(a, np.ones((1, 12, 2)))
