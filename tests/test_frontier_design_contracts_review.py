"""Root regression for a cross-TEST canonical-group hole found in draft review."""
import pytest

from opal import frontier_design_contracts as design


def library():
    guides = [
        dict(guide_id='a1', sequence='ACGT', target_id='A', role='perturbation'),
        dict(guide_id='a2', sequence='ACGA', target_id='A', role='perturbation'),
        dict(guide_id='b', sequence='CCGT', target_id='B', role='perturbation'),
        dict(guide_id='c', sequence='TCGT', target_id='C', role='perturbation'),
    ]
    constructs = [
        dict(construct_id='ab1', guide_a='a1', guide_b='b', orientation='forward'),
        dict(construct_id='ab2', guide_a='b', guide_b='a2', orientation='reverse'),
        dict(construct_id='ac', guide_a='a1', guide_b='c', orientation='forward'),
    ]
    return design.declare_library(guides, constructs)


def test_public_split_cannot_put_a_guide_variant_in_test_and_train():
    declared = library()
    split = design.DesignSplit(design.CONTRACT, ('train', 'test', 'validation'), ())
    with pytest.raises(ValueError):
        design.make_fit_view(
            declared, split, design.Provenance(True, True, False, False),
            ('A', 'B', 'C'), [0., 0., 0.], [True, True, True],
            [0], [1.], [False], [True], [2], [2.], [False], [True],
        )


def test_held_gene_removes_reversed_variants_and_incident_validation():
    declared = library()
    split = design.assign_construct_partitions(declared, ['train', 'train', 'validation'])
    held = design.apply_held_genes(declared, split, ['A'])
    assert held.partition == ('held', 'held', 'held')
    assert design.partition_index(held, 'train').tolist() == []
    assert design.partition_index(held, 'validation').tolist() == []


def test_finite_inputs_cannot_silently_create_nonfinite_train_scaler():
    with pytest.raises(ValueError):
        design.fit_train_scaler([1e308, 1e308], [False, False], ['train_reference'] * 2)


def test_declared_contrast_refuses_finite_constants_that_overflow():
    with pytest.raises(ValueError):
        design.declared_log_contrast(
            [1], [1], [False], [0], [1], [False], ['control'],
            ['independent_control'], ['train'], pseudocount=1e308, scale=1e308,
        )
