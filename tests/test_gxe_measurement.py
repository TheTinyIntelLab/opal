import gzip
import numpy as np
import pytest
from opal.gxe_measurement import assignment,rt_context,matrix_blocks,matched_keys,reference_variance


def test_exact_assignments_and_pretreatment_rt_origin():
    r=assignment('plate6_1_DDR1IN_1_rep1')
    assert matched_keys('A172',r)[1]==('A172','plate6','rep1','DDR1IN',1.,0.)
    assert rt_context('A01_G01_RT_BC_1_Lig_BC_4')=='A172'
    assert rt_context('A01_G01_RT_BC_5_Lig_BC_4')=='T98G'
    assert rt_context('A01_G01_RT_BC_10_Lig_BC_4')=='U87MG'
    with pytest.raises(ValueError):assignment('plate6_nan_DDR1IN_1_rep1')
    with pytest.raises(ValueError):rt_context('A01_G01_RT_BC_0_Lig_BC_4')


def test_bounded_matrix_stream_keeps_first_row_and_exact_count_population(tmp_path):
    p=tmp_path/'matrix.gz'
    with gzip.open(p,'wt') as f:f.write('%%MatrixMarket matrix coordinate integer general\n%fixture\n2 3 3\n1 1 2\n2 2 3\n1 3 1\n')
    stream=matrix_blocks(p,lambda:None,chunk_lines=2)
    assert next(stream)==(2,3,3)
    blocks=list(stream);np.testing.assert_array_equal(blocks[0][0],[0,1])
    assert sum(len(b[0]) for b in blocks)==3
    with gzip.open(p,'wt') as f:f.write('%%MatrixMarket matrix coordinate integer general\n2 3 1\n1 4 2\n')
    stream=matrix_blocks(p,lambda:None);next(stream)
    with pytest.raises(ValueError,match='bounds'):list(stream)


def test_author_headerless_format_requires_annotation_axes_and_preserves_first_entry(tmp_path):
    p=tmp_path/'raw.gz'
    with gzip.open(p,'wt') as f:f.write('1\t1\t2\n2\t3\t4\n')
    stream=matrix_blocks(p,lambda:None,expected_shape=(2,3),chunk_lines=1)
    assert next(stream)==(2,3,None)
    blocks=list(stream);assert sum(len(b[0]) for b in blocks)==2
    np.testing.assert_array_equal(np.concatenate([b[2] for b in blocks]),[2,4])
    with pytest.raises(ValueError,match='annotation axes'):next(matrix_blocks(p,lambda:None))


def test_reference_panel_variance_does_not_count_duplicate_strata_as_identities():
    a=np.array([1.,0.]);b=np.array([0.,3.])
    np.testing.assert_array_equal(reference_variance({'a':[a], 'b':[b]}),reference_variance({'a':[a]*4,'b':[b]}))
