"""Deposited molecular-label exclusions and backed sparse chunk boundaries."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip('h5py')

spec = importlib.util.spec_from_file_location('prepare_pertspectra_series',
         Path(__file__).parents[1] / 'tools/prepare_pertspectra_series.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_molecular_assignments_do_not_invent_guide_equivalence():
    assert module.label('CTRL00077|CTRL00146') == ((), None)
    assert module.label('BIRC2|CTRL00077') == (('BIRC2',), None)
    assert module.label('RELA|BIRC2') == (('BIRC2', 'RELA'), None)
    assert module.label('BIRC2|BIRC2')[1] == 'self_pair'
    assert module.label('BIRC2|INTERGENIC235713')[0] is None
    assert module.label('nan')[0] is None
    assert module.label('A|B|C')[0] is None


def test_sparse_backed_chunks_match_dense_rows_without_full_densification(tmp_path):
    from scipy import sparse
    source = np.asarray([[0, 2, 0], [3, 0, 4], [0, 0, 0], [5, 6, 0], [0, 7, 8]])
    csr = sparse.csr_matrix(source)
    with h5py.File(tmp_path / 'source.h5', 'w') as f:
        g = f.create_group('X')
        g.attrs['encoding-type'] = 'csr_matrix'
        for name in ['data', 'indices', 'indptr']:
            g.create_dataset(name, data=getattr(csr, name))
        f.create_dataset('dense', data=source)
        for start, stop in [(0, 2), (2, 5), (4, 5)]:
            np.testing.assert_array_equal(module.block(g, start, stop, 3).toarray(), source[start:stop])
            np.testing.assert_array_equal(module.block(f['dense'], start, stop, 3).toarray(), source[start:stop])
        g.attrs['encoding-type'] = 'csc_matrix'
        with pytest.raises(ValueError, match='full densification'):
            module.block(g, 0, 2, 3)


def test_missing_metadata_codes_are_ambiguous_not_controls(tmp_path):
    with h5py.File(tmp_path / 'metadata.h5', 'w') as f:
        g = f.create_group('labels')
        g.create_dataset('categories', data=np.asarray(['BIRC2', 'CTRL00077'], dtype=h5py.string_dtype()))
        g.create_dataset('codes', data=np.asarray([0, -1, 1]))
        decoded = module.column(g)
        assert decoded.tolist() == ['BIRC2', 'nan', 'CTRL00077']
        assert module.label(decoded[1])[0] is None


def test_csc_aggregation_preserves_library_arithmetic_and_label_means(tmp_path):
    from scipy import sparse
    source = np.asarray([[2., 0., 2.], [0., 6., 0.], [3., 3., 0.], [0., 0., 0.]])
    csc = sparse.csc_matrix(source)
    with h5py.File(tmp_path / 'expression.h5', 'w') as f:
        g = f.create_group('counts')
        for name in ['data', 'indices', 'indptr']:
            g.create_dataset(name, data=getattr(csc, name))
        indices = np.asarray(['1', '1', '2', '2'])
        contexts = np.repeat('TNFA-', 4)
        labels = [(), (), ('A',), ('A',)]
        means, counts, _ = module.aggregate_csc(g, indices, contexts, labels, 3, True)
        np.testing.assert_allclose(means[('1', 'TNFA-', ())], [2500., 5000., 2500.])
        np.testing.assert_allclose(means[('2', 'TNFA-', ('A',))], [5000., 5000., 0.])
        assert counts[('2', 'TNFA-', ('A',))] == 1
        g['data'][0] = .25
        with pytest.raises(ValueError, match='not verified raw counts'):
            module.aggregate_csc(g, indices, contexts, labels, 3, True)
