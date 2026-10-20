import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('external_eval', Path(__file__).parents[1]/'tools/evaluate_external_fitness.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_strata_keep_selection_exposure_separate():
    result = module.strata_for(np.array([['B','A'],['A','C'],['B','C'],['A','D']]),
                              np.array([[0,1],[0,2],[1,2]]), np.array([0,1,2]),
                              np.array(['A','B','C','D']))
    assert result.tolist() == ['original_train','original_validation','original_test','absent_original_eligible_bank']


def test_conditional_intervals_recover_fixed_advantage():
    names = np.array([['A','B'],['A','C'],['B','C'],['C','D']])
    intervals = module.uncertainty(names, -np.ones(4), draws=100)
    assert intervals['pair_conditional_95'] == [-1., -1.]
    assert intervals['gene_resampling_conditional_95'] == [-1., -1.]
    assert 0 < intervals['nonempty_gene_draws'] <= 100


def test_freeze_integrity_and_bias(tmp_path):
    file = tmp_path/'input'
    file.write_bytes(b'original')
    digest = module.sha256(file)
    module.verify_files({str(file): digest})
    file.write_bytes(b'modified')
    with pytest.raises(ValueError, match='Frozen input differs'):
        module.verify_files({str(file): digest})
    score = module.metrics(np.array([2.,3.]), np.array([1.,2.]))
    assert score['mse'] == score['mae'] == score['bias'] == 1.
