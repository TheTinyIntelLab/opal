"""Hand oracles for the completion experiment's reporting utilities."""
import numpy as np

from tools.wave5_completion_numerical import discrepancy, exact_rounded_sum


def test_exact_reduction_oracle_recovers_cancellation_lost_by_ascending_sum():
    values=np.array([2.**54,1.,-2.**54])
    assert np.add.accumulate(values)[-1]==0
    assert exact_rounded_sum(values)==1


def test_zero_reference_errors_and_occurrence_rows_are_explicit():
    result=discrepancy(np.array([0.,2.,1e-13]),np.array([0.,1.,0.]))
    assert result['affected_rows']==[1,2]
    assert result['zero_reference_nonzero_errors']==1
    assert result['max_abs']==result['max_relative_nonzero_reference']==1
    assert not result['atol_1e12_rtol0']
