import hashlib

import numpy as np
import pytest

from opal.evaluation_contracts import metrics_v2, strict_load_completion_v2
from opal.evaluation_precision import metrics_v3, strict_load_completion_v3


def test_nonzero_residual_cannot_silently_become_perfect_mse():
    scale=2.**-600
    y=np.array([0.,1.,-1.,2.])*scale
    p=y+scale
    assert metrics_v2(y,p)['mse']==0
    with pytest.raises(ValueError,match='representable'):metrics_v3(y,p)


@pytest.mark.parametrize('scale',[2.**-500,1.,2.**500])
def test_representable_loss_preserves_exact_v2_result(scale):
    y=np.array([0.,1.,-1.,2.])*scale
    p=np.array([1.,2.,0.,3.])*scale
    assert metrics_v3(y,p)==metrics_v2(y,p)
    assert metrics_v3(y,p)['mse']==scale*scale


def test_exact_zero_and_integer_safeguards_remain_valid():
    assert metrics_v3([0.,0.],[0.,0.])['mse']==0
    assert metrics_v3(np.array([255,0],np.uint8),np.array([0,255],np.uint8))['mse']==65025


def checkpoint(path,scale):
    np.savez(path,rank=2,penalty=.01,steps=25,seed=1,format_version=2,
             normalized_upper=np.arange(6,dtype=np.float32)/10,support=np.ones(3,bool),
             scale=scale,best_step=25,parameter_count=12)


def test_underflowed_nonzero_checkpoint_operator_is_refused(tmp_path):
    path=tmp_path/'tiny.npz';checkpoint(path,np.nextafter(0.,1.))
    old=strict_load_completion_v2(path)
    assert np.any((old.normalized!=0)&(old.matrix==0))
    with pytest.raises(ValueError,match='representable'):strict_load_completion_v3(path)


@pytest.mark.parametrize('scale',[2.**-500,1.,2.**500])
def test_accepted_checkpoint_replay_and_hash_are_unchanged(tmp_path,scale):
    path=tmp_path/'valid.npz';checkpoint(path,scale)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert np.array_equal(strict_load_completion_v3(path,sha).matrix,strict_load_completion_v2(path,sha).matrix)
    with pytest.raises(ValueError,match='hash'):strict_load_completion_v3(path,'0'*64)
