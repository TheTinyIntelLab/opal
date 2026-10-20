import numpy as np
import pytest
import torch
from opal.models.fitness import FitnessCompletion,GrowthReference
from opal.models.fitness_inference import FitnessPredictor
from opal.artifacts import sha256,write_json


def test_masked_completion_recovers_signal_and_roundtrips(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(8);n=22
    pairs=np.array([(a,b) for a in range(n) for b in range(a+1,n)])
    rng.shuffle(pairs);u=rng.normal(size=(n,2));m=u@np.diag([1.,-1.])@u.T
    y=np.stack([m[pairs[:,0],pairs[:,1]]]*2)
    model=FitnessCompletion('joint',4,.0001,400).fit(pairs[:180],y[:,:180],n,np.ones(2),validation=(pairs[180:205],y[:,180:205]))
    p=model.predict(pairs[205:]);assert np.mean((p-y[:,205:])**2)<.1*np.mean(y[:,205:]**2)
    assert np.array_equal(p,model.predict(pairs[205:,::-1]))
    path=tmp_path/'model.npz';model.save(path)
    assert np.array_equal(p,FitnessCompletion.load(path).predict(pairs[205:]))
    with pytest.raises(ValueError):model.predict([[0,0]])
    with pytest.raises(ValueError):FitnessCompletion().fit(pairs[:20],y[:,:20],n,np.ones(2),validation=(pairs[:1],y[:,:1]))


def test_reference_uses_training_targets_only():
    singles=np.array([[-.1,-.2,-.3,-.4],[-.3,-.2,-.1,-.4]])
    pairs=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    y=singles[:,pairs].sum(2)
    model=GrowthReference('additive').fit(singles,pairs[:4],y[:,:4])
    assert np.array_equal(model.predict(pairs),y)
    calibrated=GrowthReference('node',1.).fit(singles,pairs[:4],y[:,:4])
    assert np.allclose(calibrated.predict(pairs),y)


def test_named_gene_prediction_and_asset_integrity(tmp_path):
    model=FitnessCompletion('independent',8,.01,25)
    model.matrix=np.zeros((2,4,4));model.matrix[0,0,1]=model.matrix[0,1,0]=.25
    model.scales=np.ones(2);model.support=np.ones(4,dtype=bool);model.best_step=25;model.parameter_count=128
    model.save(tmp_path/'completion.npz')
    np.savez_compressed(tmp_path/'reference.npz',genes=np.array(['A','B','C','D']),
        singles=np.array([[.1,.2,.3,.4],[.1,.2,.3,.4]]),c0_center=np.zeros(3),c0_scale=np.ones(3),
        c0_coefficient=np.array([2.,0.,0.]),c0_mean=.5)
    write_json(tmp_path/'manifest.json',{'contexts':{'K562':{'context_index':0,'checkpoint':'completion.npz','reference_kind':'calibration'}},
        'files':{n:sha256(tmp_path/n) for n in ['completion.npz','reference.npz']}})
    predictor=FitnessPredictor(tmp_path)
    assert np.allclose(predictor.predict([['A','B'],['B','A']],'K562'),1.65)
    with pytest.raises(ValueError):predictor.predict([['A','Z']],'K562')
    with pytest.raises(ValueError):predictor.predict([['A','B']],'new_context')
    with (tmp_path/'reference.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError):FitnessPredictor(tmp_path)
