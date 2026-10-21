import importlib.util
from pathlib import Path
import numpy as np
import pytest
from opal.artifacts import write_json, sha256, load_npz

spec=importlib.util.spec_from_file_location('gamma_runner',Path(__file__).parents[1]/'tools/study_reagent_gamma.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def registration(tmp_path,monkeypatch):
    source=tmp_path/'input';source.write_text('synthetic')
    monkeypatch.setattr(runner,'signature',lambda:{'frozen':'recipe'})
    value=dict(code=runner.signature(),config=runner.CONFIG,dependencies=runner.environment(),inputs={str(source):sha256(source)})
    write_json(tmp_path/'registration.json',value)
    return source


def test_registration_tamper_and_environment(tmp_path,monkeypatch):
    source=registration(tmp_path,monkeypatch)
    assert runner.verify(tmp_path)['code'] == {'frozen':'recipe'}
    source.write_text('changed')
    with pytest.raises(ValueError,match='source'):runner.verify(tmp_path)
    source.write_text('synthetic')
    monkeypatch.setattr(runner,'signature',lambda:{'changed':'recipe'})
    with pytest.raises(ValueError,match='recipe'):runner.verify(tmp_path)


def test_resource_guards(tmp_path,monkeypatch):
    guard=runner.Guard(tmp_path)
    monkeypatch.setitem(runner.CONFIG,'reserve_bytes',10**30)
    with pytest.raises(RuntimeError,match='disk'):guard.check()
    monkeypatch.setitem(runner.CONFIG,'reserve_bytes',0)
    monkeypatch.setitem(runner.CONFIG,'wall_seconds',-1)
    with pytest.raises(RuntimeError,match='time'):guard.check()


def test_no_overwrite_run_lifecycle(tmp_path,monkeypatch):
    registration(tmp_path,monkeypatch)
    (tmp_path/'prerequisites.xml').write_text('qualification')
    (tmp_path/'prerequisites.log').write_text('passed')
    write_json(tmp_path/'prerequisites.json',dict(passed=True,registration_sha256=sha256(tmp_path/'registration.json'),
        junit_sha256=sha256(tmp_path/'prerequisites.xml'),log_sha256=sha256(tmp_path/'prerequisites.log')))
    (tmp_path/'run_started.json').write_text('{}')
    monkeypatch.setattr('sys.argv',['gamma','run','--output',str(tmp_path)])
    with pytest.raises(FileExistsError,match='Preserve'):runner.main()


def test_run_replay_and_nan_roundtrip(tmp_path,monkeypatch):
    monkeypatch.setitem(runner.CONFIG,'reserve_bytes',0)
    registration(tmp_path,monkeypatch)
    (tmp_path/'prerequisites.xml').write_text('qualification')
    (tmp_path/'prerequisites.log').write_text('passed')
    write_json(tmp_path/'prerequisites.json',dict(passed=True,registration_sha256=sha256(tmp_path/'registration.json'),
        junit_sha256=sha256(tmp_path/'prerequisites.xml'),log_sha256=sha256(tmp_path/'prerequisites.log')))
    monkeypatch.setattr(runner,'compute',lambda reg,check: (dict(values=np.array([0.,np.nan]),names=np.array(['a','b'])),
        dict(integrity_gate=True,q2b_original_scalar_prerequisites=True)))
    monkeypatch.setitem(runner.CONFIG,'reserve_bytes',0)
    monkeypatch.setattr('sys.argv',['gamma','run','--output',str(tmp_path)])
    runner.main()
    assert np.isnan(load_npz(tmp_path/'prepared.npz')['values'][1])
    monkeypatch.setattr('sys.argv',['gamma','replay','--output',str(tmp_path)])
    runner.main()
    assert (tmp_path/'replay.json').exists()
    with pytest.raises(FileExistsError):runner.main()

def test_compute_synthetic_population_and_historical_indexing(tmp_path,monkeypatch):
    from test_processed_gamma import guides, controls, measurement
    g=guides()
    records=controls(g)+[measurement(g,'A_one','B_one',10.),measurement(g,'B_one','A_two',20.)]
    genes=np.array(['A','B','C']);pairs=np.array([[0,1],[0,2]])
    historical_single=np.array([[[1.,3.,5.],[2.,4.,6.]],[[7.,9.,11.],[8.,10.,12.]]])
    bank=dict(genes=genes,pairs=pairs,singles=historical_single,
        replicates=np.full((2,2,2),15.), responses=np.full((2,2),15.))
    monkeypatch.setattr(runner,'library',lambda path:g)
    monkeypatch.setattr(runner,'rows',lambda *args:iter(records))
    monkeypatch.setattr(runner,'load_npz',lambda path:bank)
    arrays,result=runner.compute(dict(paths=['source','library','bank']),lambda:None)
    assert result['integrity_gate'] and result['q2b_original_scalar_prerequisites']
    assert not result['guide_aware_historical_comparison_gate']
    for c,name in enumerate(('K562','Jurkat')):
        record=result['contexts'][name]
        assert record['qualified_pairs'] == 1 and record['excluded_pairs'] == 1
        panel=record['panels']['all_qualified']
        old_min=(historical_single[c,:,0]).mean()
        assert panel['metrics']['historical_min']['mse'] == (15-old_min)**2
        assert len(panel['infection_metrics']) == 2
    np.testing.assert_array_equal(arrays['missing_matched_single_counts'],0)

def test_tampered_prerequisite_receipt_blocks_run(tmp_path,monkeypatch):
    registration(tmp_path,monkeypatch)
    (tmp_path/'prerequisites.xml').write_text('qualification')
    (tmp_path/'prerequisites.log').write_text('passed')
    write_json(tmp_path/'prerequisites.json',dict(passed=True,registration_sha256=sha256(tmp_path/'registration.json'),
        junit_sha256=sha256(tmp_path/'prerequisites.xml'),log_sha256=sha256(tmp_path/'prerequisites.log')))
    (tmp_path/'prerequisites.xml').write_text('tampered')
    monkeypatch.setattr('sys.argv',['gamma','run','--output',str(tmp_path)])
    with pytest.raises(ValueError,match='prerequisites'):runner.main()
    assert not (tmp_path/'run_started.json').exists()
