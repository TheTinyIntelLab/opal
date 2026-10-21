import importlib.util
from pathlib import Path
import numpy as np
import pytest
from opal.artifacts import write_json,sha256
from opal.processed_gamma import rows,library
from test_processed_gamma import guides,measurement,write_source
spec=importlib.util.spec_from_file_location('partner_runner',Path(__file__).parents[1]/'tools/audit_gamma_control_partners.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_exact_partner_identity_arm_context_and_finiteness():
    g=guides();first=measurement(g,'A_one','negative_control-1',0.);first[2][0,1]=np.nan
    second=measurement(g,'negative_control-2','A_one',2.)
    result=runner.collect([first,second,measurement(g,'A_one','B_one',999)],g)
    bits=result['partner_bits'];target=g['A_one']['index']
    assert result['control_names'].tolist()==['negative_control-1','negative_control-2']
    assert bits[0,0,0,target]==1 and bits[0,1,0,target]==0
    assert bits[1,1,0,target]==1 and bits[0,0,1,target]==2
    assert not bits[:,:,:,g['B_one']['index']].any()
    expected=result['counts'].transpose(0,1,3,2)
    summary=runner.summary(result,expected)
    assert summary['support_counts_exact']
    with pytest.raises(ValueError,match='support counts'):runner.summary(result,expected+1)


def test_absent_versus_nonfinite_partner_identity():
    g=guides();result=runner.collect([measurement(g,'A_one','negative_control-1',np.nan)],g)
    report=runner.summary(result,result['counts'].transpose(0,1,3,2))
    assert report['absent_from_all_source_constructs']==['negative_control-2']
    assert report['observed_but_never_finite_single_partner']==['negative_control-1']


def test_duplicate_construct_and_guide_library_rejected(tmp_path):
    import csv,gzip
    source=tmp_path/'source.gz';record=['A_one++negative_control-1']+['0']*15
    write_source(source,[record,record])
    with pytest.raises(ValueError,match='Duplicate'):runner.collect(rows(source,guides()),guides())
    path=tmp_path/'guides.gz';entry=['name','A_one','A_one','Y','unused','A'*20,'up','down','pLG18']
    with gzip.open(path,'wt',newline='') as stream:csv.writer(stream,delimiter='\t').writerows([entry,entry])
    with pytest.raises(ValueError,match='Unique'):library(path)


def synthetic_registration(out,monkeypatch):
    monkeypatch.setattr(runner,'signature',lambda:{'frozen':'recipe'})
    monkeypatch.setitem(runner.CONFIG,'reserve',0)
    source=out/'source';source.write_text('source')
    write_json(out/'registration.json',dict(code=runner.signature(),config=runner.CONFIG,environment=runner.environment(),inputs={str(source):sha256(source)}))
    (out/'prerequisites.xml').write_text('pass');(out/'prerequisites.log').write_text('pass')
    write_json(out/'prerequisites.json',dict(passed=True,registration_sha256=sha256(out/'registration.json'),
        junit_sha256=sha256(out/'prerequisites.xml'),log_sha256=sha256(out/'prerequisites.log')))
    return source


def test_tampered_frozen_source_rejected(tmp_path,monkeypatch):
    source=synthetic_registration(tmp_path,monkeypatch);runner.verify(tmp_path)
    source.write_text('tampered')
    with pytest.raises(ValueError,match='artifact'):runner.verify(tmp_path)


def test_run_replay_lifecycle_and_tamper(tmp_path,monkeypatch):
    synthetic_registration(tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'compute',lambda reg,check: (dict(bits=np.array([1,2],dtype=np.uint32)),dict(passed=True)))
    monkeypatch.setattr('sys.argv',['sidecar','run','--output',str(tmp_path)]);runner.main()
    with pytest.raises(FileExistsError):runner.main()
    monkeypatch.setattr('sys.argv',['sidecar','replay','--output',str(tmp_path)]);runner.main()
    assert (tmp_path/'replay_receipt.json').exists()
    with pytest.raises(FileExistsError):runner.main()


def test_replay_rejects_array_change(tmp_path,monkeypatch):
    synthetic_registration(tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'compute',lambda reg,check: (dict(bits=np.array([1],dtype=np.uint32)),dict(passed=True)))
    monkeypatch.setattr('sys.argv',['sidecar','run','--output',str(tmp_path)]);runner.main()
    np.savez_compressed(tmp_path/'partners.npz',bits=np.array([2],dtype=np.uint32))
    monkeypatch.setattr('sys.argv',['sidecar','replay','--output',str(tmp_path)])
    with pytest.raises(ValueError,match='replay differs'):runner.main()
    assert (tmp_path/'replay_failure.json').exists()


def test_resource_caps(tmp_path,monkeypatch):
    guard=runner.Guard(tmp_path);monkeypatch.setitem(runner.CONFIG,'seconds',-1)
    with pytest.raises(RuntimeError,match='Time'):guard.check()
