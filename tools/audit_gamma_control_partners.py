"""Identity-only sidecar for frozen Q1b control-partner support."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import shutil
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import read_json,write_json,sha256,environment,load_npz
from opal.processed_gamma import library,rows
CODE=('tools/audit_gamma_control_partners.py','tests/test_gamma_control_partners.py',
      'docs/research_reset/gamma_control_partners_protocol.md','tests/test_processed_gamma.py',
      'src/processed_gamma.py','src/artifacts.py')
CONFIG=dict(seconds=180,memory=256*1024**2,artifacts=5*1024**2,reserve=1024**3)


def collect(iterator,guides):
    controls=sorted(name for name,g in guides.items() if g['control'])
    if not controls or len(controls)>32:
        raise ValueError('One to 32 explicitly named controls required')
    lookup={guides[name]['index']:i for i,name in enumerate(controls)}
    bits=np.zeros((2,2,2,len(guides)),dtype=np.uint32)
    observed=np.zeros(len(controls),dtype=bool)
    for a,b,values in iterator:
        for guide in (a,b):
            if guide['control']: observed[lookup[guide['index']]]=True
        if a['control']==b['control']:continue
        control,target,arm=(a,b,1) if a['control'] else (b,a,0)
        bit=np.uint32(1<<lookup[control['index']])
        bits[:,:,arm,target['index']] |= np.where(np.isfinite(values),bit,np.uint32(0))
    counts=np.bitwise_count(bits)
    return dict(partner_bits=bits,observed_controls=observed,control_names=np.asarray(controls),
                guide_names=np.asarray(list(guides)),counts=counts)


def summary(arrays,expected):
    if not np.array_equal(arrays['counts'].transpose(0,1,3,2),expected):
        raise ValueError('Partner identities do not reconstruct frozen Q1b support counts')
    bits=arrays['partner_bits']
    controls={str(name):dict(observed_in_any_source_construct=bool(arrays['observed_controls'][i]),
        finite_target_single_cells=int(np.count_nonzero(bits & np.uint32(1<<i))))
        for i,name in enumerate(arrays['control_names'])}
    return dict(support_counts_exact=True,control_partners=controls,
        absent_from_all_source_constructs=[n for n,v in controls.items() if not v['observed_in_any_source_construct']],
        observed_but_never_finite_single_partner=[n for n,v in controls.items() if v['observed_in_any_source_construct'] and not v['finite_target_single_cells']],
        scope='Identity and finite-field support only; no loss, model or biological explanation')


def signature():return {name:sha256(ROOT/name) for name in CODE}


def verify(out):
    reg=read_json(out/'registration.json')
    if reg['code']!=signature() or reg['config']!=CONFIG or reg['environment']!=environment():
        raise ValueError('Frozen sidecar recipe differs')
    for name,digest in reg['inputs'].items():
        if sha256(name)!=digest:raise ValueError('Frozen Q1b/source artifact differs')
    return reg


class Guard:
    def __init__(self,out,previous=None):
        self.out=out;self.previous=previous or {};self.wall=time.monotonic();self.cpu=time.process_time();self.peak=0
    def receipt(self):return dict(wall_seconds=self.previous.get('wall_seconds',0)+time.monotonic()-self.wall,
        cpu_seconds=self.previous.get('cpu_seconds',0)+time.process_time()-self.cpu,
        peak_bytes=max(self.peak,self.previous.get('peak_bytes',0)))
    def check(self):
        memory=psutil.Process().memory_info();self.peak=max(self.peak,getattr(memory,'peak_wset',memory.rss));r=self.receipt()
        if max(r['wall_seconds'],r['cpu_seconds'])>CONFIG['seconds']:raise RuntimeError('Time cap')
        if r['peak_bytes']>CONFIG['memory']:raise RuntimeError('Memory cap')
        if shutil.disk_usage(self.out).free<CONFIG['reserve']:raise RuntimeError('Disk reserve')
        if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())>CONFIG['artifacts']:raise RuntimeError('Artifact cap')


def compute(reg,check):
    q=read_json(Path(reg['q1b'])/'registration.json')
    guides=library(q['paths'][1]);arrays=collect(rows(q['paths'][0],guides,check),guides)
    expected=load_npz(Path(reg['q1b'])/'prepared.npz')['guide_single_counts']
    return arrays,summary(arrays,expected)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','verify','run','replay'))
    p.add_argument('--q1b',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=a.output
    if a.command=='freeze':
        if out.exists():raise FileExistsError('Preserve attempts')
        q=read_json(a.q1b/'registration.json');replay=read_json(a.q1b/'replay.json')
        if not replay['passed']:raise ValueError('Passing Q1b replay required')
        if sha256(a.q1b/'prepared.npz')!=replay['arrays_sha256'] or sha256(a.q1b/'results.json')!=replay['results_sha256']:
            raise ValueError('Q1b replay artifact mismatch')
        paths=[a.q1b/name for name in ('registration.json','results.json','replay.json','prepared.npz')]+[Path(name) for name in q['paths'][:2]]
        inputs={str(path.resolve()):sha256(path) for path in paths}
        for name in q['paths'][:2]:
            if inputs[str(Path(name).resolve())]!=q['inputs'][name]:raise ValueError('Q1b raw source mismatch')
        out.mkdir(parents=True);write_json(out/'registration.json',dict(version='q1b-partners-v1',q1b=str(a.q1b.resolve()),
            inputs=inputs,code=signature(),config=CONFIG,environment=environment(),frozen_at=datetime.now(timezone.utc).isoformat()));return
    reg=verify(out)
    if a.command=='verify':
        if (out/'prerequisites.json').exists():raise FileExistsError('Preserve qualification')
        r=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_gamma_control_partners.py','--junitxml',str(out/'prerequisites.xml')],cwd=ROOT,capture_output=True,text=True,timeout=60)
        (out/'prerequisites.log').write_text(r.stdout+r.stderr,encoding='utf-8')
        write_json(out/'prerequisites.json',dict(passed=r.returncode==0,registration_sha256=sha256(out/'registration.json'),
            junit_sha256=sha256(out/'prerequisites.xml'),log_sha256=sha256(out/'prerequisites.log')))
        if r.returncode:raise RuntimeError('Synthetic qualification failed')
        return
    prereq=read_json(out/'prerequisites.json')
    if not prereq['passed'] or any(prereq[key]!=sha256(out/file) for key,file in (
        ('registration_sha256','registration.json'),('junit_sha256','prerequisites.xml'),('log_sha256','prerequisites.log'))):
        raise ValueError('Passing unchanged prerequisites required')
    replay=a.command=='replay';prefix='replay' if replay else 'run'
    if any((out/name).exists() for name in (prefix+'_started.json',prefix+'_failure.json',prefix+'_receipt.json')):
        raise FileExistsError('Preserve started attempts')
    guard=Guard(out,read_json(out/'run_receipt.json')['resources'] if replay else None)
    write_json(out/(prefix+'_started.json'),dict(started=datetime.now(timezone.utc).isoformat()))
    try:
        with threadpool_limits(1):arrays,result=compute(reg,guard.check)
        if replay:
            saved=load_npz(out/'partners.npz')
            if saved.keys()!=arrays.keys() or any(not np.array_equal(saved[k],arrays[k]) for k in arrays) or result!=read_json(out/'results.json'):
                raise ValueError('Exact sidecar replay differs')
        else:
            np.savez_compressed(out/'partners.npz',**arrays);write_json(out/'results.json',result)
        guard.check();write_json(out/(prefix+'_receipt.json'),dict(passed=True,resources=guard.receipt(),
            arrays_sha256=sha256(out/'partners.npz'),results_sha256=sha256(out/'results.json')))
    except Exception as error:
        write_json(out/(prefix+'_failure.json'),dict(error=str(error),resources=guard.receipt()));raise


if __name__=='__main__':main()
