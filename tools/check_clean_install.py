"""Install declared dependency closure in an isolated venv outside checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import platform

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel',type=Path,required=True);p.add_argument('--python',default=sys.executable)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--assets',type=Path)
    p.add_argument('--torch-index-url', help='Optional PyTorch wheel index for CPU-only validation runners')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    outside=Path(tempfile.mkdtemp(prefix='opal-clean-dependencies-'))
    env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):env.pop(key,None)
    env['PYTHONNOUSERSITE']='1'
    venv=outside/'environment'
    executable=str(Path(a.python).resolve()) if Path(a.python).exists() else a.python
    subprocess.run([executable,'-m','venv',str(venv)],check=True,env=env,cwd=outside)
    python=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    def run(label,args):
        result=subprocess.run([str(python),*args],cwd=outside,env=env,text=True,capture_output=True)
        (a.output/(label+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError('Clean validation failed: '+label+'; see '+str(a.output))
        return result.stdout
    if a.torch_index_url:
        # Install the chosen backend first; the research wheel still resolves
        # and checks its complete declared dependency closure afterward.
        run('torch_install',['-m','pip','install','torch>=2.5,<3','--index-url',a.torch_index_url])
    run('install',['-m','pip','install',str(a.wheel.resolve())])
    run('dependency_check',['-m','pip','check'])
    backend=json.loads(run('torch_backend',['-c',"import json,torch;print(json.dumps({'version':torch.__version__,'cuda_version':torch.version.cuda}))"]))
    if a.torch_index_url and a.torch_index_url.rstrip('/').endswith('/cpu') and backend['cuda_version'] is not None:
        raise RuntimeError('CPU validation requested but a CUDA PyTorch build was installed')
    imported=run('import',['-c',"import json,sys,opal;from pathlib import Path;assert Path(opal.__file__).resolve().is_relative_to(Path(sys.prefix).resolve());print(json.dumps({'path':opal.__file__,'version':opal.__version__,'python':sys.version}))"])
    run('quickstart',['-c','from opal.cli import main;main()','reproduce','--config',str(ROOT/'configs/experiments/release_quickstart.json'),'--output',str(outside/'quickstart')])
    if not (outside/'quickstart').is_dir():raise RuntimeError('Synthetic quickstart produced no artifacts')
    # Installed package must contain the new replicated workflow too.
    run('replicated_import',['-c','from opal.replicated.workflow import freeze, evaluate;from opal.replicated.data import query_references'])
    if a.assets:
        run('reference_restoration',[str(ROOT/'tools/validate_research_release.py'),'--assets',str(a.assets.resolve())])
    dependencies=run('freeze',['-m','pip','freeze'])
    report={'status':'passed','clean_dependency_install':True,'host_dependencies_reused':False,
            'outside_checkout':True,'environment':str(outside),'platform':platform.platform(),
            'import':json.loads(imported),'quickstart':'passed','reference_restoration':'passed' if a.assets else 'not_requested',
            'dependency_closure':dependencies.splitlines()}
    report['torch_index_url'] = a.torch_index_url
    report['torch_backend'] = backend
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
