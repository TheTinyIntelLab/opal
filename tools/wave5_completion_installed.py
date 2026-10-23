"""Installed-only validation of accepted precision behavior and the finite grid."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root));sys.path.insert(0,str(args.installed.resolve()))
    import opal
    import numpy as np
    from opal.evaluation_precision import metrics_v3,strict_load_completion_v3
    from opal.evaluation_contracts import metrics_v2,strict_load_completion_v2
    from tools import wave5_completion_numerical as grid
    from tools.wave5_completion_common import digest,write
    assert Path(opal.__file__).resolve().is_relative_to(args.installed.resolve())
    source=json.loads((root/'results/research-wave5-completion/A-numerical.json').read_text(encoding='utf8'))
    checks={}
    for exponent in (-500,0,500):
        scale=2.**exponent;y=np.array([0.,1.,-1.,2.])*scale;p=y+scale
        checks['accepted-scale-'+str(exponent)]=metrics_v3(y,p)==metrics_v2(y,p)
    scale=2.**-600
    try:metrics_v3(np.array([0.,1.])*scale,np.array([1.,2.])*scale);checks['metric-underflow-refused']=False
    except ValueError:checks['metric-underflow-refused']=True
    with tempfile.TemporaryDirectory(prefix='opal-installed-precision-',dir=args.output.parent) as directory:
        path=Path(directory)/'state.npz'
        payload=dict(rank=2,penalty=.01,steps=25,seed=1,format_version=2,normalized_upper=np.arange(6,dtype=np.float32)/10,support=np.ones(3,bool),scale=1.,best_step=25,parameter_count=12)
        np.savez(path,**payload)
        checks['checkpoint-exact-replay']=bool(np.array_equal(strict_load_completion_v3(path,digest(path)).matrix,strict_load_completion_v2(path,digest(path)).matrix))
        payload['scale']=np.nextafter(0.,1.);np.savez(path,**payload)
        try:strict_load_completion_v3(path);checks['checkpoint-underflow-refused']=False
        except ValueError:checks['checkpoint-underflow-refused']=True
        grid.OUT=Path(directory)
        grid.run(Path(directory)/'installed-grid.json')
        installed=json.loads((Path(directory)/'installed-grid.json').read_text(encoding='utf8'))
        fields=['width','scale_multiplier','state_sha256','production_vs_ascending','controls','stages','reduction_oracles','prediction_sha256','boundary_anchor']
        checks['nine-grid-cells-exact']=all(all(a.get(k)==b.get(k) for k in fields) for a,b in zip(source['cells'],installed['cells'],strict=True))
    modules={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if (name=='opal' or name.startswith('opal.')) and getattr(module,'__file__',None)}
    checks['all-opal-imports-installed']=all(Path(p).is_relative_to(args.installed.resolve()) for p in modules.values())
    checks['precision-code-identical']=digest(importlib.import_module('opal.evaluation_precision').__file__)==digest(root/'src/evaluation_precision.py')
    result=dict(passed=all(checks.values()),fits=0,scores=0,synthetic_only=True,checks=checks,modules=modules)
    write(args.output,result)
    print(json.dumps(result))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
