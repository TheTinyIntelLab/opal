"""Bounded native construction and asset probes in the existing legacy closure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import write_json


def run():
    local = Path('D:/opal-model-series-20261010-v2/native')
    output = ROOT / 'results/opal_model_series_v2/native/bounded'
    output.mkdir(parents=True, exist_ok=False)
    python = ROOT / '.venv-cpa/Scripts/python.exe'
    prefix = f"import sys; sys.path.insert(0,{str(local / 'PertSpectra/src')!r}); "
    wrapper = prefix + "import numpy as np; from Spectra.Spectra_Pert import SPECTRA_Model; SPECTRA_Model(X=np.ones((4,8)),labels=None,L=4,pert_idx=np.array([-1,-1]),pert_labels=['ctrl','basal'],adj_matrix=np.zeros((8,8)),use_cell_types=False)"
    # This is a numerical smoke test of upstream CPU core, not a matched score.
    core = prefix + """
import json,numpy as np,torch
from Spectra.Spectra_Pert import SPECTRA
torch.set_num_threads(1); torch.manual_seed(20261017)
X=torch.ones((12,8)); D=torch.eye(3).repeat_interleave(4,0); w=torch.ones(12)/12
m=SPECTRA(X=X.numpy(),labels=None,adj_matrix=np.array([]),L=4,pert_idx=np.array([-1,-1,-1]),pert_labels=['ctrl','A','basal'],use_cell_types=False,device=torch.device('cpu'))
o=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.001)
history=[]
for i in range(100):
 o.zero_grad(); loss=m.loss_no_cell_types(X,w,D)[0]; loss.backward(); o.step(); history.append(float(loss.detach()))
print(json.dumps({'initial_objective':history[0],'final_objective':history[-1],'epochs':100,'parameter_count':sum(p.numel() for p in m.parameters()),'scope':'synthetic native-core numerical smoke; no source benchmark score','scores':None}))
"""
    for name, code in [('pertspectra-wrapper', wrapper), ('pertspectra-cpu-core-smoke', core)]:
        start = time.monotonic()
        result = subprocess.run([str(python), '-c', code], cwd=output, capture_output=True, text=True,
                                timeout=1800, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1'))
        write_json(output / (name + '.json'), {'return_code': result.returncode, 'stdout': result.stdout,
                   'stderr': result.stderr, 'wall_seconds': time.monotonic()-start, 'scores': None,
                   'environment': str(python), 'maximum_configs': 2, 'primary_seed': 20261017})
    probes = []
    for method, url in [('GEARS', 'https://dataverse.harvard.edu/api/access/datafile/6153417'),
                        ('PRESAGE', 'https://zenodo.org/api/records/15587986')]:
        start = time.monotonic()
        try:
            request = urllib.request.Request(url, method='HEAD' if method == 'GEARS' else 'GET')
            with urllib.request.urlopen(request, timeout=60) as response:
                record = {'status': response.status, 'headers': dict(response.headers), 'url': response.url}
                if method == 'PRESAGE':
                    content = json.load(response)
                    record['files'] = [{k: f.get(k) for k in ('key','size','checksum')} for f in content.get('files', [])]
                    record['license'] = content.get('metadata', {}).get('license')
        except Exception as error:
            record = {'error': type(error).__name__ + ': ' + str(error)}
        probes.append(dict(record, method=method, wall_seconds=time.monotonic()-start, artifact_downloaded=False))
    write_json(output / 'asset_probes.json', probes)


if __name__ == '__main__':
    run()
