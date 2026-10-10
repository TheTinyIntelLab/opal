"""Bounded pinned upstream compatibility attempts, without invented scores."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, sha256, write_json


def run(local, output):
    output.mkdir(parents=True, exist_ok=False)
    cases = [
        ('PertSpectra', local / 'PertSpectra' / 'src', 'Spectra.Spectra_Pert',
         "import numpy as np; module.SPECTRA_Model(X=np.zeros((2,4),dtype=np.float32),labels=None,L=4,pert_idx=np.array([-1,-1]),pert_labels=['ctrl','basal'],adj_matrix=np.zeros((4,4)),use_cell_types=False)",
         'Unmodified wrapper construction; CPU core is audited separately before any claimed eligibility'),
        ('GEARS', ROOT / 'tools/external/GEARS', 'gears', '',
         'Requires fold-safe cell-level input and annotation/coexpression graphs; chemical Combi-seq ineligible'),
        ('TxPert', local / 'TxPert', 'gspp.models.txpert', '',
         'Public-only graph route; proprietary graph configurations excluded; no published score copied'),
        ('PRESAGE', local / 'PRESAGE' / 'src', 'presage', '',
         'Exact licensed embedding bundle not locally present; annotation projections are not native embeddings'),
        ('CPA', ROOT / 'tools/external/CPA', 'cpa', '',
         'Requires native cell-level covariates and supported Python closure; normalized aggregates cannot be recast as counts')]
    records = []
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for name, path, module, construction, scope in cases:
        pin = local / name / 'pin.json'
        if pin.exists():
            lineage = read_json(pin)
        else:
            lineage = read_json(ROOT / 'data/public_methods' / (name.lower() + '_local_manifest.json'))
        code = f"import sys,importlib; sys.path.insert(0,{str(path)!r}); module=importlib.import_module({module!r}); print('native import succeeded'); {construction}"
        started = time.monotonic()
        attempt = subprocess.run([sys.executable, '-c', code], cwd=local, env=environment,
                                 capture_output=True, text=True, timeout=1800)
        record = {'method': name, 'revision': lineage['revision'], 'status': 'compatibility_passed' if attempt.returncode == 0 else 'native_compatibility_failed',
                  'stdout': attempt.stdout, 'stderr': attempt.stderr, 'return_code': attempt.returncode,
                  'wall_seconds': time.monotonic() - started, 'scores': None, 'scope': scope,
                  'code_file_hashes': lineage['files'], 'upstream_assets_modified': False,
                  'maximum_configs': 2, 'primary_seed': 20261017, 'compatibility_cap_seconds': 1800}
        if name == 'CPA' and (ROOT / '.venv-cpa/Scripts/python.exe').exists():
            legacy = subprocess.run([str(ROOT / '.venv-cpa/Scripts/python.exe'), '-c', code], cwd=local,
                                    env=environment, capture_output=True, text=True, timeout=1800)
            record['existing_legacy_environment_attempt'] = {'return_code': legacy.returncode, 'stdout': legacy.stdout, 'stderr': legacy.stderr}
        write_json(output / (name.lower() + '.json'), record)
        records.append(record)
        print(name, record['status'], attempt.stderr.splitlines()[-1] if attempt.stderr else '', flush=True)
    write_json(output / 'status.json', {'priority_order': [c[0] for c in cases], 'records': records,
               'native_scored_comparators': 0, 'frontier_comparison_passed': False,
               'published_scores_used': False, 'native_vs_adapted': 'Compatibility records only; no aggregate Opal module relabeled as upstream model'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', type=Path, default=Path('D:/opal-model-series-20261010-v2/native'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/opal_model_series_v2/native/compatibility')
    args = parser.parse_args()
    run(args.local, args.output)
