"""Read-only hash and installed growth replay checks for research-reset delivery."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--installed', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(args.installed.resolve()))

import numpy as np
import opal
from opal.artifacts import load_npz, read_json, sha256, write_json
from opal.models.fitness import FitnessCompletion
from opal.models.fitness_inference import FitnessPredictor
from threadpoolctl import threadpool_limits

if not Path(opal.__file__).resolve().is_relative_to(args.installed.resolve()):
    raise RuntimeError('Package import did not use installed wheel')
if args.output.exists():
    raise FileExistsError('Preserve verification receipts')


def check(files):
    for name, digest in files.items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'Frozen bytes differ: {name}')
    return len(files)


with threadpool_limits(limits=1):
    checks = {}
    for directory in ('fitness-external-20261020', 'research-reset-diagnostics',
                      'research-reset-reference-swap', 'research-reset-support/v1'):
        protocol = read_json(ROOT/'results'/directory/'protocol.json')
        checks[directory] = check(protocol['files'])
    run = ROOT/'results/research-reset-development'
    protocol, result = read_json(run/'protocol.json'), read_json(run/'results.json')
    checks['development_code'] = check({str(ROOT/name): h for name, h in protocol['code'].items()})
    if sha256(Path(protocol['data_root'])/'profiles.npz') != protocol['profiles_sha256']:
        raise ValueError('Development data drift')
    for name, expected in [('protocol.json', result['protocol_sha256']),
                           ('selection.json', result['selection_sha256']),
                           ('predictions.npz', result['predictions_sha256']),
                           ('assignment.npz', protocol['assignment_sha256'])]:
        if sha256(run/name) != expected:
            raise ValueError('Development receipt/selection drift')
    prerequisite = read_json(run/'prefit_tests.json')
    started = read_json(run/'run_started.json')
    selection = read_json(run/'selection.json')
    if not (protocol['created_utc'] <= prerequisite['started_utc'] <= prerequisite['completed_utc'] <=
            started['started_utc'] <= selection['completed_utc'] <= result['completed_utc']):
        raise ValueError('Development ordering differs')
    if prerequisite['returncode'] or prerequisite['code'] != protocol['code']:
        raise ValueError('Development prerequisite failed')
    check({str(run/'protocol.json'): started['protocol_sha256'],
           str(run/'prefit_tests.json'): started['prefit_tests_sha256']})
    saved = load_npz(run/'predictions.npz')
    development = {}
    for fit in selection['fits']:
        path = run/(fit['name']+'.npz')
        check({str(path): fit['checkpoint_sha256']})
        prediction = saved[fit['name']+'_baseline'] + FitnessCompletion.load(path).predict(saved['test_pairs'])
        error = float(np.max(abs(prediction-saved[fit['name']+'_prediction'])))
        if error != 0:
            raise ValueError('Installed development composite replay differs')
        development[fit['name']] = error
    original = Path(protocol['data_root']).parent
    predictor = FitnessPredictor(original/'fitness-reference-model')
    d = load_npz(original/'prepared/profiles.npz')
    split = load_npz(original/'fitness-v1/assignment.npz')['split']
    test = np.flatnonzero(split == 2)
    target = load_npz(original/'fitness-v1/predictions.npz')['baseline']
    names = d['genes'][d['pairs'][test]]
    original_errors = {}
    for c, context in enumerate(d['contexts']):
        actual = predictor.predict(names, str(context))
        error = float(np.max(abs(actual-target[c])))
        if error > 1e-12:
            raise ValueError('Installed original held-pair replay differs')
        original_errors[str(context)] = error
    historical = read_json(ROOT/'results/fitness-external-20261020/protocol.json')
    frozen_root = next(Path(p).parent for p in historical['files'] if p.endswith('environment1_predictions.npz'))
    external_errors = {}
    for cohort in ('environment1', 'environment2', 'parp2022', 'parp2023'):
        frozen = load_npz(frozen_root/f'{cohort}_predictions.npz')
        actual = np.concatenate([predictor.predict(frozen['pair_names'][i:i+1024], 'K562')
                                 for i in range(0, len(frozen['pair_names']), 1024)])
        error = float(np.max(abs(actual-frozen['completion'])))
        if error > 1e-12:
            raise ValueError('Installed external frozen predictions differ')
        external_errors[cohort] = error
    write_json(args.output, dict(status='passed', module=opal.__file__, frozen_hash_counts=checks,
        development_composite_max_abs=development, original_test_max_abs=original_errors,
        external_frozen_max_abs=external_errors, original_test_pairs=len(test),
        selection_and_prerequisite_order_verified=True, fitting=False,
        dependency_install='wheel installed without dependencies; host dependency closure reused'))
    print(json.dumps(dict(status='passed', module=opal.__file__, original_test_pairs=len(test))))
