"""Read-only installed-wheel replay and custody closure; never fits a model."""
import argparse
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--installed', type=Path, required=True)
p.add_argument('--prior', type=Path, required=True)
p.add_argument('--stopped', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError('Preserve receipts')
sys.path.insert(0, str(a.installed.resolve()))
import opal
import numpy as np
from opal.artifacts import load_npz, read_json, write_json, sha256
from opal.flexible_singles import FlexibleSingles
from opal.scalar_screen import metrics
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if not Path(opal.__file__).resolve().is_relative_to(a.installed.resolve()):
    raise RuntimeError('Expected package-only installed import')
sys.path.insert(1, str(ROOT))
from tools import wave3_nuisance as prior
from tools.wave3_evaluator_v3 import strict_load_completion
from tools import wave3_flexible_nuisance_v2 as stopped

with threadpool_limits(1):
    registration = prior.registered(a.prior)
    selection, results = read_json(a.prior/'selection.json'), read_json(a.prior/'results.json')
    assert read_json(a.prior/'replay.json')['results_sha256'] == sha256(a.prior/'results.json')
    q = read_json(Path(registration['q1b'])/'registration.json')
    bank = load_npz(q['paths'][2])
    features = load_npz(a.prior/'features.npz')['features']
    assert sha256(a.prior/'features.npz') == selection['features_sha256']
    splits = load_npz(Path(registration['block'])/'assignments.npz')
    replay = {}
    for c, context in enumerate(('K562', 'Jurkat')):
        task = 'horlbeck_'+context
        gathered = []
        for rec in [r for r in selection['records'] if r['task'] == task]:
            dest = a.prior/task/f'fold-{rec["fold"]}'
            prior.check_files({str(dest/k): v for k, v in rec['files'].items()})
            ids = np.flatnonzero(splits[f'{task}__{rec["fold"]}'] == 2)
            pred = {}
            for family in prior.FAMILIES:
                x = features[c] if family == 'nuisance_nodes' else features[c, :, :5]
                pred[family] = prior.predict(load_npz(dest/(family+'.npz')), x, bank['pairs'], ids, len(bank['genes']))
            pred['selected'] = pred[rec['winner']]
            for name in ['aligned']+[f'shuffled-{s}' for s in prior.CONFIG['control_seeds']]:
                model = strict_load_completion(dest/(name+'.npz'), rec['files'][name+'.npz'])
                pred[name] = pred['selected']+model.predict(bank['pairs'][ids])
            gathered.append(dict(ids=ids, pred=pred))
        ids = np.concatenate([g['ids'] for g in gathered])
        saved = load_npz(a.prior/(task+'_predictions.npz'))
        np.testing.assert_array_equal(ids, saved['test_indices'])
        for name in gathered[0]['pred']:
            values = np.concatenate([g['pred'][name] for g in gathered])
            np.testing.assert_array_equal(values, saved[name])
            assert metrics(bank['responses'][c, ids], values) == results['tasks'][task]['metrics'][name]
        replay[task] = dict(pairs=len(ids), checkpoint_predictions_exact=True, metrics_exact=True)

    # Verify stopped-run state without another fit or any new test loss.
    r = read_json(a.stopped/'registration.json')
    assert r['code'] == stopped.signature()
    prior.check_files(r['inputs'])
    assert (a.stopped/'fit_failure.json').exists()
    assert not (a.stopped/'selection.json').exists()
    assert not (a.stopped/'score_started.json').exists()
    model = load_npz(a.stopped/'horlbeck_K562/fold-0/flexible_nodes.npz')
    old = FlexibleSingles.load(Path(r['flexible'])/'horlbeck_K562/fold-0/nodes-1.npz')
    for key in ('coefficient', 'center', 'scale'):
        np.testing.assert_array_equal(model[key], getattr(old, key))

    # Failed evaluator test sources are archived, not erased or rediscovered.
    archive = read_json(ROOT/'results/research-wave3-evaluator/archival_v1.json')
    for f in archive['files']:
        assert sha256(ROOT/f['archive']) == f['sha256']
    complete = read_json(ROOT/'results/research-wave3-evaluator/v3/complete.json')
    for name, digest in complete['artifact_hashes'].items():
        assert sha256(ROOT/'results/research-wave3-evaluator/v3'/name) == digest
    write_json(a.output, dict(passed=True, installed_module=str(Path(opal.__file__).resolve()),
        dependency_install='package-only; existing host closure reused', nuisance_v1=replay,
        strongest_reference='v2 stopped before completion fits and test scoring; identical fitted baseline state verified',
        evaluator_archive_and_scientific_manifest='exact',
        reserved_response_or_header_access=False, script_sha256=sha256(__file__)))
    print('Installed v1 predictions and metrics exact; stopped strongest-reference state and archives verified')
