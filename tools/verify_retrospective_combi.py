"""Verify saved retrospective predictions, nested choices and descriptive losses."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.replicated.contracts import sha, write_json


def verify(directory):
    report_path = directory / 'report.json'
    report = json.loads(report_path.read_text())
    scope = json.loads((directory / 'scope_before_fit.json').read_text())
    if report['scope_sha256'] != sha(directory / 'scope_before_fit.json'):
        raise ValueError('Scope hash mismatch')
    if report['prediction_archive_sha256'] != sha(directory / 'predictions.npz'):
        raise ValueError('Prediction hash mismatch')
    for name, expected in scope.get('code_sha256', {}).items():
        if sha(directory / 'code_snapshot' / name) != expected:
            raise ValueError('Frozen code hash mismatch')
    if any(report[key] for key in ('primary_claim_pass', 'biological_measurement_gate_pass',
                                   'new_confirmation_performed', 'confidence_intervals_computed')):
        raise ValueError('Retrospective report cannot clear scientific gates')
    checked = 0
    with np.load(directory / 'predictions.npz', allow_pickle=False) as arrays:
        y, pairs, runs = arrays['responses'], arrays['pair_ids'], arrays['run_ids']
        if not np.isfinite(y).all() or len(set(zip(runs, pairs))) != len(y):
            raise ValueError('Invalid saved endpoint identities')
        coverage = {}
        for record in report['fold_records']:
            label, run, fold = record['variant'], record['run'], record['outer_fold']
            mask = (runs == run) & np.asarray([report['fold_assignment'][str(p)] == fold for p in pairs])
            if set(pairs[mask]) != set(record['test_pairs']):
                raise ValueError('Outer pair assignment mismatch')
            key = (label, run)
            coverage.setdefault(key, []).extend(record['test_pairs'])
            selection = record['selection']
            if [r['config'] for r in selection['comparisons']] != scope['menu']:
                raise ValueError('Configuration menu mismatch')
            for role in ('reference', 'candidate'):
                best = min((r for r in selection['comparisons'] if r['config']['role'] == role),
                           key=lambda r: (r['validation_mse'], r['config']['id']))
                if selection[role] != best['config']:
                    raise ValueError('Nested selection mismatch')
                prediction = arrays[f'{label}_{role}']
                value = float(np.mean((prediction[mask] - y[mask]) ** 2))
                if not np.isfinite(prediction).all() or not np.isclose(value, record['losses'][role], rtol=1e-12):
                    raise ValueError('Fold loss mismatch')
                checked += 1
        for (_, run), identities in coverage.items():
            if len(identities) != len(set(identities)) or set(identities) != set(pairs[runs == run]):
                raise ValueError('Incomplete or repeated outer coverage')
        variants = ['aligned'] + [f'shuffle-{s}' for s in scope.get('shuffle_seeds', [])]
        if set(coverage) != {(label, str(run)) for label in variants for run in set(runs)}:
            raise ValueError('Missing expected variant or run')
        for record in report['per_run']:
            mask = runs == record['run']
            for label, role in [key.split('/') for key in record['mse'] if '/' in key]:
                value = float(np.mean((arrays[f'{label}_{role}'][mask] - y[mask]) ** 2))
                if not np.isclose(value, record['mse'][f'{label}/{role}'], rtol=1e-12):
                    raise ValueError('Run loss mismatch')
                checked += 1
        for key, value in report['equal_run_mean_mse'].items():
            if not np.isclose(value, np.mean([r['mse'][key] for r in report['per_run']]), rtol=1e-12):
                raise ValueError('Equal-run summary mismatch')
    return {'status': 'verified', 'report_sha256': sha(report_path),
            'scope_sha256': report['scope_sha256'], 'prediction_archive_sha256': report['prediction_archive_sha256'],
            'checked_prediction_losses': checked, 'outer_records': len(report['fold_records']),
            'source_rows': report['source_rows'], 'nominal_pair_count': report['nominal_pair_count'],
            'endpoint_gene_count': len(report['endpoint_genes']), 'primary_claim_pass': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    write_json(args.directory / 'verification.json', result)
    print(json.dumps(result))
