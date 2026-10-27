"""Replay a trained bundle through its public API, without fitting or scoring."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
from opal.trained_growth import TrainedGrowth


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(bundle_path, source, output, expected_manifest_sha256, installed_root=None):
    started = time.perf_counter()
    module = importlib.import_module('opal.trained_growth')
    if installed_root is not None and not Path(module.__file__).resolve().is_relative_to(Path(installed_root).resolve()):
        raise ValueError('Prediction API was not imported from the declared installation')
    bundle_path, source, output = map(Path, (bundle_path, source, output))
    if sha(bundle_path/'manifest.json') != expected_manifest_sha256:
        raise ValueError('Delivery manifest identity mismatch')
    bundle = TrainedGrowth.load(bundle_path, expected_manifest_sha256=expected_manifest_sha256)
    manifest = json.loads((bundle_path/'manifest.json').read_text())
    if sha(source/'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('Original protocol identity mismatch')
    reports = []
    for task in bundle.tasks:
        original = source/'procedures'/task/'fold-0'
        choice = json.loads((original/'selection.json').read_text())
        record, = [r for r in choice['models'] if r['name'] == choice['selected_policy']]
        model = bundle.model(task)
        with np.load(original/'queries.npz', allow_pickle=False) as queries:
            pairs = queries['pairs'].copy()
            names = tuple(str(x) for x in queries['gene_names'])
            supported = queries['supported'].copy()
        if names != model.gene_names() or not supported.all():
            raise ValueError('Replay requires the original common-supported query population')
        named = [(names[int(a)], names[int(b)]) for a,b in pairs]
        result = bundle.predict(task, named)
        if any(r['status'] != 'predicted' or r['uncertainty'] != 'unavailable' for r in result):
            raise ValueError('Supported original query abstained or claimed an interval')
        prediction = np.array([r['value'] for r in result], dtype=np.float64)
        with np.load(original/'predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(prediction, saved[record['prediction_key']]):
                raise ValueError('Named-gene API prediction replay mismatch')
        forward_result = bundle.predict(task, named[:20])
        reversed_result = bundle.predict(task, [(b,a) for a,b in named[:20]])
        if not np.array_equal([r['value'] for r in forward_result], [r['value'] for r in reversed_result]):
            raise ValueError('Orientation replay mismatch')
        unknown = '__OPAL_UNKNOWN_GENE__'
        if unknown in names:
            raise ValueError('Fixture name unexpectedly in source schema')
        invalid = bundle.predict(task, [(names[0], unknown), (names[0], names[0]), (' '+names[0], names[1])])
        if [r['reason'] for r in invalid] != ['unknown_gene','self_pair','malformed_name'] or any(r['value'] is not None for r in invalid):
            raise ValueError('Unsupported inputs did not abstain safely')
        other = bundle.predict(task, [named[0]], context='__OPAL_UNKNOWN_CONTEXT__')[0]
        if other['reason'] != 'unknown_context' or other['value'] is not None:
            raise ValueError('Wrong context produced a prediction')
        reports.append(dict(task=task, queries=len(pairs), exact=True, reversed_queries=20,
            unsupported_checks=4, selected_policy=choice['selected_policy'],
            original_checkpoint_sha256=record['checkpoint_sha256']))
    report = dict(passed=True, manifest_sha256=expected_manifest_sha256,
        imported_module=module.__file__, installed=installed_root is not None,
        checkpoint_tasks=len(reports), queries=sum(r['queries'] for r in reports),
        new_fits=0, new_scores=0, test_targets_opened=False,
        per_query_uncertainty='unavailable', wall_seconds=time.perf_counter()-started, tasks=reports)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'tasks'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--installed-root', type=Path)
    args = parser.parse_args()
    verify(args.bundle, args.source_run, args.output, args.expected_manifest_sha256, args.installed_root)


if __name__ == '__main__':
    main()
