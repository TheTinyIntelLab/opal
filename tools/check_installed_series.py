"""Exercise installed train/predict/evaluate CLI and reload all three experts."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import opal
from opal.artifacts import read_json, sha256, write_json
from opal.series import MODEL_REGISTRY, OpalBase

ROOT = Path(__file__).resolve().parents[1]


def check(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = Path(opal.__file__).resolve()
    if args.require_installed and package.is_relative_to(ROOT):
        raise ValueError('Import came from checkout, not installed wheel')
    settings = read_json(args.config or ROOT / 'configs/experiments/opal_model_series_v1.json')
    menu = args.menu or ROOT / 'configs/experiments/replicated_fixed_state_v1.json'
    if args.synthetic:
        rng = np.random.default_rng(31)
        profiles = dict(zip('ABCDEFGH', rng.normal(size=(8, 5))))
        pairs = list(itertools.combinations(profiles, 2))
        singles = np.asarray([[profiles[a], profiles[b]] for a, b in pairs])
        pair_ids = np.asarray([json.dumps(p, separators=(',', ':')) for p in pairs])
        held = pair_ids[-3:]
        development = {'singles': singles[:-3], 'pair_ids': pair_ids[:-3],
                       'culture_ids': np.repeat('synthetic', len(pairs) - 3),
                       'responses': singles[:-3].sum(1) + .2 + .1 * singles[:-3, 0] * singles[:-3, 1]}
        query = {'singles': singles[-3:], 'pair_ids': held, 'culture_ids': np.repeat('synthetic', 3)}
        outcomes = singles[-3:].sum(1) + .2 + .1 * singles[-3:, 0] * singles[-3:, 1]
        contract = {'genes': list('abcde'), 'units': 'synthetic_expression_change', 'source': 'synthetic',
                    'assay': 'synthetic', 'reference_provenance': 'original generated singles',
                    'split_lineage': 'installed-engineering-31', 'held_pair_ids': held.tolist(),
                    'acquired_pair_ids': pair_ids[:-3].tolist()}
        if settings.get('architecture_id') != 'symmetric-bilinear-nested-memory':
            settings.update(epoch_cap=3, gate_epoch_cap=3)
        np.savez_compressed(output / 'development.npz', **development)
        np.savez_compressed(output / 'query.npz', **query)
        np.savez_compressed(output / 'outcomes.npz', responses=outcomes)
        write_json(output / 'contract.json', contract)
        example = output
        references = {}
    else:
        example = args.example.resolve()
        references = {row['model']: row['path'] for row in read_json(example / 'checkpoints.json')}
        contract = read_json(example / 'contract.json')
        with np.load(example / 'query.npz', allow_pickle=False) as archive:
            query = {k: archive[k] for k in archive.files}
    write_json(output / 'settings.json', settings)
    records = []
    for name in MODEL_REGISTRY:
        checkpoint = output / name / 'checkpoint'
        commands = [
            ['series-train', '--model', name, '--inputs', str(example / 'development.npz'),
             '--contract', str(example / 'contract.json'), '--config', str(output / 'settings.json'),
             '--menu', str(menu), '--output', str(checkpoint)],
            ['series-predict', '--checkpoint', str(checkpoint), '--inputs', str(example / 'query.npz'),
             '--contract', str(example / 'contract.json'), '--output', str(output / name / 'freeze')],
            ['series-evaluate', '--freeze', str(output / name / 'freeze'), '--inputs', str(example / 'query.npz'),
             '--contract', str(example / 'contract.json'), '--outcomes', str(example / 'outcomes.npz'),
             '--output', str(output / name / 'evaluation.json')]]
        for i, command in enumerate(commands):
            result = subprocess.run([sys.executable, '-m', 'opal.cli', *command], cwd=output,
                                    capture_output=True, text=True, timeout=300)
            (output / f'{name}-{i}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        restored = opal.load_model(checkpoint)
        prediction = restored.predict(query, contract)['predictions']
        with np.load(output / name / 'freeze/predictions.npz', allow_pickle=False) as archive:
            cli_difference = float(np.max(np.abs(prediction - archive['predictions'])))
        reference_difference = 0.
        if name in references:
            expected = OpalBase.load(references[name]).predict(query, contract)['predictions']
            reference_difference = float(np.max(np.abs(prediction - expected)))
        if max(cli_difference, reference_difference) > settings['replay_atol']:
            raise ValueError('Installed CLI/API or actual source replay mismatch')
        records.append({'model': name, 'cli_train_predict_evaluate': 'passed',
                        'api_cli_max_abs': cli_difference, 'source_refit_max_abs': reference_difference,
                        'checkpoint_model_sha256': sha256(checkpoint / 'model.json'),
                        'evaluation': read_json(output / name / 'evaluation.json')})
    verification = {'status': 'verified', 'package_path': str(package), 'python': sys.version,
                    'outside_checkout': not package.is_relative_to(ROOT), 'synthetic': args.synthetic,
                    'dependency_install': 'Host dependency closure reused; not a fresh dependency installation',
                    'source': 'synthetic engineering' if args.synthetic else contract['source'], 'records': records}
    write_json(output / 'verification.json', verification)
    print(json.dumps({'status': 'verified', 'models': len(records), 'outside_checkout': verification['outside_checkout']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--example', type=Path, default=Path('E:/opal-model-series-20261010/example'))
    parser.add_argument('--require-installed', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--menu', type=Path)
    check(parser.parse_args())
