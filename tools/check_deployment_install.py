"""Verify installed deployment batch inference and refusal to assign evaluation scores."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import opal
from opal.artifacts import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.query, allow_pickle=False) as z:
        query = {k: z[k] for k in z.files}
    records = []
    for cp in read_json(args.manifest)['checkpoints']:
        checkpoint = Path(cp['checkpoint'])
        contract_path = checkpoint.parent.parent / 'contract.json'
        contract = read_json(contract_path)
        destination = args.output / cp['model']
        command = [sys.executable, '-m', 'opal.cli', 'series-predict', '--checkpoint', str(checkpoint),
                   '--inputs', str(args.query), '--contract', str(contract_path), '--output', str(destination)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr)
        prediction = opal.load_model(checkpoint).predict(query, contract)['predictions']
        with np.load(destination / 'predictions.npz', allow_pickle=False) as z:
            difference = float(np.max(np.abs(prediction - z['predictions'])))
        from opal.series import evaluate_predictions
        try:
            evaluate_predictions(destination, query, contract, np.zeros_like(prediction))
        except ValueError as error:
            if 'Deployment predictions' not in str(error):
                raise
        else:
            raise ValueError('Deployment scoring accepted')
        if difference != 0:
            raise ValueError('Deployment API/CLI replay mismatch')
        records.append(dict(model=cp['model'], api_cli_max_abs=difference, evaluation_refused=True))
    write_json(args.output / 'verification.json', dict(records=records, package_path=opal.__file__,
        python=sys.version, training=False, evaluation_score=None, status='verified'))
    print('Three installed deployment models replay; retrospective scoring refused')


if __name__ == '__main__':
    main()
