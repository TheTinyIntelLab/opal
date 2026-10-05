"""Persist fitted reference scales for an existing single-thread research export."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import load_npz, sha256, write_json
from opal.contextual.inference import ContextPredictor, export_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        prior = ContextPredictor(args.checkpoint)
        export_checkpoint(prior.model, args.output, prior.calibration,
            {'previous_manifest_sha256': sha256(args.checkpoint / 'manifest.json'),
             'upgrade': 'Persist original fitted reference scales; no fitting or evaluator access'})
    current = ContextPredictor(args.output)
    query, saved = load_npz(args.queries), load_npz(args.predictions)
    if not np.array_equal(query['identities'], saved['identities']):
        raise ValueError('Prediction identity mismatch')
    pred = current.model.predict(query['pairs'], query['contexts'])
    np.testing.assert_allclose(pred, saved['selected'], atol=1e-8, rtol=1e-8)
    write_json(args.output / 'upgrade_validation.json', {'evaluator_access': False,
        'maximum_absolute_error': float(np.max(np.abs(pred - saved['selected']))),
        'previous_predictions_sha256': sha256(args.predictions), 'checkpoint_state_version': 3})


if __name__ == '__main__':
    main()
