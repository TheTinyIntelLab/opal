"""Fit frozen development selections and export through the common safe API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import time
import argparse
import numpy as np
from threadpoolctl import threadpool_limits
from opal.artifacts import read_json, write_json, load_npz, sha256
from opal.training import fit
from opal.inference import export_members, Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-name', default='checkpoint')
    args = parser.parse_args()
    if Path(args.checkpoint_name).name != args.checkpoint_name:
        raise ValueError('Checkpoint name must be one path component')
    root = Path('results/improvements/residual-kernel-v1')
    with threadpool_limits(limits=1):
        for name in ['norman-log2', 'wessels-log2']:
            path = root / name
            data = load_npz(Path('data/prepared') / name / 'profiles.npz')
            chosen = read_json(path / 'selection.json')['winner']['config']
            config = {**chosen, 'family': 'kernel'}
            start = time.perf_counter()
            member = fit(config, data['singles'], data['pairs'], data['doubles'])
            fitting = time.perf_counter() - start
            source = read_json(Path('data/prepared') / name / 'source_manifest.json')
            output = path / args.checkpoint_name
            export_members([member], data, output, {'context': source['context'], 'scale': source['scale'],
                'research_checkpoint': True, 'validation_scope': 'exploratory nested development',
                'evaluator_access': False, 'selection_sha256': sha256(path / 'selection.json')})
            start = time.perf_counter()
            predictor = Predictor(output)
            cold = time.perf_counter() - start
            query = data['query_pairs']
            expected = load_npz(path / 'unscored_query_predictions.npz')['predictions']
            observed = np.stack([predictor.predict(*data['targets'][p])['effect'] for p in query])
            np.testing.assert_allclose(expected, observed, rtol=1e-7, atol=1e-8)
            timings = []
            a, b = data['targets'][query[0]]
            for _ in range(100):
                start = time.perf_counter()
                predictor.predict(a, b)
                timings.append(time.perf_counter() - start)
            record_name = 'export_verification.json' if args.checkpoint_name == 'checkpoint' else args.checkpoint_name + '_verification.json'
            write_json(path / record_name, {'query_predictions_reproduced': len(query),
                'fit_seconds': fitting, 'cold_seconds': cold, 'warm_median_seconds': float(np.median(timings)),
                'warm_p95_seconds': float(np.quantile(timings, .95)),
                'checkpoint_bytes': sum(p.stat().st_size for p in output.iterdir()),
                'evaluator_access': False, 'numerical_comparison': 'rtol=1e-7 atol=1e-8'})
            print(name, fitting, np.median(timings), flush=True)


if __name__ == '__main__':
    main()
