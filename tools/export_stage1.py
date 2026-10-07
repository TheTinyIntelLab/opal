"""Replay last inner-selected outer refit for engineering export verification."""
import argparse
import time
from pathlib import Path
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from opal.artifacts import read_json, load_npz, sha256, write_json
from opal.training import fit, predict
from opal.inference import export_members, Predictor


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data-root', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    records = []
    with threadpool_limits(limits=1):
        for name in ['norman-log2', 'wessels-log2']:
            data = load_npz(args.data_root / 'prepared' / name / 'profiles.npz')
            # Match registered load_data's float64 conversion before reference PCA.
            data['singles'] = data['singles'].astype(float)
            data['doubles'] = data['doubles'].astype(float)
            run = args.run / name / 'random'; result = read_json(run / 'results.json'); fold = result['folds'][-1]
            train, query = np.array(fold['train']), np.array(fold['query'])
            original = load_npz(run / f"fold-{fold['fold']}-predictions.npz")
            for family in ['shared', 'pair', 'original_network', 'original_sum_network']:
                selected = fold['selection'][family]
                config = {**selected['config'], 'epochs': selected['refit_epochs']}
                started = time.perf_counter()
                members = [fit(config, data['singles'], data['pairs'][train], data['doubles'][train], seed) for seed in [11, 29]]
                prediction = np.mean([predict(m, data['pairs'][query]) for m in members], 0)
                replay_error = float(np.max(np.abs(prediction - original['neural_' + family])))
                output = args.output / name / family
                export_members(members, {**data, 'pairs': data['pairs'][train]}, output,
                    dict(context=name, scale='log2', scope='last registered outer fold; selected on inner only; engineering replay',
                         scored_source_sha256=sha256(run / 'results.json'), fresh_confirmation=False))
                fitted_seconds = time.perf_counter() - started
                cold = time.perf_counter(); restored = Predictor(output); cold_seconds = time.perf_counter() - cold
                identities = data['targets'][data['pairs'][query]]
                warm = time.perf_counter(); restored_values = np.stack([restored.predict(*pair)['effect'] for pair in identities]); warm_seconds = time.perf_counter() - warm
                restoration_error = float(np.max(np.abs(prediction - restored_values)))
                if restoration_error > 1e-10:
                    raise ValueError('Fitted-weight restoration failed')
                pair = identities[0]; swap_error = float(np.max(np.abs(restored.predict(*pair)['effect'] - restored.predict(*pair[::-1])['effect'])))
                if swap_error > 1e-10:
                    raise ValueError('Symmetry failed')
                record = dict(source=name, family=family, fold=fold['fold'], replay_max_absolute_error=replay_error,
                    restoration_max_absolute_error=restoration_error, swap_max_absolute_error=swap_error,
                    refit_seconds=fitted_seconds, cold_load_seconds=cold_seconds, batch_seconds=warm_seconds,
                    warm_per_pair_seconds=warm_seconds / len(query), pairs_per_second=len(query) / warm_seconds,
                    export_bytes=sum(f.stat().st_size for f in output.iterdir()),
                    reference_bytes=(output / 'references.npz').stat().st_size,
                    process_peak_working_set_bytes=psutil.Process().memory_info().peak_wset,
                    checkpoint=str(output), manifest_sha256=sha256(output / 'manifest.json'))
                records.append(record); write_json(args.output / 'verification.json', records)
    print(records)


if __name__ == '__main__':
    main()
