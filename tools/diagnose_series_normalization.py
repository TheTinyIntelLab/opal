"""Sealed normalization-order and shared-single technical-half diagnostics."""
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, sha256, write_json
from prepare_pertspectra_series import column, label, csc_columns
from run_model_series_bilinear import metrics, resource_check


def run():
    output = ROOT / 'results/opal_model_series_v2/normalization-diagnostics'
    settings = read_json(ROOT / 'configs/experiments/opal_model_series_v2.json')
    resource_check(Path(settings['artifact_roots'][1]), output, settings)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'scope_before_evaluation.json', {
        'role': 'evaluator-only sensitivity; no model, representation or stopping labels',
        'primary_endpoint_replaced': False, 'altered_endpoint': 'mean per-cell log2 normalized count change',
        'same_genes_cells_and_exclusions': True, 'technical_split': 'source cell row modulo two',
        'references': 'each control and single half drawn once and reused across all affected pairs',
        'biological_noise_ceiling': False, 'runner_sha256': sha256(Path(__file__))})
    started, cpu = time.monotonic(), time.process_time()
    prepared = read_json(ROOT / 'results/opal_model_series_v1/pertspectra/access-v4/prepared.json')
    with h5py.File(ROOT / 'data/raw/pertspectra-model-series/inhouse.h5ad', 'r') as f:
        ids, contexts, targets, guides = [column(f['obs'][k]) for k in
                                        ['SampleIndex', 'Treatment', 'target_gene_name', 'feature_call']]
        assigned = [label(t)[0] if len(str(t).split('|')) == len(str(g).split('|')) else None
                    for t, g in zip(targets, guides)]
        names = column(f['var'][f['var'].attrs['_index']])
        matrix = f['layers/counts']
        libraries = np.zeros(len(ids))
        for start in range(0, len(names), 128):
            libraries += np.asarray(csc_columns(matrix, start, min(start + 128, len(names)), len(ids)).sum(1)).ravel()
        for context, info in prepared['contexts'].items():
            source = 'TNFA-positive' if context == 'TNFA+' else 'TNFA-negative'
            with np.load(Path(info['local']) / 'canonical_bank.npz', allow_pickle=False) as z:
                bank = {k: z[k] for k in ['genes', 'pair_ids', 'culture_ids', 'responses']}
            panel = [int(np.flatnonzero(names == g)[0]) for g in bank['genes']]
            counts = sparse.hstack([csc_columns(matrix, g, g + 1, len(ids)) for g in panel], format='csr')
            normalized = counts.multiply(10000 / np.maximum(libraries[:, None], 1)).tocsr()
            logged = normalized.copy()
            logged.data = np.log2(1 + logged.data)
            group = {}
            for i, pair in enumerate(assigned):
                if pair is not None and contexts[i] == context and libraries[i] > 0:
                    group.setdefault((str(ids[i]), pair), []).append(i)
            means, alternatives, halves = {}, {}, {}
            for key, rows in group.items():
                means[key] = np.log2(1 + np.asarray(normalized[rows].mean(0)).ravel())
                alternatives[key] = np.asarray(logged[rows].mean(0)).ravel()
                for half in [0, 1]:
                    selected = [i for i in rows if i % 2 == half]
                    if len(selected) >= 2:
                        halves[key + (half,)] = np.log2(1 + np.asarray(normalized[selected].mean(0)).ravel())
            technical = []
            for stratum in sorted(set(bank['culture_ids'])):
                truth, prediction, excluded = [], [], []
                for identity in bank['pair_ids'][bank['culture_ids'] == stratum]:
                    a, b = json.loads(identity)
                    required = [(stratum, pair, h) for h in [0, 1] for pair in [(), (a,), (b,), (a, b)]]
                    if not all(k in halves for k in required):
                        excluded.append(identity)
                        continue
                    residuals = []
                    for h in [0, 1]:
                        c, x, y, p = [halves[stratum, pair, h] for pair in [(), (a,), (b,), (a, b)]]
                        residuals.append(p - x - y + c)
                    prediction.append(residuals[0])
                    truth.append(residuals[1])
                truth, prediction = np.asarray(truth), np.asarray(prediction)
                technical.append({'stratum': stratum, 'pairs': len(truth), 'excluded': excluded,
                    'residual_half_reproducibility': metrics(prediction, truth, np.zeros_like(truth)),
                    'independent_cultures': False, 'single_control_draw_reuse': True})
            reports = []
            rows = read_json(ROOT / 'results/opal_model_series_v2/primary' / source / 'conditions.json')
            for row in [r for r in rows if r['budget'] == 1 and r['order'] == 'uniform'
                        and r['acquisition_seed'] == 11 and r['training_seed'] == 20261017]:
                contract = read_json(Path(row['checkpoints'][0]['path']).parent.parent / 'contract.json')
                selected = (bank['culture_ids'] == row['stratum']) & np.isin(bank['pair_ids'], contract['held_pair_ids'])
                standard = np.asarray([means[row['stratum'], tuple(json.loads(p))] - means[row['stratum'], ()]
                                       for p in bank['pair_ids'][selected]])
                alternative = np.asarray([alternatives[row['stratum'], tuple(json.loads(p))] - alternatives[row['stratum'], ()]
                                          for p in bank['pair_ids'][selected]])
                difference = float(np.max(np.abs(standard - bank['responses'][selected])))
                if difference > 1e-10:
                    raise ValueError('Normalization diagnostic failed to reproduce original primary preparation')
                evaluated = {}
                for checkpoint in row['checkpoints']:
                    with np.load(Path(checkpoint['path']).parent / 'freeze/predictions.npz', allow_pickle=False) as z:
                        prediction = z['predictions']
                    evaluated[checkpoint['model']] = metrics(prediction, alternative, np.zeros_like(alternative))
                reports.append({'stratum': row['stratum'], 'outer': row['outer'], 'held_rows': int(selected.sum()),
                    'primary_preparation_max_abs': difference, 'frozen_prediction_altered_endpoint_metrics': evaluated,
                    'endpoint_difference_rms': float(np.sqrt(np.mean((standard - alternative) ** 2))),
                    'extra_outcomes': 'all held pair cells, evaluator-only', 'primary_scores_replaced': False})
            write_json(output / (source + '.json'), {'normalization_order': reports, 'shared_single_half_controls': technical,
                'raw_source_sha256': prepared['full_file_sha256'], 'cpu_seconds': time.process_time() - cpu,
                'wall_seconds': time.monotonic() - started})
            print(source, len(reports), flush=True)


if __name__ == '__main__':
    run()
