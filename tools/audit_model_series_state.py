"""Read-only historical checkpoint replay and saved-fit diagnosis.

Writes only to a new owner-selected audit directory, never historical results.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import sha256, write_json
from opal.series import OpalBase, inner_partition, subset


def run(output):
    started, cpu = time.monotonic(), time.process_time()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sources = {
        'combi-seq': ('combi-seq', Path('E:/opal-model-series-20261010'), 'fits'),
    }
    preparation = json.loads((ROOT / 'results/opal_model_series_v1/pertspectra/access-v4/prepared.json').read_text())
    for context, info in preparation['contexts'].items():
        name = 'TNFA-positive' if context == 'TNFA+' else 'TNFA-negative'
        sources[name] = ('pertspectra/' + name, Path(info['local']), 'evaluation')
    counts, diagnostics, integrity = {}, [], []
    with threadpool_limits(limits=1):
        for name, (relative, local, fits) in sources.items():
            compact = ROOT / 'results/opal_model_series_v1' / relative
            manifest_name = 'all_budget_checkpoint_manifest.json' if name == 'combi-seq' else 'checkpoint_manifest.json'
            manifests = json.loads((compact / manifest_name).read_text())
            rows = json.loads((compact / 'conditions.json').read_text())
            with np.load(local / 'canonical_bank.npz', allow_pickle=False) as f:
                bank = {k: f[k] for k in ('singles', 'responses', 'pair_ids', 'culture_ids')}
            expected_bank = json.loads((compact / ('input_manifest.json' if name == 'combi-seq' else 'scope_before_fit.json')).read_text())
            expected_hash = expected_bank.get('bank_sha256', expected_bank.get('input_bank_sha256'))
            if sha256(local / 'canonical_bank.npz') != expected_hash:
                raise ValueError('Historical bank changed: ' + name)
            replayed, maximum, seen = 0, 0., set()
            for row in manifests:
                path = Path(row['path'])
                if sha256(path / 'model.json') != row['model_sha256'] or sha256(path / 'parameters.npz') != row['parameters_sha256']:
                    raise ValueError('Historical manifest mismatch: ' + str(path))
                model = OpalBase.load(path)
                work = path.parent.parent
                freeze = work / model.name / 'freeze'
                condition = next(r for r in rows if r['fit_id'] == work.name)
                query_mask = (bank['culture_ids'] == condition['run']) & np.isin(bank['pair_ids'], model.contract['held_pair_ids'])
                inputs = bank
                if condition['variant'] != 'aligned':
                    from opal.replicated.models import shuffled_profiles
                    inputs = shuffled_profiles(bank, int(condition['variant'].split('-')[1]))
                query = {k: v[query_mask] for k, v in inputs.items() if k != 'responses'}
                contract = json.loads((work / 'contract.json').read_text())
                result = model.predict(query, contract)
                with np.load(freeze / 'predictions.npz', allow_pickle=False) as f:
                    difference = float(np.max(np.abs(result['predictions'] - f['predictions'])))
                if difference > 1e-10:
                    raise ValueError('Historical replay mismatch: ' + str(path))
                replayed += 1
                maximum = max(maximum, difference)
                if condition['budget'] != 1 or condition['variant'] != 'aligned' or (condition['run'], condition['outer'], model.name) in seen:
                    continue
                seen.add((condition['run'], condition['outer'], model.name))
                train = (bank['culture_ids'] == condition['run']) & np.isin(bank['pair_ids'], model.contract['acquired_pair_ids'])
                development = subset(bank, train)
                training_query = {k: v for k, v in development.items() if k != 'responses'}
                base = model._base(training_query)
                residual = development['responses'] - base
                record = {'source': name, 'stratum': condition['run'], 'outer': condition['outer'], 'model': model.name,
                          'selected_reference': model.config, 'training_residual_rms': float(np.sqrt(np.mean(residual ** 2))),
                          'response_rms': float(np.sqrt(np.mean(development['responses'] ** 2))),
                          'additive_rms': float(np.sqrt(np.mean(development['singles'].sum(1) ** 2))),
                          'checkpoint_bytes': sum(p.stat().st_size for p in path.iterdir()),
                          'gradient_norms': None, 'gradient_norm_reason': 'Not recorded by historical training; no retraining substituted',
                          'oof_residual_norm': None, 'oof_reason': 'Historical OOF arrays not checkpointed; expert histories/access retained',
                          'parameter_count': sum(p.numel() for p in getattr(model, 'branch', torch.nn.Identity()).parameters())}
                features, supported = model.support(training_query)
                record['support_features'] = [{'std': float(v.std()), 'zero_rate': float(np.mean(v == 0))} for v in features.T]
                record['fallback_fraction'] = float(np.mean(~result['support']))
                if hasattr(model, 'branch'):
                    delta = model._interaction(query)
                    base_query = model._base(query)
                    record.update(branch_rms=float(np.sqrt(np.mean(delta ** 2))),
                                  correction_to_base=float(np.linalg.norm(delta) / max(np.linalg.norm(base_query), 1e-30)),
                                  output_rank=int(np.linalg.matrix_rank(delta)),
                                  branch_selection=model.branch_selection)
                if hasattr(model, 'gate_selection'):
                    gv = set(model.gate_selection['gate_validation_pairs'])
                    gt = set(model.gate_selection['gate_train_pairs'])
                    contamination = []
                    for fold in model.crossfit_records:
                        overlap = set(fold['train_pair_ids']) & gv
                        training_rows = set(fold['validation_pair_ids']) & gt
                        if overlap and training_rows:
                            contamination.append({'expert_fold': fold['fold'], 'gate_validation_labels_in_expert': len(overlap),
                                                  'affected_gate_training_rows': len(training_rows)})
                    record.update(gate_quantiles=np.quantile(result['gate'], [0, .25, .5, .75, 1]).tolist(),
                                  meta_validation_dependence=contamination,
                                  outer_test_labels_in_experts=sorted(set().union(*(set(f['train_pair_ids']) for f in model.crossfit_records)) & set(model.contract['held_pair_ids'])))
                diagnostics.append(record)
            counts[name] = {'manifest_checkpoints': len(manifests), 'replayed': replayed, 'maximum_absolute_error': maximum,
                            'bank_sha256': expected_hash}
            full = [r for r in rows if r['budget'] == 1 and r['variant'] == 'aligned' and r['order'] == 'uniform' and r['sampling_seed'] == 11]
            integrity.append({'source': name, 'full_budget_mean_mse': {m: float(np.mean([r['metrics'][m] for r in full if m in r['metrics']])) for m in sorted(set().union(*(set(r['metrics']) for r in full)))}})
    write_json(output / 'diagnosis.json', {'counts': counts, 'full_budget_metrics': integrity, 'full_budget_saved_fit_diagnostics': diagnostics,
               'meta_defect': 'Verified gate-training expert fits include gate-validation outcomes in v1; outer held pairs remain excluded. Preserve original scores. V2 must hold router validation out before generating training OOF experts.',
               'wall_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu,
               'rss_bytes': psutil.Process().memory_info().rss, 'historical_files_modified': False})
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
