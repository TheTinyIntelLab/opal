"""Run pinned native CPA on information-matched aggregate inputs with CUDA."""
import argparse
from pathlib import Path
import sys
import time
import gc
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, write_json, sha256, load_npz, environment
from opal.contextual.contracts import load_dataset


def run_native(bank, fitting, validation, queries, config, seed, *, restored_state=None):
    import pandas as pd
    import anndata as ad
    import cpa
    import torch
    a = bank.arrays
    records, expressions = [], []

    def add(context, targets, expression, split):
        interventions = [bank.metadata['intervention_records'][str(a['targets'][t])] for t in targets]
        interventions = sorted(interventions, key=lambda x: x['canonical_target'])
        perts = '+'.join(r['canonical_target'] for r in interventions) if targets else 'DMSO'
        # CPA encodes component indices in alphabetical order; doses must follow
        # that order too, regardless of the original pair index order.
        doses = '+'.join(r['dose'].split()[0] for r in interventions) if targets else '0.0'
        records.append({'perturbation': perts, 'dose': doses, 'context': str(a['contexts'][context]), 'split': split})
        expressions.append(expression)

    for ci in range(len(a['contexts'])):
        add(ci, [], a['controls'][ci], 'train')
        for ti in np.flatnonzero(a['available'][ci]):
            add(ci, [int(ti)], a['controls'][ci] + a['singles'][ci, ti], 'train')
    reference_rows = len(records)
    if validation is None:
        # Final checkpoint selection uses permitted references only. These rows
        # are removed from training rather than duplicated across the split.
        for index in [0, 1, 2, 3]:
            records[index]['split'] = 'valid'
    for p, c, response in zip(fitting['pairs'], fitting['contexts'], fitting['responses']):
        add(int(c), p.tolist(), a['controls'][c] + response, 'train')
    if validation is not None:
        for p, c, response in zip(validation['pairs'], validation['contexts'], validation['responses']):
            add(int(c), p.tolist(), a['controls'][c] + response, 'valid')
    query_start = len(records)
    for pair, ci in zip(queries['pairs'], queries['contexts']):
        # Query X contains the allowed vehicle, never the unknown response.
        add(int(ci), pair.tolist(), a['controls'][ci], 'ood')
    data = ad.AnnData(np.asarray(expressions, dtype=np.float32), obs=pd.DataFrame(records))
    data.var_names = a['genes'].tolist()
    cpa.CPA.pert_encoder = None
    cpa.CPA.covars_encoder = None
    cpa.CPA.pert_smiles_map = None
    cpa.CPA.setup_anndata(data, perturbation_key='perturbation', control_group='DMSO',
                         dosage_key='dose', categorical_covariate_keys=['context'], is_count_data=False)
    model = cpa.CPA(data, split_key='split', train_split='train', valid_split='valid', test_split='ood',
                    recon_loss='gauss', n_latent=config['latent'], n_hidden_encoder=config['width'],
                    n_hidden_decoder=config['width'], n_layers_encoder=2, n_layers_decoder=2,
                    use_batch_norm_encoder=False, use_batch_norm_decoder=False, seed=seed)
    if restored_state is not None:
        model.module.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in restored_state.items()}, strict=True)
        model.module.to('cuda')
    else:
        model.train(max_epochs=config['epochs'], use_gpu=True, batch_size=64, save_path=False,
                early_stopping_patience=config['epochs'] + 1, check_val_every_n_epoch=10,
                plan_kwargs={'n_epochs_pretrain_ae': 0, 'n_epochs_adv_warmup': 5,
                             'n_hidden_adv': 32, 'n_layers_adv': 2, 'use_batch_norm_adv': False,
                             'n_epochs_verbose': 1000},
                enable_progress_bar=False, logger=False, enable_checkpointing=False,
                num_sanity_val_steps=0)
    query_indices = np.arange(query_start, data.n_obs)
    query_data = data[query_indices].copy()
    model.predict(query_data, batch_size=64, n_samples=1)
    absolute = query_data.obsm['CPA_pred']
    prediction = absolute - a['controls'][queries['contexts']]
    state = {key: value.detach().cpu().numpy() for key, value in model.module.state_dict().items()}
    del model, data
    gc.collect(); torch.cuda.empty_cache()
    return prediction, state, reference_rows


def subset(data, rows):
    return {k: v[rows] for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/native-cpa-aggregate-v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config = read_json(args.config)
    bank, data, query, protocol = load_dataset(ROOT / config['dataset'])
    write_json(args.output / 'protocol.json', {'config': config, 'config_sha256': sha256(args.config),
               'dataset_protocol_sha256': sha256(ROOT / config['dataset'] / 'protocol.json'),
               'environment': environment(), 'smoke_only': args.smoke, 'historical_evaluator_access': False,
               'already_exposed_pilot': True, 'scope': config['scope']})
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Native comparator requires the authorized CUDA environment')
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    write_json(args.output / 'status.json', {'status': 'running', 'evaluator_access': False})
    start = time.perf_counter()
    try:
        candidates = [{'width': 32, 'latent': 8, 'epochs': 10}] if args.smoke else config['candidates']
        contexts = np.unique(data['contexts'])[:1] if args.smoke else np.unique(data['contexts'])
        search = []
        for candidate in candidates:
            errors = []
            for context in contexts:
                val = np.flatnonzero(data['contexts'] == context)
                train = np.flatnonzero(data['contexts'] != context)
                prediction, _, _ = run_native(bank, subset(data, train), subset(data, val), subset(data, val), candidate, config['seed'])
                errors.extend(np.mean((prediction - data['responses'][val]) ** 2, axis=1).tolist())
                print('Candidate', candidate, 'context', int(context), 'MSE', float(np.mean(errors)), flush=True)
            search.append({'config': candidate, 'mse': float(np.mean(errors))})
        winner = min(search, key=lambda r: r['mse'])
        write_json(args.output / 'selection.json', {'winner': winner, 'search': search, 'calibration_used': False, 'evaluator_used': False})
        if args.smoke:
            write_json(args.output / 'status.json', {'status': 'smoke_passed', 'native_score_claim': False})
            return
        prediction, state, reference_rows = run_native(bank, data, None, query, winner['config'], config['seed'])
        np.savez_compressed(args.output / 'predictions.npz', prediction=prediction, identities=query['identities'])
        np.savez_compressed(args.output / 'native_state.npz', **state)
        write_json(args.output / 'test_access.json', {'predictions_sha256': sha256(args.output / 'predictions.npz'),
                   'selection_sha256': sha256(args.output / 'selection.json'), 'previously_exposed_pilot': True})
        evaluator = load_npz(ROOT / config['dataset'] / 'evaluator/profiles.npz')
        if not np.array_equal(evaluator['identities'], query['identities']):
            raise ValueError('Evaluator identity order mismatch')
        mse = float(np.mean((prediction - evaluator['responses']) ** 2))
        with zipfile.ZipFile(args.output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(__file__, 'tools/study_native_cpa.py')
            for path in (ROOT / 'tools/external/CPA/cpa').rglob('*.py'):
                archive.write(path, path.relative_to(ROOT).as_posix())
        write_json(args.output / 'results.json', {'status': 'complete', 'method': 'native_CPA_aggregate_input', 'mse': mse,
                   'config': winner['config'], 'allowed_reference_rows': reference_rows, 'fitting_combinations': len(data['pairs']),
                   'independent_confirmation': False, 'canonical_cell_level_benchmark': False,
                   'cuda_device': torch.cuda.get_device_name(), 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                   'seconds': time.perf_counter() - start, 'source_snapshot_sha256': sha256(args.output / 'source_snapshot.zip')})
        write_json(args.output / 'status.json', {'status': 'complete', 'evaluator_access': True})
    except Exception as error:
        write_json(args.output / 'status.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
