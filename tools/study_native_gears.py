"""Native GEARS architecture/loss on aggregates with reference-derived graphs.

This is an information-aligned development adaptation, not canonical GEARS
with external gene ontology and individual cells. Historical tests stay closed.
"""
import argparse
from pathlib import Path
import sys
import time
import importlib.util
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import load_npz, write_json, sha256
from tools.improve_residual_model import fold_sets


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def graph(features, neighbors=3):
    """Bound memory while selecting positive-cosine reference neighbors."""
    x = np.asarray(features, dtype=np.float32)
    x = x - x.mean(1, keepdims=True)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    edges, weights = [], []
    for start in range(0, len(x), 128):
        similarity = x[start:start + 128] @ x.T
        for local, row in enumerate(similarity):
            row[start + local] = 0
            nearest = np.argpartition(row, -min(neighbors, len(row) - 1))[-min(neighbors, len(row) - 1):]
            for j in nearest:
                if row[j] > 0:
                    edges.append((int(j), start + local)); weights.append(float(row[j]))
    # Explicit loops also keep isolated or all-zero reference genes supported.
    edges.extend((i, i) for i in range(len(x))); weights.extend([1.] * len(x))
    return np.asarray(edges, dtype=np.int64).T, np.asarray(weights, dtype=np.float32)


def fit_predict(data, train, val, graphs, config, seed):
    import torch
    native = module('opal_native_gears_model', ROOT / 'tools/external/GEARS/gears/model.py')
    utils = module('opal_native_gears_utils', ROOT / 'tools/external/GEARS/gears/utils.py')
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    singles, pairs, responses, control = (data[k] for k in ['singles', 'pairs', 'doubles', 'control'])
    g, t = singles.shape[1], len(singles)
    config = {**config, 'num_genes': g, 'num_perts': t, 'device': 'cuda',
              'uncertainty': False, 'no_perturb': False, 'decoder_hidden_size': 16,
              'num_go_gnn_layers': 1, 'num_gene_gnn_layers': 1,
              'G_go': torch.tensor(graphs[0][0]), 'G_go_weight': torch.tensor(graphs[0][1]),
              'G_coexpress': torch.tensor(graphs[1][0]), 'G_coexpress_weight': torch.tensor(graphs[1][1])}
    model = native.GEARS_Model(config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    interventions = [[i, -1] for i in range(t)] + pairs[train].tolist()
    y = np.concatenate([singles, responses[train]]) + control
    perts = [str(i) + '+ctrl' for i in range(t)] + ['+'.join(map(str, pair)) for pair in pairs[train]]
    filters = {name: np.flatnonzero((row != 0) | (control != 0)) for name, row in zip(perts, y)}
    ctrl = torch.tensor(control, device='cuda')
    def batch(conditions):
        n = len(conditions)
        return SimpleNamespace(x=ctrl.repeat(n).reshape(-1, 1),
            batch=torch.arange(n, device='cuda').repeat_interleave(g), pert_idx=conditions)
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(config['epochs']):
        order = rng.permutation(len(y))
        for indices in np.array_split(order, max(1, int(np.ceil(len(order) / config['batch_size'])))):
            pred = model(batch([interventions[i] for i in indices]))
            target = torch.tensor(y[indices], device='cuda')
            loss = utils.loss_fct(pred, target, [perts[i] for i in indices], ctrl=ctrl,
                                   direction_lambda=.1, dict_filter=filters)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = np.concatenate([model(batch(pairs[val[start:start + config['batch_size']]].tolist())).cpu().numpy()
                      for start in range(0, len(val), config['batch_size'])]) - control
    del model, optimizer
    torch.cuda.empty_cache()
    return prediction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['norman-log2', 'wessels-log2'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1); torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this native comparison')
    started = time.perf_counter()
    path = ROOT / 'data/prepared' / args.dataset / 'profiles.npz'
    data = load_npz(path)
    config = {'hidden_size': 16, 'epochs': 2 if args.smoke else 40, 'lr': .001, 'batch_size': 8}
    write_json(args.output / 'protocol.json', {'config': config, 'seed': 20261005,
        'scope': __doc__, 'input_sha256': sha256(path), 'evaluator_access': False,
        'native_model_sha256': sha256(ROOT / 'tools/external/GEARS/gears/model.py'),
        'native_loss_sha256': sha256(ROOT / 'tools/external/GEARS/gears/utils.py'),
        'graphs': 'positive cosine nearest-three from allowed measured singles, no external ontology',
        'gene_selection': 'all prepared genes; no outcome-selected subset', 'smoke_only': args.smoke})
    graphs = graph(data['singles']), graph(data['singles'].T)
    records = []
    for kind in (['random'] if args.smoke else ['random', 'partner']):
        folds = fold_sets(data['pairs'], kind, 20261005, 5)
        for i, (train, val) in enumerate(folds[:1] if args.smoke else folds):
            prediction = fit_predict(data, train, val, graphs, config, 20261005 + i)
            mse = float(np.mean((prediction - data['doubles'][val]) ** 2))
            additive = data['singles'][data['pairs'][val]].sum(1)
            record = {'split': kind, 'fold': i, 'train': train.tolist(), 'validation': val.tolist(),
                      'mse': mse, 'additive_mse': float(np.mean((additive - data['doubles'][val]) ** 2))}
            records.append(record)
            write_json(args.output / 'progress.json', {'records': records})
            print(kind, i, 'MSE', mse, flush=True)
    write_json(args.output / 'results.json', {'status': 'smoke_passed' if args.smoke else 'complete',
        'method': 'native_GEARS_reference_graph_aggregate_adaptation', 'records': records,
        'mse': {kind: float(np.average([r['mse'] for r in records if r['split'] == kind],
                weights=[len(r['validation']) for r in records if r['split'] == kind])) for kind in {r['split'] for r in records}},
        'device': torch.cuda.get_device_name(), 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'seconds': time.perf_counter() - started, 'historical_evaluator_access': False,
        'canonical_GEARS_benchmark': False})


if __name__ == '__main__':
    main()
