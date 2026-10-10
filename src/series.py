"""Source-specific trainable Opal experts with safe, hash-bound checkpoints.

Only development outcomes enter fit. Query bundles contain measured singles,
identities and explicit assay/reference provenance, never pair outcomes.
"""
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .artifacts import array_hash, fingerprint, read_json, sha256, write_json
from .replicated.contracts import canonical_component
from .replicated.models import (ComponentAdditive, ProfileKernel, balanced_weights,
                                fit_candidate, predict_candidate, select)


class Unavailable(ValueError):
    """Prespecified insufficient acquired-label support."""


def subset(bank, mask):
    return {k: np.asarray(v)[mask].copy() for k, v in bank.items()}


def canonicalize(bank, aliases=None, outcomes=False):
    required = {'singles', 'pair_ids', 'culture_ids'} | ({'responses'} if outcomes else set())
    allowed = required | ({'partitions'} if outcomes else set())
    if set(bank) - allowed or required - set(bank):
        raise ValueError('Exact development/query schema required; query outcomes are forbidden')
    x = np.asarray(bank['singles'], dtype=float).copy()
    pairs = np.asarray(bank['pair_ids']).astype(str)
    cultures = np.asarray(bank['culture_ids']).astype(str)
    if (x.ndim != 3 or x.shape[1] != 2 or len(x) != len(pairs)
            or len(x) != len(cultures) or not np.isfinite(x).all() or not len(x)
            or any(not c.strip() for c in cultures)):
        raise ValueError('Finite measured singles and declared contexts required')
    ids = []
    for i, p in enumerate(pairs):
        components = json.loads(p)
        if not isinstance(components, list) or len(components) != 2:
            raise ValueError('Two intervention identities required')
        components = [canonical_component(c, aliases) for c in components]
        if len(set(components)) != 2:
            raise ValueError('Self pairs are unsupported')
        order = np.argsort(components)
        x[i] = x[i, order]
        ids.append(json.dumps(sorted(components), separators=(',', ':')))
    result = {'singles': x, 'pair_ids': np.asarray(ids), 'culture_ids': cultures}
    if outcomes:
        y = np.asarray(bank['responses'], dtype=float)
        if y.shape != x[:, 0].shape or not np.isfinite(y).all():
            raise ValueError('Finite development responses required')
        result['responses'] = y.copy()
        if 'partitions' in bank:
            result['partitions'] = np.asarray(bank['partitions']).astype(str)
            if result['partitions'].shape != (len(x),):
                raise ValueError('Invalid partitions')
    return result


def validate_contract(contract, dimensions):
    required = {'genes', 'units', 'source', 'assay', 'reference_provenance', 'split_lineage'}
    if required - set(contract):
        raise ValueError('Missing essential assay/reference provenance')
    if (len(contract['genes']) != dimensions or len(set(contract['genes'])) != dimensions
            or not all(isinstance(g, str) and g for g in contract['genes'])
            or contract['units'] not in {'log2_library_10000_change', 'deposited_log_change',
                                         'synthetic_expression_change'}
            or any(not contract[k] for k in required - {'genes'})):
        raise ValueError('Invalid assay scale, genes or reference contract')
    # Access declarations must already use globally canonical resolved IDs.
    # Rejecting noncanonical declarations prevents reversed/aliased holdouts
    # from bypassing the exclusion check after bank canonicalization.
    for field in ('held_pair_ids', 'acquired_pair_ids'):
        for pair in contract.get(field, []):
            components = json.loads(str(pair))
            resolved = [canonical_component(c, contract.get('aliases')) for c in components]
            if (len(resolved) != 2 or len(set(resolved)) != 2
                    or str(pair) != json.dumps(sorted(resolved), separators=(',', ':'))):
                raise ValueError('Access declarations require canonical resolved pair IDs')


def inner_partition(bank, seed, minimum):
    pairs = sorted(set(bank['pair_ids']))
    order = np.random.default_rng(seed).permutation(pairs)
    validation = set(order[:max(minimum, len(pairs) // 4)])
    if len(pairs) - len(validation) < minimum:
        raise Unavailable('Insufficient inner training/validation identities')
    result = copy.deepcopy(bank)
    result['partitions'] = np.asarray(['validation' if p in validation else 'train'
                                     for p in bank['pair_ids']])
    return result


def check_partition(bank, minimum):
    if 'partitions' not in bank:
        raise ValueError('Development partitions required')
    if set(bank['partitions']) != {'train', 'validation'}:
        raise Unavailable('Training and validation partitions required')
    train = bank['partitions'] == 'train'
    val = ~train
    if set(bank['pair_ids'][train]) & set(bank['pair_ids'][val]):
        raise ValueError('Canonical pair identity leakage')
    if min(len(set(bank['pair_ids'][train])), len(set(bank['pair_ids'][val]))) < minimum:
        raise Unavailable('Insufficient inner partition identities')
    return train, val


class Branch(nn.Module):
    def __init__(self, genes, settings):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(genes, settings['hidden_width']), nn.Tanh(),
                                     nn.Linear(settings['hidden_width'], settings['interaction_width']), nn.Tanh())
        self.output = nn.Linear(settings['interaction_width'], genes, bias=False)

    def forward(self, x):
        z = self.encoder(x)
        return self.output(z[:, 0] * z[:, 1])


def optimize(network, inputs, target, weights, settings, penalty, epochs, validation=None):
    """Full-batch weighted MSE; explicit L2, development-only stopping."""
    optimizer = torch.optim.Adam(network.parameters(), lr=settings['learning_rate'])
    best, best_state, best_epoch, stale, history = float('inf'), None, 0, 0, []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        prediction = network(inputs)
        loss = (weights * ((prediction - target) ** 2).mean(1)).sum()
        regularizer = sum(p.square().sum() for p in network.parameters()) / sum(p.numel() for p in network.parameters())
        objective = loss + penalty * regularizer
        if not torch.isfinite(objective):
            raise ValueError('Nonfinite optimization attempt')
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            if validation is None:
                score = float(loss)
            else:
                vx, vy, vw = validation
                score = float((vw * (network(vx) - vy).square().mean(1)).sum())
        history.append({'epoch': epoch, 'train_mse': float(loss.detach()), 'validation_mse': score if validation else None})
        if validation is not None:
            if score < best - settings['minimum_improvement']:
                best, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(network.state_dict())
            else:
                stale += 1
            if stale >= settings['patience']:
                break
    if validation is not None:
        network.load_state_dict(best_state)
    return {'penalty': penalty, 'selected_epoch': best_epoch if validation else epochs,
            'validation_mse': best if validation else None, 'history': history}


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


class OpalBase:
    name = 'Opal-Base'

    def __init__(self, settings, menu):
        self.settings, self.menu = copy.deepcopy(settings), copy.deepcopy(menu)

    def _prepare(self, development, contract):
        if 'pair_ids' in development and not len(development['pair_ids']):
            raise Unavailable('Zero pair labels: pair-supervised methods unavailable')
        bank = canonicalize(development, contract.get('aliases'), outcomes=True)
        validate_contract(contract, bank['singles'].shape[-1])
        pairs = set(bank['pair_ids'])
        if pairs & set(contract.get('held_pair_ids', [])):
            raise ValueError('Globally held pair outcomes cannot enter fitting')
        if pairs - set(contract.get('acquired_pair_ids', pairs)):
            raise ValueError('Unacquired labels cannot enter fitting or validation')
        if len(pairs) < self.settings['minimum_pairs']:
            raise Unavailable('Insufficient acquired pair labels')
        if 'partitions' not in bank:
            bank = inner_partition(bank, self.settings['training_seed'], self.settings['minimum_partition_pairs'])
        check_partition(bank, self.settings['minimum_partition_pairs'])
        self.contract = copy.deepcopy(contract)
        self.contract['acquired_pair_ids'] = sorted(pairs)
        self.fit_hash = array_hash(*(bank[k] for k in sorted(bank)))
        self.access = {'fitting_and_selection_pair_ids': sorted(pairs),
                       'inner_train': sorted(set(bank['pair_ids'][bank['partitions'] == 'train'])),
                       'inner_validation': sorted(set(bank['pair_ids'][bank['partitions'] == 'validation']))}
        return bank

    def fit(self, development, contract):
        bank = self._prepare(development, contract)
        self.selection = select(bank, self.menu)
        self.config = self.selection['reference']
        self.expert = fit_candidate(self.config, bank, np.ones(len(bank['pair_ids']), dtype=bool))
        self.contexts = sorted(set(bank['culture_ids']))
        self.degrees = dict(Counter(c for p in set(bank['pair_ids']) for c in json.loads(p)))
        self.train_profiles = np.unique(bank['singles'].reshape(-1, bank['singles'].shape[-1]), axis=0)
        self.input_mean = self.train_profiles.mean(0)
        self.input_scale = np.maximum(self.train_profiles.std(0), 1e-6)
        return self

    def _query(self, query, contract):
        x = canonicalize(query, self.contract.get('aliases'))
        validate_contract(contract, x['singles'].shape[-1])
        for key in ('genes', 'units', 'source', 'assay'):
            if contract[key] != self.contract[key]:
                raise ValueError('Incompatible query ' + key)
        self._query_reference = contract['reference_provenance']
        return x

    def _base(self, query):
        return predict_candidate(self.config, self.expert, query)

    def support(self, query):
        degrees = np.asarray([[self.degrees.get(c, 0) for c in json.loads(p)]
                              for p in query['pair_ids']])
        z = (query['singles'] - self.input_mean) / self.input_scale
        refs = (self.train_profiles - self.input_mean) / self.input_scale
        # Bounded query/reference distances, no outcomes or query clustering.
        distance = np.asarray([min(np.mean((s - r) ** 2) for r in refs) for s in z.reshape(-1, z.shape[-1])]).reshape(-1, 2)
        context = np.asarray([c in self.contexts for c in query['culture_ids']])
        features = np.column_stack((np.log1p(degrees.min(1)), np.log1p(degrees.max(1)),
                                    np.sqrt(distance).max(1), context.astype(float)))
        supported = (degrees.min(1) > 0) & context
        return features, supported

    def predict(self, query, contract):
        x = self._query(query, contract)
        _, supported = self.support(x)
        return self._result(x, self._base(x), supported, np.zeros(len(x['pair_ids'])))

    def _result(self, query, predictions, supported, gate):
        return {'predictions': predictions, 'pair_ids': query['pair_ids'],
                'support': supported, 'gate': gate,
                'status': ['supported' if s else 'base_fallback_unsupported_component_or_context' for s in supported],
                'provenance': {'model': self.name, 'source': self.contract['source'],
                               'assay': self.contract['assay'], 'units': self.contract['units'],
                               'genes': self.contract['genes'], 'fit_hash': self.fit_hash,
                               'split_lineage': self.contract['split_lineage'],
                               'reference_provenance': self.contract['reference_provenance'],
                               'query_reference_provenance': getattr(self, '_query_reference', None),
                               'gate_is_uncertainty': False}}

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        arrays = {'train_profiles': self.train_profiles, 'input_mean': self.input_mean, 'input_scale': self.input_scale}
        expert_meta = {}
        if self.expert is not None:
            for key, value in vars(self.expert).items():
                if isinstance(value, np.ndarray):
                    arrays['expert_' + key] = value
                else:
                    expert_meta[key] = value
        if hasattr(self, 'branch'):
            arrays.update({'branch_' + k: v.detach().numpy() for k, v in self.branch.state_dict().items()})
        if hasattr(self, 'gate_network'):
            arrays.update({'gate_' + k: v.detach().numpy() for k, v in self.gate_network.state_dict().items()})
            arrays.update(gate_mean=self.gate_mean, gate_scale=self.gate_scale)
        np.savez_compressed(path / 'parameters.npz', **arrays)
        metadata = {k: getattr(self, k) for k in ('settings', 'menu', 'contract', 'fit_hash', 'access',
                                                'selection', 'config', 'contexts', 'degrees')}
        for key in ('branch_selection', 'gate_selection', 'crossfit_records'):
            if hasattr(self, key):
                metadata[key] = getattr(self, key)
        metadata.update(model=self.name, format_version=1, expert_metadata=expert_meta,
                        architecture={'hidden': self.settings['hidden_width'], 'interaction': self.settings['interaction_width'],
                                      'encoder': 'shared two-layer tanh', 'gate': 'regularized affine sigmoid'},
                        parameters_sha256=sha256(path / 'parameters.npz'), opal_version='0.1.2',
                        replay_atol=self.settings['replay_atol'])
        write_json(path / 'model.json', metadata)
        write_json(path / 'integrity.json', {'model_sha256': sha256(path / 'model.json'),
                                            'parameters_sha256': metadata['parameters_sha256']})
        return path

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        receipt = read_json(path / 'integrity.json')
        if receipt != {'model_sha256': sha256(path / 'model.json'), 'parameters_sha256': sha256(path / 'parameters.npz')}:
            raise ValueError('Checkpoint integrity mismatch')
        meta = read_json(path / 'model.json')
        if meta['format_version'] != 1 or meta['model'] not in MODEL_REGISTRY:
            raise ValueError('Unknown checkpoint format/model')
        model = MODEL_REGISTRY[meta['model']](meta['settings'], meta['menu'])
        for key in ('settings', 'menu', 'contract', 'fit_hash', 'access', 'selection', 'config',
                    'contexts', 'degrees', 'branch_selection', 'gate_selection', 'crossfit_records'):
            if key in meta:
                setattr(model, key, meta[key])
        validate_contract(model.contract, len(model.contract['genes']))
        with np.load(path / 'parameters.npz', allow_pickle=False) as file:
            arrays = {k: file[k] for k in file.files}
        if any(v.dtype.kind not in 'fiu' or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Invalid safe checkpoint arrays')
        for key in ('train_profiles', 'input_mean', 'input_scale', 'gate_mean', 'gate_scale'):
            if key in arrays:
                setattr(model, key, arrays[key])
        kind = model.config['kind']
        model.expert = None if kind == 'additive' else (ComponentAdditive(model.config['alpha']) if kind == 'component_additive' else ProfileKernel(model.config))
        if model.expert is not None:
            for key, value in meta['expert_metadata'].items():
                setattr(model.expert, key, value)
            for key, value in arrays.items():
                if key.startswith('expert_'):
                    setattr(model.expert, key[7:], value)
        if model.name != 'Opal-Base':
            model.branch = Branch(len(model.contract['genes']), model.settings).double()
            model.branch.load_state_dict({k[7:]: tensor(v) for k, v in arrays.items() if k.startswith('branch_')})
        if model.name == 'Opal-Guard':
            model.gate_network = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid()).double()
            model.gate_network.load_state_dict({k[5:]: tensor(v) for k, v in arrays.items()
                                               if k.startswith('gate_') and k not in {'gate_mean', 'gate_scale'}})
        return model


class OpalInteraction(OpalBase):
    name = 'Opal-Interaction'

    def fit(self, development, contract):
        bank = self._prepare(development, contract)
        train, val = check_partition(bank, self.settings['minimum_partition_pairs'])
        self.selection = select(bank, self.menu)
        self.config = self.selection['reference']
        expert = fit_candidate(self.config, bank, train)
        profiles = np.unique(bank['singles'][train].reshape(-1, bank['singles'].shape[-1]), axis=0)
        mean, scale = profiles.mean(0), np.maximum(profiles.std(0), 1e-6)
        x = tensor((bank['singles'] - mean) / scale)
        residual = tensor(bank['responses'] - predict_candidate(self.config, expert, bank))
        weights = tensor(balanced_weights(bank['pair_ids'][train], bank['culture_ids'][train]))
        validation = (x[val], residual[val], tensor(balanced_weights(bank['pair_ids'][val], bank['culture_ids'][val])))
        attempts = []
        for penalty in self.settings['regularization']:
            torch.manual_seed(self.settings['training_seed'])
            branch = Branch(x.shape[-1], self.settings).double()
            attempts.append(optimize(branch, x[train], residual[train], weights, self.settings,
                                     penalty, self.settings['epoch_cap'], validation))
        best = min(attempts, key=lambda r: (r['validation_mse'], r['penalty']))
        # Refit frozen Base and residual branch on acquired development only.
        super().fit(bank, contract)
        torch.manual_seed(self.settings['training_seed'])
        self.branch = Branch(x.shape[-1], self.settings).double()
        full_x = tensor((bank['singles'] - self.input_mean) / self.input_scale)
        full_y = tensor(bank['responses'] - self._base(bank))
        refit = optimize(self.branch, full_x, full_y,
                         tensor(balanced_weights(bank['pair_ids'], bank['culture_ids'])), self.settings,
                         best['penalty'], best['selected_epoch'])
        self.branch_selection = {'attempts': attempts, 'selected_penalty': best['penalty'],
                                 'selected_epoch': best['selected_epoch'], 'refit': refit}
        return self

    def _interaction(self, query):
        with torch.no_grad():
            return self.branch(tensor((query['singles'] - self.input_mean) / self.input_scale)).numpy()

    def predict(self, query, contract):
        x = self._query(query, contract)
        _, supported = self.support(x)
        residual = self._interaction(x)
        # No supervised component/context support: explicitly use Base.
        prediction = self._base(x) + supported[:, None] * residual
        return self._result(x, prediction, supported, supported.astype(float))


class GateObjective(nn.Module):
    def __init__(self, gate, base, delta):
        super().__init__()
        self.gate, self.base, self.delta = gate, base, delta

    def forward(self, x):
        return self.base + self.gate(x) * self.delta


class OpalGuard(OpalInteraction):
    name = 'Opal-Guard'

    def fit(self, development, contract):
        bank = self._prepare(development, contract)
        pairs = sorted(set(bank['pair_ids']))
        order = np.random.default_rng(self.settings['training_seed']).permutation(pairs)
        assignment = {p: i % self.settings['crossfit_folds'] for i, p in enumerate(order)}
        folds = np.asarray([assignment[p] for p in bank['pair_ids']])
        oof_base = np.empty_like(bank['responses'])
        oof_delta = np.empty_like(oof_base)
        features = np.empty((len(oof_base), 4))
        records = []
        for fold in range(self.settings['crossfit_folds']):
            train, val = folds != fold, folds == fold
            training = subset(bank, train)
            training.pop('partitions', None)
            training = inner_partition(training, self.settings['training_seed'], self.settings['minimum_partition_pairs'])
            c = copy.deepcopy(contract)
            c['held_pair_ids'] = sorted(set(contract.get('held_pair_ids', [])) | set(bank['pair_ids'][val]))
            c['acquired_pair_ids'] = sorted(set(bank['pair_ids'][train]))
            expert = OpalInteraction(self.settings, self.menu).fit(training, c)
            query = {k: v[val] for k, v in bank.items() if k in {'singles', 'pair_ids', 'culture_ids'}}
            features[val], support = expert.support(query)
            oof_base[val] = expert._base(query)
            oof_delta[val] = expert._interaction(query) * support[:, None]
            records.append({'fold': fold, 'train_pair_ids': c['acquired_pair_ids'],
                            'validation_pair_ids': sorted(set(bank['pair_ids'][val])),
                            'expert_access': expert.access, 'expert_selection': expert.selection,
                            'branch_selection': expert.branch_selection})
        gate_bank = inner_partition(bank, self.settings['training_seed'] + 1, self.settings['minimum_partition_pairs'])
        gt, gv = check_partition(gate_bank, self.settings['minimum_partition_pairs'])
        mean, scale = features[gt].mean(0), np.maximum(features[gt].std(0), 1e-6)
        x = tensor((features - mean) / scale)
        weights = tensor(balanced_weights(bank['pair_ids'][gt], bank['culture_ids'][gt]))
        attempts = []
        for penalty in self.settings['gate_regularization']:
            torch.manual_seed(self.settings['training_seed'])
            gate = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid()).double()
            train_objective = GateObjective(gate, tensor(oof_base[gt]), tensor(oof_delta[gt]))
            val_objective = GateObjective(gate, tensor(oof_base[gv]), tensor(oof_delta[gv]))
            # Validation uses the same learned gate with different OOF expert rows.
            attempts.append(self._optimize_gate(train_objective, val_objective, x, bank,
                                                 gt, gv, weights, penalty))
        best = min(attempts, key=lambda r: (r['validation_mse'], r['penalty']))
        super().fit(bank, contract)
        self.gate_mean, self.gate_scale = features.mean(0), np.maximum(features.std(0), 1e-6)
        torch.manual_seed(self.settings['training_seed'])
        self.gate_network = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid()).double()
        objective = GateObjective(self.gate_network, tensor(oof_base), tensor(oof_delta))
        refit = optimize(objective, tensor((features - self.gate_mean) / self.gate_scale), tensor(bank['responses']),
                         tensor(balanced_weights(bank['pair_ids'], bank['culture_ids'])), self.settings,
                         best['penalty'], best['selected_epoch'])
        self.gate_selection = {'attempts': attempts, 'selected_penalty': best['penalty'],
                               'selected_epoch': best['selected_epoch'], 'refit': refit,
                               'gate_train_pairs': sorted(set(bank['pair_ids'][gt])),
                               'gate_validation_pairs': sorted(set(bank['pair_ids'][gv])),
                               'expert_predictions': 'nested grouped out of fold, never in sample'}
        self.crossfit_records = records
        return self

    def _optimize_gate(self, training, validation, x, bank, gt, gv, weights, penalty):
        optimizer = torch.optim.Adam(training.parameters(), lr=self.settings['learning_rate'])
        best, state, best_epoch, stale, history = float('inf'), None, 0, 0, []
        vw = tensor(balanced_weights(bank['pair_ids'][gv], bank['culture_ids'][gv]))
        for epoch in range(1, self.settings['gate_epoch_cap'] + 1):
            optimizer.zero_grad()
            loss = (weights * (training(x[gt]) - tensor(bank['responses'][gt])).square().mean(1)).sum()
            penalty_term = sum(p.square().sum() for p in training.parameters()) / sum(p.numel() for p in training.parameters())
            objective = loss + penalty * penalty_term
            if not torch.isfinite(objective):
                raise ValueError('Nonfinite gate attempt')
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                score = float((vw * (validation(x[gv]) - tensor(bank['responses'][gv])).square().mean(1)).sum())
            history.append({'epoch': epoch, 'train_mse': float(loss.detach()), 'validation_mse': score})
            if score < best - self.settings['minimum_improvement']:
                best, best_epoch, stale = score, epoch, 0
                state = copy.deepcopy(training.gate.state_dict())
            else:
                stale += 1
            if stale >= self.settings['patience']:
                break
        training.gate.load_state_dict(state)
        return {'penalty': penalty, 'selected_epoch': best_epoch, 'validation_mse': best, 'history': history}

    def predict(self, query, contract):
        x = self._query(query, contract)
        features, supported = self.support(x)
        with torch.no_grad():
            gate = self.gate_network(tensor((features - self.gate_mean) / self.gate_scale)).numpy().ravel()
        gate *= supported
        return self._result(x, self._base(x) + gate[:, None] * self._interaction(x), supported, gate)


MODEL_REGISTRY = {m.name: m for m in (OpalBase, OpalInteraction, OpalGuard)}


def acquisition_order(pair_ids, method, seed):
    """Nested identity-only acquisition, independent of labels and evaluator."""
    pairs = sorted(set(map(str, pair_ids)))
    shuffled = np.random.default_rng(seed).permutation(pairs).tolist()
    if method == 'uniform':
        return shuffled
    if method != 'target-balanced':
        raise ValueError('Unknown acquisition method')
    rank = {p: i for i, p in enumerate(shuffled)}
    degree, result, remaining = Counter(), [], set(pairs)
    while remaining:
        def key(p):
            a, b = json.loads(p)
            return max(degree[a], degree[b]), degree[a] + degree[b], rank[p]
        chosen = min(remaining, key=key)
        result.append(chosen)
        degree.update(json.loads(chosen))
        remaining.remove(chosen)
    return result


def freeze_predictions(directory, model, query, contract):
    """Hash predictions and inputs before evaluator outcome access."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    result = model.predict(query, contract)
    if 'held_pair_ids' in model.contract:
        pairs = set(result['pair_ids'])
        if (pairs - set(model.contract['held_pair_ids'])
                or pairs & set(model.access['fitting_and_selection_pair_ids'])):
            raise ValueError('Evaluation query is outside the frozen globally held population')
    np.savez_compressed(path / 'predictions.npz', predictions=result['predictions'],
                        pair_ids=result['pair_ids'], support=result['support'], gate=result['gate'])
    manifest = {'query_hash': array_hash(*(np.asarray(query[k]) for k in sorted(query))),
                'contract_hash': fingerprint(contract), 'contract': contract,
                'prediction_sha256': sha256(path / 'predictions.npz'), 'provenance': result['provenance'],
                'status': result['status']}
    write_json(path / 'freeze.json', manifest)
    return result


def evaluate_predictions(directory, query, contract, outcomes):
    path = Path(directory)
    freeze = read_json(path / 'freeze.json')
    if (freeze['query_hash'] != array_hash(*(np.asarray(query[k]) for k in sorted(query)))
            or freeze['contract_hash'] != fingerprint(contract)
            or freeze['prediction_sha256'] != sha256(path / 'predictions.npz')):
        raise ValueError('Evaluator rejected changed inputs, genes, units or prediction hash')
    with np.load(path / 'predictions.npz', allow_pickle=False) as file:
        prediction = file['predictions']
        pairs = file['pair_ids']
    if 'held_pair_ids' in contract and set(pairs) - set(contract['held_pair_ids']):
        raise ValueError('Evaluator rejected query membership outside held population')
    y = np.asarray(outcomes, float)
    if y.shape != prediction.shape or not np.isfinite(y).all():
        raise ValueError('Evaluator outcomes incompatible')
    error = np.mean((prediction - y) ** 2, axis=1)
    weights = balanced_weights(query['pair_ids'], query['culture_ids'])
    return {'mse': float(weights @ error), 'pair_mse': error.tolist(),
            'per_stratum': {str(c): float(balanced_weights(np.asarray(query['pair_ids'])[np.asarray(query['culture_ids']) == c],
                            np.asarray(query['culture_ids'])[np.asarray(query['culture_ids']) == c]) @ error[np.asarray(query['culture_ids']) == c])
                            for c in sorted(set(query['culture_ids']))},
            'freeze_sha256': sha256(path / 'freeze.json')}


def series_cli(args):
    """Installed-package train, batch prediction and frozen evaluation."""
    contract = read_json(args.contract)
    with np.load(args.inputs, allow_pickle=False) as file:
        bank = {k: file[k] for k in file.files}
    if args.command == 'series-train':
        settings, menu = read_json(args.config), read_json(args.menu)['menu']
        torch.set_num_threads(settings['threads'])
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=settings['threads']):
            model = MODEL_REGISTRY[args.model](settings, menu).fit(bank, contract)
        model.save(args.output)
        return {'checkpoint': str(args.output), 'model': args.model, 'fit_hash': model.fit_hash}
    if args.command == 'series-predict':
        model = OpalBase.load(args.checkpoint)
        result = freeze_predictions(args.output, model, bank, contract)
        return {'freeze': str(args.output), 'model': model.name, 'status': result['status']}
    with np.load(args.outcomes, allow_pickle=False) as file:
        result = evaluate_predictions(args.freeze, bank, contract, file['responses'])
    write_json(args.output, result)
    return result
