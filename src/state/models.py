"""Fold-fitted simple references and compact factorized state-response models."""
import numpy as np
from scipy.linalg import solve


def dose_response(x, y, query, config):
    """Fixed log-dose main-effect curve; source responses only, known intervention."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 2 or len(x) != len(y) or not len(x) or np.any(x <= 0) or query <= 0 or not np.isfinite(x).all() or not np.isfinite(query) or not np.isfinite(y).all():
        raise ValueError('Dose curve requires available finite source effects and positive nM doses')
    kind = config['kind']
    if kind == 'zero':
        return np.zeros(y.shape[1])
    if kind == 'mean':
        return y.mean(0)
    if kind == 'nearest':
        return y[np.argmin(np.abs(np.log10(x) - np.log10(query)))].copy()
    if config['alpha'] <= 0 or not np.isfinite(config['alpha']):
        raise ValueError('Positive finite regularization required')
    center, scale = np.log10(x).mean(), max(np.log10(x).std(), 1e-8)
    z, q = (np.log10(x) - center) / scale, (np.log10(query) - center) / scale
    ym = y.mean(0)
    if kind == 'ridge':
        if config['degree'] not in {1, 2}:
            raise ValueError('Registered fixed degree required')
        features = np.column_stack([z ** d for d in range(1, config['degree'] + 1)])
        fm = features.mean(0); features -= fm
        cross = np.array([q ** d for d in range(1, config['degree'] + 1)]) - fm
        return ym + cross @ np.linalg.solve(features.T @ features + config['alpha'] * np.eye(features.shape[1]), features.T @ (y - ym))
    if kind != 'kernel' or config['bandwidth'] <= 0 or not np.isfinite(config['bandwidth']):
        raise ValueError('Unknown dose model or invalid bandwidth')
    k = np.exp(-(z[:, None] - z[None]) ** 2 / config['bandwidth'])
    km = k.mean(0); kc = k - km[None] - km[:, None] + k.mean()
    cross = np.exp(-(z - q) ** 2 / config['bandwidth'])
    return ym + (cross - km - cross.mean() + km.mean()) @ np.linalg.solve(kc + config['alpha'] * np.eye(len(x)), y - ym)


class DoseReference:
    """Experimental interpolation of known-compound/state main effects in nM.

    Profiles must already have exact endpoint, assay and matched-control alignment.
    No chemical identity, context transfer, pair prediction or interval is inferred.
    """
    def __init__(self, config):
        self.config = dict(config)

    def fit(self, states, compounds, doses_nM, effects, genes, provenance):
        self.states = np.array(states, dtype=str, copy=True); self.compounds = np.array(compounds, dtype=str, copy=True)
        self.doses = np.array(doses_nM, dtype=float, copy=True); self.effects = np.array(effects, dtype=float, copy=True)
        self.genes = np.array(genes, dtype=str, copy=True); self.provenance = dict(provenance)
        if self.states.ndim != 1 or self.genes.ndim != 1 or not len(self.states) or not len(self.genes) or self.states.shape != self.compounds.shape or self.states.shape != self.doses.shape or self.effects.shape != (len(self.states), len(self.genes)):
            raise ValueError('Aligned exact source profiles and gene schema required')
        if len(set(zip(self.states, self.compounds, self.doses))) != len(self.states) or len(set(self.genes)) != len(self.genes):
            raise ValueError('Duplicate source conditions or genes')
        if not {'endpoint_hours', 'assay', 'normalization', 'source_sha256', 'license'} <= self.provenance.keys():
            raise ValueError('Exact endpoint, assay, normalization and source provenance required')
        for state, compound in set(zip(self.states, self.compounds)):
            selected = (self.states == state) & (self.compounds == compound)
            if selected.sum() < 2:
                raise ValueError('Interpolation requires at least two source doses')
            dose_response(self.doses[selected], self.effects[selected], self.doses[selected][0], self.config)
        return self

    def predict(self, state, compound, dose_nM, *, endpoint_hours):
        if endpoint_hours != self.provenance['endpoint_hours']:
            raise ValueError('Endpoint mismatch')
        selected = (self.states == state) & (self.compounds == compound)
        if not selected.any():
            raise ValueError('Unknown compound/state has no source support')
        x = self.doses[selected]
        if not np.isfinite(dose_nM) or dose_nM < x.min() or dose_nM > x.max():
            raise ValueError('Extrapolation unavailable: development gate failed')
        return {'effect': dose_response(x, self.effects[selected], dose_nM, self.config),
                'genes': self.genes.copy(), 'interval': None, 'evidence_level': 'exposed dose development',
                'source_doses_nM': x.copy(), 'state': state, 'compound': compound}

    def save(self, path):
        from pathlib import Path
        from opal.artifacts import write_json, sha256
        path = Path(path); path.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(path / 'references.npz', states=self.states, compounds=self.compounds,
                            doses_nM=self.doses, effects=self.effects, genes=self.genes)
        write_json(path / 'manifest.json', dict(schema='opal-dose-reference-v1', config=self.config,
                    provenance=self.provenance, arrays_sha256=sha256(path / 'references.npz')))

    @classmethod
    def load(cls, path):
        from pathlib import Path
        from opal.artifacts import read_json, sha256
        path = Path(path); m = read_json(path / 'manifest.json')
        if m['schema'] != 'opal-dose-reference-v1' or sha256(path / 'references.npz') != m['arrays_sha256']:
            raise ValueError('Dose export schema or checksum mismatch')
        with np.load(path / 'references.npz', allow_pickle=False) as f:
            return cls(m['config']).fit(f['states'], f['compounds'], f['doses_nM'], f['effects'], f['genes'], m['provenance'])


def basis(x, rank):
    x = np.asarray(x, dtype=float)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    keep = min(rank, int(np.sum(s > 1e-8)))
    return vt[:keep].T if keep else np.zeros((x.shape[1], 1))


class MainReference:
    def __init__(self, kind='mean', alpha=1., rank=8):
        if kind not in {'mean', 'ridge', 'kernel'} or alpha <= 0:
            raise ValueError('Unsupported main reference')
        self.kind, self.alpha, self.rank = kind, alpha, rank

    def fit(self, controls, effects, available):
        controls, effects, available = np.asarray(controls), np.asarray(effects), np.asarray(available)
        if effects.shape[:2] != available.shape or controls.shape != (len(effects), effects.shape[2]) or not available.any():
            raise ValueError('Invalid source supervision')
        self.controls = controls.copy()
        self.effects = effects.copy()
        self.available = available.copy()
        self.center = controls.mean(0)
        self.control_basis = basis(controls - self.center, self.rank)
        z = (controls - self.center) @ self.control_basis
        self.scale = np.maximum(z.std(0), 1.)
        self.z = z / self.scale
        self.means = np.zeros(effects.shape[1:])
        self.coefficients, self.sources = [], []
        for ti in range(effects.shape[1]):
            source = np.flatnonzero(available[:, ti])
            self.sources.append(source)
            if not len(source):
                self.coefficients.append(None)
                continue
            y, x = effects[source, ti], self.z[source]
            self.means[ti] = y.mean(0)
            xmean = x.mean(0)
            if self.kind == 'ridge':
                xc = x - xmean
                coef = solve(xc.T @ xc + self.alpha * np.eye(x.shape[1]), xc.T @ (y - y.mean(0)), assume_a='pos')
                self.coefficients.append((xmean, coef))
            elif self.kind == 'kernel':
                distances = np.sum((x[:, None] - x[None]) ** 2, -1)
                positive = distances[distances > 1e-12]
                length = float(np.median(positive)) if len(positive) else 1.
                k = np.exp(-distances / length)
                km = k.mean(0)
                kc = k - km[None] - km[:, None] + k.mean()
                coef = solve(kc + self.alpha * np.eye(len(x)), y - y.mean(0), assume_a='pos')
                self.coefficients.append((length, km, coef))
            else:
                self.coefficients.append(None)
        return self

    def encode_control(self, controls):
        return (np.asarray(controls) - self.center) @ self.control_basis / self.scale

    def predict(self, controls, targets):
        targets = np.asarray(targets, dtype=int)
        z = self.encode_control(controls)
        if len(z) != len(targets) or np.any(targets < 0) or np.any(targets >= len(self.means)):
            raise ValueError('Invalid main-effect queries')
        output = []
        for x, ti in zip(z, targets):
            source = self.sources[ti]
            if not len(source):
                raise ValueError('Exact intervention has no source response support')
            pred = self.means[ti].copy()
            if self.kind == 'ridge':
                xm, coef = self.coefficients[ti]
                pred += (x - xm) @ coef
            elif self.kind == 'kernel':
                length, km, coef = self.coefficients[ti]
                cross = np.exp(-np.sum((self.z[source] - x) ** 2, 1) / length)
                pred += (cross - km - cross.mean() + km.mean()) @ coef
            output.append(pred)
        return np.asarray(output)


class FactorizedResponse:
    """Separate identity, permitted descriptors, source effects and control paths.

    Latent coordinates are predictive features, never biological transitions.
    No context embedding table is used, so a compatible new baseline is accepted.
    """
    def __init__(self, width=16, rank=16, epochs=120, alpha=.01, seed=41,
                 route='combined', state=True, device='cpu'):
        if route not in {'identity', 'effect', 'descriptor', 'combined'}:
            raise ValueError('Unknown intervention input route')
        self.config = dict(width=width, rank=rank, epochs=epochs, alpha=alpha, seed=seed,
                           route=route, state=state, device=device)

    def fit(self, controls, effects, available, descriptors=None, descriptor_mask=None,
            *, resume=None, protocol_fingerprint=None):
        import torch
        from torch import nn
        from pathlib import Path
        from opal.artifacts import array_hash, fingerprint, sha256, environment
        torch.set_num_threads(1)
        torch.manual_seed(self.config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config['seed'])
        self.reference = MainReference().fit(controls, effects, available)
        rows, target = np.nonzero(available)
        y = effects[rows, target]
        self.output_basis = basis(y - self.reference.means[target], self.config['rank'])
        self.effect_basis = basis(self.reference.means, self.config['rank'])
        t, g = self.reference.means.shape
        descriptors = np.zeros((t, 1)) if descriptors is None else np.asarray(descriptors, dtype=float)
        descriptor_mask = np.zeros_like(descriptors, bool) if descriptor_mask is None else np.asarray(descriptor_mask, bool)
        if descriptors.shape != descriptor_mask.shape or descriptors.shape[0] != t or not np.isfinite(descriptors).all():
            raise ValueError('Descriptors need finite values and explicit availability masks')
        # Mask before normalization. Missing values cannot influence fitted features.
        masked = np.where(descriptor_mask, descriptors, 0.)
        self.descriptor_center = masked.mean(0)
        self.descriptor_scale = np.maximum(masked.std(0), 1.)
        desc = np.concatenate([np.where(descriptor_mask, (masked - self.descriptor_center) / self.descriptor_scale, 0.), descriptor_mask], 1)
        identities = np.eye(t)
        source_effects = self.reference.means @ self.effect_basis
        self.inputs = {'identity': identities, 'effect': source_effects, 'descriptor': desc}
        route = self.config['route']
        active = list(self.inputs) if route == 'combined' else [route]
        self.active = active
        width = self.config['width']
        self.encoders = nn.ModuleDict({k: nn.Linear(self.inputs[k].shape[1], width) for k in active})
        self.state_encoder = nn.Linear(self.reference.z.shape[1], width)
        self.head = nn.Sequential(nn.Linear(3 * width, width), nn.Tanh(), nn.Linear(width, self.output_basis.shape[1]))
        # A source-average start stabilizes very small supervision sets.
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.modules = nn.ModuleDict({'interventions': self.encoders, 'state': self.state_encoder, 'main': self.head})
        self.modules.to(self.config['device'])
        x = torch.tensor(self.reference.z[rows], dtype=torch.float32, device=self.config['device'])
        ti = torch.tensor(target, dtype=torch.long, device=self.config['device'])
        target_coeff = torch.tensor((y - self.reference.means[target]) @ self.output_basis, dtype=torch.float32, device=self.config['device'])
        optimizer = torch.optim.AdamW(self.modules.parameters(), lr=.01, weight_decay=self.config['alpha'])
        dependencies = environment()['packages']
        signature = fingerprint({'config': self.config, 'data': array_hash(controls, effects, available, masked, descriptor_mask),
            'protocol': protocol_fingerprint, 'implementation': sha256(__file__),
            'dependencies': {k: dependencies[k] for k in ['numpy', 'scipy', 'torch']}})
        start = 0
        if resume is not None and Path(resume).exists():
            saved = torch.load(resume, weights_only=True, map_location=self.config['device'])
            if saved['fingerprint'] != signature:
                raise ValueError('Resume data or protocol fingerprint mismatch')
            self.modules.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['rng'].cpu())
            if torch.cuda.is_available() and saved['cuda_rng']:
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            start = saved['epoch']
        self.loss_history = []
        for epoch in range(start, self.config['epochs']):
            optimizer.zero_grad()
            loss = (self.coefficients(ti, x) - target_coeff).square().mean()
            loss.backward(); optimizer.step()
            self.loss_history.append(float(loss.detach().cpu()))
            if resume is not None and (epoch + 1 == self.config['epochs'] or (epoch + 1) % 25 == 0):
                destination = Path(resume); destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix('.part')
                torch.save({'fingerprint': signature, 'epoch': epoch + 1, 'model': self.modules.state_dict(),
                    'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'preprocessing': self.preprocessing()}, temporary)
                temporary.replace(destination)
        self.modules.eval()
        return self

    def preprocessing(self):
        # Torch-safe primitives in resumable training state; inference uses NumPy.
        return {k: v.tolist() for k, v in self.numpy_state().items()}

    def encode_interventions(self, targets):
        import torch
        z = sum(layer(torch.as_tensor(self.inputs[k], dtype=torch.float32, device=self.config['device'])[targets])
                for k, layer in self.encoders.items())
        return torch.tanh(z)

    def coefficients(self, targets, state):
        import torch
        z = self.encode_interventions(targets)
        c = torch.tanh(self.state_encoder(state)) if self.config['state'] else torch.zeros_like(z)
        return self.head(torch.cat([z, c, z * c], 1))

    def predict(self, controls, targets):
        import torch
        targets = np.asarray(targets, dtype=int)
        # Validate exact-response support even for identity/descriptor ablations.
        self.reference.predict(controls, targets)
        with torch.no_grad():
            state = torch.tensor(self.reference.encode_control(controls), dtype=torch.float32, device=self.config['device'])
            target = torch.tensor(targets, dtype=torch.long, device=self.config['device'])
            coeff = self.coefficients(target, state).cpu().numpy()
        return self.reference.means[targets] + coeff @ self.output_basis.T

    def numpy_state(self):
        state = {'control_center': self.reference.center, 'control_basis': self.reference.control_basis,
                 'control_scale': self.reference.scale, 'source_controls': self.reference.controls,
                 'target_means': self.reference.means, 'target_support': self.reference.available.sum(0),
                 'output_basis': self.output_basis, 'effect_basis': self.effect_basis,
                 'descriptor_center': self.descriptor_center, 'descriptor_scale': self.descriptor_scale}
        state.update({'input_' + k: v for k, v in self.inputs.items()})
        state.update({'weight_' + k: v.detach().cpu().numpy() for k, v in self.modules.state_dict().items()})
        return state


class SymmetricResidual:
    """Residual head uses sum/difference or sum/difference/product, with state."""
    def __init__(self, main, construction='product', state=True, epochs=120, seed=41):
        if construction not in {'sum_difference', 'product'}:
            raise ValueError('Unsupported symmetric construction')
        self.main, self.construction, self.state, self.epochs, self.seed = main, construction, state, epochs, seed

    def features(self, pairs, state):
        import torch
        zi = self.main.encode_interventions(pairs[:, 0]); zj = self.main.encode_interventions(pairs[:, 1])
        features = [zi + zj, torch.abs(zi - zj)]
        if self.construction == 'product':
            features.append(zi * zj)
        if self.state:
            features.append(torch.tanh(self.main.state_encoder(state)))
        return torch.cat(features, 1)

    def fit(self, controls, pairs, responses, measured_singles, *, joint=False, finetune_encoders=True,
            single_controls=None, single_targets=None, single_responses=None):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        device = self.main.config['device']
        pairs_t = torch.tensor(pairs, dtype=torch.long, device=device)
        state = torch.tensor(self.main.reference.encode_control(controls), dtype=torch.float32, device=device)
        residual = responses - measured_singles.sum(1)
        self.center = residual.mean(0)
        self.decoder = basis(residual - self.center, self.main.config['rank'])
        self.head = nn.Sequential(nn.Linear(self.features(pairs_t, state).shape[1], 16), nn.Tanh(), nn.Linear(16, self.decoder.shape[1])).to(device)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        if joint and not finetune_encoders:
            for module in [self.main.encoders, self.main.state_encoder]:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        parameters = list(self.head.parameters()) + ([p for p in self.main.modules.parameters() if p.requires_grad] if joint else [])
        optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=.01)
        target = torch.tensor((residual - self.center) @ self.decoder, dtype=torch.float32, device=device)
        if joint:
            if single_controls is None or single_targets is None or single_responses is None:
                raise ValueError('Joint fitting requires independent single supervision')
            sx = torch.tensor(self.main.reference.encode_control(single_controls), dtype=torch.float32, device=device)
            st = torch.tensor(single_targets, dtype=torch.long, device=device)
            sy = torch.tensor(single_responses, dtype=torch.float32, device=device)
            mean = torch.tensor(self.main.reference.means, dtype=torch.float32, device=device)
            md = torch.tensor(self.main.output_basis, dtype=torch.float32, device=device)
            rd = torch.tensor(self.decoder, dtype=torch.float32, device=device)
            py = torch.tensor(responses, dtype=torch.float32, device=device)
            rc = torch.tensor(self.center, dtype=torch.float32, device=device)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            f = self.features(pairs_t, state)
            if not joint:
                f = f.detach()
            coeff = self.head(f)
            if joint:
                pm = sum(mean[pairs_t[:, i]] + self.main.coefficients(pairs_t[:, i], state) @ md.T for i in [0, 1])
                pair_loss = (pm + rc + coeff @ rd.T - py).square().mean()
                single_loss = (mean[st] + self.main.coefficients(st, sx) @ md.T - sy).square().mean()
                loss = pair_loss + single_loss
            else:
                loss = (coeff - target).square().mean()
            loss.backward(); optimizer.step()
        self.head.eval(); self.main.modules.eval()
        self.joint = joint
        return self

    def predict_residual(self, controls, pairs):
        import torch
        with torch.no_grad():
            state = torch.tensor(self.main.reference.encode_control(controls), dtype=torch.float32, device=self.main.config['device'])
            pairs = torch.tensor(pairs, dtype=torch.long, device=self.main.config['device'])
            coefficients = self.head(self.features(pairs, state)).cpu().numpy()
        return self.center + coefficients @ self.decoder.T
