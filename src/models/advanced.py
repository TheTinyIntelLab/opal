"""Noise-aware symmetric residual networks with frozen public priors.

All response-derived fitting objects are built from the supplied training pairs.
Source banks and reference bases use allowed observed singles only.
"""
import numpy as np
import torch
from torch import nn
from scipy.linalg import solve
from opal.models.kernel import KernelResidual


def check_pairs(pairs, targets):
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu':
        raise ValueError('Pairs must be an integer (n,2) array')
    if np.any(pairs < 0) or np.any(pairs >= targets) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError('Invalid target indices or repeated target')
    return np.ascontiguousarray(pairs)


def reference_embedding(x, rank=16):
    x = np.asarray(x, np.float64)
    mean = x.mean(0)
    centered = x - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(rank, max(1, len(x)-1), len(s))
    projection = vt[:rank].T
    features = centered @ projection
    scale = np.maximum(features.std(0), 1e-6)
    return features / scale, projection, mean, scale


def signal_weights(doubles, variance, strength=1):
    if strength == 0:
        return np.ones_like(doubles)
    reliability = doubles**2 / (doubles**2 + np.maximum(variance, 1e-6))
    magnitude = np.abs(doubles)
    reference = np.maximum(np.median(magnitude, axis=1, keepdims=True), .02)
    weights = 1 + strength * reliability * np.minimum(magnitude/reference, 4)
    return weights / weights.mean(axis=1, keepdims=True)


def make_basis(reference, residual, membership, gene_prior, structured, rank=8, transfer_basis=None):
    g = residual.shape[1]
    _, _, vt = np.linalg.svd(residual - residual.mean(0), full_matrices=False)
    components = [vt[:min(rank, max(1, len(residual)-1))].T * np.sqrt(g)]
    counts = [components[0].shape[1], 0, 0]
    if structured:
        # One reference-derived direction per supported pathway, not test-derived DEGs.
        directions = []
        for mask in membership.T:
            selected = mask > 0
            if selected.sum() < 5:
                continue
            values = reference[:, selected]
            _, _, local = np.linalg.svd(values-values.mean(0), full_matrices=False)
            direction = np.zeros(g)
            direction[selected] = local[0]
            directions.append(direction*np.sqrt(g))
        if directions:
            module = np.stack(directions, axis=1)
            components.append(module)
            counts[1] = module.shape[1]
        if gene_prior.shape[1]:
            vectors = gene_prior-gene_prior.mean(0)
            u, s, _ = np.linalg.svd(vectors, full_matrices=False)
            usable = min(8, int(np.sum(s > 1e-8)))
            if usable:
                components.append(u[:, :usable]*np.sqrt(g))
                counts[2] = usable
    if transfer_basis is not None:
        transfer_basis = np.asarray(transfer_basis, np.float64)
        norms = np.linalg.norm(transfer_basis, axis=0)
        usable = norms > 1e-8
        if usable.any():
            components.append(transfer_basis[:, usable]/norms[usable]*np.sqrt(g))
            counts[2] += int(usable.sum())
    return np.concatenate(components, axis=1).astype(np.float32), counts


class SourceFusion(nn.Module):
    def __init__(self, sources, dimension, width, attention=True):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(dimension, width) for _ in range(sources)])
        self.tokens = nn.Parameter(torch.zeros(sources, width))
        self.score = nn.Linear(width, 1)
        self.attention = attention

    def forward(self, bank, mask, reliability):
        items = torch.stack([torch.tanh(layer(bank[:, i])+self.tokens[i])
                             for i, layer in enumerate(self.projections)], dim=1)
        if self.attention:
            logits = self.score(items).squeeze(-1)
            # Only the measured-source reliability is known at deployment.
            adjustment = torch.zeros_like(logits)
            adjustment[:, 0] = torch.log(reliability.clamp_min(.02))
            logits = (logits+adjustment).masked_fill(~mask, -1e4)
            weights = torch.softmax(logits, dim=1)
        else:
            weights = mask.float()/mask.float().sum(1, keepdim=True).clamp_min(1)
        return (items*weights[:, :, None]).sum(1), weights


class SymmetricModuleNetwork(nn.Module):
    def __init__(self, sources, dimension, width, counts, attention=True, structured=False):
        super().__init__()
        self.fusion = SourceFusion(sources, dimension, width, attention)
        self.hidden = nn.Sequential(nn.Linear(3*width+2, width), nn.Tanh())
        self.output = nn.Linear(width, sum(counts))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.counts, self.structured = counts, structured
        self.module_gate = nn.Linear(width, counts[1]) if counts[1] else None

    def forward(self, bank, mask, reliability, pairs):
        latent, weights = self.fusion(bank, mask, reliability)
        a, b = latent[pairs[:, 0]], latent[pairs[:, 1]]
        ra, rb = reliability[pairs[:, 0]], reliability[pairs[:, 1]]
        features = torch.cat([a+b, (a-b).abs(), a*b, (ra+rb)[:, None], (ra-rb).abs()[:, None]], dim=1)
        hidden = self.hidden(features)
        coefficients = self.output(hidden)
        if self.module_gate is not None:
            start, stop = self.counts[0], self.counts[0]+self.counts[1]
            coefficients = torch.cat([coefficients[:, :start],
                coefficients[:, start:stop]*torch.sigmoid(self.module_gate(hidden)),
                coefficients[:, stop:]], dim=1)
        return coefficients, weights


class LinearSymmetricDecoder(nn.Module):
    """Fixed source features with a linear output-basis map, for loss controls."""
    def __init__(self,sources,dimension,counts,clip=None):
        super().__init__()
        self.clip=clip
        if clip is not None and (not np.isfinite(clip) or clip<=0):raise ValueError('Invalid linear feature bound')
        feature_size=3*sources*dimension+2
        self.register_buffer('feature_mean',torch.zeros(feature_size))
        self.register_buffer('feature_scale',torch.ones(feature_size))
        self.output=nn.Linear(feature_size,sum(counts))
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def features(self,bank,mask,reliability,pairs):
        latent=(bank*mask[:,:,None]).reshape(len(bank),-1)
        a,b=latent[pairs[:,0]],latent[pairs[:,1]]
        ra,rb=reliability[pairs[:,0]],reliability[pairs[:,1]]
        features=torch.cat([a+b,(a-b).abs(),a*b,(ra+rb)[:,None],(ra-rb).abs()[:,None]],dim=1)
        return features
    def fit_scaler(self,bank,mask,reliability,pairs):
        with torch.no_grad():
            features=self.features(bank,mask,reliability,pairs)
            self.feature_mean.copy_(features.mean(0))
            self.feature_scale.copy_(features.std(0,unbiased=False).clamp_min(1e-3))
    def forward(self,bank,mask,reliability,pairs):
        features=(self.features(bank,mask,reliability,pairs)-self.feature_mean)/self.feature_scale
        if self.clip is not None:features=features.clamp(-self.clip,self.clip)
        weights=mask.float()/mask.float().sum(1,keepdim=True).clamp_min(1)
        return self.output(features),weights


class AdvancedResidual:
    """Observed-additive output plus a simple expert and structured correction."""
    def __init__(self, singles, resources, config):
        self.singles = np.asarray(singles, np.float64)
        self.config = dict(config)
        self.resources = {k:np.asarray(v) for k,v in resources.items()}
        if self.singles.ndim != 2 or not np.isfinite(self.singles).all():
            raise ValueError('Invalid observed singles')
        for k in ['bank', 'source_mask', 'single_variance', 'membership', 'gene_prior', 'single_draw_features']:
            if k not in resources:
                raise ValueError('Missing architecture resource: '+k)
        if resources['bank'].shape[0] != len(self.singles) or resources['membership'].shape[0] != self.singles.shape[1]:
            raise ValueError('Reference resource order mismatch')
        if not np.isfinite(resources['bank']).all() or not np.isfinite(resources['single_variance']).all():
            raise ValueError('Nonfinite reference resources')
        if np.any(resources['single_variance'] < 0):
            raise ValueError('Negative single variance')

    def fit(self, pairs, doubles, double_variance=None, seed=0):
        pairs = check_pairs(pairs, len(self.singles))
        doubles = np.asarray(doubles, np.float64)
        if len(pairs) < 3 or doubles.shape != (len(pairs), self.singles.shape[1]) or not np.isfinite(doubles).all():
            raise ValueError('Invalid training responses')
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        rng = np.random.default_rng(seed)
        noisy = self.config.get('noise', False)
        sources = self.resources['bank'].shape[1] if self.config.get('priors', False) else 1
        self.bank = self.resources['bank'][:, :sources].astype(np.float32).copy()
        self.mask = self.resources['source_mask'][:, :sources].astype(bool)
        energy = np.mean(self.singles**2, axis=1)
        noise = self.resources['single_variance'].mean(1)
        self.reliability = (energy/(energy+noise+1e-8)).astype(np.float32) if noisy else np.ones(len(energy), np.float32)
        if noisy:
            self.bank[:, 0] *= self.reliability[:, None]
        self.base = KernelResidual(self.singles, {'kind':self.config.get('expert','global'),
                    'alpha':self.config.get('alpha',10), 'bandwidth':1}).fit(pairs,doubles)
        residual = doubles-self.base.predict(pairs)
        self.basis, self.counts = make_basis(self.singles,residual,self.resources['membership'],
            self.resources['gene_prior'],self.config.get('structured',False), self.config.get('rank',8),
            self.resources.get('transfer_output_basis') if self.config.get('transfer_decoder',self.config.get('transfer',False)) else None)
        self.network = SymmetricModuleNetwork(sources,self.bank.shape[2],self.config.get('width',32),
            self.counts,self.config.get('attention',True),self.config.get('structured',False))
        if self.config.get('linear',False):
            self.network=LinearSymmetricDecoder(sources,self.bank.shape[2],self.counts,self.config.get('linear_clip'))
        if self.config.get('transfer_encoder',self.config.get('transfer',False)):
            if sources < 2 or 'transfer_weight' not in self.resources:
                raise ValueError('Independent transfer encoder missing')
            layer=self.network.fusion.projections[1]
            layer.load_state_dict({'weight':torch.tensor(self.resources['transfer_weight']),
                                   'bias':torch.tensor(self.resources['transfer_bias'])})
        weights = signal_weights(doubles, np.zeros_like(doubles) if double_variance is None else double_variance,
                                 self.config.get('signal_loss',0)).astype(np.float32)
        basis = self.basis.astype(np.float64)
        # This is exactly the gene-space weighted quadratic, not a PCA-score surrogate.
        quadratic = np.einsum('gl,ng,gk->nlk',basis,weights,basis,optimize=True)/basis.shape[0]
        cross = ((weights*residual)@basis)/basis.shape[0]
        q = torch.tensor(quadratic,dtype=torch.float32)
        c = torch.tensor(cross,dtype=torch.float32)
        bank = torch.tensor(self.bank)
        mask = torch.tensor(self.mask)
        reliability = torch.tensor(self.reliability)
        p = torch.tensor(pairs,dtype=torch.long)
        if self.config.get('linear',False):self.network.fit_scaler(bank,mask,reliability,p)
        optimizer = torch.optim.AdamW(self.network.parameters(),lr=.01,weight_decay=self.config.get('decay',.01))
        penalty = torch.ones(self.basis.shape[1])
        if self.counts[2]:
            penalty[-self.counts[2]:] = 4
        if self.config.get('linear',False):
            # Solve the convex gene-space objective, avoiding an under-converged Adam control.
            from scipy.sparse.linalg import LinearOperator,cg
            with torch.no_grad():
                f=self.network.features(bank,mask,reliability,p).numpy().astype(np.float64)
            f=(f-self.network.feature_mean.numpy())/self.network.feature_scale.numpy()
            if self.network.clip is not None:f=np.clip(f,-self.network.clip,self.network.clip)
            f=np.column_stack([f,np.ones(len(f))])
            penalty_strength=self.config.get('coefficient_penalty',.01)
            qfit=quadratic+np.diag(penalty.numpy())*penalty_strength/self.basis.shape[1]
            ridge=self.config.get('linear_ridge',.01)
            shape=(f.shape[1],self.basis.shape[1])
            def multiply(vector):
                z=f@vector.reshape(shape)
                return (f.T@np.einsum('nlk,nk->nl',qfit,z)/len(f)+ridge*vector.reshape(shape)).ravel()
            operator=LinearOperator((np.prod(shape),np.prod(shape)),matvec=multiply,dtype=np.float64)
            rhs=(f.T@cross/len(f)).ravel()
            solution,info=cg(operator,rhs,rtol=1e-7,atol=1e-10,maxiter=500)
            if info!=0:raise ValueError('Linear quadratic solver did not converge')
            solution=solution.reshape(shape)
            with torch.no_grad():
                self.network.output.weight.copy_(torch.tensor(solution[:-1].T,dtype=torch.float32))
                self.network.output.bias.copy_(torch.tensor(solution[-1],dtype=torch.float32))
            self.training_log={'solver':'conjugate gradient, exact frozen-reference gene-space quadratic',
                'relative_tolerance':1e-7,'linear_ridge':ridge,'bootstrap_augmentation':False,
                'basis_counts':self.counts,'parameters':sum(p.numel() for p in self.network.parameters())}
            self.network.eval()
            return self
        history=[]
        for epoch in range(self.config.get('epochs',120)):
            training_bank=bank
            if noisy and len(self.resources['single_draw_features']):
                draw=self.resources['single_draw_features'][rng.integers(len(self.resources['single_draw_features']))]
                training_bank=bank.clone()
                # A jointly sampled reference draw is shared across all training pairs.
                training_bank[:,0]=torch.tensor(draw)*reliability[:,None]
            z,_=self.network(training_bank,mask,reliability,p)
            loss=(torch.einsum('nl,nlk,nk->n',z,q,z)-2*(z*c).sum(1)).mean()
            loss=loss+self.config.get('coefficient_penalty',.01)*(z.square()*penalty).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(),1)
            optimizer.step()
            if epoch in (0,self.config.get('epochs',120)-1): history.append(float(loss.detach()))
        self.training_log={'epochs':self.config.get('epochs',120),'loss_endpoints':history,
            'basis_counts':self.counts,'parameters':sum(p.numel() for p in self.network.parameters()),
            'uncertainty':'technical reference reliability, not calibrated predictive uncertainty'}
        self.network.eval()
        return self

    def predict(self,pairs,mixture=1):
        pairs=check_pairs(pairs,len(self.singles))
        with torch.no_grad():
            z,_=self.network(torch.tensor(self.bank),torch.tensor(self.mask),torch.tensor(self.reliability),
                             torch.tensor(pairs,dtype=torch.long))
        return self.base.predict(pairs)+mixture*z.numpy()@self.basis.T

    def attention_weights(self):
        with torch.no_grad():
            if self.config.get('linear',False):
                weights=torch.tensor(self.mask).float()
                weights=weights/weights.sum(1,keepdim=True).clamp_min(1)
            else:
                _,weights=self.network.fusion(torch.tensor(self.bank),torch.tensor(self.mask),torch.tensor(self.reliability))
        return weights.numpy()

    def state(self):
        arrays={'bank':self.bank,'mask':self.mask,'reliability':self.reliability,'basis':self.basis,
                'counts':np.array(self.counts),'singles':self.singles}
        arrays.update({'base__'+k:v for k,v in self.base.state().items()})
        arrays.update({'net__'+k:v.detach().numpy() for k,v in self.network.state_dict().items()})
        return arrays

    @classmethod
    def from_state(cls,arrays,config):
        model=cls.__new__(cls)
        model.config=dict(config)
        for k in ['bank','mask','reliability','basis','singles']:
            setattr(model,k,np.asarray(arrays[k]))
        model.counts=arrays['counts'].astype(int).tolist()
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Nonfinite advanced checkpoint')
        model.base=KernelResidual(model.singles,{'kind':config.get('expert','global'),
            'alpha':config.get('alpha',10),'bandwidth':1}).restore({k.removeprefix('base__'):v for k,v in arrays.items() if k.startswith('base__')})
        model.network=SymmetricModuleNetwork(model.bank.shape[1],model.bank.shape[2],config.get('width',32),
            model.counts,config.get('attention',True),config.get('structured',False))
        if config.get('linear',False):
            model.network=LinearSymmetricDecoder(model.bank.shape[1],model.bank.shape[2],model.counts,config.get('linear_clip'))
        model.network.load_state_dict({k.removeprefix('net__'):torch.from_numpy(v) for k,v in arrays.items() if k.startswith('net__')},strict=True)
        if model.bank.shape[0]!=len(model.singles) or model.basis.shape!=(model.singles.shape[1],sum(model.counts)):
            raise ValueError('Advanced checkpoint schema mismatch')
        model.network.eval()
        return model
