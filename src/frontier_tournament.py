"""Prospective synthetic architecture tournament, independent of legacy contracts.

No biological loader is provided. Fitters receive only charged TRAIN/VALIDATION
labels. Safe checkpoints replay on the pinned numerical stack; no historical
compatibility or biological identification is implied.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .block_completion import ScalarCompletion, checked_pairs
from .evaluation_precision import metrics_v3

CONTRACT = 'opal-frontier-synthetic-v1'
WORLDS = ('additive', 'lowrank', 'nonlinear_single', 'partner_curve', 'sparse',
          'guide_batch', 'technical_persistent', 'technical_redrawn')
SEEDS = (261101, 261102, 261103)
SHUFFLES = (261201, 261202, 261203)


def pair_order(pairs, salt):
    """Outcome-independent, orientation-independent full-digest ordering."""
    return np.array(sorted(range(len(pairs)), key=lambda i: hashlib.sha256(
        f'{salt}|{min(pairs[i])}|{max(pairs[i])}'.encode()).hexdigest()), dtype=np.int64)


def make_world(world, seed, noise, mask='complete', genes=60):
    if world not in WORLDS or mask not in ('complete', 'independent', 'input_quality'):
        raise ValueError('Unknown synthetic world/mask')
    if genes < 10 or noise not in (0., .1):
        raise ValueError('Unsupported synthetic size/noise')
    # Streams are independent of causal label, missingness and noise setting.
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(7)]
    g, factors, errors, splits, masks, sparse, guide = streams
    singles_true = g.normal(0, .2, genes)
    singles = singles_true + noise/2*g.normal(size=genes)
    z = singles_true/.2
    pairs = np.array([(a, b) for a in range(genes) for b in range(a+1, genes)], dtype=np.int64)
    a, b = pairs.T
    count = len(pairs)
    u, v, ur, vr = factors.normal(size=(4, genes, 2))
    # Population, not realized TEST, normalization. No outcome-dependent RMS.
    interaction = .15*((u[a]*u[b]).sum(1)-(v[a]*v[b]).sum(1))
    redrawn = .15*((ur[a]*ur[b]).sum(1)-(vr[a]*vr[b]).sum(1))
    effect = np.zeros((2, count))
    if world in ('lowrank', 'technical_persistent', 'technical_redrawn'):
        effect[:] = interaction
        if world == 'technical_redrawn':
            effect[1] = redrawn
    elif world == 'nonlinear_single':
        effect[:] = .2*np.tanh(z[a]+z[b])+.05*(z[a]-z[b])**2
    elif world == 'partner_curve':
        q = factors.normal(size=genes)
        effect[:] = .15*(q[a]*np.tanh(z[b])+q[b]*np.tanh(z[a]))
    elif world == 'sparse':
        effect[:] = sparse.binomial(1, .05, count)*sparse.normal(0, .5, count)
    biological_truth = singles_true[a]+singles_true[b]+effect
    # Four construct measurements are correlated, never independent initiations.
    shared = errors.normal(size=(2, count, 1))
    individual = errors.normal(size=(2, count, 4))
    constructs = biological_truth[:, :, None]+noise*np.sqrt(.5)*(shared+individual)
    guide_offsets = guide.normal(0, .12, (genes, 2))
    if world == 'guide_batch':
        for k in range(4):
            constructs[:, :, k] += guide_offsets[a, k % 2]+guide_offsets[b, (k//2) % 2]
        constructs[1] += .1
    # Assay conditional mean includes nuisance; retain latent biological mean too.
    truth = biological_truth.copy()
    if world == 'guide_batch':
        truth += guide_offsets[a].mean(1)+guide_offsets[b].mean(1)
        truth[1] += .1
    observed = constructs.mean(2)
    permutation = splits.permutation(count)
    train_end, val_end = count//2, count//2+count//5
    train, validation, test = permutation[:train_end], permutation[train_end:val_end], permutation[val_end:]
    quality = g.uniform(size=genes)
    available = np.ones(count, dtype=bool)
    if mask == 'independent':
        available = masks.uniform(size=count) >= .1
    elif mask == 'input_quality':
        # Declared pre-outcome quality rule, same across all worlds/noise levels.
        available = np.minimum(quality[a], quality[b]) >= .1
    return dict(pairs=pairs, singles=singles, features=singles[:, None].copy(),
                observed=observed, truth=truth, biological_truth=biological_truth,
                constructs=constructs, train=train, validation=validation, test=test,
                available=available, quality=quality, guide_labels=np.arange(genes)[:, None],
                guide_offsets=guide_offsets)


def budget_rows(data, fraction):
    if fraction not in (.25, 1.):
        raise ValueError('Unknown registered budget')
    selected = []
    nominal = []
    for role in ('train', 'validation'):
        rows = np.asarray(data[role])
        ordered = rows[pair_order(data['pairs'][rows], 'frontier-budget-'+role)]
        charged = ordered[:int(np.floor(len(rows)*fraction))]
        nominal.append(len(charged))
        selected.append(charged[np.asarray(data['available'])[charged]])
    return (*selected, dict(nominal_train=nominal[0], nominal_validation=nominal[1],
                           used_train=len(selected[0]), used_validation=len(selected[1]),
                           charged_pair_union=sum(nominal)))


def _base_design(singles, pairs, nodes=False):
    values = singles[pairs]
    x = np.column_stack((values.sum(1), np.abs(values[:, 0]-values[:, 1]), values.prod(1)))
    if nodes:
        incidence = np.zeros((len(pairs), len(singles)))
        np.add.at(incidence, (np.arange(len(pairs))[:, None], pairs), 1.)
        x = np.column_stack((x, incidence))
    return x


def gene_kernel(singles, p, q):
    return .5*(p[:, None] == q[None, :])+.5*np.exp(-.5*(singles[p, None]-singles[None, q])**2)


def pair_kernel(singles, pairs, landmarks):
    a, b = pairs.T
    u, v = landmarks.T
    return .5*(gene_kernel(singles, a, u)*gene_kernel(singles, b, v)+
               gene_kernel(singles, a, v)*gene_kernel(singles, b, u))


def _predict_state(state, pairs):
    kind = str(state['kind'])
    singles = state['singles']
    if kind in ('zero', 'mean', 'sum', 'minimum'):
        if kind == 'zero': return np.zeros(len(pairs))
        if kind == 'mean': return np.full(len(pairs), float(state['mean']))
        if kind == 'sum': return singles[pairs].sum(1)
        return singles[pairs].min(1)
    if kind == 'completion':
        return state['matrix'][pairs[:, 0], pairs[:, 1]]
    if kind == 'neural':
        h = np.maximum(state['embedding']@state['encoder_w1']+state['encoder_b1'], 0)
        h = h@state['encoder_w2']+state['encoder_b2']
        a, b = h[pairs[:, 0]], h[pairs[:, 1]]
        x = np.column_stack((a+b, a*b))
        x = np.maximum(x@state['head_w1']+state['head_b1'], 0)
        return (x@state['head_w2']+state['head_b2']).ravel()*float(state['target_scale'])
    if kind == 'kernel':
        x = pair_kernel(state['z'], pairs, state['landmarks'])@state['basis']
    else:
        x = _base_design(singles, pairs, bool(state['nodes']))
        if kind == 'flexible':
            coordinates = np.sort(state['z'][pairs], axis=1)
            distances = ((coordinates[:, None]-state['centers'][None])**2).sum(2)
            x = np.column_stack((x, np.exp(-distances/(2*float(state['bandwidth'])**2))))
    offset = singles[pairs].sum(1) if kind != 'kernel' else np.zeros(len(pairs))
    return offset+float(state['offset'])+((x-state['center'])/state['scale'])@state['coefficient']


class Predictor:
    """Safe array checkpoint; TEST labels are neither inputs nor stored state."""
    def __init__(self, state, base=None):
        self.state = {k: np.array(v, copy=True) for k, v in state.items()}
        self.base = base

    def supported(self, pairs):
        p = checked_pairs(pairs, len(self.state['singles']))
        return self.state['support'][p].all(1)

    def predict(self, pairs):
        p = checked_pairs(pairs, len(self.state['singles']))
        if not self.supported(p).all():
            raise ValueError('Query genes require two distinct TRAIN partners')
        result = _predict_state(self.state, p)
        if self.base is not None:
            result = self.base.predict(p)+result
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite prediction')
        return result

    def save(self, path):
        path = Path(path)
        if path.exists(): raise FileExistsError(path)
        states = {'model__'+k: v for k, v in self.state.items()}
        if self.base is not None:
            if self.base.base is not None: raise ValueError('Only one residual layer supported')
            states.update({'base__'+k: v for k, v in self.base.state.items()})
        with path.open('xb') as stream:
            np.savez_compressed(stream, contract=CONTRACT, **states)

    @classmethod
    def load(cls, path, expected_sha256=None):
        path = Path(path)
        if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('Checkpoint identity mismatch')
        with np.load(path, allow_pickle=False) as z:
            if str(z['contract']) != CONTRACT: raise ValueError('Unknown checkpoint contract')
            if any(k != 'contract' and not k.startswith(('model__', 'base__')) for k in z.files):
                raise ValueError('Unknown checkpoint field')
            states = [{k[len(prefix):]: z[k].copy() for k in z.files if k.startswith(prefix)}
                      for prefix in ('model__', 'base__')]
        model = cls(states[0], cls(states[1]) if states[1] else None)
        for obj in (model, model.base):
            if obj is None: continue
            s = obj.state
            if (str(s.get('kind')) not in {'zero','mean','sum','minimum','linear','flexible','kernel','completion','neural'}
                    or s['singles'].ndim != 1 or s['support'].shape != s['singles'].shape
                    or s['support'].dtype != bool):
                raise ValueError('Malformed checkpoint schema')
            if any(v.dtype.kind in 'fc' and not np.isfinite(v).all() for v in s.values()):
                raise ValueError('Nonfinite checkpoint')
            if 'scale' in s and np.any(s['scale'] <= 0): raise ValueError('Invalid scale')
            kind,n = str(s['kind']),len(s['singles'])
            common = {'kind','singles','support'}
            if s['singles'].dtype.kind not in 'iuf': raise ValueError('Real singles required')
            def shape(name, expected):
                if name not in s or s[name].shape != expected:
                    raise ValueError('Checkpoint shape mismatch: '+name)
            if kind in ('zero','mean','sum','minimum'):
                allowed = common|{'mean'};shape('mean',())
            elif kind == 'completion':
                allowed = common|{'matrix'};shape('matrix',(n,n))
                if not np.array_equal(s['matrix'],s['matrix'].T): raise ValueError('Asymmetric completion')
            elif kind == 'neural':
                shapes = {'embedding':(n,4),'encoder_w1':(4,8),'encoder_b1':(8,),
                          'encoder_w2':(8,4),'encoder_b2':(4,),'head_w1':(8,16),
                          'head_b1':(16,),'head_w2':(16,1),'head_b2':(1,),'target_scale':()}
                allowed = common|set(shapes)
                for name,expected in shapes.items(): shape(name,expected)
                if s['target_scale'] <= 0: raise ValueError('Nonpositive target scale')
            else:
                allowed = common|{'center','scale','coefficient','offset'}
                shape('offset',())
                if kind == 'kernel':
                    allowed |= {'z','landmarks','basis'};shape('z',(n,))
                    landmarks = checked_pairs(s['landmarks'],n,unique=True)
                    if s['basis'].ndim != 2 or s['basis'].shape[0] != len(landmarks):
                        raise ValueError('Kernel basis shape mismatch')
                    width = s['basis'].shape[1]
                else:
                    allowed |= {'nodes'};shape('nodes',())
                    if s['nodes'].dtype != bool: raise ValueError('Boolean node flag required')
                    width = 3+(n if bool(s['nodes']) else 0)
                    if kind == 'flexible':
                        allowed |= {'z','centers','bandwidth'};shape('z',(n,));shape('bandwidth',())
                        if s['centers'].ndim != 2 or s['centers'].shape[1] != 2 or s['bandwidth'] <= 0:
                            raise ValueError('Flexible centers/bandwidth mismatch')
                        width += len(s['centers'])
                for name in ('center','scale','coefficient'): shape(name,(width,))
            if set(s) != allowed: raise ValueError('Unexpected checkpoint fields')
        if model.base is not None and (not np.array_equal(model.state['singles'],model.base.state['singles'])
                or not np.array_equal(model.state['support'],model.base.state['support'])):
            raise ValueError('Residual/base input mismatch')
        return model


def _ridge(state, x, y, alpha):
    state.update(center=x.mean(0), scale=np.maximum(x.std(0), 1e-8), offset=float(y.mean()))
    normalized = (x-state['center'])/state['scale']
    state['coefficient'] = np.linalg.solve(normalized.T@normalized+alpha*np.eye(x.shape[1]),
                                           normalized.T@(y-state['offset']))
    return Predictor(state)


def _neural(state, pairs, residuals, penalty, seed, steps):
    generator = torch.Generator().manual_seed(seed)
    shapes = {'embedding': (len(state['singles']), 4), 'encoder_w1': (4, 8),
              'encoder_b1': (8,), 'encoder_w2': (8, 4), 'encoder_b2': (4,),
              'head_w1': (8, 16), 'head_b1': (16,), 'head_w2': (16, 1), 'head_b2': (1,)}
    params = {}
    for name, shape in shapes.items():
        if '_b' in name: value = torch.zeros(shape, dtype=torch.float64)
        elif name == 'embedding': value = torch.randn(shape, generator=generator, dtype=torch.float64)*.05
        else:
            bound = np.sqrt(6/(shape[0]+shape[1]))
            value = torch.empty(shape, dtype=torch.float64).uniform_(-bound, bound, generator=generator)
        params[name] = torch.nn.Parameter(value)
    scale = max(float(np.std(residuals)), 1e-6)
    truth = torch.as_tensor(residuals/scale, dtype=torch.float64)
    a, b = pairs.T
    optimizer = torch.optim.Adam(list(params.values()), lr=.01)
    for _ in range(steps):
        optimizer.zero_grad()
        p = params
        h = torch.relu(p['embedding']@p['encoder_w1']+p['encoder_b1'])
        h = h@p['encoder_w2']+p['encoder_b2']
        x = torch.cat((h[a]+h[b], h[a]*h[b]), dim=1)
        pred = (torch.relu(x@p['head_w1']+p['head_b1'])@p['head_w2']+p['head_b2']).ravel()
        loss = torch.mean((pred-truth)**2)+penalty*sum(t.square().sum() for t in params.values())/sum(t.numel() for t in params.values())
        loss.backward()
        optimizer.step()
    state.update({k: v.detach().numpy().copy() for k, v in params.items()})
    state['target_scale'] = scale
    return Predictor(state)


def residual_permutation(pairs, singles, seed):
    """Four fixed TRAIN-only strata; nonzero cyclic shifts move eligible rows."""
    degree = np.bincount(pairs.ravel(), minlength=len(singles))
    minimum_degree = degree[pairs].min(1)
    single_sum = singles[pairs].sum(1)
    strata = np.column_stack((minimum_degree >= np.median(minimum_degree),
                              single_sum >= np.median(single_sum)))
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(pairs))
    permutable = 0
    for group in np.unique(strata, axis=0):
        rows = np.flatnonzero((strata == group).all(1))
        if len(rows) >= 2:
            rows = rows[pair_order(pairs[rows], 'frontier-shuffle')]
            permutation[rows] = np.roll(rows,int(rng.integers(1,len(rows))))
            permutable += len(rows)
    if permutable/len(pairs) < .8: raise ValueError('Uninformative shuffled control')
    return permutation, permutable/len(pairs)


def fit_models(singles, features, train_pairs, train_targets, validation_pairs,
               validation_targets, seed, *, controls=True, completion_steps=500, neural_steps=300):
    """Fit bounded menu with no TEST or second-initiation outcomes in scope.

    features must be the measured-single column; extra features require a new
    versioned protocol and matched reference menu rather than silent admission.
    """
    singles = np.array(singles, dtype=float, copy=True)
    features = np.asarray(features)
    if singles.ndim != 1 or not np.isfinite(singles).all(): raise ValueError('Finite singles required')
    if features.shape != (len(singles), 1) or not np.array_equal(features[:, 0], singles):
        raise ValueError('This contract permits singles and identities only')
    tp = checked_pairs(train_pairs, len(singles), unique=True)
    vp = checked_pairs(validation_pairs, len(singles), unique=True)
    if set(map(tuple, np.sort(tp, axis=1))) & set(map(tuple, np.sort(vp, axis=1))):
        raise ValueError('TRAIN/VALIDATION pair leakage')
    ty, vy = np.array(train_targets, dtype=float, copy=True), np.array(validation_targets, dtype=float, copy=True)
    if (ty.shape != (len(tp),) or vy.shape != (len(vp),) or not len(vy) or len(ty) < 3
            or not np.isfinite(ty).all() or not np.isfinite(vy).all()):
        raise ValueError('Finite nonempty aligned fitting labels required')
    degree = np.bincount(tp.ravel(), minlength=len(singles))
    support = degree >= 2
    validation_support = support[vp].all(1)
    if not validation_support.any(): raise ValueError('Validation needs some two-partner TRAIN support')
    active_singles = singles[degree > 0]
    z = (singles-active_singles.mean())/max(float(active_singles.std()), 1e-8)
    base_state = dict(singles=singles, support=support)
    records = []

    def add(family, config, model):
        prediction = singles[vp].sum(1)
        prediction[validation_support] = model.predict(vp[validation_support])
        score = metrics_v3(vy, prediction)['mse']
        records.append(dict(family=family, config=config, model=model, validation_mse=score))

    for kind in ('zero','mean','sum','minimum'):
        add('B0', kind, Predictor(dict(base_state, kind=kind, mean=float(ty.mean()))))
    for family, nodes in (('B1',False),('B2',True)):
        for alpha in (1.,10.,100.,1000.):
            state = dict(base_state, kind='linear', nodes=nodes)
            add(family, f'alpha{alpha:g}', _ridge(state, _base_design(singles,tp,nodes), ty-singles[tp].sum(1), alpha))
    center_rows = pair_order(tp, 'frontier-centers')[:min(64,len(tp))]
    coordinates = np.sort(z[tp], axis=1)
    centers = coordinates[center_rows]
    distances = np.sqrt(((centers[:,None]-centers[None])**2).sum(2))
    positive = distances[np.triu_indices(len(centers),1)]
    positive = positive[positive > 0]
    bandwidth = max(float(np.median(positive)) if len(positive) else 1., 1e-8)
    radial = np.exp(-((coordinates[:,None]-centers[None])**2).sum(2)/(2*bandwidth**2))
    for nodes in (False,True):
        x = np.column_stack((_base_design(singles,tp,nodes),radial))
        for alpha in (10.,100.):
            state = dict(base_state, kind='flexible', nodes=nodes, z=z, centers=centers, bandwidth=bandwidth)
            add('B3', f'nodes{int(nodes)}-alpha{alpha:g}', _ridge(state,x,ty-singles[tp].sum(1),alpha))
    selected_simple = min(records, key=lambda r: r['validation_mse'])
    base = selected_simple['model']
    # TRAIN labels may include degree-one genes; inference support is a query
    # policy, not a reason to drop or replace charged training observations.
    residual = ty-_predict_state(base.state,tp)
    validation_residual = vy[validation_support]-base.predict(vp[validation_support])
    landmarks = tp[center_rows]
    gram = pair_kernel(z,landmarks,landmarks)
    values, vectors = np.linalg.eigh((gram+gram.T)*.5)
    if values.min() < -1e-8: raise ValueError('Pair kernel is not PSD')
    keep = values > 1e-10
    basis = vectors[:,keep]/np.sqrt(values[keep])
    kernel_x = pair_kernel(z,tp,landmarks)@basis
    permutations = [(None, np.arange(len(tp)), 1.)]
    if controls:
        permutations += [(i,*residual_permutation(tp,singles,s)) for i,s in enumerate(SHUFFLES)]
    for shuffle, permutation, permutable in permutations:
        target = residual[permutation]
        suffix = '' if shuffle is None else f'-shuffle{shuffle}'
        completion = ScalarCompletion(rank=8,penalty=.01,steps=completion_steps,seed=seed).fit(
            tp,target,len(singles),max(float(ty.std()),1e-6),(vp[validation_support],validation_residual))
        model = Predictor(dict(base_state,kind='completion',matrix=completion.matrix),base)
        add('I0'+suffix,'rank8-penalty0.01',model)
        records[-1].update(best_step=completion.best_step,permutable=permutable)
        for alpha in (10.,100.):
            state = dict(base_state,kind='kernel',z=z,landmarks=landmarks,basis=basis)
            model = _ridge(state,kernel_x,target,alpha)
            model.base = base
            add('I1'+suffix,f'alpha{alpha:g}',model)
        for penalty in (.001,.01):
            model = _neural(dict(base_state,kind='neural'),tp,target,penalty,seed,neural_steps)
            model.base = base
            add('I2'+suffix,f'l2-{penalty:g}',model)
    return records


def select_records(records):
    selected = {}
    for record in records:
        family = record['family']
        if family not in selected or record['validation_mse'] < selected[family]['validation_mse']:
            selected[family] = record
    simple = min((r for r in records if r['family'].startswith('B')), key=lambda r:r['validation_mse'])
    policy = min((r for r in records if '-shuffle' not in r['family']), key=lambda r:r['validation_mse'])
    return selected, simple, policy


def gene_sensitivity(pairs, target, prediction, reference, draws=2000, seed=261301):
    """Conditional graph sensitivity, never biological replication uncertainty."""
    target, prediction, reference = (np.asarray(x,dtype=float) for x in (target,prediction,reference))
    delta = (prediction-target)**2-(reference-target)**2
    genes = np.unique(pairs)
    mapped = np.searchsorted(genes,pairs)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(genes),np.full(len(genes),1/len(genes)),size=draws)
    weights = counts[:,mapped[:,0]]*counts[:,mapped[:,1]]
    sums = weights.sum(1)
    values = (weights@delta)[sums > 0]/sums[sums > 0]
    leave = [float(delta[(pairs != g).all(1)].mean()) for g in genes]
    return dict(mean_difference=float(delta.mean()),upper_975=float(np.quantile(values,.975)),
                lower_025=float(np.quantile(values,.025)),leave_gene_min=min(leave),leave_gene_max=max(leave),
                draws=draws,interpretation='conditional gene sensitivity, not independent-initiation uncertainty')
