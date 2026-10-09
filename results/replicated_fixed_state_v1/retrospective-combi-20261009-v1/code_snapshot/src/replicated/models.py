"""Bounded shared, component-additive and symmetric pair kernel menu."""
import json
import numpy as np
from scipy.linalg import solve


def balanced_weights(pair_ids, cultures):
    """Equal cultures and pairs, invariant to repeated technical rows."""
    pair_ids, cultures = np.asarray(pair_ids), np.asarray(cultures)
    w=np.zeros(len(pair_ids))
    for c in np.unique(cultures):
        pairs=np.unique(pair_ids[cultures==c])
        for p in pairs:
            idx=(cultures==c)&(pair_ids==p)
            w[idx]=1/(len(np.unique(cultures))*len(pairs)*idx.sum())
    return w


class ComponentAdditive:
    """Centered incidence ridge, zero correction for unsupported components."""
    def __init__(self, alpha=1):
        if alpha<=0: raise ValueError('Positive penalty required')
        self.alpha=alpha

    def design(self, pairs):
        index={c:i for i,c in enumerate(self.components)}
        x=np.zeros((len(pairs),len(index)))
        for i,p in enumerate(pairs):
            for c in json.loads(str(p)):
                if c in index: x[i,index[c]]+=1
        return x

    def fit(self, pairs, residuals, weights=None):
        self.components=sorted({c for p in pairs for c in json.loads(str(p))})
        x=self.design(pairs)
        y=np.asarray(residuals,float)
        w=np.ones(len(x))/len(x) if weights is None else np.asarray(weights)/np.sum(weights)
        self.center=w@x; self.mean=w@y
        xc=x-self.center
        self.coefficients=solve(xc.T@(w[:,None]*xc)+self.alpha*np.eye(x.shape[1]),
                                xc.T@(w[:,None]*(y-self.mean)),assume_a='pos')
        # Every pair has incidence sum two; centering and ridge remove the
        # intercept/component null direction. Unsupported targets stay zero.
        self.coefficients-=self.coefficients.mean(0)
        self.intercept=self.mean-self.center@self.coefficients
        return self

    def predict(self, pairs):
        return self.intercept+self.design(pairs)@self.coefficients


class ProfileKernel:
    """Existing global/summed/product kernel algebra on row-specific references."""
    def __init__(self, config):
        self.config=dict(config)
        if config['kind'] not in {'shared_offset','global','single_sum','global_plus_pair','pair_product'}:
            raise ValueError('Unknown frozen family')
        if config['alpha']<=0 or config.get('bandwidth',1)<=0:
            raise ValueError('Positive regularization/bandwidth required')

    def kernel(self,left,right):
        kind=self.config['kind']
        if kind=='shared_offset': return np.zeros((len(left),len(right)))
        l=left/self.scale; r=right/self.scale
        global_k=l.sum(1)@r.sum(1).T/l.shape[-1]
        if kind=='global':return global_k
        ks=[]
        for a,b in ((0,0),(0,1),(1,0),(1,1)):
            d=np.maximum((l[:,a]**2).sum(1)[:,None]+(r[:,b]**2).sum(1)[None,:]-2*l[:,a]@r[:,b].T,0)
            ks.append(np.exp(-d/(self.distance*self.config.get('bandwidth',1))))
        ac,ad,bc,bd=ks
        if kind=='single_sum':return (ac+ad+bc+bd)/4
        product=(ac*bd+ad*bc)/2
        return global_k+product if kind=='global_plus_pair' else product

    def fit(self, singles, responses, weights=None):
        self.references=np.asarray(singles,float).copy()
        y=np.asarray(responses,float)-self.references.sum(1)
        if self.references.ndim!=3 or self.references.shape[1]!=2 or y.shape!=self.references[:,0].shape or not np.isfinite(y).all():
            raise ValueError('Invalid fitting arrays')
        refs=self.references.reshape(-1,y.shape[1])
        self.scale=np.maximum(refs.std(0)*np.sqrt(2),1e-6)
        # Seed-fixed bounded reference sample; no pair outcomes fit scales.
        z=refs[np.linspace(0,len(refs)-1,min(len(refs),512),dtype=int)]/self.scale
        d=np.maximum((z*z).sum(1)[:,None]+(z*z).sum(1)[None,:]-2*z@z.T,0)
        self.distance=float(np.median(d[d>0])) if np.any(d>0) else 1.
        self.weights=np.ones(len(y))/len(y) if weights is None else np.asarray(weights)/np.sum(weights)
        k=self.kernel(self.references,self.references)
        self.column_mean=self.weights@k; self.grand=float(self.weights@self.column_mean)
        centered=k-self.column_mean[:,None]-self.column_mean[None,:]+self.grand
        self.mean=self.weights@y
        self.coefficients=solve(centered+self.config['alpha']*np.diag(1/self.weights),y-self.mean,assume_a='pos')
        return self

    def predict(self, singles):
        x=np.asarray(singles,float)
        if x.ndim!=3 or x.shape[1:]!=self.references.shape[1:] or not np.isfinite(x).all():
            raise ValueError('Unsupported query reference schema')
        k=self.kernel(x,self.references)
        k=k-self.column_mean[None,:]-(k@self.weights)[:,None]+self.grand
        return x.sum(1)+self.mean+k@self.coefficients


def fit_candidate(config, bank, mask):
    singles=bank['singles'][mask]; pairs=bank['pair_ids'][mask]
    y=bank['responses'][mask]; cultures=bank['culture_ids'][mask]
    w=balanced_weights(pairs,cultures)
    if config['kind']=='additive':return None
    if config['kind']=='component_additive':
        return ComponentAdditive(config['alpha']).fit(pairs,y-singles.sum(1),w)
    return ProfileKernel(config).fit(singles,y,w)


def predict_candidate(config, model, bank):
    if config['kind']=='additive':return bank['singles'].sum(1)
    if config['kind']=='component_additive':return bank['singles'].sum(1)+model.predict(bank['pair_ids'])
    return model.predict(bank['singles'])


def select(bank, menu):
    train=bank['partitions']=='train'; val=bank['partitions']=='validation'
    if not train.any() or not val.any():raise ValueError('Training/validation identities required')
    if set(bank['pair_ids'][train]) & set(bank['pair_ids'][val]):raise ValueError('Pair identity leakage')
    rows=[]
    validation={k:v[val] if k in {'singles','pair_ids','culture_ids','responses'} else v for k,v in bank.items()}
    w=balanced_weights(validation['pair_ids'],validation['culture_ids'])
    for config in menu:
        model=fit_candidate(config,bank,train)
        predictions=predict_candidate(config,model,validation)
        loss=float(w@np.mean((predictions-validation['responses'])**2,axis=1))
        rows.append({'config':config,'validation_mse':loss})
    reference=[r for r in rows if r['config']['role']=='reference']
    candidate=[r for r in rows if r['config']['role']=='candidate']
    if not reference or not candidate:raise ValueError('Both frozen candidate and reference menus required')
    key=lambda r:(r['validation_mse'],r['config']['id'])
    return {'reference':min(reference,key=key)['config'],'candidate':min(candidate,key=key)['config'],
            'comparisons':rows,'search_count':len(rows),'tie_breaker':'MSE then lexical configuration ID',
            'final_refit':'train plus validation, all development cultures'}


def shuffled_profiles(bank, seed):
    """Permute complete profiles once per culture, preserving shared references."""
    rng=np.random.default_rng(seed); result={k:v.copy() for k,v in bank.items()}
    for culture in np.unique(bank['culture_ids']):
        indices=np.flatnonzero(bank['culture_ids']==culture); profiles={}
        for i in indices:
            for j,c in enumerate(json.loads(str(bank['pair_ids'][i]))):
                value=bank['singles'][i,j]
                if c in profiles and not np.allclose(profiles[c],value):
                    raise ValueError('Shuffle requires one exact single profile per component/culture')
                profiles[c]=value
        keys=sorted(profiles); perm=rng.permutation(len(keys))
        for i in indices:
            for j,c in enumerate(json.loads(str(bank['pair_ids'][i]))):
                result['singles'][i,j]=profiles[keys[perm[keys.index(c)]]]
    return result
