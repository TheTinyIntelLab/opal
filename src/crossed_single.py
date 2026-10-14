"""Fold-local crossed-single feasibility models; no pair or biological claims."""
import numpy as np
from .context_contrast.contracts import Unsupported


FAMILIES = ('zero','source_mean','target_mean','offset','affine','pca_concat',
            'pca_mean','pca_difference','kernel','shuffled_concat')


def menu(config):
    return [dict(family=f, alpha=a) for f in FAMILIES
            for a in (config['penalties'] if f not in FAMILIES[:4] else [0.0])]


def source_features(x, family):
    if family == 'pca_mean':
        return x.mean(1)
    if family == 'pca_difference':
        return x[:,1]-x[:,0]
    return x.reshape(len(x),-1)


def fit(x, y, spec, config):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.ndim!=3 or x.shape[1]!=2 or y.shape!=(len(x),x.shape[2]) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite two-source profiles and matching target effects required')
    family=spec['family'];alpha=spec['alpha'];base=x.mean(1);residual=y-base
    state=dict(family=family,alpha=alpha,genes=y.shape[1],training_rows=len(x),rank=None)
    if family in ('zero','source_mean'):
        return state
    if family=='target_mean':
        return dict(state,mean=y.mean(0))
    offset=residual.mean(0)
    if family=='offset':return dict(state,offset=offset)
    if family=='affine':
        xm,ym=base.mean(0),y.mean(0);xc,yc=base-xm,y-ym
        slope=float(np.sum(xc*yc)/(np.sum(xc*xc)+alpha*xc.size))
        return dict(state,slope=slope,xmean=xm,mean=ym)
    feature=source_features(x,family)
    if family=='shuffled_concat':
        feature=feature[np.random.default_rng(config['alignment_seed']).permutation(len(x))]
    center=feature.mean(0);xc=feature-center
    if family=='kernel':
        distance=np.maximum(np.sum((feature[:,None]-feature[None])**2,axis=2),0)
        positive=distance[distance>config['rank_tolerance']]
        if not len(positive):raise Unsupported('Constant source kernel bank')
        width=float(np.median(positive));k=np.exp(-distance/width);km=k.mean(0)
        kc=k-km[None]-km[:,None]+k.mean()
        coef=np.linalg.solve(kc+alpha*len(x)*np.eye(len(x)),residual-offset)
        return dict(state,offset=offset,bank=feature,width=width,kmean=km,kgrand=float(k.mean()),coef=coef)
    _,s,v=np.linalg.svd(xc,full_matrices=False);rank=config['representation_rank']
    effective=int(np.sum(s>max(s[0] if len(s) else 0,1)*config['rank_tolerance']))
    if effective<rank or s[0]/s[rank-1]>config['condition_limit']:
        raise Unsupported('Fixed rank-16 source representation unsupported')
    basis=v[:rank].T;z=xc@basis;scale=np.sqrt(np.mean(z*z,axis=0))
    z/=scale
    coef=np.linalg.solve(z.T@z/len(z)+alpha*np.eye(rank),z.T@(residual-offset)/len(z))
    return dict(state,offset=offset,center=center,basis=basis,scale=scale,coef=coef,
                rank=rank,effective_rank=effective,condition=float(s[0]/s[rank-1]))


def predict(state,x):
    x=np.asarray(x,float);family=state['family'];base=x.mean(1)
    if family=='zero':return np.zeros_like(base)
    if family=='source_mean':return base
    if family=='target_mean':return np.broadcast_to(state['mean'],base.shape).copy()
    if family=='offset':return base+state['offset']
    if family=='affine':return state['mean']+state['slope']*(base-state['xmean'])
    feature=source_features(x,family)
    if family=='kernel':
        k=np.exp(-np.sum((feature[:,None]-state['bank'][None])**2,axis=2)/state['width'])
        kc=k-state['kmean']-k.mean(1,keepdims=True)+state['kgrand']
        return base+state['offset']+kc@state['coef']
    return base+state['offset']+((feature-state['center'])@state['basis']/state['scale'])@state['coef']


def grouped_folds(n,seed,folds=3):
    assignment=np.empty(n,dtype=int)
    assignment[np.random.default_rng(seed).permutation(n)]=np.arange(n)%folds
    return assignment


def select(x,y,config,record):
    folds=grouped_folds(len(x),config['split_seed']);scores=[]
    for index,spec in enumerate(menu(config)):
        losses=[];supported=True
        for inner in range(3):
            train=np.flatnonzero(folds!=inner);val=np.flatnonzero(folds==inner)
            try:
                state=fit(x[train],y[train],spec,config)
                loss=float(np.mean((predict(state,x[val])-y[val])**2));error=None
            except Unsupported as exc:
                state=None;loss=None;error=str(exc);supported=False
            record(f'inner-{inner}-spec-{index}',state,dict(spec=spec,train=train.tolist(),validation=val.tolist(),mse=loss,error=error))
            if loss is not None:losses.append(loss)
        scores.append(dict(spec=spec,supported=supported,mse=float(np.mean(losses)) if supported else None))
    choices={}
    for family in FAMILIES:
        valid=[r for r in scores if r['spec']['family']==family and r['supported']]
        choices[family]=min(valid,key=lambda r:(r['mse'],r['spec']['alpha'])) if valid else None
    refs=[r for f,r in choices.items() if f not in config['selector_excludes'] and r is not None]
    locked=min(refs,key=lambda r:(r['mse'],r['spec']['family'],r['spec']['alpha']))['spec']['family']
    return choices,locked,scores
