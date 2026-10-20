"""Measured-single growth calibration and masked symmetric matrix completion.

The task is unseen pairs of supported genes within the measured cell contexts.
It does not predict expression, new genes, or new cell types.
"""
from pathlib import Path
import numpy as np
import torch


def design(singles, pairs, nodes=False):
    a, b = np.asarray(pairs, int).T
    s = np.asarray(singles, float)
    x = np.stack([s[a]+s[b], np.abs(s[a]-s[b]), s[a]*s[b]], axis=1)
    if nodes:
        incidence = np.zeros((len(pairs), len(s)))
        incidence[np.arange(len(pairs)), a] = 1
        incidence[np.arange(len(pairs)), b] = 1
        x = np.concatenate([x, incidence], axis=1)
    return x


class GrowthReference:
    def __init__(self, kind='node', alpha=100.):
        if kind not in {'zero','additive','minimum','calibration','node'} or alpha <= 0:
            raise ValueError('Unknown reference or nonpositive penalty')
        self.kind, self.alpha = kind, float(alpha)

    def fit(self, singles, pairs, responses):
        self.singles = np.asarray(singles, float).copy()
        y = np.asarray(responses, float)
        if y.shape != (len(self.singles),len(pairs)) or not np.isfinite(y).all():
            raise ValueError('Finite context-by-pair targets required')
        self.fits = []
        for c, s in enumerate(self.singles):
            if self.kind in {'zero','additive','minimum'}: continue
            x = design(s,pairs,self.kind=='node')
            center, scale = x.mean(0), np.maximum(x.std(0),1e-8)
            z = (x-center)/scale
            offset = s[np.asarray(pairs)].sum(1)
            residual = y[c]-offset; mean = residual.mean()
            coefficient = np.linalg.solve(z.T@z+self.alpha*np.eye(z.shape[1]),z.T@(residual-mean))
            self.fits.append((center,scale,coefficient,mean))
        return self

    def predict(self, pairs):
        result=[]
        for c,s in enumerate(self.singles):
            pair_singles=s[np.asarray(pairs)]
            if self.kind=='zero': p=np.zeros(len(pairs))
            elif self.kind=='additive': p=pair_singles.sum(1)
            elif self.kind=='minimum': p=pair_singles.min(1)
            else:
                center,scale,coef,mean=self.fits[c]
                p=pair_singles.sum(1)+mean
                for start in range(0,len(pairs),4096):
                    block=slice(start,start+4096)
                    p[block]+=(design(s,pairs[block],self.kind=='node')-center)/scale@coef
            result.append(p)
        return np.asarray(result)


class FitnessCompletion:
    """Indefinite symmetric factors, with explicit shared/private ablation modes."""
    def __init__(self, mode='joint', rank=8, penalty=.001, steps=500, seed=20261020):
        if mode not in {'joint','independent','pooled'} or rank < 2 or rank % 2 or penalty <= 0 or steps < 1:
            raise ValueError('Invalid completion configuration')
        self.mode,self.rank,self.penalty,self.steps,self.seed=mode,int(rank),float(penalty),int(steps),int(seed)

    def fit(self, pairs, residuals, genes, scales, validation=None):
        pairs=np.asarray(pairs,int); y=np.asarray(residuals,float); scales=np.asarray(scales,float)
        if y.ndim!=2 or y.shape[1]!=len(pairs) or len(pairs)<3 or genes<=np.max(pairs) or np.min(pairs)<0 or not np.isfinite(y).all() or not np.isfinite(scales).all() or np.any(scales<=0):
            raise ValueError('Invalid training targets, identities or scales')
        self.genes,self.contexts,self.scales=int(genes),len(y),scales.copy()
        if self.contexts!=2: raise ValueError('The registered shared/private comparison requires two contexts')
        self.support=np.bincount(pairs.ravel(),minlength=genes)>0
        generator=torch.Generator().manual_seed(self.seed)
        parameters=[]
        def factor(shape):
            p=torch.nn.Parameter(torch.randn(shape,generator=generator)*.05);parameters.append(p);return p
        if self.mode=='independent':
            u,v=factor((2,genes,self.rank)),factor((2,genes,self.rank))
            def matrix():return (u@v.transpose(-1,-2)+v@u.transpose(-1,-2))/2
        else:
            rank=self.rank*2 if self.mode=='pooled' else self.rank
            u,v=factor((genes,rank)),factor((genes,rank))
            if self.mode=='joint': pu,pv=factor((2,genes,self.rank//2)),factor((2,genes,self.rank//2))
            def matrix():
                common=(u@v.T+v@u.T)/2
                if self.mode=='joint':return common[None]+(pu@pv.transpose(-1,-2)+pv@pu.transpose(-1,-2))/2
                return common[None].expand(2,-1,-1)
        a,b=torch.as_tensor(pairs.T,dtype=torch.long)
        truth=torch.as_tensor(y/scales[:,None],dtype=torch.float32)
        optimizer=torch.optim.Adam(parameters,lr=.03)
        best=float('inf'); best_matrix=None; self.history=[]; self.best_step=0
        if validation is not None:
            vp,vy=validation; va,vb=np.asarray(vp,int).T
            vy=np.asarray(vy,float)/scales[:,None]
            if {tuple(sorted(p)) for p in pairs.tolist()} & {tuple(sorted(p)) for p in np.asarray(vp).tolist()}:
                raise ValueError('Validation pair leakage')
        for step in range(1,self.steps+1):
            optimizer.zero_grad(); prediction=matrix()
            loss=torch.mean((prediction[:,a,b]-truth)**2)
            loss=loss+self.penalty*sum(p.square().sum() for p in parameters)/genes
            loss.backward();optimizer.step()
            if step%25==0 or step==self.steps:
                with torch.no_grad(): current=matrix().numpy().copy()
                score=float(np.mean((current[:,va,vb]-vy)**2)) if validation is not None else float(loss.detach())
                self.history.append({'step':step,'training_objective':float(loss.detach()),'validation_mse':score if validation is not None else None})
                if score<best:
                    best,best_matrix,self.best_step=score,current,step
        exact=best_matrix.astype(np.float64)
        self.matrix=(exact+exact.transpose(0,2,1))*.5*scales[:,None,None]
        self.parameter_count=sum(p.numel() for p in parameters)
        return self

    def predict(self,pairs):
        p=np.asarray(pairs,int)
        if p.ndim!=2 or p.shape[1]!=2 or np.any(p<0) or np.any(p>=self.genes) or np.any(p[:,0]==p[:,1]) or not np.all(self.support[p]):
            raise ValueError('Query must contain distinct supported training genes')
        return self.matrix[:,p[:,0],p[:,1]]

    def save(self,path):
        np.savez_compressed(Path(path),matrix=self.matrix,scales=self.scales,support=self.support,
                            mode=self.mode,rank=self.rank,penalty=self.penalty,steps=self.steps,seed=self.seed,
                            best_step=self.best_step,parameter_count=self.parameter_count)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            m=cls(str(z['mode']),int(z['rank']),float(z['penalty']),int(z['steps']),int(z['seed']))
            for k in ['matrix','scales','support']:setattr(m,k,z[k].copy())
            m.best_step=int(z['best_step']);m.parameter_count=int(z['parameter_count'])
        m.contexts,m.genes=m.matrix.shape[:2]
        return m
