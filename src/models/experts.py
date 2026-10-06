"""Small-data symmetric experts and explicitly conditional GP diagnostics."""
import numpy as np
from scipy.linalg import solve
from opal.models.advanced import check_pairs


def symmetric_features(bank,pairs):
    pairs=check_pairs(pairs,len(bank));bank=np.asarray(bank,np.float64).reshape(len(bank),-1)
    a,b=bank[pairs[:,0]],bank[pairs[:,1]]
    return np.concatenate([a+b,np.abs(a-b),a*b],axis=1)


class FeatureRidgeResidual:
    def __init__(self,singles,bank,alpha=10):
        if alpha<=0:raise ValueError('Positive regularization required')
        self.singles=np.asarray(singles,np.float64);self.bank=np.asarray(bank,np.float64);self.alpha=alpha
    def fit(self,pairs,doubles):
        self.pairs=check_pairs(pairs,len(self.singles)).copy()
        features=symmetric_features(self.bank,self.pairs)
        self.center=features.mean(0);self.scale=np.maximum(features.std(0),1e-5)
        self.train_features=(features-self.center)/self.scale
        residual=doubles-self.singles[self.pairs[:,0]]-self.singles[self.pairs[:,1]]
        self.mean=residual.mean(0)
        kernel=self.train_features@self.train_features.T/self.train_features.shape[1]
        self.coefficients=solve(kernel+self.alpha*np.eye(len(pairs)),residual-self.mean,assume_a='pos')
        return self
    def predict(self,pairs):
        pairs=check_pairs(pairs,len(self.singles))
        x=(symmetric_features(self.bank,pairs)-self.center)/self.scale
        return self.singles[pairs[:,0]]+self.singles[pairs[:,1]]+self.mean+x@self.train_features.T@self.coefficients/x.shape[1]
    def state(self):
        return {k:getattr(self,k) for k in ['singles','bank','pairs','center','scale','train_features','mean','coefficients']}
    @classmethod
    def from_state(cls,arrays,alpha):
        model=cls(arrays['singles'],arrays['bank'],alpha)
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite feature-ridge checkpoint')
        for k,v in arrays.items():setattr(model,k,v)
        if model.coefficients.shape!=(len(model.pairs),model.singles.shape[1]):raise ValueError('Feature-ridge shape mismatch')
        return model


class MeasurementGP:
    """Fixed-kernel multigene predictor; variance conditional on fixed hyperparameters."""
    def __init__(self,singles,bank,alpha=.1,bandwidth=1):
        self.singles=np.asarray(singles,np.float64);self.bank=np.asarray(bank,np.float64)
        self.alpha,self.bandwidth=alpha,bandwidth
        if alpha<=0 or bandwidth<=0:raise ValueError('Invalid GP hyperparameters')
    def kernel(self,x,y):
        distance=np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T,0)/x.shape[1]
        return np.exp(-distance/self.bandwidth)+.1*x@y.T/x.shape[1]
    def fit(self,pairs,doubles,variance):
        self.pairs=check_pairs(pairs,len(self.singles)).copy()
        features=symmetric_features(self.bank,pairs)
        self.center=features.mean(0);self.scale=np.maximum(features.std(0),1e-4)
        self.x=(features-self.center)/self.scale
        residual=doubles-self.singles[pairs[:,0]]-self.singles[pairs[:,1]]
        self.mean=residual.mean(0);self.output_scale=np.maximum(residual.std(0),.001)
        # Condition-average technical residual noise is known for training pairs only.
        noise=np.mean(variance,axis=1)/max(float(np.mean(self.output_scale**2)),1e-8)
        self.training_noise=np.clip(noise,.01,10)+self.alpha
        matrix=self.kernel(self.x,self.x)+np.diag(self.training_noise)
        self.inverse=solve(matrix,np.eye(len(pairs)),assume_a='pos')
        self.coefficients=self.inverse@((residual-self.mean)/self.output_scale)
        return self
    def predict(self,pairs,return_std=False):
        pairs=check_pairs(pairs,len(self.singles))
        x=(symmetric_features(self.bank,pairs)-self.center)/self.scale
        cross=self.kernel(x,self.x)
        prediction=self.singles[pairs[:,0]]+self.singles[pairs[:,1]]+self.mean+(cross@self.coefficients)*self.output_scale
        if not return_std:return prediction
        variance=np.maximum(np.diag(self.kernel(x,x))-np.einsum('ni,ij,nj->n',cross,self.inverse,cross),0)
        return prediction,np.sqrt(variance)[:,None]*self.output_scale


def retrieve_residual(singles,bank,pairs,doubles,train,query,neighbors=3):
    identities=lambda indices:{tuple(sorted(p)) for p in np.asarray(pairs)[indices]}
    if identities(train)&identities(query):raise ValueError('Retrieval query identity overlaps training pairs')
    features=symmetric_features(bank,pairs)
    scale=np.maximum(features[train].std(0),1e-4)
    x=features/scale
    distance=np.maximum((x[query]**2).sum(1)[:,None]+(x[train]**2).sum(1)[None,:]-2*x[query]@x[train].T,0)
    residual=doubles-singles[pairs[:,0]]-singles[pairs[:,1]]
    prediction=[]
    for row in distance:
        selected=np.argsort(row)[:min(neighbors,len(train))]
        weights=1/(row[selected]+max(float(np.median(row)),1e-4)*.01)
        weights/=weights.sum()
        prediction.append(weights@residual[train[selected]])
    return singles[pairs[query,0]]+singles[pairs[query,1]]+np.array(prediction)
