"""Heteroscedastic residual kernels with an unpenalized shared correction."""
import numpy as np
from scipy.linalg import solve
from opal.models.kernel import KernelResidual
from opal.models.advanced import check_pairs


def noise_diagonal(variance,strength):
    if not np.isfinite(strength) or not 0<=strength<=1:raise ValueError('Invalid noise strength')
    variance=np.asarray(variance,np.float64)
    if variance.ndim==2:variance=variance.mean(1)
    if variance.ndim!=1 or not np.isfinite(variance).all() or np.any(variance<0):raise ValueError('Invalid training measurement variance')
    scale=max(float(np.median(variance)),1e-8)
    return (1-strength)+strength*np.clip(variance/scale,.05,20)


def measurement_smoother(kernel,train,query,alpha,variance,strength=1):
    if not np.isfinite(alpha) or not 0<=strength<=1 or alpha<=0:raise ValueError('Invalid noise regularization')
    diagonal=noise_diagonal(variance,strength)
    if len(diagonal)!=len(train):raise ValueError('Training variance order mismatch')
    matrix=kernel[np.ix_(train,train)]+alpha*np.diag(diagonal)
    inverse=solve(matrix,np.eye(len(train)),assume_a='pos')
    intercept=inverse.sum(0)/inverse.sum()
    weights=kernel[np.ix_(query,train)]@inverse
    return weights+(1-weights.sum(1))[:,None]*intercept[None,:]


class NoiseAwareKernel(KernelResidual):
    def fit(self,pairs,doubles,variance):
        pairs=check_pairs(pairs,len(self.singles));doubles=np.asarray(doubles,np.float64)
        if doubles.shape!=(len(pairs),self.singles.shape[1]) or not np.isfinite(doubles).all():raise ValueError('Invalid fitting responses')
        self.train_pairs=pairs.copy()
        residual=doubles-self.singles[pairs[:,0]]-self.singles[pairs[:,1]]
        self.training_noise=noise_diagonal(variance,self.config.get('noise_strength',1))
        if len(self.training_noise)!=len(pairs) or not np.isfinite(self.config['alpha']) or self.config['alpha']<=0:raise ValueError('Invalid training noise configuration')
        inverse=solve(self.kernel(pairs,pairs)+self.config['alpha']*np.diag(self.training_noise),np.eye(len(pairs)),assume_a='pos')
        intercept=inverse.sum(0)/inverse.sum()
        self.mean=intercept@residual
        self.coefficients=inverse@(residual-self.mean)
        return self
    def predict(self,pairs):
        pairs=check_pairs(pairs,len(self.singles))
        return self.singles[pairs[:,0]]+self.singles[pairs[:,1]]+self.mean+self.kernel(pairs,self.train_pairs)@self.coefficients
    def state(self):
        return {k:getattr(self,k) for k in ['train_pairs','mean','coefficients','training_noise']}
    def restore(self,arrays):
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite noise-kernel checkpoint')
        self.train_pairs=check_pairs(arrays['train_pairs'],len(self.singles))
        for k in ['mean','coefficients','training_noise']:setattr(self,k,arrays[k])
        if self.mean.shape!=(self.singles.shape[1],) or self.training_noise.shape!=(len(self.train_pairs),) or self.coefficients.shape!=(len(self.train_pairs),self.singles.shape[1]) or np.any(self.training_noise<=0):raise ValueError('Noise-kernel schema mismatch')
        return self
