"""Residual kernels with shared-reference measurement covariance."""
import numpy as np
from scipy.linalg import solve
from opal.models.noise_kernel import NoiseAwareKernel,noise_diagonal
from opal.models.advanced import check_pairs


def noise_covariance(covariance,strength=1,correlation=1):
    c=np.asarray(covariance,np.float64)
    if c.ndim!=2 or c.shape[0]!=c.shape[1] or not len(c) or not np.isfinite(c).all():raise ValueError('Invalid measurement covariance')
    if not np.isfinite(strength) or not np.isfinite(correlation) or not 0<=strength<=1 or not 0<=correlation<=1:raise ValueError('Invalid covariance shrinkage')
    if not np.allclose(c,c.T,atol=1e-9,rtol=1e-7):raise ValueError('Covariance is not symmetric')
    c=(c+c.T)/2
    eigenvalues,eigenvectors=np.linalg.eigh(c)
    if eigenvalues.min() < -1e-8*max(float(np.abs(c).max()),1e-12):raise ValueError('Covariance is not positive semidefinite')
    if eigenvalues.min()<0:c=(eigenvectors*np.maximum(eigenvalues,0))@eigenvectors.T
    diagonal=noise_diagonal(np.maximum(np.diag(c),0),1)
    factors=np.sqrt(diagonal/np.maximum(np.diag(c),1e-12))
    normalized=c*factors[:,None]*factors[None,:]
    normalized+=np.diag(np.maximum(diagonal-np.diag(normalized),0))
    return (1-strength)*np.eye(len(c))+strength*((1-correlation)*np.diag(diagonal)+correlation*normalized)


def covariance_smoother(kernel,train,query,alpha,regularizer):
    r=np.asarray(regularizer,np.float64)
    if r.shape!=(len(train),len(train)) or not np.isfinite(r).all() or not np.isfinite(alpha) or alpha<=0:raise ValueError('Invalid covariance fitting block')
    inverse=solve(kernel[np.ix_(train,train)]+alpha*r,np.eye(len(train)),assume_a='pos')
    intercept=inverse.sum(0)/inverse.sum()
    weights=kernel[np.ix_(query,train)]@inverse
    return weights+(1-weights.sum(1))[:,None]*intercept


class CovarianceKernel(NoiseAwareKernel):
    def fit(self,pairs,doubles,covariance):
        pairs=check_pairs(pairs,len(self.singles));doubles=np.asarray(doubles,np.float64)
        if doubles.shape!=(len(pairs),self.singles.shape[1]) or not np.isfinite(doubles).all():raise ValueError('Invalid fitting responses')
        self.training_noise=noise_covariance(covariance,self.config.get('noise_strength',1),self.config.get('correlation',1))
        if self.training_noise.shape!=(len(pairs),len(pairs)):raise ValueError('Covariance order mismatch')
        self.train_pairs=pairs.copy();residual=doubles-self.singles[pairs[:,0]]-self.singles[pairs[:,1]]
        inverse=solve(self.kernel(pairs,pairs)+self.config['alpha']*self.training_noise,np.eye(len(pairs)),assume_a='pos')
        intercept=inverse.sum(0)/inverse.sum();self.mean=intercept@residual;self.coefficients=inverse@(residual-self.mean)
        return self
    def restore(self,arrays):
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite covariance checkpoint')
        self.train_pairs=check_pairs(arrays['train_pairs'],len(self.singles))
        for k in ['mean','coefficients','training_noise']:setattr(self,k,np.asarray(arrays[k]))
        n,g=len(self.train_pairs),self.singles.shape[1]
        if self.mean.shape!=(g,) or self.coefficients.shape!=(n,g) or self.training_noise.shape!=(n,n):raise ValueError('Covariance checkpoint schema mismatch')
        if not np.allclose(self.training_noise,self.training_noise.T) or np.linalg.eigvalsh(self.training_noise).min() < -1e-8*max(float(np.abs(self.training_noise).max()),1e-12):raise ValueError('Invalid stored covariance regularizer')
        return self
