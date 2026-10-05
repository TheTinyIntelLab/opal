"""Public-prior symmetric kernels with an observed-response global expert."""
import numpy as np
from opal.models.kernel import KernelResidual,single_kernel,symmetric_kernel
from opal.models.advanced import check_pairs


class PriorKernelResidual(KernelResidual):
    def __init__(self,singles,bank,mask,config):
        self.singles=np.asarray(singles,np.float64);self.bank=np.asarray(bank,np.float64)
        self.mask=np.asarray(mask,bool);self.config=dict(config)
        indices=config['sources']
        if not indices or any(i<0 or i>=self.bank.shape[1] for i in indices):raise ValueError('Invalid prior sources')
        if any(not np.isfinite(config.get(k,d)) for k,d in [('alpha',0),('bandwidth',1),('prior_weight',1)]) or config.get('prior_weight',1)<0 or config.get('alpha',0)<=0 or config.get('bandwidth',1)<=0:raise ValueError('Invalid regularization or bandwidth')
        if self.bank.shape[:2]!=self.mask.shape or len(self.bank)!=len(self.singles):raise ValueError('Prior bank schema mismatch')
        if not np.isfinite(self.bank).all():raise ValueError('Nonfinite prior features')
        kernels=[]
        for i in indices:
            gram=single_kernel(self.bank[:,i],bandwidth=config.get('bandwidth',1))
            gram*=self.mask[:,i,None]*self.mask[None,:,i]
            kernels.append(gram)
        self.single_gram=np.mean(kernels,axis=0)
        scale=np.maximum(self.singles.std(0)*np.sqrt(2),1e-6)
        x=self.singles/scale;self.linear_gram=x@x.T/self.singles.shape[1]
    def kernel(self,left,right):
        left=check_pairs(left,len(self.singles));right=check_pairs(right,len(self.singles))
        return 4*symmetric_kernel(self.linear_gram,left,right)+self.config.get('prior_weight',1)*symmetric_kernel(self.single_gram,left,right,True)
    def state(self):
        return {**super().state(),'bank':self.bank,'mask':self.mask}
