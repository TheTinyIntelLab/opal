"""Gene-local response composition with shrinkage toward pooled coefficients.

This is a predictive calibration model, not an identifiable causal interaction
model. Inputs must be matched-control single effects in a fixed gene order.
"""
from pathlib import Path
import numpy as np


class Composition:
    def __init__(self, alpha=1., overlap=True, pooled=False):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        self.alpha, self.overlap, self.pooled = float(alpha), bool(overlap), bool(pooled)

    def features(self, singles):
        x = np.asarray(singles, dtype=np.float64)
        if x.ndim != 3 or x.shape[1] != 2 or not np.isfinite(x).all():
            raise ValueError('Expected finite [pairs, 2, genes] single effects')
        a, b = x[:, 0], x[:, 1]
        total = a + b
        parts = [total]
        if self.overlap:
            overlap = np.sign(total) * np.minimum(np.abs(a), np.abs(b))
            parts += [overlap * (a * b > 0), overlap * (a * b < 0)]
        return np.stack(parts, axis=-1)

    def fit(self, singles, responses):
        x, y = self.features(singles), np.asarray(responses, dtype=np.float64)
        if y.shape != x.shape[:2] or len(y) < 3 or not np.isfinite(y).all():
            raise ValueError('At least three finite, matched training responses required')
        self.xmean, self.ymean = x.mean(0), y.mean(0)
        xc, yc = x - self.xmean, y - self.ymean
        gram = np.einsum('ngk,ngl->gkl', xc, xc) / len(y)
        rhs = np.einsum('ngk,ng->gk', xc, yc) / len(y)
        pooled_gram, pooled_rhs = gram.mean(0), rhs.mean(0)
        # A positive floor supports constant features without a pseudoinverse.
        regularizer = np.diag(np.maximum(np.diag(pooled_gram), 1e-10))
        prior = np.zeros(x.shape[-1]); prior[0] = 1.
        self.global_coefficient = np.linalg.solve(pooled_gram + 1e-8 * regularizer,
                                                  pooled_rhs + 1e-8 * regularizer @ prior)
        if self.pooled:
            self.coefficient = np.broadcast_to(self.global_coefficient, rhs.shape).copy()
        else:
            self.coefficient = np.linalg.solve(
                gram + self.alpha * regularizer,
                (rhs + self.alpha * (regularizer @ self.global_coefficient))[..., None])[..., 0]
        self.genes = y.shape[1]
        return self

    def predict(self, singles):
        x = self.features(singles)
        if x.shape[1] != self.genes:
            raise ValueError('Gene count differs from the training schema')
        return self.ymean + np.einsum('ngk,gk->ng', x - self.xmean, self.coefficient)

    def save(self, path):
        """Safe numeric checkpoint; callers must retain their gene-order manifest."""
        np.savez_compressed(Path(path), alpha=self.alpha, overlap=self.overlap, pooled=self.pooled,
                            xmean=self.xmean, ymean=self.ymean, coefficient=self.coefficient,
                            global_coefficient=self.global_coefficient)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            model = cls(float(z['alpha']), bool(z['overlap']), bool(z['pooled']))
            for name in ('xmean', 'ymean', 'coefficient', 'global_coefficient'):
                setattr(model, name, z[name].copy())
        model.genes = len(model.ymean)
        return model
