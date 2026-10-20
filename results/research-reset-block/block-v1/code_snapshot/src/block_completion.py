"""Scalar references and a fixed supported-gene completion diagnostic.

This module does not change any historically registered model implementation.
It supports one scalar screen per fit, without fabricating a second context.
"""
from pathlib import Path

import numpy as np
import torch

from .models.fitness import design


def reference_statistics(singles, pairs, targets, nodes=False, check=None):
    """Train-only ridge sufficient statistics, without a full edge-by-gene matrix."""
    width = 3+(len(singles) if nodes else 0)
    total = np.zeros(width)
    square = np.zeros(width)
    gram = np.zeros((width, width))
    cross = np.zeros(width)
    residual_sum = 0.
    for start in range(0, len(pairs), 4096):
        if check is not None:
            check()
        rows = slice(start, start+4096)
        x = design(singles, pairs[rows], nodes)
        residual = targets[rows]-singles[pairs[rows]].sum(1)
        total += x.sum(0)
        square += np.square(x).sum(0)
        gram += x.T@x
        cross += x.T@residual
        residual_sum += float(residual.sum())
    center = total/len(pairs)
    scale = np.maximum(np.sqrt(np.maximum(square/len(pairs)-center**2, 0)), 1e-8)
    return dict(center=center, scale=scale, mean=residual_sum/len(pairs),
                gram=(gram-len(pairs)*np.outer(center, center))/np.outer(scale, scale),
                cross=(cross-center*residual_sum)/scale)


def checked_pairs(pairs, genes, *, unique=False):
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError('Integer pair-by-two identities required')
    if np.any(pairs < 0) or np.any(pairs >= genes) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError('Distinct in-schema identities required')
    canonical = np.sort(pairs, axis=1)
    if unique and len(np.unique(canonical, axis=0)) != len(pairs):
        raise ValueError('Duplicate canonical pairs')
    return pairs.astype(np.int64, copy=False)


class ScalarReference:
    """Existing ridge design, plus an explicit training-mean null predictor."""

    def __init__(self, kind='node', alpha=100.):
        if kind not in {'mean', 'zero', 'additive', 'minimum', 'calibration', 'node'}:
            raise ValueError('Unknown scalar reference')
        self.kind, self.alpha = kind, float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError('Positive finite ridge penalty required')

    def fit(self, singles, pairs, targets, check=None, _statistics=None):
        self.singles = np.asarray(singles, float).copy()
        if self.singles.ndim != 1:
            raise ValueError('One gene-level single per schema gene required')
        pairs = checked_pairs(pairs, len(self.singles), unique=True)
        if not np.isfinite(self.singles[pairs]).all():
            raise ValueError('Finite queried gene-level singles required')
        y = np.asarray(targets, float)
        if y.shape != (len(pairs),) or not len(y) or not np.isfinite(y).all():
            raise ValueError('Finite aligned training targets required')
        self.support = np.bincount(pairs.ravel(), minlength=len(self.singles)) > 0
        self.mean = float(y.mean())
        if self.kind in {'calibration', 'node'}:
            stats = _statistics if _statistics is not None else reference_statistics(
                self.singles, pairs, y, self.kind == 'node', check)
            self.center, self.scale, self.offset = stats['center'].copy(), stats['scale'].copy(), float(stats['mean'])
            self.coefficient = np.linalg.solve(stats['gram']+self.alpha*np.eye(len(self.center)), stats['cross'])
        return self

    def predict(self, pairs):
        p = checked_pairs(pairs, len(self.singles))
        if not np.all(self.support[p]):
            raise ValueError('Query genes need training-pair support')
        if self.kind == 'mean':
            return np.full(len(p), self.mean)
        if self.kind == 'zero':
            return np.zeros(len(p))
        if self.kind == 'minimum':
            return self.singles[p].min(1)
        prediction = self.singles[p].sum(1)
        if self.kind in {'calibration', 'node'}:
            prediction += self.offset
            for start in range(0, len(p), 4096):
                rows = slice(start, start+4096)
                x = design(self.singles, p[rows], self.kind == 'node')
                prediction[rows] += (x-self.center)/self.scale@self.coefficient
        return prediction

    def save(self, path):
        values = dict(kind=self.kind, alpha=self.alpha, singles=self.singles,
                      support=self.support, mean=self.mean)
        if self.kind in {'calibration', 'node'}:
            values.update(center=self.center, scale=self.scale, coefficient=self.coefficient, offset=self.offset)
        np.savez_compressed(Path(path), **values)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            model = cls(str(z['kind']), float(z['alpha']))
            model.singles, model.support, model.mean = z['singles'].copy(), z['support'].copy(), float(z['mean'])
            if model.kind in {'calibration', 'node'}:
                model.center, model.scale, model.coefficient = z['center'].copy(), z['scale'].copy(), z['coefficient'].copy()
                model.offset = float(z['offset'])
        return model


def fit_reference_grid(singles, pairs, targets, alphas=(1., 10., 100., 1000.), check=None):
    """Fit the finite reference menu, sharing train-only sufficient statistics."""
    models = []
    for kind in ('zero', 'mean', 'additive', 'minimum'):
        if check is not None:
            check()
        models.append(ScalarReference(kind).fit(singles, pairs, targets))
    for kind in ('calibration', 'node'):
        stats = reference_statistics(np.asarray(singles), np.asarray(pairs), np.asarray(targets), kind == 'node', check)
        for alpha in alphas:
            if check is not None:
                check()
            models.append(ScalarReference(kind, alpha).fit(singles, pairs, targets, _statistics=stats))
    return models


class ScalarCompletion:
    """Symmetric rank-eight factors with train-only scales and validation steps.

    Half the scalar data loss preserves the per-context error/penalty ratio of
    the historical two-context independent objective. Contexts now select their
    checkpoints separately rather than sharing a normalized validation mean.
    """

    def __init__(self, rank=8, penalty=.01, steps=500, seed=20261022):
        if rank < 2 or rank % 2 or not np.isfinite(penalty) or penalty <= 0 or steps < 1:
            raise ValueError('Invalid completion configuration')
        self.rank, self.penalty, self.steps, self.seed = int(rank), float(penalty), int(steps), int(seed)

    def fit(self, pairs, residuals, genes, scale, validation, check=None):
        p = checked_pairs(pairs, genes, unique=True)
        vp = checked_pairs(validation[0], genes, unique=True)
        y, vy = np.asarray(residuals, float), np.asarray(validation[1], float)
        if y.shape != (len(p),) or vy.shape != (len(vp),) or len(p) < 3 or not len(vp):
            raise ValueError('Nonempty aligned train/validation outcomes required')
        if not np.isfinite(y).all() or not np.isfinite(vy).all() or not np.isfinite(scale) or scale <= 0:
            raise ValueError('Finite outcomes and positive training scale required')
        if set(map(tuple, np.sort(p, axis=1))) & set(map(tuple, np.sort(vp, axis=1))):
            raise ValueError('Training/validation pair leakage')
        self.genes, self.scale = int(genes), float(scale)
        self.support = np.bincount(p.ravel(), minlength=genes) > 0
        if not self.support[vp].all():
            raise ValueError('Validation genes need training support')
        generator = torch.Generator().manual_seed(self.seed)
        u = torch.nn.Parameter(torch.randn((genes, self.rank), generator=generator)*.05)
        v = torch.nn.Parameter(torch.randn((genes, self.rank), generator=generator)*.05)
        a, b = torch.as_tensor(p.T, dtype=torch.long)
        va, vb = vp.T
        truth = torch.as_tensor(y/self.scale, dtype=torch.float32)
        validation_truth = vy/self.scale
        optimizer = torch.optim.Adam([u, v], lr=.03)
        best, matrix = float('inf'), None
        self.history, self.best_step = [], 0
        for step in range(1, self.steps+1):
            if check is not None:
                check()
            optimizer.zero_grad()
            prediction = (u@v.T+v@u.T)*.5
            loss = .5*torch.mean((prediction[a, b]-truth)**2)
            loss = loss+self.penalty*(u.square().sum()+v.square().sum())/genes
            loss.backward()
            optimizer.step()
            if step % 25 == 0 or step == self.steps:
                with torch.no_grad():
                    current = ((u@v.T+v@u.T)*.5).numpy().copy()
                score = float(np.mean((current[va, vb]-validation_truth)**2))
                self.history.append(dict(step=step, objective=float(loss.detach()), validation_mse=score))
                if score < best:
                    best, matrix, self.best_step = score, current, step
        self.matrix = matrix.astype(np.float64)*self.scale
        self.matrix = (self.matrix+self.matrix.T)*.5
        self.parameter_count = int(u.numel()+v.numel())
        return self

    def predict(self, pairs):
        p = checked_pairs(pairs, self.genes)
        if not self.support[p].all():
            raise ValueError('Query genes need training-pair support')
        return self.matrix[p[:, 0], p[:, 1]]

    def save(self, path):
        np.savez_compressed(Path(path), matrix=self.matrix, support=self.support,
                            rank=self.rank, penalty=self.penalty, steps=self.steps,
                            seed=self.seed, scale=self.scale, best_step=self.best_step,
                            parameter_count=self.parameter_count)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            model = cls(int(z['rank']), float(z['penalty']), int(z['steps']), int(z['seed']))
            model.matrix, model.support = z['matrix'].copy(), z['support'].copy()
            model.genes, model.scale = len(model.support), float(z['scale'])
            model.best_step, model.parameter_count = int(z['best_step']), int(z['parameter_count'])
        return model
