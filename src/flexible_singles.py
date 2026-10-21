"""One fixed nonlinear singles feature map for exposed scalar diagnostics."""
from pathlib import Path

import numpy as np

from .block_completion import checked_pairs
from .models.fitness import design


class FlexibleSingles:
    """Ridge on a train-only symmetric RBF Nystrom map, optionally with nodes."""

    def __init__(self, nodes=False, alpha=100., centers=64, seed=20261022):
        if not np.isfinite(alpha) or alpha <= 0 or centers < 2:
            raise ValueError('Positive ridge and at least two centers required')
        self.nodes, self.alpha = bool(nodes), float(alpha)
        self.n_centers, self.seed = int(centers), int(seed)

    def _coordinates(self, pairs):
        return (np.sort(self.singles[pairs], axis=1)-self.input_center)/self.input_scale

    def _kernel(self, x, z):
        distance = np.square(x[:, None, :]-z[None, :, :]).sum(2)
        return np.exp(-distance/(2*self.bandwidth**2))

    def _features(self, pairs):
        nonlinear = self._kernel(self._coordinates(pairs), self.centers)@self.whitener
        return np.concatenate((design(self.singles, pairs, self.nodes), nonlinear), axis=1)

    def prepare(self, singles, pairs, targets, check=None):
        self.singles = np.asarray(singles, float).copy()
        if self.singles.ndim != 1:
            raise ValueError('One single per schema gene required')
        pairs = checked_pairs(pairs, len(self.singles), unique=True)
        y = np.asarray(targets, float)
        if (len(pairs) < self.n_centers or y.shape != (len(pairs),) or
                not np.isfinite(y).all() or not np.isfinite(self.singles[pairs]).all()):
            raise ValueError('Enough finite training rows and exact queried singles required')
        self.support = np.bincount(pairs.ravel(), minlength=len(self.singles)) > 0
        sorted_singles = np.sort(self.singles[pairs], axis=1)
        self.input_center = sorted_singles.mean(0)
        self.input_scale = np.maximum(sorted_singles.std(0), 1e-8)
        coordinates = self._coordinates(pairs)
        self.center_rows = np.random.default_rng(self.seed).choice(len(pairs), self.n_centers, replace=False)
        self.centers = coordinates[self.center_rows].copy()
        distances = np.sqrt(np.square(self.centers[:, None]-self.centers[None, :]).sum(2))
        nonzero = distances[np.triu_indices(self.n_centers, 1)]
        nonzero = nonzero[nonzero > 0]
        if not len(nonzero):
            raise ValueError('Training centers have no nonzero distance')
        self.bandwidth = float(np.median(nonzero))
        values, vectors = np.linalg.eigh(self._kernel(self.centers, self.centers))
        self.whitener = vectors/np.sqrt(np.maximum(values, 1e-10))
        width = 3+(len(self.singles) if self.nodes else 0)+self.n_centers
        total, squares = np.zeros(width), np.zeros(width)
        gram, cross = np.zeros((width, width)), np.zeros(width)
        residual = y-self.singles[pairs].sum(1)
        for start in range(0, len(pairs), 2048):
            if check is not None:
                check()
            rows = slice(start, start+2048)
            x = self._features(pairs[rows])
            total += x.sum(0)
            squares += np.square(x).sum(0)
            gram += x.T@x
            cross += x.T@residual[rows]
        self.center = total/len(pairs)
        self.scale = np.maximum(np.sqrt(np.maximum(squares/len(pairs)-self.center**2, 0)), 1e-8)
        self.offset = float(residual.mean())
        self.gram = (gram-len(pairs)*np.outer(self.center, self.center))/np.outer(self.scale, self.scale)
        self.cross = (cross-self.center*residual.sum())/self.scale
        return self

    def solve(self, alpha=None):
        if alpha is not None:
            if not np.isfinite(alpha) or alpha <= 0:
                raise ValueError('Positive finite ridge required')
            self.alpha = float(alpha)
        self.coefficient = np.linalg.solve(self.gram+self.alpha*np.eye(len(self.center)), self.cross)
        return self

    def fit(self, singles, pairs, targets, check=None):
        return self.prepare(singles, pairs, targets, check).solve()

    def predict(self, pairs):
        pairs = checked_pairs(pairs, len(self.singles))
        if not self.support[pairs].all():
            raise ValueError('Query genes need training-pair support')
        if not np.isfinite(self.singles[pairs]).all():
            raise ValueError('Finite queried singles required')
        prediction = self.singles[pairs].sum(1)+self.offset
        for start in range(0, len(pairs), 2048):
            rows = slice(start, start+2048)
            prediction[rows] += (self._features(pairs[rows])-self.center)/self.scale@self.coefficient
        return prediction

    def save(self, path):
        names = ('singles', 'support', 'input_center', 'input_scale', 'center_rows',
                 'centers', 'whitener', 'bandwidth', 'center', 'scale', 'offset', 'coefficient')
        np.savez_compressed(Path(path), format_version=1, nodes=self.nodes, alpha=self.alpha,
                            n_centers=self.n_centers, seed=self.seed, **{n: getattr(self, n) for n in names})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if int(z['format_version']) != 1:
                raise ValueError('Unknown flexible singles format')
            obj = cls(bool(z['nodes']), float(z['alpha']), int(z['n_centers']), int(z['seed']))
            for n in ('singles', 'support', 'input_center', 'input_scale', 'center_rows',
                      'centers', 'whitener', 'center', 'scale', 'coefficient'):
                setattr(obj, n, z[n].copy())
            for n in ('bandwidth', 'offset'):
                setattr(obj, n, float(z[n]))
        if (not np.isfinite(obj.coefficient).all() or obj.bandwidth <= 0 or
                obj.coefficient.shape != obj.center.shape or obj.scale.shape != obj.center.shape):
            raise ValueError('Invalid flexible singles checkpoint')
        return obj
