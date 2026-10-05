"""Symmetric residual kernels with a shared offset and development-only tuning."""
import numpy as np
from scipy.linalg import solve


def single_kernel(singles, kind='rbf', bandwidth=1.0):
    x = np.asarray(singles, dtype=np.float64)
    gram = x @ x.T
    if kind == 'linear':
        return gram / max(float(np.trace(gram) / len(x)), 1e-12)
    distance = np.maximum(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2 * gram, 0)
    positive = distance[np.triu_indices(len(x), 1)]
    scale = max(float(np.median(positive[positive > 0])) if np.any(positive > 0) else 1, 1e-12)
    return np.exp(-distance / (scale * bandwidth))


def symmetric_kernel(single_gram, left, right, interaction=False):
    a, b = np.asarray(left).T
    c, d = np.asarray(right).T
    ac, ad = single_gram[np.ix_(a, c)], single_gram[np.ix_(a, d)]
    bc, bd = single_gram[np.ix_(b, c)], single_gram[np.ix_(b, d)]
    return (ac * bd + ad * bc) / 2 if interaction else (ac + ad + bc + bd) / 4


def global_kernel(singles, pairs):
    additive = singles[pairs[:, 0]] + singles[pairs[:, 1]]
    # Reference-only scale, so outer/inner response holdouts cannot fit it.
    scale = np.maximum(np.std(singles, axis=0) * np.sqrt(2), 1e-6)
    x = additive / scale
    return x @ x.T / x.shape[1]


def smoother(kernel, train, query, alpha):
    """Return query-to-training weights, including an unpenalized intercept."""
    k = np.asarray(kernel[np.ix_(train, train)], np.float64)
    cross = np.asarray(kernel[np.ix_(query, train)], np.float64)
    column_mean = k.mean(0)
    centered = k - column_mean[None, :] - column_mean[:, None] + k.mean()
    cross = cross - column_mean[None, :] - cross.mean(1)[:, None] + k.mean()
    weights = solve(centered + alpha * np.eye(len(train)), cross.T, assume_a='pos').T
    return weights - weights.mean(1)[:, None] + 1 / len(train)


def prediction_error(weights, train, query, residual_gram):
    """Exact equal-pair MSE in gene units without reconstructing every candidate."""
    yy = residual_gram[np.ix_(train, train)]
    cross = residual_gram[np.ix_(query, train)]
    error = np.einsum('ij,jk,ik->i', weights, yy, weights)
    error += np.diag(residual_gram)[query] - 2 * np.sum(weights * cross, axis=1)
    return float(np.mean(error))


class KernelResidual:
    """Closed-form observed-single model; pair responses are fitting targets only."""
    def __init__(self, singles, config):
        self.singles = np.asarray(singles, np.float64)
        self.config = dict(config)
        kinds = {'shared_offset', 'global', 'single_sum', 'pair_product',
                 'global_plus_pair', 'global_plus_pair_10'}
        if config['kind'] not in kinds or config.get('alpha', 0) <= 0:
            raise ValueError('Unknown kernel kind or nonpositive regularization')
        if config.get('bandwidth', 1) <= 0 or not np.isfinite(self.singles).all():
            raise ValueError('Invalid bandwidth or single profiles')
        self.scale = np.maximum(self.singles.std(0) * np.sqrt(2), 1e-6)
        self.single_gram = single_kernel(self.singles, bandwidth=config.get('bandwidth', 1))
        scaled = self.singles / self.scale
        self.linear_gram = scaled @ scaled.T / self.singles.shape[1]

    def kernel(self, left, right):
        kind = self.config['kind']
        if kind == 'shared_offset':
            return np.zeros((len(left), len(right)))
        if kind == 'single_sum':
            return symmetric_kernel(self.single_gram, left, right)
        if kind == 'pair_product':
            return symmetric_kernel(self.single_gram, left, right, True)
        kernel = 4 * symmetric_kernel(self.linear_gram, left, right)
        if kind.startswith('global_plus_pair'):
            weight = 10 if kind.endswith('_10') else 1
            kernel += weight * symmetric_kernel(self.single_gram, left, right, True)
        return kernel

    def fit(self, pairs, doubles):
        pairs = np.asarray(pairs, int)
        doubles = np.asarray(doubles, np.float64)
        if not len(pairs) or doubles.shape != (len(pairs), self.singles.shape[1]):
            raise ValueError('Empty or mismatched fitting profiles')
        if not np.isfinite(doubles).all():
            raise ValueError('Nonfinite pair responses')
        self.train_pairs = pairs.copy()
        residual = doubles - self.singles[pairs[:, 0]] - self.singles[pairs[:, 1]]
        self.mean = residual.mean(0)
        kernel = self.kernel(pairs, pairs)
        self.column_mean = kernel.mean(0)
        centered = kernel - self.column_mean[None, :] - self.column_mean[:, None] + kernel.mean()
        self.coefficients = solve(centered + self.config['alpha'] * np.eye(len(pairs)),
                                  residual - self.mean, assume_a='pos')
        return self

    def predict(self, pairs):
        pairs = np.asarray(pairs, int)
        kernel = self.kernel(pairs, self.train_pairs)
        centered = kernel - self.column_mean[None, :] - kernel.mean(1)[:, None] + self.column_mean.mean()
        return (self.singles[pairs[:, 0]] + self.singles[pairs[:, 1]]
                + centered @ self.coefficients + self.mean)

    def state(self):
        return {name: getattr(self, name) for name in ['train_pairs', 'mean', 'column_mean', 'coefficients']}

    def restore(self, arrays):
        shapes = {'train_pairs': (len(arrays['train_pairs']), 2),
                  'mean': (self.singles.shape[1],), 'column_mean': (len(arrays['train_pairs']),),
                  'coefficients': (len(arrays['train_pairs']), self.singles.shape[1])}
        for name, shape in shapes.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError('Kernel checkpoint shape or finiteness mismatch')
            setattr(self, name, arrays[name])
        if self.train_pairs.dtype.kind not in 'iu' or np.any(self.train_pairs < 0) or np.any(self.train_pairs >= len(self.singles)):
            raise ValueError('Invalid kernel checkpoint target indices')
        return self
