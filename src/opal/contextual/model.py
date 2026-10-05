"""Symmetric context-conditioned residual kernels and explicit main effects."""
import numpy as np
from scipy.linalg import solve


class ContextualKernel:
    FAMILIES = {"shared_offset", "global", "pooled_pair", "contextual_pair", "pooled_sum", "contextual_sum"}

    def __init__(self, bank, config=None):
        self.bank = bank
        self.config = {"kind": "contextual_pair", "alpha": 1., "bandwidth": 1.,
                       "context_bandwidth": 1., "context_weight": 1., "global_weight": 1., **(config or {})}
        if self.config["kind"].endswith("_sum") and "global_weight" not in (config or {}):
            self.config["global_weight"] = 0.
        if self.config["kind"] not in self.FAMILIES:
            raise ValueError("Unknown contextual kernel family")
        for key in ["alpha", "bandwidth", "context_bandwidth"]:
            if not np.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not np.isfinite(self.config["context_weight"]) or self.config["context_weight"] < 0:
            raise ValueError("Context weight must be nonnegative and finite")
        if not np.isfinite(self.config["global_weight"]) or self.config["global_weight"] < 0:
            raise ValueError("Global weight must be nonnegative and finite")
        a = bank.arrays
        # All declared measured references are allowed, including query-context
        # singles. No pair outcomes fit these transductive reference scales.
        profiles = np.asarray(a["singles"], dtype=np.float64)
        observed = profiles[a["available"]]
        if not len(observed):
            raise ValueError("No measured references")
        self.gene_scale = np.maximum(observed.std(0), 1e-6)
        self.scaled = (profiles / self.gene_scale).reshape(-1, profiles.shape[-1])
        indices = np.flatnonzero(a["available"].ravel())
        # Bound bandwidth estimation memory rather than allocating a square
        # matrix over every single in a multi-context corpus.
        sampled = indices if len(indices) <= 512 else np.random.default_rng(0).choice(indices, 512, replace=False)
        profiles = self.scaled[sampled]
        norms = np.sum(profiles ** 2, axis=1)
        positive = np.maximum(norms[:, None] + norms[None] - 2 * profiles @ profiles.T, 0)
        positive = positive[positive > 1e-12]
        self.reference_distance = float(np.median(positive)) if len(positive) else 1.

    def _reference_indices(self, pairs, contexts):
        return pairs + contexts[:, None] * len(self.bank.arrays["targets"])

    def _components(self, left, lc, right, rc):
        left, lc = self.bank.validate_rows(left, lc)
        right, rc = self.bank.validate_rows(right, rc)
        li, ri = self._reference_indices(left, lc), self._reference_indices(right, rc)
        return self._profile_components(self.scaled[li], lc, self.scaled[ri], rc)

    def _profile_components(self, left, lc, right, rc):
        lx, rx = left.sum(1), right.sum(1)
        global_term = lx @ rx.T / lx.shape[1]
        lprofile, rprofile = left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1])
        sd = np.maximum(np.sum(lprofile ** 2, axis=1)[:, None] + np.sum(rprofile ** 2, axis=1)[None] - 2 * lprofile @ rprofile.T, 0)
        single = np.exp(-sd / (self.reference_distance * self.config["bandwidth"]))
        cube = single.reshape(len(left), 2, len(right), 2)
        pair = (cube[:, 0, :, 0] * cube[:, 1, :, 1] + cube[:, 0, :, 1] * cube[:, 1, :, 0]) / 2
        if self.config["kind"].endswith("_sum"):
            pair = cube.sum(axis=(1, 3)) / 4
        f = self.bank.arrays["features"]
        lf, rf = (f[lc] - self.context_mean) / self.context_scale, (f[rc] - self.context_mean) / self.context_scale
        dist = np.maximum(np.sum(lf ** 2, 1)[:, None] + np.sum(rf ** 2, 1)[None] - 2 * lf @ rf.T, 0)
        ck = np.exp(-dist / (self.context_distance * self.config["context_bandwidth"]))
        return global_term, pair, pair * ck

    def kernel(self, left, lc, right, rc):
        g, p, cp = self._components(left, lc, right, rc)
        return self._combine(g, p, cp)

    def _combine(self, g, p, cp):
        kind = self.config["kind"]
        if kind == "shared_offset":
            return np.zeros_like(g)
        if kind == "global":
            return g / self.global_scale
        result = self.config["global_weight"] * g / self.global_scale + p / self.pair_scale
        if kind.startswith("contextual_"):
            result += self.config["context_weight"] * cp / self.pair_scale
        return result

    def fit(self, pairs, contexts, responses):
        self.train_pairs, self.train_contexts = self.bank.validate_rows(pairs, contexts)
        y = np.asarray(responses, dtype=np.float64)
        if not len(pairs) or y.shape != (len(pairs), len(self.bank.arrays["genes"])) or not np.isfinite(y).all():
            raise ValueError("Invalid fitting responses")
        source = self.bank.arrays["features"][np.unique(self.train_contexts)]
        self.context_mean = source.mean(0)
        std = source.std(0)
        self.context_scale = np.where(std > 1e-6, std, 1.)
        z = (source - self.context_mean) / self.context_scale
        distances = np.sum((z[:, None] - z[None]) ** 2, axis=-1)
        positive = distances[distances > 1e-12]
        self.context_distance = float(np.median(positive)) if len(positive) else 1.
        g, p, _ = self._components(self.train_pairs, self.train_contexts, self.train_pairs, self.train_contexts)
        self.global_scale, self.pair_scale = max(float(np.diag(g).mean()), 1e-12), max(float(np.diag(p).mean()), 1e-12)
        k = self.kernel(self.train_pairs, self.train_contexts, self.train_pairs, self.train_contexts)
        self.column_mean = k.mean(0)
        centered = k - self.column_mean[None] - self.column_mean[:, None] + k.mean()
        residual = y - self.bank.additive(self.train_pairs, self.train_contexts)
        self.mean = residual.mean(0)
        self.coefficients = solve(centered + self.config["alpha"] * np.eye(len(k)), residual - self.mean, assume_a="pos")
        return self

    def weights(self, pairs, contexts):
        k = self.kernel(pairs, contexts, self.train_pairs, self.train_contexts)
        cross = k - self.column_mean[None] - k.mean(1)[:, None] + self.column_mean.mean()
        train = self.kernel(self.train_pairs, self.train_contexts, self.train_pairs, self.train_contexts)
        train = train - self.column_mean[None] - self.column_mean[:, None] + self.column_mean.mean()
        w = solve(train + self.config["alpha"] * np.eye(len(train)), cross.T, assume_a="pos").T
        return w - w.mean(1)[:, None] + 1 / len(train)

    def predict(self, pairs, contexts):
        additive = self.bank.additive(pairs, contexts)
        k = self.kernel(pairs, contexts, self.train_pairs, self.train_contexts)
        centered = k - self.column_mean[None] - k.mean(1)[:, None] + self.column_mean.mean()
        return additive + self.mean + centered @ self.coefficients

    def predict_profiles(self, profiles, contexts):
        """Explicit estimated-single route; never changes measured availability."""
        profiles = np.asarray(profiles, dtype=float)
        contexts = np.asarray(contexts)
        if profiles.ndim != 3 or profiles.shape[1:] != (2, len(self.bank.arrays['genes'])) or not np.isfinite(profiles).all():
            raise ValueError('Estimated profiles must have shape (queries, 2, genes) and be finite')
        if contexts.shape != (len(profiles),) or contexts.dtype.kind not in 'iu' or np.any(contexts < 0) or np.any(contexts >= len(self.bank.arrays['contexts'])):
            raise ValueError('Invalid estimated-profile contexts')
        indices = self._reference_indices(self.train_pairs, self.train_contexts)
        components = self._profile_components(profiles / self.gene_scale, contexts, self.scaled[indices], self.train_contexts)
        k = self._combine(*components)
        centered = k - self.column_mean[None] - k.mean(1)[:, None] + self.column_mean.mean()
        return profiles.sum(1) + self.mean + centered @ self.coefficients

    def state(self):
        names = ["train_pairs", "train_contexts", "context_mean", "context_scale", "context_distance",
                 "global_scale", "pair_scale", "column_mean", "mean", "coefficients", "gene_scale", "reference_distance"]
        return {k: np.asarray(getattr(self, k)) for k in names}

    def restore(self, state):
        expected = {"train_pairs", "train_contexts", "context_mean", "context_scale", "context_distance",
                    "global_scale", "pair_scale", "column_mean", "mean", "coefficients"}
        persisted_reference = {'gene_scale', 'reference_distance'}
        if set(state) not in [expected, expected | persisted_reference] or any(not np.isfinite(v).all() for v in state.values()):
            raise ValueError("Invalid contextual checkpoint state")
        pairs, contexts = self.bank.validate_rows(state["train_pairs"], state["train_contexts"])
        n, g, d = len(pairs), len(self.bank.arrays["genes"]), self.bank.arrays["features"].shape[1]
        shapes = {"column_mean": (n,), "mean": (g,), "coefficients": (n, g), "context_mean": (d,), "context_scale": (d,),
                  "context_distance": (), "global_scale": (), "pair_scale": ()}
        if persisted_reference <= set(state):
            shapes.update({'gene_scale': (g,), 'reference_distance': ()})
        if not n or any(state[k].shape != shape for k, shape in shapes.items()):
            raise ValueError("Contextual checkpoint shape mismatch")
        if any(np.any(state[k] <= 0) for k in ["context_scale", "context_distance", "global_scale", "pair_scale"] + list(persisted_reference & set(state))):
            raise ValueError("Invalid contextual checkpoint scales")
        for k, v in state.items():
            setattr(self, k, v.copy())
        self.scaled = (np.asarray(self.bank.arrays['singles'], dtype=np.float64) / self.gene_scale).reshape(-1, g)
        return self
