"""A separate estimated-single contract, with no inferred measurement flags."""
import numpy as np
from scipy.linalg import solve


class MainEffectImputer:
    def __init__(self, bank, kind='target_mean', alpha=1., bandwidth=1.):
        if kind not in {'target_mean', 'control_kernel'}:
            raise ValueError('Unknown main-effect model')
        if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError('Invalid main-effect regularization or bandwidth')
        self.bank, self.kind, self.alpha, self.bandwidth = bank, kind, alpha, bandwidth

    def predict(self, target, context):
        a = self.bank.arrays
        if target not in range(len(a['targets'])) or context not in range(len(a['contexts'])):
            raise ValueError('Unknown main-effect identity')
        if a['available'][context, target]:
            return a['singles'][context, target].copy(), {'observed': True, 'source_contexts': [int(context)]}
        source = np.flatnonzero(a['available'][:, target])
        if not len(source):
            raise ValueError('No measured source for the exact intervention and dose; prediction must abstain')
        y = np.asarray(a['singles'][source, target], dtype=np.float64)
        mean = y.mean(0)
        if self.kind == 'target_mean' or len(source) == 1:
            prediction = mean
        else:
            features = np.asarray(a['features'][source], dtype=np.float64)
            scale = np.maximum(features.std(0), 1e-6)
            z = (features - features.mean(0)) / scale
            q = (np.asarray(a['features'][context], dtype=np.float64) - features.mean(0)) / scale
            distances = np.sum((z[:, None] - z[None]) ** 2, axis=-1)
            positive = distances[distances > 1e-12]
            length = (float(np.median(positive)) if len(positive) else 1.) * self.bandwidth
            k = np.exp(-distances / length)
            km = k.mean(0)
            centered = k - km[None] - km[:, None] + k.mean()
            cross = np.exp(-np.sum((z - q) ** 2, axis=1) / length)
            cross = cross - km - cross.mean() + k.mean()
            prediction = mean + cross @ solve(centered + self.alpha * np.eye(len(source)), y - mean, assume_a='pos')
        return prediction, {'observed': False, 'source_contexts': source.tolist(),
                            'method': self.kind, 'formal_coverage_guarantee': False}


class MixedSinglePredictor:
    """Use a fitted residual model while exposing every estimated main effect."""
    def __init__(self, model, imputer):
        if model.bank.fingerprint != imputer.bank.fingerprint:
            raise ValueError('Main and residual reference banks differ')
        self.model, self.imputer = model, imputer

    def predict(self, a, b, *, context):
        bank = self.model.bank
        targets = {str(t): i for i, t in enumerate(bank.arrays['targets'])}
        contexts = {str(c): i for i, c in enumerate(bank.arrays['contexts'])}
        if a == b or a not in targets or b not in targets or context not in contexts:
            raise ValueError('Unsupported pair or context')
        ci = contexts[context]
        profiles, provenance = zip(*(self.imputer.predict(targets[t], ci) for t in [a, b]))
        profiles = np.array(profiles)
        effect = self.model.predict_profiles(profiles[None], np.array([ci]))[0]
        additive = profiles.sum(0)
        return {'effect': effect, 'additive': additive, 'residual': effect - additive,
                'single_profiles': profiles, 'single_provenance': list(provenance),
                'genes': bank.arrays['genes'].copy(), 'context': context,
                'route': 'estimated_single_pair' if any(not p['observed'] for p in provenance) else 'observed_single_pair',
                'schema': 'opal-contextual-estimated-singles-v1',
                'reliability_status': 'experimental_no_calibrated_interval',
                'formal_coverage_guarantee': False}


def export_estimated_checkpoint(model, imputer, directory, provenance=None):
    from pathlib import Path
    from opal.artifacts import sha256, write_json
    from .inference import export_checkpoint
    MixedSinglePredictor(model, imputer)
    directory = Path(directory)
    export_checkpoint(model, directory, provenance=provenance)
    write_json(directory / 'estimated_manifest.json', {
        'schema': 'opal-contextual-estimated-singles-v1',
        'observed_manifest_sha256': sha256(directory / 'manifest.json'),
        'main_effect_config': {'kind': imputer.kind, 'alpha': imputer.alpha, 'bandwidth': imputer.bandwidth},
        'uncertainty': 'unavailable; observed-single calibration does not apply',
        'combination_outcomes_included': False})


def load_estimated_checkpoint(directory):
    from pathlib import Path
    from opal.artifacts import read_json, sha256
    from .inference import ContextPredictor
    directory = Path(directory)
    manifest = read_json(directory / 'estimated_manifest.json')
    if manifest.get('schema') != 'opal-contextual-estimated-singles-v1' or sha256(directory / 'manifest.json') != manifest['observed_manifest_sha256']:
        raise ValueError('Estimated-single checkpoint schema or fingerprint mismatch')
    observed = ContextPredictor(directory)
    return MixedSinglePredictor(observed.model, MainEffectImputer(observed.bank, **manifest['main_effect_config']))
