"""NumPy-only exports for baseline-conditioned, exact-intervention predictions."""
from pathlib import Path
import numpy as np
from opal.artifacts import load_npz, read_json, sha256, write_json

SCHEMA = 'opal-state-response-v1'
ENSEMBLE_SCHEMA = 'opal-state-response-ensemble-v1'


def export_state(main, directory, *, genes, targets, assay, scale, intervention_records,
                 residual=None, provenance=None):
    directory = Path(directory)
    if scale not in {'log2', 'count'} or not assay or set(intervention_records) != set(map(str, targets)):
        raise ValueError('Export requires exact intervention and assay contracts')
    directory.mkdir(parents=True, exist_ok=False)
    state = main.numpy_state()
    if residual is not None:
        state.update({'residual_center': residual.center, 'residual_decoder': residual.decoder})
        state.update({'residual_weight_' + k: v.detach().cpu().numpy() for k, v in residual.head.state_dict().items()})
    np.savez_compressed(directory / 'fitted.npz', **state)
    write_json(directory / 'manifest.json', {'schema': SCHEMA, 'files': {'fitted.npz': sha256(directory / 'fitted.npz')},
        'genes': list(map(str, genes)), 'targets': list(map(str, targets)), 'scale': scale, 'assay': assay,
        'intervention_records': intervention_records, 'active_paths': main.active,
        'config': main.config, 'residual': None if residual is None else {
            'construction': residual.construction, 'state': residual.state, 'joint': residual.joint},
        'population_mean_only': True, 'control_input': 'processed_matched_baseline_in_frozen_gene_order',
        'query_reference_adaptation': False, 'evaluation_labels_included': False,
        'reliability_status': 'experimental_unavailable', 'descriptor_only_unknown_identity': 'unsupported',
        'provenance': provenance or {}})
    manifest = read_json(directory / 'manifest.json')
    manifest['source_evidence'] = getattr(main, 'source_evidence', {})
    manifest['combination_support'] = getattr(residual, 'training_records', []) if residual is not None else []
    write_json(directory / 'manifest.json', manifest)


class StatePredictor:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = read_json(directory / 'manifest.json')
        m = self.manifest
        if m.get('schema') == ENSEMBLE_SCHEMA:
            names = m.get('members', [])
            if names != [f'member-{i}' for i in range(len(names))] or not names:
                raise ValueError('Invalid ensemble member paths')
            if set(m.get('files', {})) != {name + '/manifest.json' for name in names}:
                raise ValueError('Unexpected ensemble files')
            for name, checksum in m['files'].items():
                if sha256(directory / name) != checksum:
                    raise ValueError('Ensemble manifest checksum mismatch')
            self.members = [StatePredictor(directory / name) for name in names]
            for member in self.members:
                if any(member.manifest[key] != m[key] for key in ['genes', 'targets', 'assay', 'scale', 'residual']):
                    raise ValueError('Ensemble member information contracts differ')
            return
        if m.get('schema') != SCHEMA or set(m.get('files', {})) != {'fitted.npz'}:
            raise ValueError('Unsupported state-response checkpoint')
        if sha256(directory / 'fitted.npz') != m['files']['fitted.npz']:
            raise ValueError('State checkpoint checksum mismatch')
        self.arrays = load_npz(directory / 'fitted.npz')
        a = self.arrays
        if any(not np.isfinite(v).all() for v in a.values()):
            raise ValueError('Nonfinite fitted state')
        self.targets = {t: i for i, t in enumerate(m['targets'])}
        if len(self.targets) != len(m['targets']) or len(set(m['genes'])) != len(m['genes']):
            raise ValueError('Repeated checkpoint identifiers')
        if a['target_means'].shape != (len(self.targets), len(m['genes'])) or a['control_center'].shape != (len(m['genes']),):
            raise ValueError('Invalid fitted reference shapes')
        if np.any(a['control_scale'] <= 0):
            raise ValueError('Invalid fitted normalization')
        # Exercise every persisted layer at restoration to detect shape failures.
        self.predict(None, baseline=a['control_center'], genes=m['genes'], assay=m['assay'], scale=m['scale'])
        supported = [t for t, i in self.targets.items() if a['target_support'][i] > 0]
        if not supported:
            raise ValueError('No supported interventions')
        self._main(self.targets[supported[0]], self._state(a['control_center']))

    def _linear(self, x, name, residual=False):
        prefix = 'residual_weight_' if residual else 'weight_'
        return x @ self.arrays[prefix + name + '.weight'].T + self.arrays[prefix + name + '.bias']

    def _state(self, baseline):
        a = self.arrays
        return (baseline - a['control_center']) @ a['control_basis'] / a['control_scale']

    def _intervention(self, ti):
        return np.tanh(sum(self._linear(self.arrays['input_' + k][ti], 'interventions.' + k)
                           for k in self.manifest['active_paths']))

    def _main(self, ti, state):
        z = self._intervention(ti)
        c = np.tanh(self._linear(state, 'state')) if self.manifest['config']['state'] else np.zeros_like(z)
        features = np.concatenate([z, c, z * c])
        coefficient = self._linear(np.tanh(self._linear(features, 'main.0')), 'main.2')
        return self.arrays['target_means'][ti] + coefficient @ self.arrays['output_basis'].T

    def predict(self, a=None, b=None, *, baseline, genes, assay, scale, measured_singles=None,
                return_absolute=True, context=None):
        if hasattr(self, 'members'):
            results = [member.predict(a, b, baseline=baseline, genes=genes, assay=assay, scale=scale,
                       measured_singles=measured_singles, return_absolute=return_absolute, context=context)
                       for member in self.members]
            result = dict(results[0])
            for key in ['effect', 'main_effects', 'additive', 'residual'] + (['absolute_response'] if return_absolute else []):
                result[key] = np.mean([r[key] for r in results], axis=0)
            result['model_variation'] = np.std([r['effect'] for r in results], axis=0)
            result['model_variation_status'] = 'uncalibrated_fitted_model_variation'
            result['ensemble_members'] = len(results)
            return result
        m, fitted = self.manifest, self.arrays
        baseline = np.asarray(baseline, dtype=float)
        if assay != m['assay'] or scale != m['scale'] or list(genes) != m['genes']:
            raise ValueError('Incompatible assay, scale or gene order')
        if baseline.shape != (len(m['genes']),) or not np.isfinite(baseline).all():
            raise ValueError('Baseline must be a finite processed gene vector')
        if (a is None and b is not None) or (a is not None and a == b):
            raise ValueError('Unsupported repeated target or incomplete intervention query')
        interventions = [t for t in [a, b] if t is not None]
        if measured_singles is not None and (not isinstance(measured_singles, dict) or set(measured_singles) != set(interventions)):
            raise ValueError('Measured-single route requires exactly the declared interventions')
        for t in interventions:
            if t not in self.targets or fitted['target_support'][self.targets[t]] <= 0:
                raise ValueError('Unknown intervention or no exact-intervention response support')
        if b is not None and m['residual'] is None:
            raise ValueError('Combination model unavailable in this export')
        state = self._state(baseline)
        profiles, provenance = [], []
        for t in interventions:
            if measured_singles is not None:
                value = np.asarray(measured_singles[t], dtype=float)
                if value.shape != baseline.shape or not np.isfinite(value).all():
                    raise ValueError('Measured singles must be finite matched-control-relative effects')
                profiles.append(value)
                provenance.append({'intervention': t, 'kind': 'measured', 'denoising_correction': 'none'})
            else:
                profiles.append(self._main(self.targets[t], state))
                provenance.append({'intervention': t, 'kind': 'predicted_exact_intervention',
                                   'source_reference_strata_count': int(fitted['target_support'][self.targets[t]]),
                                   'reference_support': m.get('source_evidence', {}).get(t, []),
                                   'reference_support_status': 'identified' if m.get('source_evidence', {}).get(t) else 'legacy_count_only'})
        main = np.sum(profiles, axis=0) if profiles else np.zeros_like(baseline)
        residual = np.zeros_like(baseline)
        if b is not None:
            z1, z2 = (self._intervention(self.targets[t]) for t in interventions)
            features = [z1 + z2, np.abs(z1 - z2)]
            if m['residual']['construction'] == 'product':
                features.append(z1 * z2)
            if m['residual']['state']:
                features.append(np.tanh(self._linear(state, 'state')))
            coef = self._linear(np.tanh(self._linear(np.concatenate(features), '0', True)), '2', True)
            residual = fitted['residual_center'] + coef @ fitted['residual_decoder'].T
        distance = np.linalg.norm((fitted['source_controls'] - baseline) @ fitted['control_basis'] / fitted['control_scale'], axis=1)
        effect = main + residual
        result = {'effect': effect, 'main_effects': np.asarray(profiles).reshape(len(profiles), len(baseline)),
                  'additive': main, 'residual': residual, 'genes': np.asarray(m['genes']),
                  'single_provenance': provenance, 'context': context,
                  'scale': scale, 'route': 'control' if not interventions else
                      ('observed' if measured_singles is not None else 'estimated') + ('_pair' if b is not None else '_single'),
                  'support': {'nearest_source_state_distance': float(distance.min()),
                              'assay_compatible': True, 'exact_intervention_support': True,
                              'distance_status': 'diagnostic_not_calibrated', 'baseline_reference_available': True},
                  'reliability_status': 'experimental_unavailable', 'interval': None,
                  'formal_coverage_guarantee': False}
        result['combination_support'] = m.get('combination_support', []) if b is not None else []
        if return_absolute:
            result['absolute_response'] = baseline + effect
        return result


def export_ensemble(mains, directory, *, residuals=None, **kwargs):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    if not mains or residuals is not None and len(residuals) != len(mains):
        raise ValueError('Ensemble needs aligned main and residual members')
    names = []
    for i, main in enumerate(mains):
        name = f'member-{i}'; names.append(name)
        export_state(main, directory / name, residual=None if residuals is None else residuals[i], **kwargs)
    manifest = read_json(directory / names[0] / 'manifest.json')
    manifest.update({'schema': ENSEMBLE_SCHEMA, 'members': names,
        'files': {name + '/manifest.json': sha256(directory / name / 'manifest.json') for name in names},
        'model_variation': 'uncalibrated; not measurement uncertainty or a prediction interval'})
    write_json(directory / 'manifest.json', manifest)
    StatePredictor(directory)
