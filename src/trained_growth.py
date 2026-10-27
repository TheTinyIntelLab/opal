"""Named-gene API for one exported assay-local growth checkpoint.

The checkpoint bytes and stored singles come from a local delivery bundle.
Queries do not supply new singles, and this module does not fit, score, or
calibrate an interval. Per-query uncertainty is unavailable.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from opal.frontier_packing_v2 import PACKING_CONTRACT, MixedLoader
from opal.frontier_tournament import CONTRACT
from opal.scalar_screen import (
    EXTERNAL_HASHES, EXTERNAL_PREPARATION_HASH, ORIGINAL_HASH, ORIGINAL_SOURCE_HASH, TASKS,
)


DELIVERY_CONTRACT = 'opal-trained-growth-delivery-v1'
RUN_CONTRACT = 'opal-frontier-growth-v3'
LOADER = 'opal.frontier_packing_v2.MixedLoader'
LABEL = 'assay-local research model; all-task promotion gate failed'
TRAIN_LABELS = 16000
VALIDATION_LABELS = 4000
FOLD = 0
PER_QUERY_UNCERTAINTY = 'unavailable'
BENCHMARK_RELATION = (
    'pooled five-fold training-procedure benchmark; not a fold-0-only score '
    'and not a score of this single fold-0 checkpoint')
SUPPORT_RULE = (
    'TRAIN degree >= 2 distinct partners; validation and test rows do not grant support')
ENDPOINT = 'mean of two infection-specific gamma measurements'
ENDPOINT_QUALITY = 'frozen processed bank; no independent-infection claim'

ASSAYS = {
    'horlbeck_K562': dict(
        source='Horlbeck 2018', accession='GSE116198', context='K562', units='gamma',
        assay='Horlbeck CRISPRi growth screen, K562 context',
        expected_bank_sha256=ORIGINAL_HASH),
    'horlbeck_Jurkat': dict(
        source='Horlbeck 2018', accession='GSE116198', context='Jurkat', units='gamma',
        assay='Horlbeck CRISPRi growth screen, Jurkat context',
        expected_bank_sha256=ORIGINAL_HASH),
    'environment1': dict(
        source='environmental', accession='GSE312636', context='K562_environment1', units='gamma',
        assay='Environmental CRISPRi growth cohort, K562 context',
        expected_bank_sha256=EXTERNAL_HASHES['environment1']),
    'environment2': dict(
        source='environmental', accession='GSE312636', context='K562_environment2', units='gamma',
        assay='Environmental CRISPRi growth cohort, K562 context',
        expected_bank_sha256=EXTERNAL_HASHES['environment2']),
    'parp2022': dict(
        source='PARPi', accession='GSE284018', context='K562_parp2022', units='gamma',
        assay='PARPi CRISPRi growth cohort, K562 context',
        expected_bank_sha256=EXTERNAL_HASHES['parp2022']),
    'parp2023': dict(
        source='PARPi', accession='GSE284018', context='K562_parp2023', units='gamma',
        assay='PARPi CRISPRi growth cohort, K562 context',
        expected_bank_sha256=EXTERNAL_HASHES['parp2023']),
}
SOURCE_HASHES = dict(
    identity='original_pre_fit_snapshot',
    original_hash=ORIGINAL_HASH, original_source_hash=ORIGINAL_SOURCE_HASH,
    external_preparation_hash=EXTERNAL_PREPARATION_HASH, external_hashes=dict(EXTERNAL_HASHES))

if tuple(ASSAYS) != TASKS:
    raise RuntimeError('Assay table drifted from the registered task order')
for _assay in ASSAYS.values():
    _assay['endpoint'] = ENDPOINT
    _assay['endpoint_quality'] = ENDPOINT_QUALITY


def train_degree_support(pairs, train_index, n_genes):
    """Registered TRAIN support rule.

    A gene is supported only when it has two distinct partners inside the
    charged TRAIN rows. Validation and test rows are not an argument.
    """
    pairs = np.asarray(pairs)
    train_index = np.asarray(train_index)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError('Integer pair matrix required')
    if isinstance(n_genes, (bool, np.bool_)) or not isinstance(n_genes, (int, np.integer)) or n_genes < 1:
        raise ValueError('Gene schema is empty')
    if train_index.ndim != 1 or not np.issubdtype(train_index.dtype, np.integer):
        raise ValueError('Integer TRAIN index required')
    if not len(train_index) or np.any(train_index < 0) or np.any(train_index >= len(pairs)):
        raise ValueError('TRAIN index outside the pair roster')
    if len(np.unique(train_index)) != len(train_index):
        raise ValueError('Duplicate TRAIN rows')
    used = np.asarray(pairs[train_index], dtype=np.int64)
    if np.any(used < 0) or np.any(used >= n_genes) or np.any(used[:, 0] == used[:, 1]):
        raise ValueError('TRAIN pairs leave the gene schema')
    if len(np.unique(np.sort(used, axis=1), axis=0)) != len(used):
        raise ValueError('Duplicate canonical TRAIN pairs')
    degree = np.bincount(used.ravel(), minlength=int(n_genes)).astype(np.int64, copy=False)
    return degree, degree >= 2


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _member(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or any(part in ('', '.', '..') for part in relative.parts):
        raise ValueError('Bundle member path is not local')
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('Bundle member escapes the bundle')
    return path


def _pair_rows(pairs):
    if isinstance(pairs, (str, bytes)):
        raise TypeError('Pairs must be a sequence of name pairs')
    if isinstance(pairs, tuple) and len(pairs) == 2 and all(isinstance(item, str) for item in pairs):
        return [pairs]
    if isinstance(pairs, np.ndarray):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise TypeError('Pairs must be a sequence of name pairs')
        return [tuple(row) for row in pairs.tolist()]
    rows = []
    for row in pairs:
        if isinstance(row, (str, bytes)):
            raise TypeError('Each query must contain two gene names')
        members = tuple(row)
        if len(members) != 2:
            raise TypeError('Each query must contain two gene names')
        rows.append(members)
    return rows


def _clean_name(value):
    if not isinstance(value, str):
        return None
    name = value.strip()
    if not name or '|' in name or name != value:
        return None
    return name


def _same_state(left, right):
    if set(left) != set(right):
        return False
    for key in left:
        a, b = np.asarray(left[key]), np.asarray(right[key])
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
            return False
    return True


def same_predictor(left, right):
    """Exact array and dtype equality, including a residual base when present."""
    if not _same_state(left.state, right.state):
        return False
    if (left.base is None) != (right.base is None):
        return False
    if left.base is not None and not _same_state(left.base.state, right.base.state):
        return False
    return True


class TaskModel:
    """One task, one context, and the stored singles from its export."""

    def __init__(self, bundle, record, model, names, dropped, degree):
        self.bundle = bundle
        self.task = record['task']
        self.context = record['context']
        self.assay = record.get('assay')
        self.source = record.get('source')
        self.accession = record.get('accession')
        self.units = record.get('units')
        self.endpoint = record.get('endpoint')
        self._model = model
        self._names = tuple(names)
        self._dropped = set(dropped)
        self._index = {name: i for i, name in enumerate(self._names)}
        self._degree = np.array(degree, dtype=np.int64, copy=True)
        self._support = np.array(model.state['support'], dtype=bool, copy=True)

    def gene_names(self):
        return self._names

    def support(self):
        return self._support.copy()

    def stored_singles(self):
        """Copy of the source-local singles stored with the checkpoint."""
        return np.array(self._model.state['singles'], copy=True)

    def uncertainty(self):
        return dict(
            per_query_calibrated_uncertainty=PER_QUERY_UNCERTAINTY,
            wrapper_fit=False,
            conditional_gene_interval='aggregate_only',
            conditional_gene_interval_scope=BENCHMARK_RELATION,
            value=None)

    def predict(self, pairs, *, context=None):
        """Predict named gene pairs, or abstain explicitly.

        ``context`` must be omitted or equal this task's assay context.
        A different context produces no number. Stored singles are not an
        argument.
        """
        rows = _pair_rows(pairs)
        if context is not None and context != self.context:
            return [self._record(row, 'abstain', 'unknown_context', None) for row in rows]
        prepared, results = [], [None]*len(rows)
        for index, row in enumerate(rows):
            reason, endpoints = self._classify(row)
            if reason is not None:
                results[index] = self._record(row, 'abstain', reason, None)
            else:
                prepared.append((index, endpoints))
        if prepared:
            query = np.asarray([item[1] for item in prepared], dtype=np.int64)
            values = self._model.predict(query)
            if values.shape != (len(prepared),) or not np.isfinite(values).all():
                raise ValueError('Nonfinite prediction')
            for (index, _), value in zip(prepared, values, strict=True):
                results[index] = self._record(rows[index], 'predicted', None, float(value))
        return results

    def _classify(self, row):
        left, right = (_clean_name(row[0]), _clean_name(row[1]))
        if left is None or right is None:
            return 'malformed_name', None
        if left == right:
            return 'self_pair', None
        problems, indices = [], []
        for name in (left, right):
            if name in self._index:
                problems.append(None)
                indices.append(self._index[name])
            elif name in self._dropped:
                problems.append('absent_gene')
            else:
                problems.append('unknown_gene')
        if 'unknown_gene' in problems:
            return 'unknown_gene', None
        if 'absent_gene' in problems:
            return 'absent_gene', None
        if not bool(self._support[indices].all()):
            return 'unsupported_gene', None
        return None, (indices[0], indices[1])

    def _record(self, row, status, reason, value):
        shown = [item if isinstance(item, str) else None for item in row]
        return dict(
            status=status, reason=reason, value=value, genes=shown, task=self.task,
            context=self.context, uncertainty=PER_QUERY_UNCERTAINTY,
            label=self.bundle.label, biological_delivery=self.bundle.biological_delivery)


class TrainedGrowth:
    """Local bundle of validation-selected fold-0 checkpoints."""

    def __init__(self, root, manifest, models):
        self.root = Path(root)
        self.manifest = manifest
        self.label = manifest['label']
        self.biological_delivery = bool(manifest['biological_delivery'])
        self._models = models
        self.tasks = tuple(record['task'] for record in manifest['tasks'])

    def model(self, task):
        try:
            return self._models[task]
        except KeyError:
            raise ValueError('Unknown task') from None

    def predict(self, task, pairs, *, context=None):
        return self.model(task).predict(pairs, context=context)

    def uncertainty(self, task=None):
        if task is None:
            return self.model(self.tasks[0]).uncertainty()
        return self.model(task).uncertainty()

    @classmethod
    def load(cls, root, expected_manifest_sha256=None):
        """Load a local bundle.

        Pass ``expected_manifest_sha256`` to pin the manifest bytes before they
        are parsed. Checkpoint and schema hashes do not by themselves stop a
        swap of one self-consistent manifest for another.
        """
        root = Path(root)
        manifest_path = root / 'manifest.json'
        payload = manifest_path.read_bytes()
        if expected_manifest_sha256 is not None:
            if hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
                raise ValueError('Manifest receipt mismatch')
        manifest = json.loads(payload)
        _require_manifest(manifest)
        models = {}
        for record in manifest['tasks']:
            task = record['task']
            if task in models:
                raise ValueError('Duplicate task in delivery manifest')
            checkpoint = _member(root, record['checkpoint'])
            schema_path = _member(root, record['schema'])
            if _sha256(checkpoint) != record['checkpoint_sha256']:
                raise ValueError('Checkpoint integrity mismatch')
            if _sha256(schema_path) != record['schema_sha256']:
                raise ValueError('Schema integrity mismatch')
            with np.load(schema_path, allow_pickle=False) as schema:
                names = _names(schema['gene_names'])
                dropped = _names(schema['dropped_names'])
                singles = schema['singles'].copy()
                support = schema['support'].copy()
                degree = schema['train_degree'].copy()
            _require_schema_vectors(names, singles, support, degree)
            if set(dropped) & set(names):
                raise ValueError('Dropped genes overlap the schema')
            historical = (checkpoint.parent.resolve(), root.resolve())
            model = MixedLoader(historical_roots=historical).load(
                checkpoint, record['checkpoint_sha256'])
            _bind_checkpoint(model, singles, support)
            models[task] = TaskModel(None, record, model, names, dropped, degree)
        bundle = cls(root, manifest, models)
        for model in models.values():
            model.bundle = bundle
        return bundle


def _names(values):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind != 'U':
        raise ValueError('Unicode name vector required')
    names = [str(item) for item in array.tolist()]
    if any(not name or name != name.strip() or '|' in name for name in names):
        raise ValueError('Canonical stored names required')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate stored names')
    return names


def _require_schema_vectors(names, singles, support, degree):
    if singles.ndim != 1 or support.ndim != 1 or degree.ndim != 1:
        raise ValueError('Gene arrays must be one-dimensional')
    if not (len(names) == len(singles) == len(support) == len(degree)) or not len(names):
        raise ValueError('Gene, single, and support counts differ')
    if support.dtype != bool or degree.dtype != np.int64:
        raise ValueError('Support schema is malformed')
    if singles.dtype.kind != 'f' or not np.isfinite(singles).all():
        raise ValueError('Stored singles must be a finite real vector')
    if np.any(degree < 0):
        raise ValueError('TRAIN degree is negative')
    if not np.array_equal(degree >= 2, support):
        raise ValueError('Stored degree does not implement the TRAIN degree rule')


def _bind_checkpoint(model, singles, support):
    if model is None:
        raise ValueError('Checkpoint did not load')
    for layer in (model, model.base):
        if layer is None:
            continue
        state = layer.state
        stored = state['singles']
        if stored.ndim != 1 or stored.dtype.kind != 'f' or not np.isfinite(stored).all():
            raise ValueError('Checkpoint singles must be a finite real vector')
        if stored.shape != singles.shape or not np.array_equal(stored, singles):
            raise ValueError('Checkpoint singles do not match the stored source-local singles')
        if state['support'].shape != support.shape or not np.array_equal(state['support'], support):
            raise ValueError('Checkpoint support does not match the stored TRAIN support')
    if model.base is not None and str(model.state.get('kind')) == 'completion':
        if not np.array_equal(model.state['singles'], model.base.state['singles']):
            raise ValueError('Residual and base singles differ')


def _require_manifest(manifest):
    if manifest.get('contract') != DELIVERY_CONTRACT:
        raise ValueError('Unknown delivery contract')
    if manifest.get('loader') != LOADER or manifest.get('packing_contract') != PACKING_CONTRACT:
        raise ValueError('Delivery loader mismatch')
    if manifest.get('model_contract') != CONTRACT:
        raise ValueError('Predictor contract mismatch')
    if manifest.get('per_query_uncertainty') != PER_QUERY_UNCERTAINTY:
        raise ValueError('Per-query uncertainty must be unavailable')
    if manifest.get('uncertainty_wrapper_fit') is not False:
        raise ValueError('An uncertainty wrapper is not part of this delivery')
    if manifest.get('conditional_gene_interval') != 'aggregate_only':
        raise ValueError('Conditional gene intervals are aggregate only')
    benchmark = manifest.get('benchmark') or {}
    if benchmark.get('metrics_copied') is not False or benchmark.get('fold0_checkpoint_is_the_benchmark') is not False:
        raise ValueError('Fold-0 bundle cannot carry the pooled benchmark')
    if benchmark.get('attribution') != BENCHMARK_RELATION:
        raise ValueError('Benchmark attribution mismatch')
    if manifest.get('new_fits') != 0 or manifest.get('new_scores') != 0:
        raise ValueError('Delivery does not record a new fit or score')
    if manifest.get('refit') is not False or manifest.get('test_dependent_selection') is not False:
        raise ValueError('Delivery refuses refit and test-dependent selection')
    if manifest.get('other_folds_exported') is not False or manifest.get('fold') != FOLD:
        raise ValueError('Only the fixed fold-0 checkpoint is exported')
    rights = manifest.get('rights') or {}
    if rights.get('status') != 'unresolved' or rights.get('redistribution') != 'not_authorized':
        raise ValueError('Weight redistribution is not authorized')
    if rights.get('git') is not False or rights.get('wheel') is not False or rights.get('sdist') is not False:
        raise ValueError('Checkpoints are not package payload')
    budget = manifest.get('information_budget') or {}
    biological = manifest.get('biological_delivery')
    if biological is True:
        if manifest.get('label') != LABEL or manifest.get('run_contract') != RUN_CONTRACT:
            raise ValueError('Biological delivery label mismatch')
        if int(budget.get('train_labels')) != TRAIN_LABELS or int(budget.get('validation_labels')) != VALIDATION_LABELS:
            raise ValueError('Biological information budget mismatch')
        if manifest.get('training_reproduction_substituted') is not False:
            raise ValueError('Reproduction weights are not substituted for the original checkpoint')
        if manifest.get('expected_source_hashes') != SOURCE_HASHES:
            raise ValueError('Source snapshot mismatch')
        records = manifest['tasks']
        if tuple(record.get('task') for record in records) != TASKS:
            raise ValueError('Biological task roster mismatch')
        for record in records:
            assay = ASSAYS[record['task']]
            for key, value in assay.items():
                if record.get(key) != value:
                    raise ValueError('Assay record does not match the registered task')
        gate = manifest.get('program_gate') or {}
        if gate.get('passes') is not False or gate.get('biological_confirmation') is not False or gate.get('rescue') is not False:
            raise ValueError('Biological delivery is the failed all-task gate')
        alignment = manifest.get('protocol_alignment') or {}
        if alignment.get('executed_run_contract') != RUN_CONTRACT:
            raise ValueError('Executed run contract mismatch')
        if benchmark.get('source_result_sha256') in (None, ''):
            raise ValueError('Biological benchmark source hash is required')
    elif biological is False:
        if manifest.get('label') == LABEL:
            raise ValueError('A fixture cannot use the biological delivery label')
        if manifest.get('program_gate') is not None:
            raise ValueError('A fixture cannot carry the biological program gate')
    else:
        raise ValueError('biological_delivery must be explicit')
    if not manifest.get('endpoint_caveats') or not manifest.get('tasks'):
        raise ValueError('Assay caveats and tasks are required')
    for record in manifest['tasks']:
        if int(record.get('fold')) != FOLD:
            raise ValueError('Task record is not fold 0')
        for key in ('task', 'context', 'checkpoint', 'checkpoint_sha256', 'schema', 'schema_sha256'):
            if not record.get(key):
                raise ValueError('Task binding is incomplete')
        if not isinstance(record['context'], str):
            raise ValueError('Assay context is required')
