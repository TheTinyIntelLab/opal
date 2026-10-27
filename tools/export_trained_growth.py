"""Export original validation-selected fold-0 growth checkpoints.

The biological source is the frozen growth-v3 tree. Bytes are copied. Nothing
is refit, rescored, or replaced by a training reproduction. Per-query
uncertainty is left unavailable. Checkpoint files stay in an ignored local
directory.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from opal.frontier_packing_v2 import PACKING_CONTRACT, MixedLoader
from opal.frontier_tournament import CONTRACT
from opal.trained_growth import (
    ASSAYS, BENCHMARK_RELATION, DELIVERY_CONTRACT, ENDPOINT, ENDPOINT_QUALITY, FOLD,
    LABEL, LOADER, PER_QUERY_UNCERTAINTY, RUN_CONTRACT, SOURCE_HASHES, SUPPORT_RULE,
    TASKS, TRAIN_LABELS, VALIDATION_LABELS, TrainedGrowth, same_predictor,
    train_degree_support,
)
from tools.report_trained_growth_benchmark import EXPECTED_RESULT


BUNDLE_CAP_BYTES = 80 * 1024 ** 2
RESERVE_BYTES = 1024 ** 3
EXPECTED_PROTOCOL = 'b825df936f3a5c830e0cba85d34f15f11fc42ee7f2229a51a61bd57dbd595f43'
EXPECTED_INPUTS = '836076c90f8d4c96f5127322aac39c1f40cfed3de034153780683f9c43b272bb'
EXPECTED_BANKS = '87b1e4752197edb1a30a2861a481fcb1082da4bcbe489a67ce0730227889d5a6'
FIT_KEYS = ('pairs', 'singles', 'gene_names', 'dropped_names', 'train_index',
            'validation_index', 'test_index')


@dataclass(frozen=True)
class DeliverySpec:
    registered: bool
    run_contract: str
    packing_contract: str
    model_contract: str
    tasks: tuple
    fold: int
    train_labels: int
    validation_labels: int
    protocol_sha256: str | None
    inputs_sealed_sha256: str | None
    results_sha256: str | None
    bank_expectations_sha256: str | None
    bundle_cap_bytes: int
    reserve_bytes: int


REGISTERED_SPEC = DeliverySpec(
    registered=True, run_contract=RUN_CONTRACT, packing_contract=PACKING_CONTRACT,
    model_contract=CONTRACT, tasks=tuple(TASKS), fold=FOLD, train_labels=TRAIN_LABELS,
    validation_labels=VALIDATION_LABELS, protocol_sha256=EXPECTED_PROTOCOL,
    inputs_sealed_sha256=EXPECTED_INPUTS, results_sha256=EXPECTED_RESULT,
    bank_expectations_sha256=EXPECTED_BANKS, bundle_cap_bytes=BUNDLE_CAP_BYTES,
    reserve_bytes=RESERVE_BYTES)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _save_npz(path, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def _copy_bytes(source, dest):
    data = Path(source).read_bytes()
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    with dest.open('xb') as stream:
        stream.write(data)
    if dest.read_bytes() != data:
        raise ValueError('Checkpoint copy drifted')
    return hashlib.sha256(data).hexdigest()


def _bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob('*') if path.is_file())


def _free(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _refuse_locations(source, output, checkout):
    source, output, checkout = source.resolve(), output.resolve(), checkout.resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError('Export output must not overlap the frozen source tree')
    if output == checkout or checkout in output.parents:
        relative = output.relative_to(checkout)
        if not relative.parts or relative.parts[0] != 'tmp':
            raise ValueError('Checkpoint bundles stay in an ignored local tmp directory')
    if output.exists():
        raise FileExistsError(output)


def _names(values):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind != 'U':
        raise ValueError('Unicode gene names required')
    names = [str(item) for item in array.tolist()]
    if len(set(names)) != len(names) or any(not name or name != name.strip() or '|' in name for name in names):
        raise ValueError('Canonical gene names required')
    return names


def _require_budget(spec, accounting, train_index, validation_index):
    expected = dict(
        nominal_train=spec.train_labels, nominal_validation=spec.validation_labels,
        charged_train=spec.train_labels, charged_validation=spec.validation_labels,
        charged_pair_union=spec.train_labels + spec.validation_labels, replaced_rows=0)
    for key, value in expected.items():
        if int(accounting[key]) != value:
            raise ValueError('Information budget mismatch: ' + key)
    if len(train_index) != spec.train_labels or len(validation_index) != spec.validation_labels:
        raise ValueError('Charged row counts do not match the information budget')


def _require_disjoint(train_index, validation_index, test_index):
    train, validation = set(map(int, train_index)), set(map(int, validation_index))
    if train & validation:
        raise ValueError('TRAIN/VALIDATION overlap')
    if len(test_index):
        test = set(map(int, test_index))
        if test & train or test & validation:
            raise ValueError('TEST identities overlap the fit split')


def _require_protocol(protocol, spec):
    if protocol.get('id') != spec.run_contract:
        raise ValueError('Run contract mismatch')
    if int(protocol['train_labels']) != spec.train_labels or int(protocol['validation_labels']) != spec.validation_labels:
        raise ValueError('Information budget mismatch')
    if tuple(protocol['tasks']) != spec.tasks:
        raise ValueError('Task roster mismatch')
    if protocol.get('packing_contract') != spec.packing_contract:
        raise ValueError('Packing contract mismatch')
    if protocol.get('model_contract') != spec.model_contract:
        raise ValueError('Predictor contract mismatch')
    if spec.registered:
        if protocol.get('synthetic_only') is not False:
            raise ValueError('Registered delivery requires the biological growth-v3 run')
        if int(protocol.get('folds')) != 5:
            raise ValueError('Five procedure folds are part of the growth-v3 run')
        if 'opal-budget-v1' not in str(protocol.get('budget')):
            raise ValueError('Budget identifier mismatch')


def _published(path, expected, label):
    digest = _sha(path)
    if expected is not None and digest != expected:
        raise ValueError(label + ' identity mismatch')
    return digest


def _program(path, expected):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Published result ledger identity mismatch')
    data = json.loads(payload)
    policies = {}
    for row in data['procedures']:
        if int(row['fold']) == FOLD:
            if row['task'] in policies:
                raise ValueError('Duplicate fold-0 policy')
            policies[row['task']] = row['selected_policy']
    program = {key: data['program'][key] for key in (
        'passes', 'failed_tasks', 'biological_confirmation', 'rescue')}
    if program['passes'] is not False or program['biological_confirmation'] is not False or program['rescue'] is not False:
        raise ValueError('Published program gate does not match this failed-gate delivery')
    if 'horlbeck_Jurkat' not in program['failed_tasks']:
        raise ValueError('Published failed-task record drifted')
    del data
    return policies, program


def _registration_binding():
    registration = ROOT / 'docs' / 'model_delivery' / 'registration.md'
    reproduction = ROOT / 'docs' / 'model_delivery' / 'training_reproduction.md'
    registration_text = registration.read_text(encoding='utf-8')
    reproduction_text = reproduction.read_text(encoding='utf-8')
    if 'opal-frontier-growth-v1' not in registration_text:
        raise ValueError('Frozen delivery registration does not name its protocol')
    if 'per-query calibrated uncertainty is unavailable' not in registration_text:
        raise ValueError('Frozen delivery registration does not state the uncertainty rule')
    if 'The fixed export remains the original validation-selected fold-0 model.' not in reproduction_text:
        raise ValueError('Training reproduction registration does not keep the original checkpoint')
    if 'Never substitute a numerically different reproduction' not in reproduction_text:
        raise ValueError('Training reproduction registration does not forbid substitution')
    return dict(
        delivery_registration='docs/model_delivery/registration.md',
        delivery_registration_sha256=_sha(registration),
        training_reproduction='docs/model_delivery/training_reproduction.md',
        training_reproduction_sha256=_sha(reproduction))


def _caveats(spec):
    rows = [
        ENDPOINT + '.',
        ENDPOINT_QUALITY + '.',
        'Those two measurements are treated as dependent assay readings, not as independent experimental replications.',
        (f'{spec.train_labels} TRAIN and {spec.validation_labels} VALIDATION labels are the charged '
         'information budget inside one fold. They are not a count of raw-assay measurements.'),
        'Stored singles are source-local measured inputs. Queries cannot replace them or infer them from pair outcomes.',
        'Fold 0 is a fixed deployment convention, not a test-score comparison. ' + BENCHMARK_RELATION + '.',
        'Per-query calibrated uncertainty is unavailable. No prediction-interval wrapper is fit. Conditional gene-sensitivity intervals remain aggregate evidence.',
        'Weight redistribution is not authorized. Item rights for public redistribution remain unresolved.',
        'The exported checkpoint is the original validation-selected fold-0 model. A training reproduction is not substituted for it.',
    ]
    if spec.registered:
        rows.append(LABEL + '.')
        rows.append('Each task keeps its registered assay context. Checkpoints do not transfer across tasks or contexts.')
    return rows


def _replay(model, pairs, singles, saved):
    from tools.frontier_tournament import predictions
    prediction, supported = predictions(model, pairs, singles)
    if prediction.shape != saved.shape or not np.array_equal(prediction, saved) or not np.isfinite(prediction).all():
        raise ValueError('Prediction replay mismatch')
    if supported.any() and not np.array_equal(model.predict(pairs[supported]), saved[supported]):
        raise ValueError('Supported query does not reproduce the original predictor')
    return int(supported.sum()), int((~supported).sum())


def _export_task(source, output, task, spec, published_policy):
    fold = spec.fold
    procedure = source / 'procedures' / task / f'fold-{fold}'
    selection_path = procedure / 'selection.json'
    selection = _read(selection_path)
    policy = selection['selected_policy']
    if published_policy is not None and published_policy != policy:
        raise ValueError('Selection does not match the published fold-0 policy')
    matches = [row for row in selection['models'] if row.get('name') == policy]
    if len(matches) != 1:
        raise ValueError('Selected policy checkpoint is not unique')
    chosen = matches[0]
    checkpoint = procedure / 'checkpoints' / chosen['checkpoint']
    digest = _published(checkpoint, chosen['checkpoint_sha256'], 'Checkpoint')
    with np.load(checkpoint, allow_pickle=False) as archive:
        if str(archive['contract']) != PACKING_CONTRACT or str(archive['model_contract']) != CONTRACT:
            raise ValueError('Checkpoint contract is not packing v2 on the original predictor')
        if int(archive['format_version']) != 2:
            raise ValueError('Packing format mismatch')
    fit_path = procedure / 'fit-args.npz'
    fit_key = f'procedures/{task}/fold-{fold}/fit-args.npz'
    sealed = _read(source / 'inputs-sealed.json')
    if sealed['fit_args'][fit_key] != _sha(fit_path):
        raise ValueError('Fit-input identity mismatch')
    with np.load(fit_path, allow_pickle=False) as fit:
        arrays = {key: fit[key].copy() for key in FIT_KEYS}
    names = _names(arrays['gene_names'])
    dropped = _names(arrays['dropped_names'])
    if set(dropped) & set(names):
        raise ValueError('Dropped genes overlap the schema')
    singles = np.asarray(arrays['singles'])
    if singles.ndim != 1 or singles.dtype.kind != 'f' or singles.shape != (len(names),) or not np.isfinite(singles).all():
        raise ValueError('Fit singles must be a finite real vector')
    _require_budget(spec, _read(procedure / 'accounting.json'), arrays['train_index'], arrays['validation_index'])
    _require_disjoint(arrays['train_index'], arrays['validation_index'], arrays['test_index'])
    degree, support = train_degree_support(arrays['pairs'], arrays['train_index'], len(names))
    original = MixedLoader(historical_roots=(source.resolve(),)).load(checkpoint, digest)
    for layer in (original, original.base):
        if layer is None:
            continue
        if not np.array_equal(layer.state['support'], support):
            raise ValueError('Checkpoint support is not the TRAIN degree rule')
        stored = layer.state['singles']
        if stored.ndim != 1 or stored.dtype.kind != 'f' or not np.array_equal(stored, singles):
            raise ValueError('Checkpoint singles are not the stored source-local singles')
    with np.load(procedure / 'queries.npz', allow_pickle=False) as queries:
        query_pairs = queries['pairs'].copy()
        query_singles = queries['singles'].copy()
        if 'gene_names' in queries.files and _names(queries['gene_names']) != names:
            raise ValueError('Query gene names differ from fit inputs')
        saved_support = queries['supported'].copy() if 'supported' in queries.files else None
    if not np.array_equal(query_singles, singles):
        raise ValueError('Replay singles are not the stored source-local singles')
    if len(arrays['test_index']) and not np.array_equal(query_pairs, arrays['pairs'][arrays['test_index']]):
        raise ValueError('Saved queries are not the procedure query roster')
    with np.load(procedure / 'predictions.npz', allow_pickle=False) as predictions:
        saved = predictions[chosen['prediction_key']].copy()
    supported_rows, unsupported_rows = _replay(original, query_pairs, singles, saved)
    if saved_support is not None and not np.array_equal(saved_support, original.supported(query_pairs)):
        raise ValueError('Saved support mask does not match TRAIN support')
    destination = output / 'tasks' / task
    destination.mkdir(parents=True)
    copied = _copy_bytes(checkpoint, destination / 'checkpoint.npz')
    if copied != digest:
        raise ValueError('Copied checkpoint hash drifted')
    loaded = MixedLoader(historical_roots=(destination.resolve(), output.resolve())).load(
        destination / 'checkpoint.npz', copied)
    if not same_predictor(original, loaded):
        raise ValueError('Copied checkpoint state drifted')
    _save_npz(destination / 'schema.npz', gene_names=np.asarray(names, dtype='U'),
              dropped_names=np.asarray(dropped, dtype='U'), singles=np.array(singles, copy=True),
              support=np.array(support, copy=True), train_degree=degree)
    record = dict(ASSAYS[task]) if task in ASSAYS else dict(
        context=f'{task}-context', source=None, accession=None, units=None, assay=None,
        endpoint=ENDPOINT, endpoint_quality=ENDPOINT_QUALITY)
    record.update(
        task=task, fold=FOLD, selected_policy=policy,
        checkpoint=f'tasks/{task}/checkpoint.npz', checkpoint_sha256=copied,
        schema=f'tasks/{task}/schema.npz', schema_sha256=_sha(destination / 'schema.npz'),
        selection_sha256=_sha(selection_path), fit_args_sha256=_sha(fit_path),
        information_budget=dict(train_labels=spec.train_labels, validation_labels=spec.validation_labels,
                                charged_train=spec.train_labels, charged_validation=spec.validation_labels,
                                replaced_rows=0),
        genes=len(names), supported_genes=int(support.sum()), dropped_unused_genes=len(dropped),
        replay=dict(equal=True, saved_replay_rows=int(len(saved)), supported_rows=supported_rows,
                    unsupported_rows=unsupported_rows, new_metrics=False, outcomes_opened=False,
                    population='saved procedure queries; identity comparison only; not a new score'))
    return record


def deliver_tree(source, output, spec):
    """Write one local bundle. The registered biological tree accepts only REGISTERED_SPEC."""
    source, output = Path(source), Path(output)
    _refuse_locations(source, output, ROOT)
    if spec.registered and spec != REGISTERED_SPEC:
        raise ValueError('Registered export spec drifted')
    protocol_path = source / 'protocol.json'
    protocol = _read(protocol_path)
    if protocol.get('id') == RUN_CONTRACT and int(protocol.get('train_labels', -1)) == TRAIN_LABELS:
        if spec != REGISTERED_SPEC:
            raise ValueError('Biological growth-v3 export requires the registered delivery spec')
    protocol_sha = _published(protocol_path, spec.protocol_sha256, 'Protocol')
    _require_protocol(protocol, spec)
    published_policies, program = {}, None
    banks_sha = inputs_sha = results_sha = None
    registration = None
    if spec.registered:
        inputs_sha = _published(source / 'inputs-sealed.json', spec.inputs_sealed_sha256, 'Fit-input seal')
        banks_sha = _published(source / 'bank-expectations.json', spec.bank_expectations_sha256, 'Bank expectations')
        banks = _read(source / 'bank-expectations.json')
        if banks.get('original_hash') != SOURCE_HASHES['original_hash']:
            raise ValueError('Original bank hash mismatch')
        if banks.get('original_source_hash') != SOURCE_HASHES['original_source_hash']:
            raise ValueError('Original source hash mismatch')
        if banks.get('external_preparation_hash') != SOURCE_HASHES['external_preparation_hash']:
            raise ValueError('External preparation hash mismatch')
        if banks.get('external_hashes') != SOURCE_HASHES['external_hashes']:
            raise ValueError('External bank hash mismatch')
        results_sha = spec.results_sha256
        published_policies, program = _program(source / 'results.json', results_sha)
        if tuple(published_policies) != tuple(TASKS):
            raise ValueError('Published fold-0 roster mismatch')
        registration = _registration_binding()
    if spec.reserve_bytes and _free(output) < spec.reserve_bytes:
        raise RuntimeError('Free-disk reserve')
    output.mkdir(parents=True, exist_ok=False)
    records = [_export_task(source, output, task, spec, published_policies.get(task)) for task in spec.tasks]
    if _bytes(output) > spec.bundle_cap_bytes:
        raise RuntimeError('Bundle cap exceeded before the manifest')
    manifest = dict(
        contract=DELIVERY_CONTRACT, label=LABEL if spec.registered else
        'synthetic delivery fixture; not the biological checkpoint',
        biological_delivery=bool(spec.registered), run_contract=spec.run_contract, loader=LOADER,
        packing_contract=PACKING_CONTRACT, model_contract=CONTRACT, fold=FOLD,
        other_folds_exported=False, refit=False, test_dependent_selection=False,
        training_reproduction_substituted=False,
        checkpoint_authority='original validation-selected fold-0 checkpoint',
        new_fits=0, new_scores=0, per_query_uncertainty=PER_QUERY_UNCERTAINTY,
        uncertainty_wrapper_fit=False, conditional_gene_interval='aggregate_only',
        endpoint=ENDPOINT, endpoint_quality=ENDPOINT_QUALITY, endpoint_caveats=_caveats(spec),
        information_budget=dict(
            train_labels=spec.train_labels, validation_labels=spec.validation_labels,
            budget=protocol.get('budget') if spec.registered else 'fixture'),
        support_rule=SUPPORT_RULE,
        rights=dict(status='unresolved', redistribution='not_authorized', git=False, wheel=False, sdist=False,
                    scope='historical growth registration authorizes local scientific reuse only'),
        benchmark=dict(attribution=BENCHMARK_RELATION, metrics_copied=False,
                       fold0_checkpoint_is_the_benchmark=False, owned_by='root',
                       source_result_sha256=EXPECTED_RESULT if spec.registered else None),
        program_gate=program, expected_source_hashes=SOURCE_HASHES if spec.registered else None,
        protocol_sha256=protocol_sha, inputs_sealed_sha256=inputs_sha,
        bank_expectations_sha256=banks_sha, results_sha256=results_sha,
        protocol_alignment=(None if registration is None else dict(
            registration_protocol_name='opal-frontier-growth-v1',
            executed_run_contract=RUN_CONTRACT, loader=LOADER, **registration,
            note=('The frozen delivery registration names opal-frontier-growth-v1. '
                  'This bundle binds the executed growth-v3 checkpoints and the frontier_packing_v2 loader.'))),
        tasks=records)
    _write_json(output / 'manifest.json', manifest)
    if _bytes(output) > spec.bundle_cap_bytes:
        raise RuntimeError('Bundle cap exceeded')
    loaded = TrainedGrowth.load(output)
    if loaded.tasks != tuple(spec.tasks) or loaded.biological_delivery != spec.registered:
        raise ValueError('Reloaded bundle does not match the export')
    digest = _sha(output / 'manifest.json')
    TrainedGrowth.load(output, digest)
    return dict(manifest=loaded.manifest, manifest_sha256=digest, bundle_bytes=_bytes(output),
                biological_delivery=spec.registered, new_fits=0, new_scores=0)


def export_trained_growth(source, output):
    """Export the six original growth-v3 fold-0 checkpoints. Root runs this."""
    return deliver_tree(source, output, REGISTERED_SPEC)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'tmp' / 'frontier-execution-20261026' / 'growth-v3')
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp' / 'model-delivery-20261027' / 'trained-growth')
    args = parser.parse_args()
    result = export_trained_growth(args.source, args.output)
    print(json.dumps(dict(
        output=str(args.output), manifest_sha256=result['manifest_sha256'],
        bundle_bytes=result['bundle_bytes'], tasks=list(TASKS), biological_delivery=True,
        new_fits=0, new_scores=0, per_query_uncertainty=PER_QUERY_UNCERTAINTY,
        training_reproduction_substituted=False, run_contract=RUN_CONTRACT, fold=FOLD)))


if __name__ == '__main__':
    main()
