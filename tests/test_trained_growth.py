"""Source-free checks for the fold-0 delivery. Root runs the biological export."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from opal.frontier_packing_v2 import PACKING_CONTRACT, save_checkpoint
from opal.frontier_tournament import CONTRACT, Predictor
from opal.trained_growth import (
    ASSAYS, ENDPOINT, ENDPOINT_QUALITY, LABEL, SOURCE_HASHES, TASKS, TRAIN_LABELS,
    VALIDATION_LABELS, TrainedGrowth, train_degree_support,
)
from tools.export_trained_growth import (
    EXPECTED_RESULT, REGISTERED_SPEC, DeliverySpec, deliver_tree, export_trained_growth,
)
from tools.frontier_tournament import predictions


def test_registered_labels_and_pre_fit_snapshot():
    assert ENDPOINT == 'mean of two infection-specific gamma measurements'
    assert ENDPOINT_QUALITY == 'frozen processed bank; no independent-infection claim'
    assert SOURCE_HASHES['identity'] == 'original_pre_fit_snapshot'
    assert 'banks_opened' not in SOURCE_HASHES
    assert TRAIN_LABELS == 16000 and VALIDATION_LABELS == 4000
    assert tuple(ASSAYS) == TASKS and len(TASKS) == 6
    for assay in ASSAYS.values():
        assert assay['endpoint'] == ENDPOINT
        assert assay['endpoint_quality'] == ENDPOINT_QUALITY
        assert assay['units'] == 'gamma'
    assert LABEL == 'assay-local research model; all-task promotion gate failed'
    assert REGISTERED_SPEC.results_sha256 == EXPECTED_RESULT
    assert REGISTERED_SPEC.run_contract == 'opal-frontier-growth-v3'
    assert REGISTERED_SPEC.packing_contract == PACKING_CONTRACT


def test_support_uses_train_degree_only():
    pairs = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    degree, support = train_degree_support(pairs, np.array([0, 1]), 4)
    assert degree.tolist() == [1, 2, 1, 0]
    assert support.tolist() == [False, True, False, False]
    added = train_degree_support(pairs, np.array([0, 1, 2]), 4)[1]
    assert added[2] and not support[2]
    with pytest.raises(ValueError, match='outside'):
        train_degree_support(pairs, np.array([3]), 4)


def test_manifest_pin_is_checked_before_parsing(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_bytes(b'not-json')
    with pytest.raises(ValueError, match='Manifest receipt mismatch'):
        TrainedGrowth.load(tmp_path, '0' * 64)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(json.JSONDecodeError):
        TrainedGrowth.load(tmp_path, digest)


def _biological_manifest(tasks):
    records = []
    for task in tasks:
        record = dict(ASSAYS[task])
        record.update(task=task, fold=0, context=record['context'],
                      checkpoint=f'tasks/{task}/checkpoint.npz', checkpoint_sha256='a' * 64,
                      schema=f'tasks/{task}/schema.npz', schema_sha256='b' * 64)
        records.append(record)
    return dict(
        contract='opal-trained-growth-delivery-v1', label=LABEL, biological_delivery=True,
        run_contract='opal-frontier-growth-v3', loader='opal.frontier_packing_v2.MixedLoader',
        packing_contract=PACKING_CONTRACT, model_contract=CONTRACT, fold=0,
        other_folds_exported=False, refit=False, test_dependent_selection=False,
        training_reproduction_substituted=False, new_fits=0, new_scores=0,
        per_query_uncertainty='unavailable', uncertainty_wrapper_fit=False,
        conditional_gene_interval='aggregate_only', endpoint_caveats=['frozen processed bank'],
        information_budget=dict(train_labels=16000, validation_labels=4000),
        rights=dict(status='unresolved', redistribution='not_authorized', git=False, wheel=False, sdist=False),
        benchmark=dict(
            attribution=('pooled five-fold training-procedure benchmark; not a fold-0-only score '
                         'and not a score of this single fold-0 checkpoint'),
            metrics_copied=False, fold0_checkpoint_is_the_benchmark=False,
            source_result_sha256='c' * 64),
        program_gate=dict(passes=False, failed_tasks=['horlbeck_Jurkat'],
                          biological_confirmation=False, rescue=False),
        protocol_alignment=dict(executed_run_contract='opal-frontier-growth-v3'),
        expected_source_hashes=SOURCE_HASHES, tasks=records)


def test_biological_roster_and_context_are_exact(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(_biological_manifest(tuple(reversed(TASKS)))), encoding='utf-8')
    with pytest.raises(ValueError, match='Biological task roster mismatch'):
        TrainedGrowth.load(tmp_path)
    payload = _biological_manifest(TASKS)
    payload['tasks'][0]['context'] = 'Jurkat'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='Assay record does not match the registered task'):
        TrainedGrowth.load(tmp_path)


def test_schema_name_count_degree_and_singles(tmp_path):
    task = tmp_path / 'tasks' / 'toy'
    task.mkdir(parents=True)
    checkpoint = task / 'checkpoint.npz'
    checkpoint.write_bytes(b'not-a-model')
    schema = task / 'schema.npz'

    def store(**arrays):
        if schema.exists():
            schema.unlink()
        with schema.open('xb') as stream:
            np.savez_compressed(stream, dropped_names=np.array([], dtype='U'), **arrays)

    def manifest():
        body = dict(
            contract='opal-trained-growth-delivery-v1',
            label='synthetic delivery fixture; not the biological checkpoint',
            biological_delivery=False, run_contract='opal-trained-growth-fixture',
            loader='opal.frontier_packing_v2.MixedLoader', packing_contract=PACKING_CONTRACT,
            model_contract=CONTRACT, fold=0, other_folds_exported=False, refit=False,
            test_dependent_selection=False, new_fits=0, new_scores=0,
            per_query_uncertainty='unavailable', uncertainty_wrapper_fit=False,
            conditional_gene_interval='aggregate_only', endpoint_caveats=['fixture'],
            program_gate=None,
            rights=dict(status='unresolved', redistribution='not_authorized', git=False, wheel=False, sdist=False),
            benchmark=dict(
                attribution=('pooled five-fold training-procedure benchmark; not a fold-0-only score '
                             'and not a score of this single fold-0 checkpoint'),
                metrics_copied=False, fold0_checkpoint_is_the_benchmark=False,
                source_result_sha256=None),
            tasks=[dict(task='toy', fold=0, context='toy-context',
                        checkpoint='tasks/toy/checkpoint.npz',
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        schema='tasks/toy/schema.npz',
                        schema_sha256=hashlib.sha256(schema.read_bytes()).hexdigest())])
        (tmp_path / 'manifest.json').write_text(json.dumps(body), encoding='utf-8')

    store(gene_names=np.array(['A', 'B'], dtype='U'), singles=np.array([0.1]),
          support=np.array([True]), train_degree=np.array([2], dtype=np.int64))
    manifest()
    with pytest.raises(ValueError, match='counts differ'):
        TrainedGrowth.load(tmp_path)
    store(gene_names=np.array(['A'], dtype='U'), singles=np.array([np.nan]),
          support=np.array([False]), train_degree=np.array([0], dtype=np.int64))
    manifest()
    with pytest.raises(ValueError, match='finite real'):
        TrainedGrowth.load(tmp_path)
    store(gene_names=np.array(['A'], dtype='U'), singles=np.array([0.2]),
          support=np.array([False]), train_degree=np.array([-1], dtype=np.int64))
    manifest()
    with pytest.raises(ValueError, match='negative'):
        TrainedGrowth.load(tmp_path)


def _fixture_source(root):
    pairs = np.array([[0, 1], [0, 2], [1, 2], [0, 4], [1, 4], [2, 3], [3, 4], [2, 4], [1, 3]], dtype=np.int64)
    train = np.arange(6, dtype=np.int64)
    validation = np.array([6], dtype=np.int64)
    test = np.array([7, 8], dtype=np.int64)
    names = np.array(['G0', 'G1', 'G2', 'G3', 'G4'], dtype='U')
    singles = np.array([0.1, -0.2, 0.3, 0.4, -0.5])
    degree, support = train_degree_support(pairs, train, len(names))
    procedure = root / 'procedures' / 'toy' / 'fold-0'
    (procedure / 'checkpoints').mkdir(parents=True)
    (root / 'procedures' / 'toy' / 'fold-1').mkdir()
    (root / 'procedures' / 'toy' / 'fold-1' / 'marker.txt').write_text('leave', encoding='utf-8')
    matrix = np.arange(25, dtype=float).reshape(5, 5)
    matrix = np.triu(matrix) + np.triu(matrix, 1).T
    base = Predictor(dict(kind='sum', singles=singles, support=support, mean=0.))
    model = Predictor(dict(kind='completion', singles=singles, support=support, matrix=matrix), base)
    save_checkpoint(model, procedure / 'checkpoints' / 'model-00.npz')
    checkpoint = procedure / 'checkpoints' / 'model-00.npz'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with (procedure / 'fit-args.npz').open('xb') as stream:
        np.savez_compressed(stream, pairs=pairs, singles=singles, gene_names=names,
                            dropped_names=np.array(['GONE'], dtype='U'), train_index=train,
                            validation_index=validation, test_index=test)
    query = pairs[test]
    predicted, supported = predictions(model, query, singles)
    with (procedure / 'queries.npz').open('xb') as stream:
        np.savez_compressed(stream, pairs=query, singles=singles, gene_names=names, supported=supported)
    with (procedure / 'predictions.npz').open('xb') as stream:
        np.savez_compressed(stream, prediction_0=predicted)
    selection = dict(selected_policy='I0--rank8-penalty0.01', models=[dict(
        name='I0--rank8-penalty0.01', family='I0', checkpoint='model-00.npz',
        checkpoint_sha256=digest, prediction_key='prediction_0')])
    accounting = dict(nominal_train=6, nominal_validation=1, charged_train=6, charged_validation=1,
                      charged_pair_union=7, replaced_rows=0, collapsed_genes=5)
    for name, payload in (('selection.json', selection), ('accounting.json', accounting)):
        with (procedure / name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(payload, stream)
            stream.write('\n')
    fit_sha = hashlib.sha256((procedure / 'fit-args.npz').read_bytes()).hexdigest()
    sealed = dict(fit_args={'procedures/toy/fold-0/fit-args.npz': fit_sha})
    protocol = dict(id='opal-trained-growth-fixture', synthetic_only=True, train_labels=6,
                    validation_labels=1, tasks=['toy'], packing_contract=PACKING_CONTRACT,
                    model_contract=CONTRACT, budget='fixture')
    for name, payload in (('inputs-sealed.json', sealed), ('protocol.json', protocol)):
        with (root / name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(payload, stream)
            stream.write('\n')
    return predicted


def _spec():
    return DeliverySpec(
        registered=False, run_contract='opal-trained-growth-fixture', packing_contract=PACKING_CONTRACT,
        model_contract=CONTRACT, tasks=('toy',), fold=0, train_labels=6, validation_labels=1,
        protocol_sha256=None, inputs_sealed_sha256=None, results_sha256=None,
        bank_expectations_sha256=None, bundle_cap_bytes=80 * 1024 ** 2, reserve_bytes=0)


def test_fixture_export_matches_checkpoint_and_abstains(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'bundle'
    saved = _fixture_source(source)
    marker = (source / 'procedures' / 'toy' / 'fold-1' / 'marker.txt').read_bytes()
    result = deliver_tree(source, output, _spec())
    assert result['new_fits'] == 0 and result['new_scores'] == 0
    assert result['biological_delivery'] is False
    bundle = TrainedGrowth.load(output, result['manifest_sha256'])
    assert bundle.label != LABEL
    assert bundle.manifest['endpoint'] == ENDPOINT
    assert bundle.manifest['endpoint_quality'] == ENDPOINT_QUALITY
    assert bundle.manifest['training_reproduction_substituted'] is False
    assert bundle.manifest['information_budget']['train_labels'] == 6
    model = bundle.model('toy')
    assert model.gene_names() == ('G0', 'G1', 'G2', 'G3', 'G4')
    assert model.support().tolist() == [True, True, True, False, True]
    predicted = model.predict(('G2', 'G4'), context='toy-context')
    assert predicted[0]['status'] == 'predicted'
    assert predicted[0]['value'] == float(saved[0])
    assert predicted[0]['uncertainty'] == 'unavailable'
    abstained = model.predict([('G1', 'G3'), ('GONE', 'G0'), ('NOPE', 'G0'), ('G0', 'G0'),
                               ('G0|x', 'G1'), (1, 'G0')])
    assert [row['reason'] for row in abstained] == [
        'unsupported_gene', 'absent_gene', 'unknown_gene', 'self_pair', 'malformed_name', 'malformed_name']
    assert all(row['value'] is None for row in abstained)
    assert abstained[0]['value'] != float(saved[1])
    assert model.predict(('G0', 'G1'), context='Jurkat')[0]['reason'] == 'unknown_context'
    uncertainty = model.uncertainty()
    assert uncertainty['per_query_calibrated_uncertainty'] == 'unavailable'
    assert uncertainty['value'] is None and uncertainty['wrapper_fit'] is False
    with pytest.raises(TypeError):
        model.predict(('G0', 'G1'), singles=model.stored_singles())
    flipped = bytearray((output / 'tasks' / 'toy' / 'checkpoint.npz').read_bytes())
    flipped[-1] ^= 1
    (output / 'tasks' / 'toy' / 'checkpoint.npz').write_bytes(flipped)
    with pytest.raises(ValueError, match='Checkpoint integrity mismatch'):
        TrainedGrowth.load(output)
    assert (source / 'procedures' / 'toy' / 'fold-1' / 'marker.txt').read_bytes() == marker
    with pytest.raises(FileExistsError):
        deliver_tree(source, output, _spec())


def test_registered_export_does_not_start_on_an_unbound_tree(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'protocol.json').write_text('{"id":"opal-frontier-growth-v3","train_labels":1}\n', encoding='utf-8')
    output = tmp_path / 'bundle'
    with pytest.raises(ValueError, match='Protocol identity mismatch'):
        export_trained_growth(source, output)
    assert not output.exists()
    protocol = dict(id='opal-frontier-growth-v3', train_labels=16000, validation_labels=4000, tasks=list(TASKS))
    (source / 'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    with pytest.raises(ValueError, match='registered delivery spec'):
        deliver_tree(source, tmp_path / 'other', _spec())
    assert not (tmp_path / 'other').exists()


def test_delivery_sources_do_not_fit_or_score():
    root = Path(__file__).resolve().parents[1]
    for name in ('src/trained_growth.py', 'tools/export_trained_growth.py'):
        text = (root / name).read_text(encoding='utf-8')
        for banned in ('fit_models', 'fit_procedure', 'metrics_v3', 'train_targets', 'sealed-test'):
            assert banned not in text
