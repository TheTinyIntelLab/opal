"""Independent contract tests for the synthetic tournament.

Fixture seeds are 77111, 88021 and 88022. Evaluation seeds and the exposed
pilot seed are not used. Menu fits below use controls=False with 5 completion
steps and 5 neural steps. The registered experiment remains 500 and 300 with
controls enabled. Root executes this file; it is not a pilot.
"""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from opal.block_completion import ScalarCompletion
from opal.evaluation_precision import metrics_v3
from opal.frontier_tournament import (CONTRACT, SEEDS, Predictor, budget_rows, fit_models,
    make_world, select_records)

ROOT = Path(__file__).resolve().parents[1]


def _lifecycle():
    path = ROOT / 'tools' / 'frontier_tournament.py'
    spec = importlib.util.spec_from_file_location('frontier_tournament_lifecycle_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def life():
    return _lifecycle()


@pytest.fixture(scope='module')
def fitted():
    torch.set_num_threads(1)
    data = make_world('additive', 77111, 0.0, genes=12)
    train, validation, _ = budget_rows(data, 1.)
    records = fit_models(data['singles'], data['features'], data['pairs'][train], data['observed'][0, train],
                         data['pairs'][validation], data['observed'][0, validation], 71,
                         controls=False, completion_steps=5, neural_steps=5)
    return data, train, validation, records


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, obj):
    Path(path).write_text(json.dumps(obj), encoding='utf-8')


def _bind(path, *, pilot, cells):
    path.mkdir()
    protocol = dict(pilot=pilot, cells=cells, cap_seconds=600, artifact_cap_bytes=64 * 1024 * 1024,
                    contract=CONTRACT)
    _write(path / 'protocol.json', protocol)
    source = ROOT / 'src' / 'frontier_tournament.py'
    _write(path / 'code-binding.json', dict(
        protocol_sha256=_digest(path / 'protocol.json'),
        files={'src/frontier_tournament.py': _digest(source)},
        environment=dict(numpy=np.__version__, torch=torch.__version__)))
    return path


def _seal(path, cells):
    models = path / 'models'
    models.mkdir()
    artifact = models / 'marker.txt'
    artifact.write_text('sealed', encoding='utf-8')
    _write(path / 'all-predictions-sealed.json', dict(
        cells=[dict(cell=cell, procedures=[]) for cell in cells],
        files={'models/marker.txt': _digest(artifact)}))


def test_generator_alias_budget_and_outcome_blind_prefix():
    seed = 88021
    low = make_world('lowrank', seed, 0.1, 'independent')
    persistent = make_world('technical_persistent', seed, 0.1, 'independent')
    redrawn = make_world('technical_redrawn', seed, 0.1, 'independent')
    for key in low:
        assert np.array_equal(low[key], persistent[key]), key
    assert np.array_equal(low['observed'][0], redrawn['observed'][0])
    assert np.array_equal(low['constructs'][0], redrawn['constructs'][0])
    assert not np.array_equal(low['observed'][1], redrawn['observed'][1])
    quiet = make_world('additive', seed, 0.0)
    other = make_world('sparse', seed, 0.0)
    noisy = make_world('additive', seed, 0.1)
    assert np.array_equal(quiet['train'], other['train'])
    assert np.array_equal(quiet['validation'], other['validation'])
    assert np.array_equal(quiet['test'], other['test'])
    assert np.array_equal(quiet['singles'], other['singles'])
    assert not np.array_equal(quiet['observed'], other['observed'])
    assert np.array_equal(quiet['biological_truth'], noisy['biological_truth'])
    assert np.array_equal(quiet['train'], noisy['train'])
    assert not np.array_equal(quiet['singles'], noisy['singles'])
    assert quiet['features'].shape == (60, 1)
    assert np.array_equal(quiet['features'][:, 0], quiet['singles'])
    assert quiet['constructs'].shape[-1] == 4
    assert np.array_equal(quiet['constructs'].mean(2), quiet['observed'])
    roles = np.concatenate([quiet[name] for name in ('train', 'validation', 'test')])
    assert np.array_equal(np.sort(roles), np.arange(1770))
    small_train, small_validation, small_cost = budget_rows(quiet, .25)
    full_train, full_validation, full_cost = budget_rows(quiet, 1.)
    assert (small_cost['nominal_train'], small_cost['nominal_validation'], small_cost['charged_pair_union']) == (221, 88, 309)
    assert (full_cost['nominal_train'], full_cost['nominal_validation'], full_cost['charged_pair_union']) == (885, 354, 1239)
    assert np.array_equal(full_train[:221], small_train)
    assert np.array_equal(full_validation[:88], small_validation)
    poisoned = dict(quiet, observed=np.full_like(quiet['observed'], 999.), truth=np.zeros_like(quiet['truth']))
    again, _, _ = budget_rows(poisoned, .25)
    assert np.array_equal(again, small_train)
    assert 'path' not in inspect.signature(make_world).parameters
    signature = inspect.signature(fit_models)
    assert 'test_targets' not in signature.parameters
    assert signature.parameters['completion_steps'].default == 500
    assert signature.parameters['neural_steps'].default == 300
    assert signature.parameters['controls'].default is True
    # Check actual fixture seed values. Scanning this source for forbidden
    # literals would also find the scanner's own list, so cannot test isolation.
    assert all(value not in SEEDS for value in (77111, 88021, 88022))


def test_refuses_feature_leakage_and_duplicate_pairs():
    data = make_world('additive', 88022, 0.0, genes=12)
    train, validation, _ = budget_rows(data, 1.)
    arguments = [data['singles'], data['features'], data['pairs'][train], data['observed'][0, train],
                 data['pairs'][validation], data['observed'][0, validation]]
    kwargs = dict(seed=71, controls=False, completion_steps=1, neural_steps=1)
    widened = list(arguments)
    widened[1] = np.column_stack((data['singles'], data['singles']))
    with pytest.raises(ValueError, match='identities only'):
        fit_models(*widened, **kwargs)
    overlap = list(arguments)
    overlap[4] = data['pairs'][train[:2], ::-1]
    overlap[5] = data['observed'][0, train[:2]]
    with pytest.raises(ValueError, match='leakage'):
        fit_models(*overlap, **kwargs)
    duplicated = list(arguments)
    duplicated[2] = np.vstack((data['pairs'][train[:3]], data['pairs'][train[:1]]))
    duplicated[3] = data['observed'][0, train[:4]]
    with pytest.raises(ValueError, match='Duplicate'):
        fit_models(*duplicated, **kwargs)
    missing = list(arguments)
    missing[0] = data['singles'].copy()
    missing[0][0] = np.nan
    with pytest.raises(ValueError, match='Finite singles'):
        fit_models(*missing, **kwargs)


def test_shuffle_loser_cannot_become_deployed_policy():
    records = [
        dict(family='B0', config='zero', validation_mse=1.),
        dict(family='B0', config='sum', validation_mse=1.),
        dict(family='I0-shuffle0', config='rank8-penalty0.01', validation_mse=0.),
        dict(family='I0', config='rank8-penalty0.01', validation_mse=.5),
    ]
    selected, simple, policy = select_records(records)
    assert simple['config'] == 'zero'
    assert policy['family'] == 'I0'
    assert '-shuffle' not in policy['family']
    assert selected['I0-shuffle0']['validation_mse'] == 0.


def test_support_refusal_uses_single_sum_fallback(life):
    singles = np.array([.2, -.1, .4, 0.])
    support = np.array([True, True, False, False])
    model = Predictor(dict(kind='sum', singles=singles, support=support, mean=0.))
    query = np.array([[0, 1], [0, 2], [1, 0]])
    assert model.supported(query).tolist() == [True, False, True]
    with pytest.raises(ValueError, match='TRAIN partners'):
        model.predict(query)
    with pytest.raises(ValueError, match='Distinct'):
        model.predict(np.array([[0, 0]]))
    predicted, mask = life.predictions(model, query, singles)
    assert mask.tolist() == [True, False, True]
    assert predicted[1] == singles[query[1]].sum()
    assert np.array_equal(predicted[[0, 2]], model.predict(query[[0, 2]]))


def test_checkpoint_schema_and_metric_identity(tmp_path):
    singles = np.zeros(4)
    support = np.ones(4, dtype=bool)
    ok = Predictor(dict(kind='mean', singles=singles, support=support, mean=1.))
    ok.save(tmp_path / 'ok.npz')
    loaded = Predictor.load(tmp_path / 'ok.npz', _digest(tmp_path / 'ok.npz'))
    query = np.array([[0, 1], [0, 1], [1, 0]])
    assert np.array_equal(loaded.predict(query), np.ones(3))
    with pytest.raises(FileExistsError):
        ok.save(tmp_path / 'ok.npz')
    with pytest.raises(ValueError, match='identity'):
        Predictor.load(tmp_path / 'ok.npz', '0' * 64)
    matrix = np.eye(4)
    matrix[0, 1] = 2.
    Predictor(dict(kind='completion', singles=singles, support=support, matrix=matrix)).save(tmp_path / 'asym.npz')
    with pytest.raises(ValueError, match='Asymmetric'):
        Predictor.load(tmp_path / 'asym.npz')
    broken = singles.copy()
    broken[0] = np.nan
    Predictor(dict(kind='sum', singles=broken, support=support, mean=0.)).save(tmp_path / 'nan.npz')
    with pytest.raises(ValueError, match='Nonfinite'):
        Predictor.load(tmp_path / 'nan.npz')
    Predictor(dict(kind='linear', singles=singles, support=support, nodes=False, center=np.zeros(3),
                   scale=np.ones(3), coefficient=np.zeros(3), offset=0.)).save(tmp_path / 'good-linear.npz')
    Predictor.load(tmp_path / 'good-linear.npz')
    Predictor(dict(kind='linear', singles=singles, support=support, nodes=False, center=np.zeros(3),
                   scale=np.array([1., 1., 0.]), coefficient=np.zeros(3), offset=0.)).save(tmp_path / 'scale.npz')
    with pytest.raises(ValueError, match='Invalid scale'):
        Predictor.load(tmp_path / 'scale.npz')
    Predictor(dict(kind='sum', singles=singles, support=np.ones(4), mean=0.)).save(tmp_path / 'dtype.npz')
    with pytest.raises(ValueError, match='Malformed checkpoint schema'):
        Predictor.load(tmp_path / 'dtype.npz')
    centers = np.zeros((1, 2))
    Predictor(dict(kind='flexible', singles=singles, support=support, nodes=False, z=np.zeros(4),
                   centers=centers, bandwidth=0., center=np.zeros(4), scale=np.ones(4),
                   coefficient=np.zeros(4), offset=0.)).save(tmp_path / 'bandwidth.npz')
    with pytest.raises(ValueError, match='Flexible centers/bandwidth'):
        Predictor.load(tmp_path / 'bandwidth.npz')
    with np.load(tmp_path / 'ok.npz') as stored:
        payload = {key: stored[key].copy() for key in stored.files}
    payload['contract'] = np.array('other-contract')
    np.savez_compressed(tmp_path / 'contract.npz', **payload)
    with pytest.raises(ValueError, match='Unknown checkpoint contract'):
        Predictor.load(tmp_path / 'contract.npz')
    payload['contract'] = np.array(CONTRACT)
    payload['model__extra'] = np.zeros(1)
    np.savez_compressed(tmp_path / 'extra.npz', **payload)
    with pytest.raises(ValueError, match='Unexpected checkpoint fields'):
        Predictor.load(tmp_path / 'extra.npz')
    target = np.array([1., 2., 4.])
    prediction = np.array([1., 0., 2.])
    manual = float(np.square(target - prediction).sum(dtype=np.float64) / len(target))
    assert metrics_v3(target, prediction)['mse'] == manual
    draw = np.random.default_rng(7)
    wide_target, wide_prediction = draw.normal(size=531), draw.normal(size=531)
    wide_manual = float(np.square(wide_target - wide_prediction).sum(dtype=np.float64) / 531)
    assert metrics_v3(wide_target, wide_prediction)['mse'] == wide_manual


def test_inaccessible_labels_leave_fit_unchanged_and_validation_changes_selection(fitted):
    data, train, validation, records = fitted
    assert set(train.tolist()).isdisjoint(data['test'].tolist())
    assert set(train.tolist()).isdisjoint(validation.tolist())
    poisoned = data['observed'].copy()
    poisoned[1] = np.nan
    poisoned[0, data['test']] = np.nan
    other = fit_models(data['singles'], data['features'], data['pairs'][train], poisoned[0, train],
                       data['pairs'][validation], poisoned[0, validation], 71,
                       controls=False, completion_steps=5, neural_steps=5)
    hidden = {'observed', 'truth', 'test', 'constructs', 'biological_truth', 'guide_offsets'}
    for first, second in zip(records, other, strict=True):
        assert first['validation_mse'] == second['validation_mse']
        assert hidden.isdisjoint(first['model'].state)
        for key in first['model'].state:
            assert np.array_equal(first['model'].state[key], second['model'].state[key])
        if first['model'].base is not None:
            for key in first['model'].base.state:
                assert np.array_equal(first['model'].base.state[key], second['model'].base.state[key])
    _, simple, _ = select_records(records)
    assert simple['family'] == 'B0' and simple['config'] == 'sum' and simple['validation_mse'] == 0
    controlled = fit_models(data['singles'], data['features'], data['pairs'][train], data['observed'][0, train],
                            data['pairs'][validation], np.zeros(len(validation)), 71,
                            controls=False, completion_steps=5, neural_steps=5)
    _, changed, _ = select_records(controlled)
    assert changed['family'] == 'B0' and changed['config'] == 'zero' and changed['validation_mse'] == 0
    original_sum = next(record for record in records if record['config'] == 'sum')
    poisoned_sum = next(record for record in controlled if record['config'] == 'sum')
    assert original_sum['validation_mse'] == 0
    assert poisoned_sum['validation_mse'] != original_sum['validation_mse']
    for key in original_sum['model'].state:
        assert np.array_equal(original_sum['model'].state[key], poisoned_sum['model'].state[key])
    supports = [record['model'].state['support'] for record in records]
    for support in supports[1:]:
        assert np.array_equal(support, supports[0])


def test_validation_targets_change_completion_checkpoint():
    draw = np.random.default_rng(11)
    genes = 10
    pairs = np.array([(i, j) for i in range(genes) for j in range(i + 1, genes)], dtype=np.int64)
    train, validation = pairs[:30], pairs[30:40]
    residuals = draw.normal(size=len(train))

    def fit(targets):
        return ScalarCompletion(rank=8, penalty=.01, steps=50, seed=3).fit(
            train, residuals, genes, 1., (validation, targets))

    early = fit(-np.ones(len(validation)) * 3)
    late = fit(np.zeros(len(validation)))
    assert early.best_step == 25
    assert late.best_step == 50
    assert not np.array_equal(early.matrix, late.matrix)
    assert [row['validation_mse'] for row in early.history] != [row['validation_mse'] for row in late.history]


def test_lifecycle_seal_roster_skips_and_binding(life, tmp_path):
    cells = [dict(world='additive', seed=88022, noise=0.0, mask='complete')]
    missing = _bind(tmp_path / 'missing-seal', pilot=True, cells=cells)
    with pytest.raises(FileNotFoundError):
        life.score(missing)
    sealed = _bind(tmp_path / 'sealed', pilot=True, cells=cells)
    _seal(sealed, cells)
    marker = sealed / 'models' / 'marker.txt'
    marker.write_text('altered', encoding='utf-8')
    with pytest.raises(ValueError, match='Sealed artifact drift'):
        life.verify_seal(sealed)
    marker.write_text('sealed', encoding='utf-8')
    life.verify_seal(sealed)
    (sealed / 'models' / 'extra.txt').write_text('extra', encoding='utf-8')
    with pytest.raises(ValueError, match='Unsealed roster artifact'):
        life.verify_seal(sealed)
    (sealed / 'models' / 'extra.txt').unlink()
    partial = _bind(tmp_path / 'partial', pilot=True, cells=cells)
    _seal(partial, [])
    with pytest.raises(ValueError, match='Partial roster'):
        life.verify_seal(partial)
    held = _bind(tmp_path / 'held', pilot=True, cells=cells)
    _seal(held, cells)
    _write(held / 'results.json', dict(preserved=True))
    with pytest.raises(FileExistsError, match='Preserve scoring'):
        life.score(held)
    with pytest.raises(FileNotFoundError):
        life.replay(_bind(tmp_path / 'unscored', pilot=True, cells=cells))
    unscored = _bind(tmp_path / 'unscored-sealed', pilot=True, cells=cells)
    _seal(unscored, cells)
    with pytest.raises(FileNotFoundError):
        life.replay(unscored)
    _write(unscored / 'results.json', dict(preserved=True))
    _write(unscored / 'replay-start.json', dict(already=True))
    with pytest.raises(FileExistsError):
        life.replay(unscored)
    drifted = _bind(tmp_path / 'drift', pilot=True, cells=cells)
    binding = json.loads((drifted / 'code-binding.json').read_text(encoding='utf-8'))
    binding['files']['src/frontier_tournament.py'] = '0' * 64
    _write(drifted / 'code-binding.json', binding)
    with pytest.raises(ValueError, match='Code binding drift'):
        life.verify_binding(drifted)
    shifted = _bind(tmp_path / 'protocol-drift', pilot=False, cells=cells)
    _write(shifted / 'protocol.json', dict(pilot=False, cells=cells, cap_seconds=1, artifact_cap_bytes=1))
    with pytest.raises(ValueError, match='Protocol drift'):
        life.verify_binding(shifted)
    environment = _bind(tmp_path / 'environment', pilot=True, cells=cells)
    binding = json.loads((environment / 'code-binding.json').read_text(encoding='utf-8'))
    binding['environment']['numpy'] = '0'
    _write(environment / 'code-binding.json', binding)
    with pytest.raises(ValueError, match='Numerical environment drift'):
        life.verify_binding(environment)
    with pytest.raises(ValueError, match='Forecast requires pilot'):
        life.forecast(shifted)
    with pytest.raises(FileNotFoundError):
        life.forecast(missing)
    unqualified = _bind(tmp_path / 'unqualified', pilot=False, cells=cells)
    with pytest.raises(ValueError, match='Full run requires measured pilot forecast'):
        life.fit(unqualified)
    occupied = _bind(tmp_path / 'occupied', pilot=True, cells=cells)
    (occupied / 'models').mkdir()
    with pytest.raises(FileExistsError):
        life.fit(occupied)
