"""Fixture checks for the v2 growth adapter. No profile refit and no biological load."""
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

import tools.frontier_growth as growth_v1
import tools.frontier_growth_v2 as growth


_V1_SHA256 = {
    'tools/frontier_growth.py': '8f67b81de7a36ffeced2574397ba1e1134a5f962f210d32b583d085af6f3b6b9',
    'src/frontier_tournament.py': 'a1c54b4668de6fc64de358cfaac5048c282ab8b88576c7ee1f59db6330686b98',
    'tools/frontier_tournament.py': '15f527107d006d39b9bcfe03c389093aebfc34bc080b10e513762340c0637318',
}


def test_original_modules_and_caps_are_unchanged():
    root = Path(__file__).resolve().parents[1]
    for name, expected in _V1_SHA256.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    assert growth_v1.PROTOCOL_ID == 'opal-frontier-growth-v1'
    assert growth.PROTOCOL_ID == 'opal-frontier-growth-v2'
    assert growth.PROGRAM_CAP == growth_v1.PROGRAM_CAP
    assert growth.PROFILE_CAP == growth_v1.PROFILE_CAP
    assert growth.RAW_COMPLETION_BYTES == growth_v1.RAW_COMPLETION_BYTES == 345_600_000
    assert growth.TRAIN_LABELS == 16000 and growth.VALIDATION_LABELS == 4000
    assert growth.PROFILE_QUERY_COUNT == 10000 and growth.PROFILE_BOUND_GENES == 600
    assert growth.FIT_SEED == growth_v1.FIT_SEED
    assert growth.PACKING_CONTRACT != growth.CONTRACT


def test_guard_charges_retained_profile_bytes(tmp_path):
    import json
    (tmp_path / 'packed-qualification.json').write_text(json.dumps({
        'retained_profile_bytes': growth.PROGRAM_CAP['artifact_bytes']}), encoding='utf-8')
    guard = growth.Guard(tmp_path, growth.PROGRAM_CAP)
    with pytest.raises(RuntimeError, match='Artifact cap'):
        guard.check()


def test_schema_bounds_stop_before_a_fit(tmp_path):
    growth.require_schema_bounds(600, 10000, where='edge')
    with pytest.raises(growth.QualificationError, match='not dropped'):
        growth.require_schema_bounds(601, 10000, where='edge')
    with pytest.raises(growth.QualificationError, match='not truncated'):
        growth.require_schema_bounds(600, 10001, where='edge')
    wide = tmp_path / 'wide'
    wide.mkdir()
    _write_fit_args(wide, genes=601, test=np.array([2], dtype=np.int64))
    with pytest.raises(growth.QualificationError, match='gene'):
        growth.fit_procedure(wide)
    assert not (wide / 'checkpoints').exists()
    long = tmp_path / 'long'
    long.mkdir()
    _write_fit_args(long, genes=10, test=np.arange(2, 10003, dtype=np.int64))
    with pytest.raises(growth.QualificationError, match='not truncated'):
        growth.fit_procedure(long)
    assert not (long / 'checkpoints').exists()


def _write_fit_args(proc, genes, test):
    pairs = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    growth.save_arrays(
        proc / 'fit-args.npz', pairs=pairs, singles=np.zeros(genes),
        gene_names=np.array([f'g{i}' for i in range(genes)]),
        train_index=np.array([0], dtype=np.int64),
        validation_index=np.array([1], dtype=np.int64), test_index=test,
        train_targets=np.array([0.0]), validation_targets=np.array([0.0]))


def test_packed_forecast_replaces_completion_bytes_under_the_same_caps():
    procedures = []
    for artifact, full, packed in ((10000, 4000, 1000), (8000, 3000, 900), (7000, 2000, 800), (6000, 1000, 700)):
        procedures.append(dict(
            v1_artifact_bytes=artifact, full_completion_bytes=full, packed_completion_bytes=packed,
            new_artifact_bytes=artifact - full + packed, v1_wall_seconds=10.0, v1_cpu_seconds=9.0,
            packing_wall_seconds=1.0, packing_cpu_seconds=0.5))
    result = growth.forecast_packed_program(
        procedures, 1000, 100, 1024, 512, 10.0, 9.0, 2.0, 1.0)
    assert result['qualified'] is True
    assert result['forecast_artifact_bytes'] == 7000 * 30 * 1.25 + 1000 + 100
    assert result['source_cap_changed'] is False
    assert result['artifact_cap_bytes'] == 512 * 1024 ** 2
    over_profile = growth.forecast_packed_program(
        procedures, 49_336_536, 20_000_000, 1024, 512, 10.0, 9.0, 2.0, 1.0)
    assert over_profile['qualified'] is False
    assert '64MiB' in over_profile['blocker']
    assert over_profile['combined_profile_bytes'] == 49_336_536 + 20_000_000
    late = growth.forecast_packed_program(
        procedures, 1000, 100, 1024, 512, 590.0, 590.0, 20.0, 20.0)
    assert late['qualified'] is False
    assert '600' in late['blocker']
    with pytest.raises(ValueError, match='exceed'):
        growth.forecast_packed_program([
            dict(v1_artifact_bytes=10, full_completion_bytes=11, packed_completion_bytes=1,
                 v1_wall_seconds=1, v1_cpu_seconds=1, packing_wall_seconds=0, packing_cpu_seconds=0)
            for _ in range(4)], 1, 1, 1, 1, 1, 1, 1, 1)


def test_profile_refit_is_refused_and_packing_is_the_only_new_stage(tmp_path):
    with pytest.raises(RuntimeError, match='qualify-packed'):
        growth.profile(tmp_path / 'unused')
    source = inspect.getsource(growth.fit_procedure)
    assert 'save_checkpoint(' in source
    assert "record['model'].save" not in source
    qualify = inspect.getsource(growth.qualify_packed)
    assert 'load_screen_roster' not in qualify
    assert 'fit_models' not in qualify
    assert 'make_world' not in qualify
    loader = growth.active_predictor()
    assert loader.load is not None
    with pytest.raises(ValueError, match='outside the checkout'):
        growth.active_predictor(growth.ROOT)


def test_qualify_packed_freezes_before_refusing_biology(tmp_path):
    out = tmp_path / 'v2'
    prior = tmp_path / 'prior'
    prior.mkdir()
    marker = prior / 'fit-started.json'
    marker.write_text('{}\n', encoding='utf-8')
    before = sorted(path.name for path in prior.iterdir())
    with pytest.raises(growth.QualificationError, match='biological'):
        growth.qualify_packed(out, prior)
    assert (out / 'protocol.json').is_file()
    assert sorted(path.name for path in prior.iterdir()) == before
    assert marker.read_text(encoding='utf-8') == '{}\n'
    assert not any(prior.rglob('*.npz'))


def test_prior_binding_and_missing_profile_are_refused(tmp_path):
    with pytest.raises(ValueError, match='Code binding drift'):
        growth._verify_bound_hashes({'tools/frontier_growth.py': '0' * 64})
    prior = tmp_path / 'empty'
    prior.mkdir()
    with pytest.raises(ValueError, match='missing'):
        growth.verify_prior_profile(prior)
