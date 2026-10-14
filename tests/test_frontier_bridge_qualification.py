"""Metadata failure must remain auditable when historical banks disappear."""
import json
from pathlib import Path

import numpy as np
import pytest

from opal.frontier_bridge import contracts
from opal.frontier_bridge.contracts import ContractError, fingerprint, load_config, qualify, sha256

ROOT = Path(__file__).parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return dict(path=str(path), sha256=sha256(path))


def metadata_fixture(tmp_path, gene_minimum=3):
    cfg = dict(contexts=["TNFA+", "TNFA-"], aliases={}, minimum_common_pairs=1,
               minimum_common_genes=gene_minimum)
    contexts, sources, support = {}, [], []
    for context, suffix in [("TNFA+", "TNFA-positive"), ("TNFA-", "TNFA-negative")]:
        local = tmp_path / "missing-banks" / suffix
        checkpoint = write(tmp_path / f"{suffix}-checkpoint.json", [dict(path=str(local / "evaluation/checkpoint"))])
        sources.append(dict(source_version=f"example/PertSpectra/{suffix}", manifest=checkpoint["path"],
                            manifest_sha256=checkpoint["sha256"]))
        contexts[context] = dict(units="log2_library_10000_change", genes=["G1", "G2"],
                                 strata=["s1"], pairs=1, local=str(local), bank_sha256="expected-historical-hash")
        for targets in [[], ["A"], ["B"], ["A", "B"]]:
            support.append(dict(context=context, sample_index="s1", targets=targets, cells=10,
                                guide_mixtures={"guide":10}, included_pair=len(targets)==2,
                                reference_supported=len(targets)<2))
    cfg["source_registry"] = write(tmp_path / "registry.json", dict(sources=sources))
    cfg["prepared_source"] = write(tmp_path / "prepared.json", dict(contexts=contexts, counts_verified=True,
                                                                   matrix_key="layers/counts", full_file_sha256="raw-hash"))
    cfg["source_access"] = write(tmp_path / "access.json", dict(full_file_sha256="raw-hash"))
    cfg["source_support"] = write(tmp_path / "support.json", support)
    return cfg


def test_metadata_failure_never_opens_bank(tmp_path, monkeypatch):
    cfg = metadata_fixture(tmp_path)
    monkeypatch.setattr(contracts.np, "load", lambda *a, **k: pytest.fail("Bank opened despite failed gene minimum"))
    result = qualify(tmp_path, cfg)
    assert result["common_gene_count"] == 2
    assert result["common_pair_count"] == 1
    assert result["reasons"] == ["insufficient_common_reference_selected_genes"]
    assert result["qualified"] is False
    assert result["configuration_hash"] == fingerprint(cfg)
    assert result["current_bank_hashes_verified"] is False
    assert result["current_bank_arrays_opened"] is False
    assert all(not b["current_bank_available"] for b in result["banks"].values())


def test_sufficient_metadata_cannot_qualify_missing_banks(tmp_path):
    result = qualify(tmp_path, metadata_fixture(tmp_path, gene_minimum=2))
    assert result["metadata_support_passed"] is True
    assert result["qualified"] is False
    assert set(result["reasons"]) == {"prepared_bank_unavailable:TNFA+", "prepared_bank_unavailable:TNFA-"}


def test_supported_current_banks_qualify_without_decoding_responses(tmp_path, monkeypatch):
    cfg = metadata_fixture(tmp_path, gene_minimum=2)
    metadata_path = Path(cfg["prepared_source"]["path"])
    prepared = json.loads(metadata_path.read_text())
    for entry in prepared["contexts"].values():
        path = Path(entry["local"]) / "canonical_bank.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, genes=np.asarray(entry["genes"]), pair_ids=np.asarray(['["A","B"]']),
                 culture_ids=np.asarray(["s1"]), singles=np.zeros((1, 2, 2)), responses=np.zeros((1, 2)))
        entry["bank_sha256"] = sha256(path)
    cfg["prepared_source"] = write(metadata_path, prepared)
    original = np.load

    class ReferenceOnlyArchive:
        def __init__(self, archive):
            self.archive = archive
            self.files = archive.files

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            assert key != "responses", "Qualification decoded pair responses"
            return self.archive[key]

    monkeypatch.setattr(contracts.np, "load", lambda *a, **k: ReferenceOnlyArchive(original(*a, **k)))
    result = qualify(tmp_path, cfg)
    assert result["qualified"] is True
    assert result["current_bank_hashes_verified"] is True
    assert result["current_bank_arrays_opened"] is True
    assert result["query_expression_accessed"] is False


def test_included_pair_requires_supported_reference(tmp_path):
    cfg = metadata_fixture(tmp_path)
    path = Path(cfg["source_support"]["path"])
    support = json.loads(path.read_text())
    support[1]["cells"] = 9
    cfg["source_support"] = write(path, support)
    with pytest.raises(ContractError, match="Missing matched reference"):
        qualify(tmp_path, cfg)


def test_metadata_hash_tampering_fails_before_bank_access(tmp_path):
    cfg = metadata_fixture(tmp_path)
    Path(cfg["prepared_source"]["path"]).write_text("{}")
    with pytest.raises(ContractError, match="tampered"):
        qualify(tmp_path, cfg)


def test_resource_amendment_tampering_and_execution_expansion_refused(tmp_path):
    cfg = json.loads((ROOT / "configs/experiments/opal_frontier_bridge_v1.json").read_text())
    amendment = dict(experiment_id="opal_frontier_bridge_v1", authorization_basis="Owner permits measured storage continuation",
                     new_retained_bytes=250000000, artifact_cap_bytes=3250000000)
    cfg["resources"].update(new_retained_bytes=250000000, artifact_cap_bytes=3250000000)
    cfg["resources"]["amendment"] = write(tmp_path / "amendment.json", amendment)
    config_path = tmp_path / "config.json"
    write(config_path, cfg)
    assert load_config(config_path)["resources"]["new_retained_bytes"] == 250000000
    Path(cfg["resources"]["amendment"]["path"]).write_text("{}")
    with pytest.raises(ContractError, match="tampered"):
        load_config(config_path)
    for alteration in [dict(workers=2), dict(cpu_core_hours=25), dict(raw_downloads=True), dict(new_retained_bytes=True)]:
        changed = dict(amendment, **alteration)
        cfg["resources"]["amendment"] = write(tmp_path / "amendment.json", changed)
        write(config_path, cfg)
        with pytest.raises(ContractError):
            load_config(config_path)
