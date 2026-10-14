"""Safe numerical states with all inference dependencies and manifest hashes."""
import json
import hashlib
from pathlib import Path

import numpy as np

from .contracts import ContractError, read_json, sha256, write_json


def save_state(directory, state, budget=None):
    path = Path(directory)
    arrays = {}

    def pack(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ContractError("Object arrays forbidden")
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"__array__": key}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(v) for v in value]
        return value

    structure = pack(state)
    if budget:
        budget.check(sum(v.nbytes for v in arrays.values()) + len(json.dumps(structure)) + 8192)
    path.mkdir(parents=True, exist_ok=False)
    shared = path.parent / "shared_arrays"
    shared.mkdir(exist_ok=True)
    assets = {}
    for key, value in arrays.items():
        digest = hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()
        destination = shared / f"{digest}.npz"
        if not destination.exists():
            np.savez_compressed(destination, value=value)
        assets[key] = dict(path=str(destination.resolve()), sha256=sha256(destination))
    (budget.write_json if budget else write_json)(path / "state.json", structure)
    record = dict(path=str(path.resolve()), array_assets=assets, state_sha256=sha256(path / "state.json"))
    (budget.write_json if budget else write_json)(path / "manifest.json", record)
    if budget:
        budget.check()
    return record


def load_state(record):
    path = Path(record["path"])
    files = [("state.json", "state_sha256")]
    if "arrays_sha256" in record:
        files.append(("arrays.npz", "arrays_sha256"))
    for name, key in files:
        if sha256(path / name) != record[key]:
            raise ContractError(f"Checkpoint tampered: {name}")
    arrays = {}
    if "array_assets" in record:
        for key, asset in record["array_assets"].items():
            if sha256(asset["path"]) != asset["sha256"]:
                raise ContractError("Shared checkpoint array tampered")
            with np.load(asset["path"], allow_pickle=False) as archive:
                arrays[key] = archive["value"].copy()
    else:
        with np.load(path / "arrays.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}

    def unpack(value):
        if isinstance(value, dict):
            if set(value) == {"__array__"}:
                return arrays[value["__array__"]]
            return {k: unpack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [unpack(v) for v in value]
        return value

    return unpack(read_json(path / "state.json"))
