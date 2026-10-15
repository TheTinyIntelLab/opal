"""Content-addressed, portable JSON/NPZ states without pickle."""
from pathlib import Path
import numpy as np
from ..artifacts import array_hash, read_json, sha256, write_json
from .contracts import ContractError


def save_state(path, state, budget=None):
    path = Path(path)
    write = budget.write_json if budget is not None and hasattr(budget, "write_json") else write_json
    if budget:
        budget.check()
    path.mkdir(parents=True, exist_ok=False)
    store = path.parent / "arrays"
    store.mkdir(exist_ok=True)
    assets = {}
    def pack(v):
        if isinstance(v, np.ndarray):
            if v.dtype.hasobject or (v.dtype.kind in "fc" and not np.isfinite(v).all()):
                raise ContractError("Unsafe state array")
            digest = array_hash(v)
            target = store / (digest + ".npz")
            if budget:
                budget.check(v.nbytes + 8192)
            if not target.exists():
                np.savez_compressed(target, value=v)
            assets[digest] = dict(path="../arrays/" + target.name, sha256=sha256(target), bytes=target.stat().st_size,
                logical_bytes=v.nbytes)
            return {"__array__": digest}
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, dict):
            return {k: pack(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [pack(x) for x in v]
        return v
    write(path / "state.json", pack(state))
    record = dict(schema="opal-context-contrast-state-v1", assets=assets, state_sha256=sha256(path / "state.json"))
    write(path / "manifest.json", record)
    if budget:
        budget.check()
    return record


def load_state(path):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    if manifest.get("schema") != "opal-context-contrast-state-v1" or sha256(path / "state.json") != manifest["state_sha256"]:
        raise ContractError("State schema/hash mismatch")
    arrays = {}
    for key, record in manifest["assets"].items():
        asset = (path / record["path"]).resolve()
        if not asset.is_relative_to(path.parent) or sha256(asset) != record["sha256"]:
            raise ContractError("State asset mismatch")
        with np.load(asset, allow_pickle=False) as archive:
            # Keep the NPY layout: SVD bases are commonly column-major.
            # A default C-order copy can change BLAS summation and break
            # bitwise replay despite identical serialized float values.
            arrays[key] = archive["value"].copy(order="K")
        if array_hash(arrays[key]) != key:
            raise ContractError("Array content mismatch")
    def unpack(v):
        if isinstance(v, dict):
            if set(v) == {"__array__"}:
                return arrays[v["__array__"]]
            return {k: unpack(x) for k, x in v.items()}
        if isinstance(v, list):
            return [unpack(x) for x in v]
        return v
    return unpack(read_json(path / "state.json"))


def save_npz(path, budget=None, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if budget:
        budget.check(sum(np.asarray(a).nbytes for a in arrays.values()) + 8192)
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise ContractError("Object role arrays forbidden")
    np.savez_compressed(path, **arrays)
    if budget:
        budget.check()
    return dict(path=path.name, sha256=sha256(path), bytes=path.stat().st_size)


def load_npz(root, record):
    path = (Path(root) / record["path"]).resolve()
    if not path.is_relative_to(Path(root).resolve()) or sha256(path) != record["sha256"]:
        raise ContractError("Role file hash/path mismatch")
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}
