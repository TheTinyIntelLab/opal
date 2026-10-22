"""No-fit stage trace for the frozen wave-three validation mismatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def difference(a, b):
    import numpy as np
    delta = np.abs(a-b)
    index = tuple(int(i) for i in np.unravel_index(np.argmax(delta), delta.shape)) if delta.size else ()
    return dict(max_abs=float(delta.max(initial=0)), index=index,
                unequal=int(np.count_nonzero(delta)), exact=bool(np.array_equal(a, b)))


def stages(model, pairs):
    import numpy as np
    from opal.models.fitness import design
    ordered = np.sort(model.singles[pairs], axis=1)
    coordinates = (ordered-model.input_center)/model.input_scale
    distance = np.square(coordinates[:, None, :]-model.centers[None, :, :]).sum(2)
    kernel = np.exp(-distance/(2*model.bandwidth**2))
    whitened = kernel@model.whitener
    features = np.concatenate((design(model.singles, pairs, model.nodes), whitened), axis=1)
    standardized = (features-model.center)/model.scale
    dot = standardized@model.coefficient
    additive = model.singles[pairs].sum(1)
    return dict(ordered=ordered, coordinates=coordinates, distance=distance, kernel=kernel,
                whitened=whitened, features=features, standardized=standardized, dot=dot,
                prediction=(additive+model.offset)+dot)


def query_trace(model, pairs, chunk=2048, check=lambda: None):
    import numpy as np
    result = {}
    for start in range(0, len(pairs), chunk):
        check()
        current = stages(model, pairs[start:start+chunk])
        if not result:
            result = {k: np.empty((len(pairs),)+v.shape[1:], dtype=v.dtype) for k, v in current.items()}
        for key, value in current.items():
            result[key][start:start+len(value)] = value
    return result


def global_trace(model, pairs, ids, check=lambda: None):
    import numpy as np
    result = {}
    for start in range(0, len(pairs), 2048):
        check()
        current_ids = np.flatnonzero((ids >= start) & (ids < start+2048))
        if not len(current_ids):
            continue
        current = stages(model, pairs[start:start+2048])
        if not result:
            result = {k: np.empty((len(ids),)+v.shape[1:], dtype=v.dtype) for k, v in current.items()}
        for key, value in current.items():
            result[key][current_ids] = value[ids[current_ids]-start]
    return result


def projection(model, features, pairs, chunk=2048, grouping='historical'):
    import numpy as np
    dot = np.concatenate([((features[s:s+chunk]-model.center)/model.scale)@model.coefficient
                          for s in range(0, len(pairs), chunk)])
    additive = model.singles[pairs].sum(1)
    return (additive+model.offset)+dot if grouping == 'historical' else additive+(model.offset+dot)


def diagnose(a, freeze, check):
    import numpy as np
    import opal
    from opal.flexible_singles import FlexibleSingles
    from threadpoolctl import threadpool_info, threadpool_limits
    if a.installed:
        package = a.installed.resolve()
        assert Path(opal.__file__).resolve().is_relative_to(package), opal.__file__
        for name, module in list(sys.modules.items()):
            if (name == 'opal' or name.startswith('opal.')) and getattr(module, '__file__', None):
                assert Path(module.__file__).resolve().is_relative_to(package), name
    reg_path = a.stopped/'registration.json'
    reg = read(reg_path)
    q_path = Path(reg['q1b'])/'registration.json'
    bank_path = Path(read(q_path)['paths'][2])
    split_path = Path(reg['block'])/'assignments.npz'
    folder = Path(reg['flexible'])/'horlbeck_K562/fold-0'
    selection_path = folder/'selection.json'
    selection = read(selection_path)
    old_path = folder/f"nodes-{selection['chosen']['nodes']}.npz"
    transform_path = a.stopped/'horlbeck_K562/fold-0/transform.npz'
    baseline_path = a.stopped/'horlbeck_K562/fold-0/flexible_nodes.npz'
    paths = [reg_path, q_path, bank_path, split_path, selection_path, old_path, transform_path, baseline_path]
    paths += [ROOT/name for name in reg['code']]
    hashes = {str(p): digest(p) for p in paths}
    for p in paths:
        expected = reg['inputs'].get(str(p))
        if expected:
            assert hashes[str(p)] == expected, str(p)
    for name, expected in reg['code'].items():
        assert digest(ROOT/name) == expected, name
    assert digest(old_path) == selection['files'][old_path.name]
    write(a.output.with_suffix('.inputs.json'), dict(input_sha256=hashes,
          allowed_bank_fields=['genes', 'pairs', 'singles'], outcome_arrays_loaded=False))
    with np.load(bank_path, allow_pickle=False) as z:
        bank = {k: z[k].copy() for k in ('genes', 'pairs', 'singles')}
    with np.load(split_path, allow_pickle=False) as z:
        split = z['horlbeck_K562__0'].copy()
    with np.load(baseline_path, allow_pickle=False) as z:
        baseline = {k: z[k].copy() for k in z.files}
    pairs = bank['pairs']
    ids = np.flatnonzero(split == 1)
    train = np.flatnonzero(split == 0)
    old, transform = FlexibleSingles.load(old_path), FlexibleSingles.load(transform_path)
    state = {}
    for key in ('singles', 'support', 'input_center', 'input_scale', 'center_rows', 'centers', 'whitener', 'bandwidth'):
        state[key] = bool(np.array_equal(getattr(old, key), getattr(transform, key)))
        assert state[key], key
    for key in ('center', 'scale', 'coefficient'):
        state[key] = bool(np.array_equal(getattr(old, key), baseline[key]))
        assert state[key], key
    assert old.offset == float(baseline['mean'])
    assert old.alpha == float(baseline['alpha'])
    assert np.array_equal(old.singles, bank['singles'][0].mean(0))
    assert len(old.singles) == len(bank['genes'])
    assert np.array_equal(old.centers, old._coordinates(pairs[train])[old.center_rows])
    assert old.support[pairs[ids]].all()
    state.update(offset_mean=True, alpha=True, bank_singles=True, centers_train_mapping=True,
                 gene_schema_length=True, support=True)
    metadata = {key: dict(shape=list(np.asarray(getattr(old, key)).shape),
                         dtype=str(np.asarray(getattr(old, key)).dtype),
                         strides=list(np.asarray(getattr(old, key)).strides))
                for key in ('singles', 'centers', 'whitener', 'center', 'scale', 'coefficient')}
    with threadpool_limits(1):
        reference = old.predict(pairs[ids])
        local = query_trace(old, pairs[ids], check=check)
        global_map = global_trace(old, pairs, ids, check)
        assert np.array_equal(reference, local['prediction'])
        legacy = projection(old, global_map['features'], pairs[ids], grouping='v2')
        stage_differences = {k: difference(local[k], global_map[k]) for k in local}
        worst = int(np.argmax(abs(legacy-reference)))
        weighted = (global_map['features'][worst]-local['features'][worst])/old.scale*old.coefficient
        order = np.argsort(abs(weighted))[::-1][:12]
        causal = {
            'global_features_same_projection': difference(projection(old, global_map['features'], pairs[ids]), reference),
            'local_features_v2_assembly': difference(projection(old, local['features'], pairs[ids], grouping='v2'), reference),
            'global_features_v2_assembly': difference(legacy, reference),
            'same_features_F_layout': difference(projection(old, np.asfortranarray(local['features']), pairs[ids]), reference),
            'same_features_full_projection': difference(projection(old, local['features'], pairs[ids], chunk=len(ids)), reference),
        }
        partitions = {}
        row_min, row_max = reference.copy(), reference.copy()
        for size in (1, 2, 63, 2047, 2048, 2049, len(ids)):
            current = query_trace(old, pairs[ids], size, check)['prediction']
            partitions[str(size)] = difference(current, reference)
            row_min = np.minimum(row_min, current)
            row_max = np.maximum(row_max, current)
        permutation = np.random.default_rng(41).permutation(len(ids))
        permuted = old.predict(pairs[ids][permutation])[np.argsort(permutation)]
        partitions['permuted_inverse'] = difference(permuted, reference)
        partitions['reversed_orientation'] = difference(old.predict(pairs[ids][:, ::-1]), reference)
        noncontiguous = np.repeat(pairs[ids], 2, axis=0)[::2]
        partitions['noncontiguous'] = difference(old.predict(noncontiguous), reference)
        repeated = np.repeat(pairs[ids][worst:worst+1], 2049, axis=0)
        repeat_pred = old.predict(repeated)
        partitions['repeated_worst_row_boundary'] = difference(repeat_pred, np.full(2049, reference[worst]))
        # Fixed per-row contractions are a diagnostic alternative, not an accepted fix.
        stable = np.empty(len(ids))
        from opal.models.fitness import design
        for start in range(0, len(ids), 2048):
            check()
            p = pairs[ids[start:start+2048]]
            kernel = local['kernel'][start:start+len(p)]
            white = np.einsum('ij,jk->ik', kernel, old.whitener, optimize=False)
            x = np.concatenate((design(old.singles, p, old.nodes), white), axis=1)
            dot = np.einsum('ij,j->i', (x-old.center)/old.scale, old.coefficient, optimize=False)
            stable[start:start+len(p)] = (old.singles[p].sum(1)+old.offset)+dot
        candidate = difference(stable, reference)
    check()
    assert all(digest(p) == h for p, h in hashes.items()), 'Input mutation'
    tol = freeze['numerical']['absolute_tolerance']
    return dict(schema='opal-wave4-numerical-diagnosis-v1', fitting=False, scoring=False,
        reserved_response_or_header_access=False, immutable_historical_target='Q2b nodes selected state; validation sequence; legacy internal 2048',
        validation_rows=len(ids), state_exact=state, array_metadata=metadata,
        input_sha256=hashes, ordered_gene_schema_sha256=hashlib.sha256(bank['genes'].tobytes()).hexdigest(),
        validation_indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        validation_pair_keys_sha256=hashlib.sha256(pairs[ids].tobytes()).hexdigest(),
        stage_differences=stage_differences, causal_swaps=causal, legacy_partitions=partitions,
        legacy_all_variant_interval_intersection=bool(np.all(row_max-row_min <= 2*tol)),
        worst_row=dict(validation_position=worst, bank_index=int(ids[worst]), pair=pairs[ids[worst]].tolist(),
                       legacy_delta=float(legacy[worst]-reference[worst]),
                       weighted_feature_delta_sum=float(weighted.sum()),
                       columns=[dict(column=int(j), scale=float(old.scale[j]), coefficient=float(old.coefficient[j]),
                                     raw_delta=float(global_map['features'][worst,j]-local['features'][worst,j]),
                                     weighted_delta=float(weighted[j])) for j in order]),
        fixed_order_candidate=candidate, fixed_order_candidate_compatible=candidate['max_abs'] <= tol,
        absolute_tolerance=tol, relative_tolerance=0, successor_authorized=False,
        environment=dict(python=sys.version, numpy=np.__version__, module=str(Path(opal.__file__).resolve()),
                         flexible_module=str(Path(sys.modules[FlexibleSingles.__module__].__file__).resolve()),
                         flexible_module_sha256=digest(sys.modules[FlexibleSingles.__module__].__file__),
                         threadpools=threadpool_info(), dependency_reuse=True),
        engineering_decision='diagnosis_only; no accepted invariant repair' if candidate['max_abs'] > tol else 'candidate needs independent review and invariance tests')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--stopped', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--installed', type=Path)
    a = parser.parse_args()
    if a.installed:
        sys.path.insert(0, str(a.installed.resolve()))
    else:
        sys.path.insert(0, str(ROOT))
    freeze = read(a.freeze)
    n = freeze['numerical']
    assert n['authorized'] and n['saved_state_only'] and n['synthetic_no_fit']
    assert n['absolute_tolerance'] == 1e-12 and n['relative_tolerance'] == 0
    assert not freeze['successor_authorized'] and not freeze['scope']['biological_fit'] and not freeze['scope']['test_scoring']
    for name, expected in freeze['files'].items():
        assert digest(ROOT/name) == expected, name
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    import psutil
    process = psutil.Process()
    began, cpu = time.perf_counter(), time.process_time()
    peak = 0
    def check():
        nonlocal peak
        memory = process.memory_info()
        peak = max(peak, memory.rss, getattr(memory, 'peak_wset', 0))
        if time.perf_counter()-began > n['wall_seconds'] or peak > n['peak_bytes']:
            raise RuntimeError('Diagnostic resource cap exceeded')
        if shutil.disk_usage(a.output.parent).free < n['reserve_bytes']:
            raise RuntimeError('Writable volume reserve required')
    start = a.output.with_suffix('.started.json')
    write(start, dict(command=sys.argv, freeze_sha256=digest(a.freeze), code_sha256=digest(__file__),
                      guard='cooperative checks; caller must enforce interrupting wall timeout'))
    try:
        check()
        result = diagnose(a, freeze, check)
        result['resources'] = dict(wall_seconds=time.perf_counter()-began, cpu_seconds=time.process_time()-cpu,
                                   peak_rss_bytes=peak, guard='cooperative')
        encoded_size = len(json.dumps(result).encode('utf-8'))
        if encoded_size > n['output_bytes']:
            raise RuntimeError('Output cap exceeded')
        result['code_sha256'] = digest(__file__)
        result['freeze_sha256'] = digest(a.freeze)
        write(a.output, result)
        print(json.dumps({k: result[k] for k in ('engineering_decision', 'fixed_order_candidate', 'causal_swaps', 'resources')}))
    except BaseException as exc:
        write(a.output.with_suffix('.failure.json'), dict(error=repr(exc), wall_seconds=time.perf_counter()-began,
              cpu_seconds=time.process_time()-cpu, peak_rss_bytes=peak, code_sha256=digest(__file__)))
        raise


if __name__ == '__main__':
    main()
