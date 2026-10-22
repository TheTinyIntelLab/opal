"""Wave-five experiment-only saved-state numerical matrix. Never fits or scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-12
STAGES = ('ordered', 'coordinates', 'distance', 'kernel', 'whitened', 'features',
          'standardized', 'dot', 'additive', 'prediction')
VARIANTS = ['legacy'] + [f'{method}_{stage}' for method in
    ('matmul', 'einsum_false', 'einsum_true', 'ascending') for stage in ('white', 'dot', 'joint')]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def array_hash(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def difference(a, b, rows=None):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Shape mismatch or nonfinite arithmetic')
    delta = np.abs(a-b)
    nonzero = b != 0
    rel = np.zeros_like(delta)
    np.divide(delta, np.abs(b), out=rel, where=nonzero)
    mask = delta > ATOL
    rowmask = np.any(mask.reshape(len(a), -1), axis=1) if len(a) else np.zeros(0, bool)
    unequal_rows = np.any((delta != 0).reshape(len(a), -1), axis=1) if len(a) else np.zeros(0, bool)
    worst = tuple(int(i) for i in np.unravel_index(np.argmax(delta), delta.shape)) if delta.size else ()
    indices = np.flatnonzero(rowmask)
    return dict(max_abs=float(delta.max(initial=0)), max_relative_nonzero_reference=float(rel.max(initial=0)),
        zero_reference_nonzero_error=int(np.count_nonzero((~nonzero) & (delta != 0))),
        unequal_elements=int(np.count_nonzero(delta)), over_atol_elements=int(mask.sum()),
        unequal_rows=int(unequal_rows.sum()), affected_rows=int(rowmask.sum()),
        affected_row_positions=indices.tolist(),
        affected_bank_indices=(np.asarray(rows)[indices].tolist() if rows is not None else None),
        worst_index=worst, exact=bool(np.array_equal(a, b)), within_atol=bool(not mask.any()))


def metadata(a):
    import numpy as np
    a = np.asarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype), byteorder=a.dtype.byteorder,
                strides=list(a.strides), c_contiguous=bool(a.flags.c_contiguous),
                f_contiguous=bool(a.flags.f_contiguous), sha256=array_hash(a))


def contract_product(a, b, method):
    import numpy as np
    if method == 'legacy':
        return a @ b
    if method == 'matmul':
        return np.matmul(a, b)
    if method.startswith('einsum'):
        return np.einsum('ij,jk->ik' if b.ndim == 2 else 'ij,j->i', a, b,
                         optimize=method == 'einsum_true')
    if method != 'ascending':
        raise ValueError(method)
    # Product is rounded in operand dtype, then an ascending-index accumulator.
    if b.ndim == 1:
        products = np.multiply(a, b, dtype=a.dtype)
        return np.add.accumulate(products, axis=1, dtype=a.dtype)[:, -1]
    result = np.zeros((len(a), b.shape[1]), dtype=a.dtype)
    for j in range(a.shape[1]):
        result = np.add(result, np.multiply(a[:, j:j+1], b[j], dtype=a.dtype), dtype=a.dtype)
    return result


def stage(model, pairs, variant='legacy', dtype='float64', feature_layout='C'):
    import numpy as np
    from opal.models.fitness import design
    method, _, target = variant.rpartition('_')
    if variant == 'legacy':
        method, target = 'legacy', 'joint'
    white_method = method if target in ('white', 'joint') else 'legacy'
    dot_method = method if target in ('dot', 'joint') else 'legacy'
    dt = np.dtype(dtype)
    singles = np.asarray(model.singles, dtype=dt)
    ordered = np.sort(singles[pairs], axis=1)
    coordinates = (ordered-np.asarray(model.input_center, dtype=dt))/np.asarray(model.input_scale, dtype=dt)
    distance = np.square(coordinates[:, None, :]-np.asarray(model.centers, dtype=dt)[None, :, :]).sum(2)
    kernel = np.exp(-distance/(dt.type(2)*dt.type(model.bandwidth)**2))
    whitener = np.asarray(model.whitener, dtype=dt)
    if feature_layout == 'F':
        kernel, whitener = np.asfortranarray(kernel), np.asfortranarray(whitener)
    white = contract_product(kernel, whitener, white_method)
    features = np.concatenate((np.asarray(design(singles, pairs, model.nodes), dtype=dt), white), axis=1)
    if feature_layout == 'F':
        features = np.asfortranarray(features)
    elif feature_layout == 'strided':
        features = np.repeat(features, 2, axis=1)[:, ::2]
    standardized = (features-np.asarray(model.center, dtype=dt))/np.asarray(model.scale, dtype=dt)
    dot = contract_product(standardized, np.asarray(model.coefficient, dtype=dt), dot_method)
    additive = singles[pairs].sum(1)
    prediction = (additive+dt.type(model.offset))+dot
    result = dict(zip(STAGES, (ordered, coordinates, distance, kernel, white, features,
                               standardized, dot, additive, prediction)))
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite stage arithmetic')
    return result


def trace(model, pairs, sizes=(2048,), variant='legacy', dtype='float64', layout='C', full=False):
    import numpy as np
    selected = STAGES if full else ('prediction',)
    result, start, count = {}, 0, 0
    while start < len(pairs):
        size = sizes[count % len(sizes)]
        current = stage(model, pairs[start:start+size], variant, dtype, layout)
        if not result:
            result = {k: np.empty((len(pairs),)+current[k].shape[1:], dtype=current[k].dtype) for k in selected}
        for k in selected:
            result[k][start:start+len(current[k])] = current[k]
        start += size
        count += 1
    return result


def global_trace(model, pairs, ids):
    import numpy as np
    result = {}
    for start in range(0, len(pairs), 2048):
        chosen = np.flatnonzero((ids >= start) & (ids < start+2048))
        if not len(chosen):
            continue
        current = stage(model, pairs[start:start+2048])
        if not result:
            result = {k: np.empty((len(ids),)+v.shape[1:], dtype=v.dtype) for k, v in current.items()}
        for k, value in current.items():
            result[k][chosen] = value[ids[chosen]-start]
    return result


def project(model, features, pairs, grouping='historical'):
    import numpy as np
    parts = [((features[s:s+2048]-model.center)/model.scale) @ model.coefficient
             for s in range(0, len(pairs), 2048)]
    dot = np.concatenate(parts)
    additive = model.singles[pairs].sum(1)
    return (additive+model.offset)+dot if grouping == 'historical' else additive+(model.offset+dot)


def input_paths():
    paths = {str(p): h for p, h in read(ROOT/'results/research-wave4/numerical-diagnosis.inputs.json')['input_sha256'].items()}
    # New v1 state was whole-file bound by root in the plan freeze.
    for p in (Path('E:/opal-wave3-20261021/flexible-nuisance-v1/registration.json'),
              Path('E:/opal-wave3-20261021/flexible-nuisance-v1/horlbeck_K562/fold-0/transform.npz'),
              Path('E:/opal-wave3-20261021/flexible-nuisance-v1/horlbeck_K562/fold-0/flexible_nodes.npz')):
        if p.exists():
            paths[str(p)] = digest(p)
    return paths


def load_allowed(binding):
    import numpy as np
    from opal.flexible_singles import FlexibleSingles
    reg = read('E:/opal-wave3-20261021/flexible-nuisance-v2/registration.json')
    q = read(Path(reg['q1b'])/'registration.json')
    with np.load(q['paths'][2], allow_pickle=False) as z:
        bank = {k: z[k].copy() for k in ('genes', 'pairs', 'singles')}
    with np.load(Path(reg['block'])/'assignments.npz', allow_pickle=False) as z:
        split = z['horlbeck_K562__0'].copy()
    folder = Path(reg['flexible'])/'horlbeck_K562/fold-0'
    selection = read(folder/'selection.json')
    model = FlexibleSingles.load(folder/f"nodes-{selection['chosen']['nodes']}.npz")
    ids = np.flatnonzero(split == 1)
    if len(ids) != 15779 or not model.support[bank['pairs'][ids]].all():
        raise ValueError('Validation identity/support drift')
    if not np.array_equal(model.singles, bank['singles'][0].mean(0)):
        raise ValueError('Saved singles drift')
    return model, bank, ids


def history(model, bank, ids):
    import numpy as np
    from types import SimpleNamespace
    from opal.flexible_singles import FlexibleSingles
    pairs, query = bank['pairs'], bank['pairs'][ids]
    local = trace(model, query, full=True)
    reference = model.predict(query)
    if not np.array_equal(reference, local['prediction']):
        raise ValueError('Historical query procedure reconstruction differs')
    result = {}
    for version, expected in (('v1', 1.2463305998355878e-8), ('v2', 1.245199382493567e-8)):
        folder = Path(f'E:/opal-wave3-20261021/flexible-nuisance-{version}/horlbeck_K562/fold-0')
        if not (folder/'transform.npz').exists() or not (folder/'flexible_nodes.npz').exists():
            result[version] = dict(status='blocked_missing_saved_state', expected_max_abs=expected)
            continue
        transform = FlexibleSingles.load(folder/'transform.npz')
        with np.load(folder/'flexible_nodes.npz', allow_pickle=False) as z:
            baseline = {k: z[k].copy() for k in ('center', 'scale', 'coefficient', 'mean', 'nodes', 'alpha')}
        global_map = global_trace(transform, pairs, ids)
        # V1 map has three basic columns + nonlinear; prior.design appends incidence.
        if version == 'v1':
            n = len(model.singles)
            incidence = np.zeros((len(ids), n))
            incidence[np.arange(len(ids))[:, None], query] = 1
            actual = np.column_stack((global_map['features'], incidence))
            canonical = np.r_[np.arange(3), np.arange(3+len(transform.centers), actual.shape[1]),
                              np.arange(3, 3+len(transform.centers))]
        else:
            actual, canonical = global_map['features'], np.arange(global_map['features'].shape[1])
        state = SimpleNamespace(**{k: baseline[k][canonical] for k in ('center', 'scale', 'coefficient')},
                                offset=float(baseline['mean']), singles=model.singles)
        actual_state = SimpleNamespace(**{k: baseline[k] for k in ('center', 'scale', 'coefficient')},
                                      offset=float(baseline['mean']), singles=model.singles)
        prediction = project(actual_state, actual, query, 'nuisance')
        canonical_features = actual[:, canonical]
        stage_deltas = {k: difference(global_map[k], local[k], ids) for k in ('ordered', 'coordinates', 'distance', 'kernel', 'whitened')}
        stage_deltas['features'] = difference(canonical_features, local['features'], ids)
        for k, array in (('standardized', (canonical_features-state.center)/state.scale),
                         ('dot', (prediction-model.singles[query].sum(1))-state.offset), ('prediction', prediction)):
            stage_deltas[k] = difference(array, local[k], ids)
        delta = difference(prediction, reference, ids)
        worst = int(np.argmax(abs(prediction-reference)))
        weighted = (canonical_features[worst]-local['features'][worst])/model.scale*model.coefficient
        causal_pred = project(model, canonical_features, query)
        signed = float(weighted.sum())
        result[version] = dict(status='reproduced' if delta['max_abs'] == expected else 'observed_different',
            expected_max_abs=expected, expected_observed_abs_error=abs(delta['max_abs']-expected), delta=delta,
            state_deltas={k: difference(getattr(state, k)[None, :], getattr(model, k)[None, :]) for k in ('center', 'scale', 'coefficient')},
            state_metadata={k: metadata(v) for k, v in baseline.items()},
            map_state_equal={k: bool(np.array_equal(getattr(model, k), getattr(transform, k))) for k in
                ('singles', 'support', 'input_center', 'input_scale', 'center_rows', 'centers', 'whitener', 'bandwidth')},
            canonical_column_permutation=canonical.tolist(), stages=stage_deltas,
            first_unequal_stage=next((k for k, v in stage_deltas.items() if not v['exact']), None),
            causal_swaps=dict(map_only_old_state=difference(causal_pred, reference, ids),
                state_only_old_map=difference(project(state, local['features'], query), reference, ids),
                addition_only_old_map=difference(project(model, local['features'], query, 'nuisance'), reference, ids),
                canonicalized_historical_projection=difference(project(state, canonical_features, query, 'nuisance'), prediction, ids)),
            implicated_row=dict(validation_position=worst, bank_index=int(ids[worst]), pair=query[worst].tolist(),
                map_signed_sum=signed, map_triangle_bound=float(abs(weighted).sum()),
                map_final_delta=float(causal_pred[worst]-reference[worst]),
                reconstruction_residual=float(causal_pred[worst]-reference[worst])-signed,
                columns=[dict(column=int(j), scale=float(model.scale[j]), coefficient=float(model.coefficient[j]),
                    touches_floor=bool(model.scale[j] == 1e-8), raw_delta=float(canonical_features[worst, j]-local['features'][worst, j]),
                    weighted_delta=float(weighted[j])) for j in np.argsort(abs(weighted))[::-1][:16]]))
        del global_map, actual, canonical_features
    return dict(target_provenance='reconstructed Q2b selected checkpoint; ordered validation; internal query-local2048; no archived prediction vector loaded',
        target_sha256=array_hash(reference), validation_indices_sha256=array_hash(ids), pair_keys_sha256=array_hash(query),
        rows=len(ids), histories=result,
        state_metadata={k: metadata(getattr(model, k)) for k in ('singles', 'support', 'input_center', 'input_scale',
            'center_rows', 'centers', 'whitener', 'bandwidth', 'center', 'scale', 'coefficient', 'offset')})


def candidate(model, bank, ids, variant, dtype, threads):
    import numpy as np
    import psutil
    query = bank['pairs'][ids]
    reference = model.predict(query)
    initial = trace(model, query, variant=variant, dtype=dtype, full=True)
    legacy_stages = trace(model, query, full=True)
    baseline = initial['prediction'].copy()
    first = next((k for k in STAGES if not np.array_equal(initial[k], legacy_stages[k])), None)
    result = dict(variant=variant, dtype=dtype, threads=threads, first_unequal_stage=first,
                  canonical_sha256=array_hash(baseline), compatibility=difference(baseline, reference, ids), controls={})
    del initial, legacy_stages
    full_controls = dtype == 'float64' and threads == 1
    controls = [(f'chunk-{s}', (s,), query, None, 'C') for s in
                ((1, 2, 63, 2047, 2048, 2049, len(ids)) if full_controls else (1, 2048, len(ids)))]
    controls.append(('irregular', (1, 63, 2049, 2, 2047), query, None, 'C'))
    if threads == 2:
        controls = [x for x in controls if x[0] in ('chunk-2048', f'chunk-{len(ids)}', 'irregular')]
    if full_controls:
        permutation = np.random.default_rng(41).permutation(len(ids))
        reverse = np.arange(len(ids))[::-1]
        controls += [('permuted', (2048,), query[permutation], np.argsort(permutation), 'C'),
                     ('reverse', (2048,), query[reverse], reverse, 'C'),
                     ('orientation', (2048,), query[:, ::-1], None, 'C'),
                     ('F-layout', (2048,), np.asfortranarray(query), None, 'F'),
                     ('noncontiguous', (2048,), np.repeat(query, 2, axis=0)[::2], None, 'strided')]
    process = psutil.Process()
    for name, sizes, current, inverse, layout in controls:
        began, cpu = time.perf_counter(), time.process_time()
        before_rss = process.memory_info().rss
        output = trace(model, current, sizes, variant, dtype, layout)['prediction']
        if inverse is not None:
            output = output[inverse]
        result['controls'][name] = dict(self_invariance=difference(output, baseline, ids),
            compatibility=difference(output, reference, ids), sha256=array_hash(output),
            wall_seconds=time.perf_counter()-began, cpu_seconds=time.process_time()-cpu,
            rss_before_bytes=before_rss, rss_after_bytes=process.memory_info().rss,
            os_process_peak_bytes=getattr(process.memory_info(), 'peak_wset', None))
    if full_controls:
        worst = int(np.argmax(abs(baseline-reference)))
        repeated = np.repeat(query[worst:worst+1], 2049, axis=0)
        repeated_output = trace(model, repeated, variant=variant, dtype=dtype)['prediction']
        result['repeated_boundary'] = dict(row=worst, bank_index=int(ids[worst]),
            self_invariance=difference(repeated_output, np.full(2049, baseline[worst], dtype=baseline.dtype)),
            compatibility=difference(repeated_output, np.full(2049, reference[worst])))
    result['all_self_exact'] = all(v['self_invariance']['exact'] for v in result['controls'].values()) and result.get('repeated_boundary', {}).get('self_invariance', {}).get('exact', True)
    result['all_self_within_atol'] = all(v['self_invariance']['within_atol'] for v in result['controls'].values()) and result.get('repeated_boundary', {}).get('self_invariance', {}).get('within_atol', True)
    return result


def check_binding(a):
    freeze, auth, binding = read(a.freeze), read(a.authorization), read(a.binding)
    if digest(a.freeze) != auth['freeze_sha256'] or binding['freeze_sha256'] != digest(a.freeze):
        raise ValueError('Freeze/authorization drift')
    if not auth['scope']['saved_state_no_fit_validation_arithmetic'] or not auth['A_numerical_slot_authorized']:
        raise ValueError('A numerical slot not authorized')
    for key in ('biological_fitting', 'new_biological_scoring', 'reserved_response_access', 'reserved_response_header_access'):
        if auth['scope'][key]:
            raise ValueError('Unexpected broadened scope')
    for path, expected in binding['files'].items():
        if digest(path) != expected:
            raise ValueError('Bound input/code drift: '+path)
    return freeze, binding


def worker(a):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(a.threads)
    sys.path.insert(0, str(a.installed.resolve() if a.installed else ROOT))
    import numpy as np
    import opal
    import psutil
    from opal.flexible_singles import FlexibleSingles
    from threadpoolctl import threadpool_info, threadpool_limits
    freeze, binding = check_binding(a)
    if a.installed and not Path(opal.__file__).resolve().is_relative_to(a.installed.resolve()):
        raise ValueError('Wrong installed module')
    # Fit/prepare/solve are prohibited even if an accidentally imported path calls them.
    def forbidden(*args, **kwargs):
        raise RuntimeError('No fit/prepare/solve in numerical experiment')
    FlexibleSingles.fit = FlexibleSingles.prepare = FlexibleSingles.solve = forbidden
    start, cpu = time.perf_counter(), time.process_time()
    output = Path(a.output)
    write(output.with_suffix('.started.json'), dict(command=sys.argv, cwd=os.getcwd(), binding_sha256=digest(a.binding),
        freeze_sha256=digest(a.freeze), implementation_sha256=digest(__file__), numerical=False))
    try:
        with threadpool_limits(a.threads):
            model, bank, ids = load_allowed(binding)
            result = history(model, bank, ids) if a.worker == 'history' else candidate(model, bank, ids, a.worker, a.dtype, a.threads)
        check_binding(a)
        modules = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                   if (name == 'opal' or name.startswith('opal.')) and getattr(module, '__file__', None)}
        if a.installed and any(not Path(p).is_relative_to(a.installed.resolve()) for p in modules.values()):
            raise ValueError('Mixed package provenance')
        result.update(schema='opal-wave5-numerical-experiment-v1', fit=False, score=False,
            allowed_bank_fields=['genes', 'pairs', 'singles'], outcome_arrays_loaded=False,
            absolute_tolerance=ATOL, relative_tolerance=0, input_code_hashes=binding['files'],
            environment=dict(python=sys.version, numpy=np.__version__, threadpools=threadpool_info(),
                modules=modules, dependency_reuse=True, dtype_mantissa_bits=int(np.finfo(np.dtype(a.dtype)).nmant)),
            resources=dict(wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
                os_peak_working_set_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', None)))
        if result['resources']['cpu_seconds'] > freeze['limits']['A']['cpu_seconds_per_diagnostic']:
            raise RuntimeError('CPU cap exceeded')
        write(output, result)
    except BaseException as error:
        write(output.with_suffix('.failure.json'), dict(error=repr(error), traceback=traceback.format_exc(),
            wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu))
        raise


def controller(a):
    import psutil
    freeze, binding = check_binding(a)
    limits = freeze['limits']['A']
    directory = Path(a.output)
    directory.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(directory).free < limits['reserve_bytes']:
        raise RuntimeError('Disk reserve')
    began, cpu = time.perf_counter(), time.process_time()
    write(directory/'started.json', dict(command=sys.argv, cwd=os.getcwd(), freeze_sha256=digest(a.freeze),
        binding_sha256=digest(a.binding), limits=limits))
    configs = [('history', 'float64', 1, 1)]
    configs += [(variant, 'float64', 1, 1) for variant in VARIANTS]
    configs += [('legacy', dtype, 1, 1) for dtype in ('float32', 'longdouble')]
    configs += [('legacy', 'float64', threads, repeat) for threads in (1, 2) for repeat in (2, 3)]
    records = []
    try:
        for variant, dtype, threads, repeat in configs:
            name = f'{variant}-{dtype}-t{threads}-r{repeat}'
            path = directory/(name+'.json')
            command = [sys.executable, str(Path(__file__).resolve()), '--freeze', str(Path(a.freeze).resolve()),
                '--authorization', str(Path(a.authorization).resolve()), '--binding', str(Path(a.binding).resolve()),
                '--output', str(path.resolve()), '--worker', variant, '--dtype', dtype, '--threads', str(threads)]
            if a.installed:
                command += ['--installed', str(a.installed.resolve())]
            start = time.perf_counter()
            peak = 0
            killed = None
            with (directory/(name+'.stdout.txt')).open('x') as stdout, (directory/(name+'.stderr.txt')).open('x') as stderr:
                child = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=os.getcwd())
                monitor = psutil.Process(child.pid)
                while child.poll() is None:
                    try:
                        processes = [monitor]+monitor.children(recursive=True)
                        peak = max(peak, sum(p.memory_info().rss for p in processes if p.is_running()))
                    except psutil.Error:
                        pass
                    if peak > limits['peak_bytes']:
                        killed = 'sampled process-tree RSS cap'
                    elif time.perf_counter()-start > limits['wall_seconds_per_diagnostic']:
                        killed = 'interrupting wall timeout'
                    if killed:
                        child.kill()
                        break
                    time.sleep(.05)
                code = child.wait()
            record = dict(name=name, command=command, cwd=os.getcwd(), exit_code=code, killed=killed,
                wall_seconds=time.perf_counter()-start, sampled_peak_process_tree_rss_bytes=peak)
            records.append(record)
            write(directory/(name+'.supervisor.json'), record)
            if code or killed:
                raise RuntimeError('Stopped failed subprocess '+name)
            result = read(path)
            cumulative_cpu = sum(read(directory/(r['name']+'.json'))['resources']['cpu_seconds'] for r in records)+time.process_time()-cpu
            if time.perf_counter()-began > limits['suite_seconds'] or cumulative_cpu > limits['suite_seconds']:
                raise RuntimeError('Suite cumulative cap')
            if sum(p.stat().st_size for p in directory.iterdir()) > limits['output_bytes']:
                raise RuntimeError('Output cap')
            write(directory/(name+'.completed.json'), dict(result_sha256=digest(path), cumulative_cpu_seconds=cumulative_cpu))
        numerical = {}
        for record in records:
            result = read(directory/(record['name']+'.json'))
            numerical[record['name']] = {k: result[k] for k in
                ('histories', 'target_sha256', 'canonical_sha256', 'compatibility', 'all_self_exact', 'all_self_within_atol') if k in result}
        deterministic = {}
        for threads in (1, 2):
            one = read(directory/f'legacy-float64-t{threads}-r2.json')
            two = read(directory/f'legacy-float64-t{threads}-r3.json')
            deterministic[str(threads)] = dict(canonical_exact=one['canonical_sha256'] == two['canonical_sha256'],
                control_exact={k: one['controls'][k]['sha256'] == two['controls'][k]['sha256'] for k in one['controls']})
        write(directory/'summary.json', dict(records=records, numerical=numerical, repeat_determinism=deterministic,
            wall_seconds=time.perf_counter()-began, cumulative_cpu_seconds=cumulative_cpu,
            all_registered_diagnostics_finished=True, implementation_authorized=False))
    except BaseException as error:
        write(directory/'failure.json', dict(error=repr(error), records=records, wall_seconds=time.perf_counter()-began))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--installed', type=Path)
    parser.add_argument('--worker', choices=['history']+VARIANTS)
    parser.add_argument('--dtype', choices=['float64', 'float32', 'longdouble'], default='float64')
    parser.add_argument('--threads', type=int, choices=[1, 2], default=1)
    parser.add_argument('--synthetic-counterexamples', action='store_true')
    args = parser.parse_args()
    if args.synthetic_counterexamples:
        synthetic_counterexamples(args)
    elif args.worker:
        worker(args)
    else:
        controller(args)


def synthetic_counterexamples(a):
    """C's independent C_initial fixed constructions, with no biological input."""
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    import math
    import numpy as np
    import psutil
    from fractions import Fraction
    freeze, auth, binding = read(a.freeze), read(a.authorization), read(a.binding)
    if digest(a.freeze) != auth['freeze_sha256'] or binding['freeze_sha256'] != digest(a.freeze):
        raise ValueError('C freeze drift')
    slot = ROOT/'docs/research_reset/wave5/C_numerical_slot.json'
    if not auth.get('C_numerical_slot_authorized') and not (slot.exists() and read(slot).get('authorized')):
        raise ValueError('C slot not authorized')
    for p, h in binding['files'].items():
        if digest(p) != h:
            raise ValueError('C code drift')
    began, cpu = time.perf_counter(), time.process_time()
    write(a.output.with_suffix('.started.json'), dict(command=sys.argv, cwd=os.getcwd(),
        binding_sha256=digest(a.binding), freeze_sha256=digest(a.freeze)))
    terms = np.array([2.**54, 1., -2.**54])
    left = float((terms[0]+terms[1])+terms[2])
    cancellation_first = float((terms[0]+terms[2])+terms[1])
    exact = sum(Fraction(float(x)) for x in terms)
    perturbation, scale = 2.**-40, 2.**-20
    epsilon = 2.**-26
    x = np.array([[1., 1.], [1., 1.+epsilon], [1., 1.-epsilon], [1., 1.]])
    exact_gram = [[sum(Fraction(float(r[i]))*Fraction(float(r[j])) for r in x) for j in range(2)] for i in range(2)]
    controls = {}
    for dtype in ('float64', 'float32'):
        rows = np.tile(terms.astype(dtype), (512, 1))
        base = contract_product(rows, np.ones(3, dtype=dtype), 'ascending')
        current = {}
        for size in (1, 3, 7, 64, 512):
            output = np.concatenate([contract_product(rows[s:s+size], np.ones(3, dtype=dtype), 'ascending') for s in range(0, len(rows), size)])
            current[f'partition-{size}'] = difference(output, base)
        perm = np.random.default_rng(41).permutation(512)
        for name, value, restore in (('reverse', rows[::-1], np.arange(512)[::-1]),
                                    ('permutation', rows[perm], np.argsort(perm)),
                                    ('F', np.asfortranarray(rows), np.arange(512)),
                                    ('strided', np.repeat(rows, 2, axis=1)[:, ::2], np.arange(512))):
            current[name] = difference(contract_product(value, np.ones(3, dtype=dtype), 'ascending')[restore], base)
        controls[dtype] = current
    result = dict(schema='opal-wave5-C-counterexamples-v1', fit=False, score=False,
        C1=dict(exact_sum=str(exact), left_associated=left, cancellation_first=cancellation_first,
            fixed_ascending=float(contract_product(terms[None, :], np.ones(3), 'ascending')[0]),
            compensated=math.fsum(terms.tolist()), longdouble_sum=str(np.sum(terms.astype(np.longdouble))),
            longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant)),
        C2=dict(raw_perturbation=perturbation, scale=scale, amplified=perturbation/scale,
            unit_scale=perturbation, amplification_factor=1/scale),
        C3=dict(epsilon=epsilon, computed_gram=(x.T@x).tolist(),
            exact_gram=[[str(v) for v in row] for row in exact_gram],
            erased_bottom_right_increment=str(exact_gram[1][1]-Fraction(float((x.T@x)[1, 1]))),
            fixed_coefficient_prediction=(x@np.array([1., -1.])).tolist()),
        C4=controls,
        C5=dict(first_interval=[-ATOL, ATOL], second_interval=[3e-12-ATOL, 3e-12+ATOL],
            intersection_empty=bool(ATOL < 3e-12-ATOL),
            one_target_invariant_constant_matches=True, implication='all-target incompatibility does not reject one target'),
        resources=dict(wall_seconds=time.perf_counter()-began, cpu_seconds=time.process_time()-cpu,
            peak_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)),
        environment=dict(python=sys.version, numpy=np.__version__), input_code_hashes=binding['files'])
    for p, h in binding['files'].items():
        if digest(p) != h:
            raise ValueError('C post-execution drift')
    if result['resources']['peak_bytes'] > freeze['limits']['C']['peak_bytes']:
        raise RuntimeError('C memory cap')
    write(a.output, result)


if __name__ == '__main__':
    main()
