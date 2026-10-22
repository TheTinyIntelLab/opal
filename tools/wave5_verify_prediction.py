"""No-fit acceptance of the new prediction contract against the measured candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.installed:
        sys.path.insert(0, str(args.installed.resolve()))
    import numpy as np
    import opal
    from threadpoolctl import threadpool_limits
    from opal.flexible_prediction import FixedOrderFlexiblePredictor, CONTRACT
    from opal.flexible_singles import FlexibleSingles
    from tools import wave5_numerical_matrix as old
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    binding = old.read(ROOT/'results/research-wave5/A-code-input-binding.json')
    for p, expected in binding['files'].items():
        if sha(p) != expected:
            raise ValueError('Historical code/input drift')
    if args.output.exists():
        raise FileExistsError('Preserve verification')
    checkpoint = Path('E:/opal-wave2-20261021/q2b-v1/horlbeck_K562/fold-0/nodes-1.npz')
    expected = binding['files'][str(checkpoint)]
    def forbidden(*args, **kwargs):
        raise RuntimeError('No fitting in contract verification')
    FlexibleSingles.fit = FlexibleSingles.prepare = FlexibleSingles.solve = forbidden
    with threadpool_limits(1):
        legacy, bank, ids = old.load_allowed(binding)
        query = bank['pairs'][ids]
        model = FixedOrderFlexiblePredictor.from_legacy_checkpoint(checkpoint, expected)
        output = model.predict(query)
        candidate = old.trace(legacy, query, variant='ascending_joint')['prediction']
        historical = legacy.predict(query)
        controls = {}
        for size in (1, 63, 2047, 2048, 2049, len(query)):
            value = np.concatenate([model.predict(query[s:s+size]) for s in range(0, len(query), size)])
            controls[str(size)] = bool(np.array_equal(value, output))
        perm = np.random.default_rng(41).permutation(len(query))
        controls['permutation'] = bool(np.array_equal(model.predict(query[perm])[np.argsort(perm)], output))
        with tempfile.TemporaryDirectory(prefix='opal-wave5-prediction-') as tmp:
            path = Path(tmp)/'model.npz'
            model.save(path)
            exact_replay = bool(np.array_equal(FixedOrderFlexiblePredictor.load(path, sha(path)).predict(query), output))
    modules = {n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
               if (n=='opal' or n.startswith('opal.')) and getattr(m,'__file__',None)}
    if args.installed and any(not Path(p).is_relative_to(args.installed.resolve()) for p in modules.values()):
        raise ValueError('Installed provenance mismatch')
    result = dict(contract=CONTRACT, rows=len(ids), fits=0, scores=0, outcome_arrays_loaded=False,
        exact_measured_candidate=bool(np.array_equal(candidate, output)), self_exact=controls,
        exact_serialized_replay=exact_replay, legacy_max_abs=float(np.max(abs(output-historical))),
        legacy_atol_1e12_rtol0=bool(np.all(abs(output-historical)<=1e-12)), modules=modules,
        checkpoint_sha256=expected, prediction_sha256=hashlib.sha256(output.tobytes()).hexdigest())
    if not result['exact_measured_candidate'] or not all(controls.values()) or not exact_replay:
        raise RuntimeError('Prospective prediction contract failed')
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':
    main()
