"""Outside-checkout installed regression for the new evaluator contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve installed verification')
    sys.path.insert(0, str(args.installed.resolve()))
    import numpy as np
    from opal.evaluation_contracts import metrics_v2, strict_load_completion_v2
    from opal.block_completion import ScalarCompletion
    result = dict(unsigned_mse=metrics_v2(np.array([255, 0], np.uint8), np.array([0, 255], np.uint8))['mse'])
    assert result['unsigned_mse'] == 65025
    with tempfile.TemporaryDirectory(prefix='opal-wave5-installed-evaluation-') as tmp:
        path = Path(tmp)/'state.npz'
        data = dict(rank=2, penalty=.01, steps=25, seed=1, format_version=2,
                    normalized_upper=np.arange(6, dtype=np.float32), support=np.ones(3, bool),
                    scale=1., best_step=25, parameter_count=12)
        np.savez(path, **data)
        old = ScalarCompletion.load(path)
        loaded = strict_load_completion_v2(path, hashlib.sha256(path.read_bytes()).hexdigest())
        result['valid_checkpoint_exact'] = bool(np.array_equal(old.matrix, loaded.matrix))
        data['penalty'] = True
        np.savez(path, **data)
        try:
            strict_load_completion_v2(path)
            raise AssertionError('Boolean penalty accepted')
        except ValueError:
            result['boolean_penalty_refused'] = True
    modules = {n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
               if (n=='opal' or n.startswith('opal.')) and getattr(m,'__file__',None)}
    assert all(Path(p).is_relative_to(args.installed.resolve()) for p in modules.values())
    result.update(modules=modules, synthetic_only=True, biological_fits=0, biological_scores=0)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
