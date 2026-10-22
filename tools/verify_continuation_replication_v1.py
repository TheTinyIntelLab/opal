"""Read-only custody, split, arithmetic and checkpoint audit; never refit."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.installed_root.resolve() if args.installed_root else root))
    import numpy as np
    from opal.block_completion import ScalarCompletion, ScalarReference
    module = importlib.import_module("opal.block_completion")
    if args.installed_root:
        assert Path(module.__file__).resolve().is_relative_to(args.installed_root.resolve())
    assert digest(module.__file__) == digest(root / "src/block_completion.py")
    run = root / "results/research-continuation/replication-v1"
    freeze = json.loads((root / "results/research-continuation/replication-freeze-v1.json").read_text(encoding="utf-8"))
    for name, expected in freeze["code_and_protocol_sha256"].items():
        assert digest(root / name) == expected, name
    manifest = json.loads((run / "all-predictions-frozen.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        assert digest(run / name) == expected, name
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert len(results["cells"]) == len(manifest["cells"]) == 30
    counts = dict(cells=0, checkpoint_replays=0, independently_recomputed_metrics=0,
                  canonical_split_checks=0, exact_causal_aliases=0)
    for label, result in zip(manifest["cells"], results["cells"], strict=True):
        seed, noise, world = label.split("-", 2)
        assert int(seed) == result["seed"] and float(noise) == result["noise"] and world == result["world"]
        with np.load(run / (label + "-inputs.npz"), allow_pickle=False) as inputs:
            pairs = inputs["pairs"]
            roles = [inputs[k] for k in ("train", "validation", "test")]
            assert [len(r) for r in roles] == [885, 354, 531]
            assert np.array_equal(np.sort(np.concatenate(roles)), np.arange(1770))
            assert len(np.unique(np.sort(pairs, axis=1), axis=0)) == 1770
            assert np.all(np.bincount(pairs[roles[0]].ravel(), minlength=60))
            counts["canonical_split_checks"] += 1
            test_pairs = pairs[roles[2]]
            targets = {"observed": inputs["observed"][:, roles[2]], "noiseless": inputs["truth"][:, roles[2]]}
        reference = ScalarReference.load(run / (label + "-reference.npz"))
        baseline = reference.predict(test_pairs)
        with np.load(run / (label + "-predictions.npz"), allow_pickle=False) as predictions:
            assert np.array_equal(baseline, predictions["reference"])
            for name in ("aligned", "shuffled"):
                checkpoint = ScalarCompletion.load(run / (label + "-" + name + ".npz"))
                assert np.array_equal(baseline + checkpoint.predict(test_pairs), predictions[name])
                counts["checkpoint_replays"] += 1
            for name in predictions.files:
                assert np.isfinite(predictions[name]).all()
                for kind, target in targets.items():
                    for initiation in (0, 1):
                        prediction = predictions[name][initiation] if predictions[name].ndim == 2 else predictions[name]
                        independent_mse = float(np.sum(np.square(target[initiation] - prediction), dtype=np.float64) / len(prediction))
                        assert independent_mse == result["metrics"][name][kind][initiation]
                        counts["independently_recomputed_metrics"] += 1
        counts["cells"] += 1
    for seed in (6101, 6102, 6103):
        for noise in (0., .1):
            first = run / f"{seed}-{noise:g}-biological_persistent-inputs.npz"
            second = run / f"{seed}-{noise:g}-technical_persistent-inputs.npz"
            with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
                assert all(np.array_equal(a[k], b[k]) for k in a.files)
            counts["exact_causal_aliases"] += 1
    summary = dict(passed=True, synthetic_only=True, fits=0, biological_scores=0,
                   module_path=module.__file__, module_sha256=digest(module.__file__),
                   manifest_sha256=digest(run / "all-predictions-frozen.json"),
                   results_sha256=digest(run / "results.json"), counts=counts)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
