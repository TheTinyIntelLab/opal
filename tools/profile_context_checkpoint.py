"""Fresh-process contextual inference and total reference/state resource accounting."""
from pathlib import Path
import argparse
import sys
import time
import os
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Resource measurements are immutable")
    import numpy as np
    from threadpoolctl import threadpool_limits
    from opal.artifacts import environment, write_json
    from opal.contextual import ContextPredictor
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        predictor = ContextPredictor(args.checkpoint)
        cold = time.perf_counter() - start
        arrays = predictor.bank.arrays
        ci = len(arrays["contexts"]) - 1
        targets = np.flatnonzero(arrays["available"][ci])[:2]
        if len(targets) < 2:
            raise ValueError("Profiling context needs two observed singles")
        pair, context = arrays["targets"][targets], str(arrays["contexts"][ci])
        predictor.predict(*pair, context=context)
        times = []
        for _ in range(50):
            start = time.perf_counter()
            predictor.predict(*pair, context=context)
            times.append(time.perf_counter() - start)
        memory = None
        try:
            import psutil
            current = psutil.Process(os.getpid()).memory_info()
            memory = {"rss_bytes": current.rss, "peak_working_set_bytes": getattr(current, "peak_wset", None)}
        except ImportError:
            pass
        output = {"environment": environment(), "cold_load_seconds": cold, "warm_prediction_median_seconds": statistics.median(times),
                  "warm_prediction_p95_seconds": float(np.quantile(times, .95)), "repetitions": len(times), "threads": 1,
                  "checkpoint_bytes": sum(p.stat().st_size for p in args.checkpoint.iterdir() if p.is_file()),
                  "reference_array_bytes": sum(a.nbytes for a in arrays.values()),
                  "fitted_state_array_bytes": sum(a.nbytes for a in predictor.model.state().values()),
                  "fitted_numeric_values": sum(a.size for a in predictor.model.state().values()),
                  "memory": memory, "synthetic": predictor.bank.metadata["synthetic"],
                  "scope": "includes support metadata and observed-reference bank; this fixture does not estimate large-corpus costs"}
        write_json(args.output, output)
        print(output)


if __name__ == "__main__":
    main()
